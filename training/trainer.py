import os
import json
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from training.losses import CompositePancreasLoss
from training.metrics import evaluate_pancreas_and_tumor
from evaluation.sliding_window_infer import predict_full_volume

class PancreasTrainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"])
        )
        
        epochs = config["training"]["epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        
        _amp_enabled = config["training"].get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=_amp_enabled)
        self.criterion = CompositePancreasLoss(
            num_classes=config["model"]["num_classes"],
            class_weights=config["loss"]["class_weights"],
            dice_weight=config["loss"]["dice_weight"],
            ce_weight=config["loss"]["ce_weight"],
            focal_gamma=config["loss"]["focal_gamma"]
        )
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        self.log_dir = config["training"].get("log_dir", "./logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_tumor_dice = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Train]")
        use_amp = self.config["training"].get("amp", True) and self.device.type == "cuda"
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                seg_logits, loc_heatmap, _ = self.model(images)
                loss, loss_dict = self.criterion(seg_logits, loc_heatmap, labels)
                
            if use_amp:
                self.scaler.scale(loss).backward()
                if self.config["training"].get("grad_clip"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config["training"].get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])
                self.optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "DiceLoss": f"{loss_dict['dice_loss']:.4f}"})
            
        avg_loss = total_loss / max(1, len(self.train_loader))
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        panc_dices = []
        tumor_dices = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = batch["image"].to(self.device)
                labels = batch["label"].squeeze(0).squeeze(0).cpu().numpy()
                pred_labels, _ = predict_full_volume(
                    self.model, images,
                    roi_size=tuple(self.config["preprocessing"]["roi_size"]),
                    sw_batch_size=self.config["inference"]["sw_batch_size"],
                    overlap=self.config["inference"]["overlap"]
                )
                metrics = evaluate_pancreas_and_tumor(pred_labels, labels)
                panc_dices.append(metrics["pancreas_dice"])
                tumor_dices.append(metrics["tumor_dice"])
                
        avg_panc_dice = float(sum(panc_dices) / max(1, len(panc_dices)))
        avg_tumor_dice = float(sum(tumor_dices) / max(1, len(tumor_dices)))
        return avg_panc_dice, avg_tumor_dice

    def fit(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config["training"]["epochs"]
            
        print(f"[*] Starting PancreasGATUNet Training for {num_epochs} Epochs...")
        history = []
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_panc_dice, val_tumor_dice = self.validate(epoch)
            self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Val/Pancreas_Dice", val_panc_dice, epoch)
            self.writer.add_scalar("Val/Tumor_Dice", val_tumor_dice, epoch)
            self.writer.add_scalar("Train/Learning_Rate", lr, epoch)
            
            print(f"--> Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Pancreas Dice: {val_panc_dice:.4f} | Val Tumor Dice: {val_tumor_dice:.4f} | LR: {lr:.6f}")
            
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_pancreas_dice": val_panc_dice,
                "val_tumor_dice": val_tumor_dice,
                "lr": lr
            }
            history.append(epoch_metrics)
            
            # Save best checkpoint
            if val_tumor_dice >= self.best_tumor_dice:
                self.best_tumor_dice = val_tumor_dice
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_tumor_dice": self.best_tumor_dice,
                    "config": self.config
                }, checkpoint_path)
                print(f"  [*] Saved new best checkpoint (Tumor Dice: {val_tumor_dice:.4f}) -> {checkpoint_path}")
                
            # Save latest checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "latest_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config
            }, latest_path)
            
        with open(os.path.join(self.log_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
            
        self.writer.close()
        print(f"[*] Training finished. Best Val Tumor Dice: {self.best_tumor_dice:.4f}")
        return history
