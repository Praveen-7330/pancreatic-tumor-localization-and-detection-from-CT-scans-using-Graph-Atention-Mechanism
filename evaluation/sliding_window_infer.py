import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

def predict_full_volume(model, image_tensor, roi_size=(96, 96, 96), sw_batch_size=2, overlap=0.5):
    model.eval()
    with torch.no_grad():
        def predictor(patch):
            seg_logits, _, _ = model(patch)
            return seg_logits
        val_outputs = sliding_window_inference(
            inputs=image_tensor, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=predictor, overlap=overlap, mode="gaussian"
        )
        pred_probs = F.softmax(val_outputs, dim=1)
        pred_labels = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()
    return pred_labels, pred_probs.squeeze(0).cpu().numpy()
