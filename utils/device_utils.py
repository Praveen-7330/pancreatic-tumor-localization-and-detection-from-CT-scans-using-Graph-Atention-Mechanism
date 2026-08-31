def get_device():
    """Return the best available torch device.
    - CUDA if available
    - MPS (Apple Silicon) if available
    - CPU fallback
    """
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
