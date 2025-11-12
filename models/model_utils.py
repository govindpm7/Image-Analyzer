"""
Shared utilities for models
"""
import torch

def load_model_checkpoint(model, checkpoint_path, device):
    """Safely load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


