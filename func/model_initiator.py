import torch
import torch.nn as nn

def get_dino_finetuned(model_type, weights_path, device='cuda'):
    """
    Load a DINOv2 model with user-selected architecture and finetuned weights.

    Parameters:
        model_type (str): 'vits14' or 'vitg14' to specify the DINOv2 architecture.
        weights_path (str): Path to the finetuned weights file.
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: Loaded and prepared DINOv2 model.
    """
    # Validate model type
    if model_type not in ['vits14', 'vitg14']:
        raise ValueError("Invalid model type. Choose 'vits14' or 'vitg14'.")

    # Load the correct base model
    print(f"Loading DINOv2 model: {model_type}...")
    model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_type}')

    # Load the finetuned weights
    print(f"Loading finetuned weights from: {weights_path}...")
    pretrained = torch.load(weights_path, map_location=torch.device(device))

    # Create a new state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            # Skip DINO head weights
            continue
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    # Adjust positional embedding shape based on model type
    if model_type == 'vits14':
        model.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    elif model_type == 'vitg14':
        model.pos_embed = nn.Parameter(torch.zeros(1, 257, 1536))

    # Load the state dict
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model
