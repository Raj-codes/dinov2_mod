import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the ImageDataset to dynamically load images
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform if transform else transforms.ToTensor()  # Default to ToTensor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Feature extraction function
def extract_features_batchwise(loader, model, device):
    """
    Extract features dynamically batch-by-batch, making it memory efficient.

    Args:
        loader (DataLoader): DataLoader for images.
        model (torch.nn.Module): Model for feature extraction.
        device (torch.device): Device to run the model on.

    Returns:
        np.ndarray: Extracted features for all images in the loader.
    """
    all_features = []
    print("Extracting features from images...")

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch_idx, images in enumerate(tqdm(loader, desc="Processing batches")):
            images = images.to(device)  
            outputs = model(images)  
            all_features.append(outputs.cpu().numpy())  # Move outputs to CPU and convert to NumPy

            # Free memory
            del images, outputs
            torch.cuda.empty_cache()

    print("Feature extraction complete.")
    all_features = np.vstack(all_features)  # Combine all features into a single array
    return all_features
