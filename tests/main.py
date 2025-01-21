from feature_extraction import ImageDataset, extract_features_batchwise, transform
from torch.utils.data import DataLoader

# Paths to all images
dataroot = 'C:/Users/rrai/dinov2/WSI/NEW_tiles_128'
all_image_paths = [os.path.join(dataroot, img) for img in os.listdir(dataroot) if img.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset and dataloader
dataset = ImageDataset(all_image_paths, transform=transform)
loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)  # num_workers can be increased for faster loading

# Load your pre-trained DINO model (ensure it's defined elsewhere)
# Example: model = get_dino_finetuned(model_type='vits14', weights_path='path_to_weights.pth', device=device)

# Extract features
all_features = extract_features_batchwise(loader, model, device)
print(f"Extracted features shape: {all_features.shape}")
