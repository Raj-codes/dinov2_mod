import argparse
import json
import os
from torch.utils.data import DataLoader
from func.model_initiator import get_dino_finetuned
from func.image_processing import crop_image_to_tiles, create_image_tiles
from func.feature_extraction import extract_features_batchwise, ImageDataset, transform
from func.predict_labels import predict_labels

def main():
    
    parser = argparse.ArgumentParser(description="Run WSI to DinoV2 Pipeline")
    
    # Add argument for config file path
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file")
    
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Extract arguments 
    input_file = config["wsi_path"]
    tile_size = config["patch_size"]
    output_folder = config["output_dir"]
    model_type = config["model_type"]
    weights_path = config["model_checkpoint"]
    predict_model_path = config["predict_model_path"]
    label_map_path = config["label_map_path"]
    output_file = config["output_json_path"]
    batch_size = config["model_batch_size"]  

    print("Cropping and splitting image into patches...")
    cropped_image = crop_image_to_tiles(input_file, tile_size)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    create_image_tiles(cropped_image, input_file, output_folder, tile_size)
    
    print("Image tiles successfully created...\nLoading model, datasets and dataloaders")
    model = get_dino_finetuned(model_type, weights_path, device='cuda')
    print("Model loaded successfully!")
    
    test_image_paths = [os.path.join(output_folder, img) for img in os.listdir(output_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print(f"Processing {len(test_image_paths)} image tiles for feature extraction.")
    dataset = ImageDataset(image_paths=test_image_paths, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    print("Extracting feature representations....")
    all_features = extract_features_batchwise(loader=loader, model=model, device='cuda')
    print(f"Extracted features shape: {all_features.shape}")

    print("Predicting image tiles using logistic regression...")
    results = predict_labels(predict_model_path, label_map_path, all_features, test_image_paths, output_file)
    print("Completed..")

if __name__ == "__main__":
    main()
