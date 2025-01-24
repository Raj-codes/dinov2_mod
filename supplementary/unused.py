import argparse
import os
from torch.utils.data import DataLoader
from func.model_initiator import get_dino_finetuned
from func.image_processing import crop_image_to_tiles, create_image_tiles
from func.feature_extraction import extract_features_batchwise, ImageDataset, transform
from func.predict_labels import predict_labels

def main():
    
    parser = argparse.ArgumentParser(description="Run WSI to DinoV2 Pipeline")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI file")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the image patches (default: 224x224)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the tiles")
    parser.add_argument("--model_type", type=str, default="vits14", help="Vision Tranformer model(vits14 or vitg14)")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the DINOv2 pretrained weights")
    parser.add_argument("--predict_model_path", type=str, required=True, help="Path to the prediction model")
    parser.add_argument("--label_map_path", type=str, required=True, help="Path to the mapped labels to predictions")
    parser.add_argument("--output_json_path", type=str, required=True, help="Path to the embedding/predictions output")
    args = parser.parse_args()

    input_file=args.wsi_path
    tile_size=args.patch_size
    output_folder=args.output_dir
    model_type=args.model_type
    weights_path=args.model_checkpoint
    predict_model_path=args.predict_model_path
    label_map_path=args.label_map_path
    output_file=args.output_json_path
    
    print("Cropping and splitting image into patches...")
    cropped_image = crop_image_to_tiles(input_file, tile_size)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    create_image_tiles(cropped_image,input_file, output_folder, tile_size)
    
    print("Image tiles successfully created...\nLoading model, datasets and dataloaders")
    model = get_dino_finetuned(model_type, weights_path, device='cuda')
    print("Model loaded successfully!")
    
    
    test_image_paths = [os.path.join(output_folder, img) for img in os.listdir(output_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print(f"Processing {len(test_image_paths)} image tiles for feature extraction.")
    dataset = ImageDataset(test_image_paths, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=64, num_workers=0, shuffle=False)

    print("Extracting feature representations....")
    all_features = extract_features_batchwise(loader=loader, model=model, device='cuda')
    print(f"Extracted features shape: {all_features.shape}")

    print("Predicting image tiles using logistic regression...")
    results = predict_labels(predict_model_path, label_map_path, all_features, test_image_paths, output_file)
    print("Completed..")

if __name__ == "__main__":
    main()
