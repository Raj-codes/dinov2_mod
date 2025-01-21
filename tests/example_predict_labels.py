from predict_labels import predict_labels
import torch

# Paths to required files
model_path = 'logistic_regression_model.joblib'
label_map_path = 'label_map.json'
output_file = 'predictions.json'

# Example test data
test_tensor_feats = torch.rand((10, 128))  # Replace with actual test features
test_image_paths = [f"image_{i}.png" for i in range(10)]  # Replace with actual paths

# Get predictions and save results
results, saved_file = predict_labels(model_path, label_map_path, test_tensor_feats, test_image_paths, output_file)

print("Results:", results)
print("Saved JSON file:", saved_file)
