from joblib import load
import json

def predict_labels(model_path, label_map_path, test_tensor_feats, test_image_paths, output_file="predictions.json"):
    """
    Predict labels for test features, save results to a JSON file, and return the results.

    Parameters:
        model_path (str): Path to the saved logistic regression model file.
        label_map_path (str): Path to the saved label map JSON file.
        test_tensor_feats (torch.Tensor): Tensor containing test features.
        test_image_paths (list): List of paths to the test images.
        output_file (str): File path to save the predictions in JSON format (default: 'predictions.json').

    Returns:
        list: Zipped list of image paths and named predictions.
        str: Path to the saved JSON file.
    """
    # Load the logistic regression model
    loaded_clf = load(model_path)
    print("Model loaded successfully.")

    # Load the label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    # Convert test features to NumPy
    test_feats = test_tensor_feats.numpy()

    # Predict with the loaded model
    predictions = loaded_clf.predict(test_feats)
    print("Predictions:", predictions)

    # Map predictions to their text labels
    predictions_named = [label_map[str(pred)] for pred in predictions]

    # Combine test image paths with predictions
    results = list(zip(test_image_paths, predictions_named))

    # Save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(dict(results), f, indent=4)
    print(f"Results saved to {output_file}.")

    return results, output_file
