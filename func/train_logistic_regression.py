from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

def train_logistic_regression(train_feats, train_labels, max_iter=100, solver='lbfgs', verbose=1, save_path=None):
    """
    Train a logistic regression classifier.

    Parameters:
        train_feats (np.ndarray): Training feature matrix (NumPy array).
        train_labels (np.ndarray): Training labels (NumPy array).
        max_iter (int): Maximum number of iterations for the solver (default: 100).
        solver (str): Optimization solver to use ('lbfgs', 'saga', etc., default: 'lbfgs').
        verbose (int): Verbosity level for training progress (default: 1).
        save_path (str): Optional file path to save the trained model (default: None).

    Returns:
        LogisticRegression: Trained logistic regression model.
    """
    # Define the logistic regression classifier
    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class='multinomial',  # Multinomial for multiclass classification
        solver=solver,
        verbose=verbose
    )

    # Train the logistic regression classifier
    print("Training logistic regression...")
    clf.fit(train_feats, train_labels)
    print("Logistic regression training complete.")

    # Save the model if a save path is provided
    if save_path:
        joblib.dump(clf, save_path)
        print(f"Trained logistic regression model saved to: {save_path}")

    return clf
