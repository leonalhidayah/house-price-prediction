import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Function to train a single model
def train_model(model, X_train, y_train):
    """
    Train a given model and return the training time.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time


# Function to evaluate a model
def evaluate_model(model, X, y):
    """
    Evaluate a model using MAE, MSE, RMSE, and R2.
    """
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return mae, mse, rmse, r2


# Function to print evaluation results
def print_evaluation_results(model_name, train_metrics, test_metrics):
    """
    Print the evaluation results for both training and testing datasets.
    """
    print("=" * 100)
    print(f" {model_name} Regressor ".center(100, "="))
    print("=" * 100)

    print("Training")
    print(f"Train MAE: {train_metrics['MAE']}")
    print(f"Train MSE: {train_metrics['MSE']}")
    print(f"Train RMSE: {train_metrics['RMSE']}")
    print(f"Train R2: {train_metrics['R2']}")

    print("\nTesting")
    print(f"Test MAE: {test_metrics['MAE']}")
    print(f"Test MSE: {test_metrics['MSE']}")
    print(f"Test RMSE: {test_metrics['RMSE']}")
    print(f"Test R2: {test_metrics['R2']}")


# Main function to train and evaluate multiple models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    """
    Train and evaluate multiple models and return a summary of evaluation results.
    """
    evaluation_results = []

    for model_name, model in models.items():
        # Train the model
        trained_model, training_time = train_model(model, X_train, y_train)

        # Evaluate on training data
        train_mae, train_mse, train_rmse, train_r2 = evaluate_model(
            trained_model, X_train, y_train
        )

        # Evaluate on testing data
        test_mae, test_mse, test_rmse, test_r2 = evaluate_model(
            trained_model, X_test, y_test
        )

        # Print results
        print_evaluation_results(
            model_name,
            {"MAE": train_mae, "MSE": train_mse, "RMSE": train_rmse, "R2": train_r2},
            {"MAE": test_mae, "MSE": test_mse, "RMSE": test_rmse, "R2": test_r2},
        )

        # Store results
        evaluation_results.append(
            {
                "Model": model_name,
                "Train MAE": train_mae,
                "Test MAE": test_mae,
                "Train MSE": train_mse,
                "Test MSE": test_mse,
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "Train R2": train_r2,
                "Test R2": test_r2,
                "Training Time (seconds)": training_time,
            }
        )

    result_df = pd.DataFrame(evaluation_results)
    return result_df


def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    import joblib

    joblib.dump(model, file_path)
