import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocessing import load_remote_data, prepare_and_split_data

def run_training_pipeline(raw_data):
    """
    Splits data using the project's preprocessing module, extracts correct features,
    trains the Linear Regression model, and saves the joblib artifact.
    """
    # 1. Split data chronologically using our custom source function
    X_train, X_test, y_train, y_test = prepare_and_split_data(raw_data, target_col='Signal_Y')

    # 2. Select only signal features (exclude TimeStamp from training)
    features = [f'Signal_X{i}' for i in range(1, 12)]
    
    X_train_signals = X_train[features]
    X_test_signals = X_test[features]

    print("[INFO] Training Baseline Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_signals, y_train)

    # 3. Performance Evaluation
    preds = model.predict(X_test_signals)
    print("\n--- Model Performance Metrics ---")
    print(f"Mean Squared Error: {mean_squared_error(y_test, preds):.6f}")
    print(f"R-squared Score: {r2_score(y_test, preds):.6f}")

    # 4. Persist the model artifact for dashboard usage
    os.makedirs("models", exist_ok=True)
    save_path = 'models/linear_model.joblib'
    joblib.dump(model, save_path)
    print(f"\n[SUCCESS] Model saved to: {save_path}")

    return model

if __name__ == "__main__":
    URL = "https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/data/Finalized_Dataset.csv"
    try:
        dataset = load_remote_data(URL)
        run_training_pipeline(dataset)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")