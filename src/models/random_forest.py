import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocessing import load_remote_data, prepare_and_split_data

def run_advanced_training_pipeline(raw_data):
    """
    Handles feature selection, performs a strictly chronological train-test split
    using the project preprocessing module, and trains an optimized Random Forest model.
    """
    # 1. Split data chronologically using our custom source function
    X_train, X_test, y_train, y_test = prepare_and_split_data(raw_data, target_col='Signal_Y')

    # 2. Select only signal features (exclude TimeStamp from training)
    features = [f'Signal_X{i}' for i in range(1, 12)]
    
    X_train_signals = X_train[features]
    X_test_signals = X_test[features]

    print(f"[INFO] Advanced Model - Selected Features: {features}")
    print("[INFO] Training Random Forest Regressor...")

    # 3. Dedicated hyperparameters matching your Colab experiment
    model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train_signals, y_train)

    # 4. Performance Evaluation
    preds = model.predict(X_test_signals)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n--- Random Forest Performance ---")
    print(f"MSE: {mse:.6f}")
    print(f"R-squared Score: {r2:.6f}")

    # 5. Persist the model artifact for dashboard usage
    os.makedirs("models", exist_ok=True)
    save_path = 'models/advanced_model.joblib'
    joblib.dump(model, save_path)
    print(f"\n[SUCCESS] Advanced model saved to: {save_path}")

    return model

if __name__ == "__main__":
    URL = "https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/data/Finalized_Dataset.csv"
    try:
        dataset = load_remote_data(URL)
        run_advanced_training_pipeline(dataset)
    except Exception as e:
        print(f"[ERROR] Advanced tree pipeline failed: {e}")