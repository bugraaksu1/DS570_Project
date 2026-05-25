import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_and_split_data(df, target_col='Signal_Y', test_size=0.2):
    """
    Splits the dataframe chronologically to maintain time-series integrity
    and prevent data leakage.
    """
    print(f"[INFO] Splitting dataset chronologically (target: {target_col})...")

    if target_col not in df.columns:
        available = df.columns.tolist()
        raise KeyError(f"Target column '{target_col}' not found. Available columns: {available}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # shuffle=False is critical for time-series data to preserve the order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    print(f"[SUCCESS] Split complete.")
    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    return X_train, X_test, y_train, y_test

def load_remote_data(url):
    """
    Loads the dataset from GitHub, handling specific CSV formatting issues.
    """
    print(f"[INFO] Fetching data from: {url}")

    # The dataset uses ';' as separator and ',' for decimals
    df = pd.read_csv(url, sep=';', decimal=',')

    # Clean up any trailing whitespace in column names
    df.columns = df.columns.str.strip()

    return df