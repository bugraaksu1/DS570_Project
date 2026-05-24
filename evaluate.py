{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPqJ0Vvp50N26N1fxb6c9OV",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bugraaksu1/DS570_Project/blob/Development_Workspace/evaluate.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "62787b6d",
        "outputId": "7b74c247-38a7-45e4-8d4d-260d1765d10c"
      },
      "source": [
        "import os\n",
        "import joblib\n",
        "import pandas as pd\n",
        "import urllib.request\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "def fetch_remote_model(url, filename):\n",
        "    \"\"\"Attempts to download a pre-trained model from a remote repository.\"\"\"\n",
        "    if not os.path.exists('models'):\n",
        "        os.makedirs('models')\n",
        "\n",
        "    target_path = os.path.join('models', filename)\n",
        "    try:\n",
        "        print(f'[INFO] Fetching {filename} from remote storage...')\n",
        "        urllib.request.urlretrieve(url, target_path)\n",
        "        return True\n",
        "    except Exception as e:\n",
        "        print(f'[ERROR] Failed to download {filename}: {e}')\n",
        "        return False\n",
        "\n",
        "def load_and_preprocess(url):\n",
        "    \"\"\"Loads dataset and extracts features/target.\"\"\"\n",
        "    dataset = pd.read_csv(url, sep=';', decimal=',')\n",
        "    feature_cols = [f'Signal_X{i}' for i in range(1, 12)]\n",
        "    X = dataset[feature_cols]\n",
        "    y = dataset['Signal_Y']\n",
        "    return X, y\n",
        "\n",
        "def run_model_benchmarking():\n",
        "    # Configuration for data and model sources\n",
        "    DATASET_ENDPOINT = 'https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/data/Finalized_Dataset.csv'\n",
        "    MODELS_BASE_URL = 'https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/src/models/'\n",
        "\n",
        "    print('[PHASE 1] Initializing data ingestion...')\n",
        "    X, y = load_and_preprocess(DATASET_ENDPOINT)\n",
        "\n",
        "    # Time-series split (no shuffle as per signal data nature)\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)\n",
        "\n",
        "    # Define model configurations with optimized local fallbacks\n",
        "    experiments = [\n",
        "        ('Linear Regression', MODELS_BASE_URL + 'baseline_linear_reg.joblib', 'linear_model.joblib', LinearRegression()),\n",
        "        ('Random Forest', MODELS_BASE_URL + 'random_forest.joblib', 'advanced_model.joblib',\n",
        "         RandomForestRegressor(n_estimators=10, max_depth=8, n_jobs=-1, random_state=42))\n",
        "    ]\n",
        "\n",
        "    performance_metrics = []\n",
        "    print('[PHASE 2] Evaluating model performance...')\n",
        "\n",
        "    for label, remote_url, local_fname, fallback_engine in experiments:\n",
        "        is_downloaded = fetch_remote_model(remote_url, local_fname)\n",
        "        model_path = os.path.join('models', local_fname)\n",
        "\n",
        "        if is_downloaded and os.path.exists(model_path):\n",
        "            predictor = joblib.load(model_path)\n",
        "            print(f'[SUCCESS] Using production model for {label}.')\n",
        "        else:\n",
        "            print(f'[FALLBACK] {label} not found. Running localized training...')\n",
        "            predictor = fallback_engine.fit(X_train, y_train)\n",
        "            joblib.dump(predictor, model_path)\n",
        "\n",
        "        y_pred = predictor.predict(X_test)\n",
        "        performance_metrics.append({\n",
        "            'Model': label,\n",
        "            'MSE': mean_squared_error(y_test, y_pred),\n",
        "            'R2_Score': r2_score(y_test, y_pred)\n",
        "        })\n",
        "\n",
        "    if performance_metrics:\n",
        "        summary_df = pd.DataFrame(performance_metrics)\n",
        "        print('\\n' + '-'*45)\n",
        "        print(summary_df.to_string(index=False))\n",
        "        print('-'*45 + '\\n')\n",
        "\n",
        "        # Persist metrics for downstream analysis (e.g., Dashboard usage)\n",
        "        metrics_dir = 'data/metrics'\n",
        "        os.makedirs(metrics_dir, exist_ok=True)\n",
        "        summary_df.to_csv(f'{metrics_dir}/model_comparison.csv', index=False)\n",
        "        print(f'[FINISH] Metrics archived at {metrics_dir}/model_comparison.csv')\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    run_model_benchmarking()"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[PHASE 1] Initializing data ingestion...\n",
            "[PHASE 2] Evaluating model performance...\n",
            "[INFO] Fetching linear_model.joblib from remote storage...\n",
            "[ERROR] Failed to download linear_model.joblib: HTTP Error 404: Not Found\n",
            "[FALLBACK] Linear Regression not found. Running localized training...\n",
            "[INFO] Fetching advanced_model.joblib from remote storage...\n",
            "[ERROR] Failed to download advanced_model.joblib: HTTP Error 404: Not Found\n",
            "[FALLBACK] Random Forest not found. Running localized training...\n",
            "\n",
            "---------------------------------------------\n",
            "            Model      MSE  R2_Score\n",
            "Linear Regression 0.000007  0.999760\n",
            "    Random Forest 0.000006  0.999813\n",
            "---------------------------------------------\n",
            "\n",
            "[FINISH] Metrics archived at data/metrics/model_comparison.csv\n"
          ]
        }
      ]
    }
  ]
}