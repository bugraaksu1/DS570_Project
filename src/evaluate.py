{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPApyPY+15JwV262w/BhJ7u",
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
        "outputId": "fe337a10-516b-4ab0-da22-e37713a8c26c"
      },
      "source": [
        "import os\n",
        "import joblib\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "import urllib.request\n",
        "\n",
        "def download_model(url, filename):\n",
        "    if not os.path.exists('models'):\n",
        "        os.makedirs('models')\n",
        "    path = os.path.join('models', filename)\n",
        "    try:\n",
        "        print(f'[INFO] Downloading {filename}...')\n",
        "        urllib.request.urlretrieve(url, path)\n",
        "        return True\n",
        "    except Exception as e:\n",
        "        print(f'[WARNING] GitHub download failed for {filename}: {e}')\n",
        "        return False\n",
        "\n",
        "def load_data(url):\n",
        "    df = pd.read_csv(url, sep=';', decimal=',')\n",
        "    features = [f'Signal_X{i}' for i in range(1, 12)]\n",
        "    X = df[features]\n",
        "    y = df['Signal_Y']\n",
        "    return X, y\n",
        "\n",
        "def run_evaluation():\n",
        "    DATA_URL = 'https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/data/Finalized_Dataset.csv'\n",
        "    BASE_GITHUB_PATH = 'https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/src/models/'\n",
        "\n",
        "    print('[1/3] Loading data from cloud...')\n",
        "    X, y = load_data(DATA_URL)\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)\n",
        "\n",
        "    # Fallback modellerine n_jobs=-1 ve hız optimizasyonları eklendi\n",
        "    model_configs = [\n",
        "        ('Linear Regression', BASE_GITHUB_PATH + 'baseline_linear_reg.joblib', 'linear_model.joblib', LinearRegression()),\n",
        "        ('Random Forest', BASE_GITHUB_PATH + 'random_forest.joblib', 'advanced_model.joblib', RandomForestRegressor(n_estimators=10, max_depth=8, n_jobs=-1, random_state=42))\n",
        "    ]\n",
        "\n",
        "    results = []\n",
        "    print('[2/3] Evaluating models...')\n",
        "    for name, url, fname, fallback_model in model_configs:\n",
        "        success = download_model(url, fname)\n",
        "        path = os.path.join('models', fname)\n",
        "\n",
        "        if success and os.path.exists(path):\n",
        "            model = joblib.load(path)\n",
        "            print(f'[OK] {name} loaded successfully from remote store.')\n",
        "        else:\n",
        "            print(f'[INFO] {name} download failed. Training an optimized fallback local version...')\n",
        "            model = fallback_model.fit(X_train, y_train)\n",
        "            joblib.dump(model, path)\n",
        "\n",
        "        preds = model.predict(X_test)\n",
        "        results.append({\n",
        "            'Model': name,\n",
        "            'MSE': mean_squared_error(y_test, preds),\n",
        "            'R2': r2_score(y_test, preds)\n",
        "        })\n",
        "\n",
        "    if results:\n",
        "        df_results = pd.DataFrame(results)\n",
        "        print('\\n' + '='*40)\n",
        "        print(df_results.to_string(index=False))\n",
        "        print('='*40 + '\\n')\n",
        "\n",
        "        # [CRUCIAL] Dashboard'un okuyabilmesi için metrikleri CSV olarak saklıyoruz\n",
        "        os.makedirs('data/metrics', exist_ok=True)\n",
        "        df_results.to_csv('data/metrics/model_comparison.csv', index=False)\n",
        "        print(\"[SUCCESS] Evaluation metrics persisted to data/metrics/model_comparison.csv\")\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    run_evaluation()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1/3] Loading data from cloud...\n",
            "[2/3] Evaluating models...\n",
            "[INFO] Downloading linear_model.joblib...\n",
            "[WARNING] GitHub download failed for linear_model.joblib: HTTP Error 404: Not Found\n",
            "[INFO] Linear Regression download failed. Training an optimized fallback local version...\n",
            "[INFO] Downloading advanced_model.joblib...\n",
            "[WARNING] GitHub download failed for advanced_model.joblib: HTTP Error 404: Not Found\n",
            "[INFO] Random Forest download failed. Training an optimized fallback local version...\n",
            "\n",
            "========================================\n",
            "            Model      MSE       R2\n",
            "Linear Regression 0.000007 0.999760\n",
            "    Random Forest 0.000006 0.999813\n",
            "========================================\n",
            "\n",
            "[SUCCESS] Evaluation metrics persisted to data/metrics/model_comparison.csv\n"
          ]
        }
      ]
    }
  ]
}
