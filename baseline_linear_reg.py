{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNLJ9BjszfmXDLceq23EVIC",
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
        "<a href=\"https://colab.research.google.com/github/bugraaksu1/DS570_Project/blob/Development_Workspace/baseline_linear_reg.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f222f507",
        "outputId": "eb7b83f6-aea6-4909-e82e-0f1b6a881a50"
      },
      "source": [
        "import os\n",
        "import importlib\n",
        "import src.preprocessing\n",
        "\n",
        "os.makedirs('src', exist_ok=True)\n",
        "\n",
        "# Define the preprocessing module content\n",
        "preprocessing_code = \"\"\"\n",
        "import pandas as pd\n",
        "import requests\n",
        "from io import StringIO\n",
        "\n",
        "def load_remote_data(url):\n",
        "    \\\"\\\"\\\"\n",
        "    Downloads data from a remote URL and cleans formatting issues like\n",
        "    non-standard decimal separators (commas and spaces).\n",
        "    \\\"\\\"\\\"\n",
        "    print(f'[INFO] Fetching data from: {url}')\n",
        "    response = requests.get(url)\n",
        "\n",
        "    if response.status_code != 200:\n",
        "        raise ConnectionError(f'Could not retrieve data. Status: {response.status_code}')\n",
        "\n",
        "    # Load using semicolon delimiter\n",
        "    df = pd.read_csv(StringIO(response.text), sep=';')\n",
        "\n",
        "    # Sanitize numeric columns by normalizing decimal separators\n",
        "    numeric_cols = [col for col in df.columns if col != 'TimeStamp']\n",
        "    for col in numeric_cols:\n",
        "        df[col] = (df[col].astype(str)\n",
        "                   .str.replace(' ', '', regex=False)\n",
        "                   .str.replace(',', '.', regex=False)\n",
        "                   .astype(float))\n",
        "\n",
        "    print(f'[SUCCESS] Loaded dataset with {df.shape[0]} rows.')\n",
        "    return df\n",
        "\"\"\"\n",
        "\n",
        "with open('src/preprocessing.py', 'w') as f:\n",
        "    f.write(preprocessing_code)\n",
        "\n",
        "importlib.reload(src.preprocessing)\n",
        "print(\"Project structure updated and module reloaded.\")"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Project structure updated and module reloaded.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import joblib\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "from src.preprocessing import load_remote_data\n",
        "\n",
        "def run_training_pipeline(data):\n",
        "    \"\"\"\n",
        "    Handles feature selection, performs a chronological train-test split\n",
        "    to avoid data leakage, and saves the trained model artifact.\n",
        "    \"\"\"\n",
        "    features = [f'Signal_X{i}' for i in range(1, 12)]\n",
        "    X = data[features]\n",
        "    y = data['Signal_Y']\n",
        "\n",
        "    # Chronological split is vital for time-series data to maintain temporal integrity\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)\n",
        "\n",
        "    model = LinearRegression()\n",
        "    model.fit(X_train, y_train)\n",
        "\n",
        "    # Performance Evaluation\n",
        "    preds = model.predict(X_test)\n",
        "    print(\"\\n--- Model Performance Metrics ---\")\n",
        "    print(f\"Mean Squared Error: {mean_squared_error(y_test, preds):.6f}\")\n",
        "    print(f\"R-squared Score: {r2_score(y_test, preds):.6f}\")\n",
        "\n",
        "    # Persist the model\n",
        "    os.makedirs(\"models\", exist_ok=True)\n",
        "    save_path = 'models/linear_model.joblib'\n",
        "    joblib.dump(model, save_path)\n",
        "    print(f\"\\n[SUCCESS] Model saved to: {save_path}\")\n",
        "\n",
        "    return model\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    URL = \"https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/data/Finalized_Dataset.csv\"\n",
        "    try:\n",
        "        dataset = load_remote_data(URL)\n",
        "        run_training_pipeline(dataset)\n",
        "    except Exception as e:\n",
        "        print(f\"[ERROR] Pipeline failed: {e}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e4d0b703-80ea-482d-a937-ee3ed1c53f4f",
        "id": "pJ9lDCzdz9v3"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[INFO] Fetching data from: https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/data/Finalized_Dataset.csv\n",
            "[SUCCESS] Loaded dataset with 104840 rows.\n",
            "\n",
            "--- Model Performance Metrics ---\n",
            "Mean Squared Error: 0.000007\n",
            "R-squared Score: 0.999760\n",
            "\n",
            "[SUCCESS] Model saved to: models/linear_model.joblib\n"
          ]
        }
      ]
    }
  ]
}