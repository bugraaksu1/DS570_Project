{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMNXmuu5K3WsIiVOM3Rhvxb",
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
        "<a href=\"https://colab.research.google.com/github/bugraaksu1/DS570_Project/blob/Development_Workspace/random_forest.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2d1f25da",
        "outputId": "8b26125a-1966-4018-ac2f-6e819fca98f5"
      },
      "source": [
        "import os\n",
        "import joblib\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "\n",
        "def run_advanced_training_pipeline(data):\n",
        "    \"\"\"\n",
        "    Handles feature selection, performs a strictly chronological train-test split\n",
        "    to prevent data leakage, and trains an optimized Random Forest model.\n",
        "    \"\"\"\n",
        "    features = [f'Signal_X{i}' for i in range(1, 12)]\n",
        "    X = data[features]\n",
        "    y = data['Signal_Y']\n",
        "\n",
        "    print(f\"[INFO] Advanced Model - Selected Features: {features}\")\n",
        "\n",
        "    # Chronological split: test_size=0.2, shuffle=False\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)\n",
        "\n",
        "    print(\"[INFO] Training Random Forest Regressor...\")\n",
        "    model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)\n",
        "    model.fit(X_train, y_train)\n",
        "\n",
        "    preds = model.predict(X_test)\n",
        "    mse = mean_squared_error(y_test, preds)\n",
        "    r2 = r2_score(y_test, preds)\n",
        "\n",
        "    print(\"\\n--- Random Forest Performance ---\")\n",
        "    print(f\"MSE: {mse:.6f}\")\n",
        "    print(f\"R-squared Score: {r2:.6f}\")\n",
        "\n",
        "    os.makedirs(\"models\", exist_ok=True)\n",
        "    save_path = 'models/advanced_model.joblib'\n",
        "    joblib.dump(model, save_path)\n",
        "    print(f\"\\n[SUCCESS] Advanced model saved to: {save_path}\")\n",
        "\n",
        "    return model\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    URL = \"https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/data/Finalized_Dataset.csv\"\n",
        "    try:\n",
        "        # Updated to handle both semicolon separator and comma decimal separator\n",
        "        dataset = pd.read_csv(URL, sep=';', decimal=',')\n",
        "        run_advanced_training_pipeline(dataset)\n",
        "    except Exception as e:\n",
        "        print(f\"[ERROR] Advanced tree pipeline failed: {e}\")"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[INFO] Advanced Model - Selected Features: ['Signal_X1', 'Signal_X2', 'Signal_X3', 'Signal_X4', 'Signal_X5', 'Signal_X6', 'Signal_X7', 'Signal_X8', 'Signal_X9', 'Signal_X10', 'Signal_X11']\n",
            "[INFO] Training Random Forest Regressor...\n",
            "\n",
            "--- Random Forest Performance ---\n",
            "MSE: 0.000006\n",
            "R-squared Score: 0.999807\n",
            "\n",
            "[SUCCESS] Advanced model saved to: models/advanced_model.joblib\n"
          ]
        }
      ]
    }
  ]
}