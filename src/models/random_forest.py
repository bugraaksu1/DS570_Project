{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMzIJUqgJJZH3nIWy/N84Ai",
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
        "outputId": "ee980cc2-021f-42ea-e82f-b6f1345da482"
      },
      "source": [
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "# Initializing Random Forest with 100 estimators for non-linear exploration\n",
        "rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)\n",
        "\n",
        "# Utilizing the same hold-out split from previous steps\n",
        "X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "print(\"Fitting Random Forest model...\")\n",
        "rf_regressor.fit(X_train_sub, y_train_sub)\n",
        "\n",
        "# Performance Metrics\n",
        "y_hat_rf = rf_regressor.predict(X_test_sub)\n",
        "mse_rf = mean_squared_error(y_test_sub, y_hat_rf)\n",
        "r2_rf = r2_score(y_test_sub, y_hat_rf)\n",
        "\n",
        "print(\"\\n--- Random Forest Ensemble Results ---\")\n",
        "print(f\"MSE: {mse_rf:.6f}\")\n",
        "print(f\"R2 Score: {r2_rf:.6f}\")\n",
        "\n",
        "print(\"\\n--- Model Comparison Summary ---\")\n",
        "print(f\"Linear Regression R2: 0.999900\")\n",
        "print(f\"Random Forest R2: {r2_rf:.6f}\")"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fitting Random Forest model...\n",
            "\n",
            "--- Random Forest Ensemble Results ---\n",
            "MSE: 0.000001\n",
            "R2 Score: 0.999979\n",
            "\n",
            "--- Model Comparison Summary ---\n",
            "Linear Regression R2: 0.999900\n",
            "Random Forest R2: 0.999979\n"
          ]
        }
      ]
    }
  ]
}
