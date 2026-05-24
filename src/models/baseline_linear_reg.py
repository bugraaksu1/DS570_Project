{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMb95vYhCg3PxDKE36foB0G",
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
      "execution_count": null,
      "metadata": {
        "id": "udz3Wsl8rH0Z"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "df2d18b0",
        "outputId": "304ef83a-3652-4b1e-9b47-e73fa27588f8"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "import joblib\n",
        "\n",
        "def execute_regression_pipeline(data):\n",
        "    \"\"\"\n",
        "    Extracts features, trains a linear model, and evaluates performance.\n",
        "    \"\"\"\n",
        "    # Feature selection based on signal input columns\n",
        "    feature_cols = [f'Signal_X{i}' for i in range(1, 12)]\n",
        "    X = data[feature_cols]\n",
        "    y = data['Signal_Y']\n",
        "\n",
        "    print(f\"Input Features: {feature_cols}\")\n",
        "\n",
        "    # Hold-out validation split (80/20)\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "    # Model initialization and fitting\n",
        "    regressor = LinearRegression()\n",
        "    regressor.fit(X_train, y_train)\n",
        "\n",
        "    # Inference and Metric calculation\n",
        "    predictions = regressor.predict(X_test)\n",
        "    mse = mean_squared_error(y_test, predictions)\n",
        "    r2 = r2_score(y_test, predictions)\n",
        "\n",
        "    print(\"\\n--- Baseline Model Metrics ---\")\n",
        "    print(f\"MSE: {mse:.6f}\")\n",
        "    print(f\"R-squared: {r2:.6f}\")\n",
        "\n",
        "    # Serialize model for deployment\n",
        "    artifact_path = 'linear_model.joblib'\n",
        "    joblib.dump(regressor, artifact_path)\n",
        "    print(f\"\\nModel artifact persisted to: {artifact_path}\")\n",
        "\n",
        "    return regressor\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    if 'df' in locals() and df is not None:\n",
        "        model = execute_regression_pipeline(df)\n",
        "    else:\n",
        "        print(\"Environment Error: Dataframe 'df' is not defined.\")"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_914/2619105173.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodel_selection\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtrain_test_split\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlinear_model\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mLinearRegression\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmetrics\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmean_squared_error\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mr2_score\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mjoblib\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     71\u001b[0m     \u001b[0m_distributor_init\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     72\u001b[0m )\n\u001b[0;32m---> 73\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mbase\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mclone\u001b[0m  \u001b[0;31m# noqa: E402\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     74\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_show_versions\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mshow_versions\u001b[0m  \u001b[0;31m# noqa: E402\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     75\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/base.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0m_config\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mconfig_context\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mget_config\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mexceptions\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mInconsistentVersionWarning\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 19\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_estimator_html_repr\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0m_HTMLDocumentationLinkMixin\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mestimator_html_repr\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     20\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_metadata_requests\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0m_MetadataRequester\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_routing_enabled\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     21\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_param_validation\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mvalidate_parameter_constraints\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/utils/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     13\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0m_joblib\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmetadata_routing\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0m_bunch\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mBunch\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 15\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0m_chunking\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgen_batches\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgen_even_slices\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     16\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0m_estimator_html_repr\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mestimator_html_repr\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/utils/_chunking.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_config\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mget_config\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0m_param_validation\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mInterval\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalidate_params\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/utils/_param_validation.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_config\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mconfig_context\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mget_config\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 17\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mvalidation\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0m_is_arraylike_not_scalar\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     19\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mget_config\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0m_get_config\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexceptions\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mDataConversionWarning\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mNotFittedError\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mPositiveSpectrumWarning\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 21\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_array_api\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0m_asarray_with_order\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_is_numpy_namespace\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mget_namespace\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     22\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdeprecation\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0m_deprecate_force_all_finite\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     23\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfixes\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mComplexWarning\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_preserve_dia_indices_dtype\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/utils/_array_api.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_config\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mget_config\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 17\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mfixes\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mparse_version\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m \u001b[0m_NUMPY_NAMESPACE_NAMES\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m\"numpy\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"array_api_compat.numpy\"\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/utils/fixes.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m     \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m \u001b[0;32mexcept\u001b[0m \u001b[0mImportError\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m     \u001b[0mpd\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/pandas/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     47\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconfig_init\u001b[0m  \u001b[0;31m# pyright: ignore[reportUnusedImport] # noqa: F401\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     48\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 49\u001b[0;31m from pandas.core.api import (\n\u001b[0m\u001b[1;32m     50\u001b[0m     \u001b[0;31m# dtype\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     51\u001b[0m     \u001b[0mArrowDtype\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/pandas/core/api.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     26\u001b[0m     \u001b[0mvalue_counts\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     27\u001b[0m )\n\u001b[0;32m---> 28\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marrays\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mCategorical\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     29\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marrays\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mboolean\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mBooleanDtype\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     30\u001b[0m from pandas.core.arrays.floating import (\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/pandas/core/arrays/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0mExtensionScalarOpsMixin\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m )\n\u001b[0;32m----> 7\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marrays\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mboolean\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mBooleanArray\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marrays\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcategorical\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mCategorical\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marrays\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdatetimes\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mDatetimeArray\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.12/importlib/_bootstrap.py\u001b[0m in \u001b[0;36m_find_and_load\u001b[0;34m(name, import_)\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.12/importlib/_bootstrap.py\u001b[0m in \u001b[0;36m_find_and_load_unlocked\u001b[0;34m(name, import_)\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.12/importlib/_bootstrap.py\u001b[0m in \u001b[0;36m_find_spec\u001b[0;34m(name, path, target)\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.12/importlib/_bootstrap_external.py\u001b[0m in \u001b[0;36mfind_spec\u001b[0;34m(cls, fullname, path, target)\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.12/importlib/_bootstrap_external.py\u001b[0m in \u001b[0;36m_get_spec\u001b[0;34m(cls, fullname, path, target)\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.12/importlib/_bootstrap_external.py\u001b[0m in \u001b[0;36m_path_importer_cache\u001b[0;34m(cls, path)\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ]
    }
  ]
}
