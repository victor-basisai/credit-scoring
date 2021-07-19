"""
Python script for training a model version
"""
# Core Packages
import os
import json

# Third Party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn import metrics
import utils.credit as utils

# Bedrock
from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
import pickle
import logging

# ---------------------------------
# Constants
# ---------------------------------

OUTPUT_MODEL_PATH = "/artefact/model.pkl"
FEATURE_COLS_PATH = "/artefact/feature_cols.pkl"

CONFIG_FAI = {
    'SEX': {
        'privileged_attribute_values': [1],
        'privileged_group_name': 'Male',  # privileged group name corresponding to values=[1]
        'unprivileged_attribute_values': [2],
        'unprivileged_group_name': 'Female',  # unprivileged group name corresponding to values=[0]
    }
}

# Retrieving environmental params from Bedrock HCL
DATA_DIR_LOCAL = os.getenv("DATA_DIR_LOCAL") 
SEED = int(os.getenv("SEED"))
TH = float(os.getenv("TH"))
LR_REGULARIZER = float(os.getenv("LR_REGULARIZER")) 
RF_N_ESTIMATORS = float(os.getenv("RF_N_ESTIMATORS")) 
CB_ITERATIONS = float(os.getenv("CB_ITERATIONS")) 

# ---------------------------------
# Bedrock functions
# ---------------------------------

def compute_log_metrics(model, x_train, 
                        x_test, y_test, 
                        best_th=0.5,
                        model_name="tree_model", 
                        model_type=ModelTypes.TREE):
    """Compute and log metrics."""
    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = np.where(test_prob > best_th, 1, 0)

    acc = metrics.accuracy_score(y_test, test_pred)
    precision = metrics.precision_score(y_test, test_pred)
    recall = metrics.recall_score(y_test, test_pred)
    f1_score = metrics.f1_score(y_test, test_pred)
    roc_auc = metrics.roc_auc_score(y_test, test_prob)
    avg_prc = metrics.average_precision_score(y_test, test_prob)
    print("Evaluation\n"
          f"  Accuracy          = {acc:.4f}\n"
          f"  Precision         = {precision:.4f}\n"
          f"  Recall            = {recall:.4f}\n"
          f"  F1 score          = {f1_score:.4f}\n"
          f"  ROC AUC           = {roc_auc:.4f}\n"
          f"  Average precision = {avg_prc:.4f}")

    # --- Bedrock-native Integrations ---
    # Bedrock Logger: captures model metrics
    bedrock = BedrockApi(logging.getLogger(__name__))

    # Log into a chart
    bedrock.log_chart_data(y_test.astype(int).tolist(),
                           test_prob.flatten().tolist())

    # Log key-value pairs
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)

    # Bedrock Model Analyzer: generates model explainability and fairness metrics
    # Analyzer (optional): generate explainability metrics
    analyzer = ModelAnalyzer(model[1], model_name=model_name, model_type=model_type)\
                    .train_features(x_train)\
                    .test_features(x_test)
    
    # Analyzer (optional): generate fairness metrics
    analyzer.fairness_config(CONFIG_FAI)\
        .test_labels(y_test)\
        .test_inference(test_pred)
    
    # Return the 4 metrics
    return analyzer.analyze()


def main():
    # Extraneous columns (as might be determined through feature selection)
    drop_cols = ['ID']

    # --- Data ETL ---
    # Load into pandas dataframes
    # x_<name> : features
    # y_<name> : labels
    x_train, y_train = utils.load_dataset(os.path.join(DATA_DIR_LOCAL, 'creditdata_train_v2.csv'), drop_columns=drop_cols)
    x_test, y_test = utils.load_dataset(os.path.join(DATA_DIR_LOCAL, 'creditdata_test_v2.csv'), drop_columns=drop_cols)

    
    # --- Candidate Binary Classification Algos ---
    # MODEL 1: LOGISTIC REGRESSION
    # Use best parameters from a model selection and threshold tuning process
    # model = utils.train_log_reg_model(x_train, y_train, seed=SEED, C=LR_REGULARIZER, upsample=True, verbose=True)
    # model_name = "logreg_model"
    # model_type = ModelTypes.LINEAR

    # MODEL 2: RANDOM FOREST
    # Uses default threshold of 0.5 and model parameters
    model = utils.train_rf_model(x_train, y_train, seed=SEED, upsample=True, verbose=True)
    model_name = "randomforest_model"
    model_type = ModelTypes.TREE

    # MODEL 3: CATBOOST
    # Uses default threshold of 0.5 and model parameters
    model = utils.train_catboost_model(x_train, y_train, seed=SEED, upsample=True, verbose=True)
    model_name = "catboost_model"
    model_type = ModelTypes.TREE


    # --- Bedrock-native Integrations ---
    # Bedrock Model Analyzer: generated values
    (
        shap_values, 
        base_shap_values, 
        global_explainability, 
        fairness_metrics,
    ) = compute_log_metrics(model=model, x_train=x_train, 
                            x_test=x_test, y_test=y_test, 
                            best_th=TH,
                            model_name=model_name, model_type=model_type)

    # IMPORTANT: Saving the Model Artefact  Bedrock
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    
    # Bedrock Model Monitoring: pre-requisite for monitoring concept drift
    # Prepare the inference probabilities
    train_prob = model.predict_proba(x_train)[:, 1]
    train_pred = np.where(train_prob > TH, 1, 0)

    # This step initialises the distribution from model training     
    ModelMonitoringService.export_text(
        features=x_train.iteritems(),
        inference=train_prob.tolist(),
    )
    # --- End of Bedrock-native Integrations ---

    print("Done!")

if __name__ == "__main__":
    try:
        print("Hello Bedrock!")
        main()
    except Exception as e:
        print(e)
        print("Something went wrong...")