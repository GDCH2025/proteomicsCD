#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flaml import AutoML
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

def bootstrap_metrics(y_true, y_proba, n_bootstrap=1000):
    np.random.seed(100)
    auc_list = []
    n_samples = len(y_true)    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue        
        y_true_boot = y_true.iloc[indices]
        y_proba_boot = y_proba[indices]        
        auc_value = roc_auc_score(y_true_boot, y_proba_boot)
        auc_list.append(auc_value)    
    summary_metrics = {
        "auc_mean": np.mean(auc_list),
        "auc_ci_lower": np.percentile(auc_list, 2.5),
        "auc_ci_upper": np.percentile(auc_list, 97.5)
    }    
    return summary_metrics
    
data = pd.read_csv("/GDPH/proteinAI/data/revise/CDpredictor_external_validation.csv", sep=",")
column_name = ["CD",  'CHI3L1', 'CD274', 'ITGAV', 'REG1B',  'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6']
data = data[column_name]
X_test_new = data.drop(["CD"], axis=1)
y_test_new = data["CD"]

avg_result = pd.DataFrame()
model_path_template = "/GDPH/proteinAI/model1/protein_{label}_{estimator}_repeat{repeat}.joblib"
label = ["CD"]
estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree']
n_repeats = 10

for lab in label:
    for estimator in estimator_list:
        results = []        
        for repeat in range(1, n_repeats + 1):
            model_path = model_path_template.format(label=lab, estimator=estimator, repeat=repeat)
            model = joblib.load(model_path)
            y_pred_test_proba = model.predict_proba(X_test_new)[:, 1]
            test_metrics = bootstrap_metrics(y_test_new, y_pred_test_proba)

            result = {
                "data": label,
                "estimator": [estimator],
                "test_auc_mean": test_metrics["auc_mean"],
                "test_auc_ci_lower": test_metrics["auc_ci_lower"],
                "test_auc_ci_upper": test_metrics["auc_ci_upper"]
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'/GDPH/proteinAI/result1/9protein_external_validation_repeat10.csv', index=False)

        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        numerical_results_avg = results_df[numeric_columns].mean()
        test_auc_mean_std = results_df["test_auc_mean"].std()   
        numerical_results_avg["test_auc_ci_lower"] = numerical_results_avg["test_auc_mean"] - 1.96 * test_auc_mean_std
        numerical_results_avg["test_auc_ci_upper"] = numerical_results_avg["test_auc_mean"] + 1.96 * test_auc_mean_std
        numerical_results_avg_df = pd.DataFrame(numerical_results_avg).T
        text_columns = results_df.drop(columns=numeric_columns)
        text_columns_first_row = text_columns.iloc[[0]]
        avg_res = pd.concat([text_columns_first_row, numerical_results_avg_df], axis=1)
        avg_res = pd.DataFrame(avg_res)
        avg_result = pd.concat([avg_result, avg_res], ignore_index=True)

avg_result.to_csv(f'/GDPH/proteinAI/result1/9protein_external_validation_average_repeat10.csv', index=False)

