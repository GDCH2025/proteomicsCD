#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


def Multimachine_with_cv(data, estimator, label, iter, n_repeats, model_name, model):
    column_name = [label, "Region_code"]+ model 
    data = data[column_name]
    train_data = data[data['Region_code'] == 0]
    test_data = data[data['Region_code'] == 1]
    X_train = train_data.drop([label, 'Region_code'], axis=1)
    y_train = train_data[label]
    X_test = test_data.drop([label, 'Region_code'], axis=1)
    y_test = test_data[label]

    classes = np.unique(y_train)
    sklearn_weights = compute_class_weight(class_weight = 'balanced',
                                                    classes = classes,
                                                    y = y_train)    
    y_weight = y_train.copy()
    y_weight = np.where (y_weight, sklearn_weights[1], sklearn_weights[0])
        
    results = []

    for repeat in range(n_repeats):
        automl = AutoML()
        settings = {
            "time_budget": 100000, 
            "eval_method": "cv",
            "n_splits": 5,
            "metric": 'roc_auc',
            "estimator_list": [estimator],
            "task": 'classification',
            "seed": 7654321 + repeat,
            "n_jobs": 32,
            "early_stop": True,
            "max_iter": iter,
            'sample_weight': y_weight 
        }
    
        automl.fit(X_train=X_train, y_train=y_train, **settings)
        model_filename = f"/GDCH/proteinAI/model1/{model_name}_{label}_{estimator}_repeat{repeat + 1}.joblib"
        joblib.dump(automl, model_filename)
        
        y_pred_test_proba = automl.predict_proba(X_test)[:, 1]
        y_pred_train_proba = automl.predict_proba(X_train)[:, 1]
        test_metrics = bootstrap_metrics(y_test, y_pred_test_proba)
        train_metrics = bootstrap_metrics(y_train, y_pred_train_proba)

        dic = {
            "data": label,
            "estimator": [estimator],
            "iter": iter,
            "auc_val": 1 - automl.best_result["val_loss"],
            "train_cases": X_train.shape[0],
            "test_cases": X_test.shape[0],
            "total_cases": data.shape[0],
            "event_cases": (data[data[label] == 1]).shape[0],
            "test_auc_mean": test_metrics["auc_mean"],
            "test_auc_ci_lower": test_metrics["auc_ci_lower"],
            "test_auc_ci_upper": test_metrics["auc_ci_upper"],
            "train_auc_mean": train_metrics["auc_mean"],
            "train_auc_ci_lower": train_metrics["auc_ci_lower"],
            "train_auc_ci_upper": train_metrics["auc_ci_upper"]
        }
        results.append(dic)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'/GDCH/proteinAI/result1/{model_name}_{label}_{estimator}_repeat10.csv', index=False)

    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    numerical_results_avg = results_df[numeric_columns].mean()
    test_auc_mean_std = results_df["test_auc_mean"].std()
    train_auc_mean_std = results_df["train_auc_mean"].std()    
    numerical_results_avg["test_auc_ci_lower"] = numerical_results_avg["test_auc_mean"] - 1.96 * test_auc_mean_std
    numerical_results_avg["test_auc_ci_upper"] = numerical_results_avg["test_auc_mean"] + 1.96 * test_auc_mean_std
    numerical_results_avg["train_auc_ci_lower"] = numerical_results_avg["train_auc_mean"] - 1.96 * train_auc_mean_std
    numerical_results_avg["train_auc_ci_upper"] = numerical_results_avg["train_auc_mean"] + 1.96 * train_auc_mean_std
    numerical_results_avg_df = pd.DataFrame(numerical_results_avg).T
    text_columns = results_df.drop(columns=numeric_columns)
    text_columns_first_row = text_columns.iloc[[0]]
    results_avg_full = pd.concat([text_columns_first_row, numerical_results_avg_df], axis=1)
    results_avg_full = pd.DataFrame(results_avg_full)

    return results_avg_full

avg_result = pd.DataFrame()

data = pd.read_csv('/GDCH/proteinAI/data/revise/CDallpredictor.csv', sep=",")
iter = 100
label = ["CD_10","CD_11","CD_12","CD_13","CD_14","CD_15","CD_16"]
estimator_list = ['lgbm', 'xgboost','rf', 'extra_tree']
n_repeats=10

protein = ['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6']
demo=['age']
serum=["LP", "PLC", "S2", "NLR", "PLR", "RDW", "Alb", "Creatinine", "TC", "LDL_C", "HDL_C"]
prs=['CD_prs']
protein_demo = ['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6','age']
protein_serum = ['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6',"LP", "PLC", "S2", "NLR", "PLR", "RDW", "Alb", "Creatinine", "TC", "LDL_C", "HDL_C"]
protein_prs = ['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6', 'CD_prs']
protein_demo_serum = ['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6','age',"LP", "PLC", "S2", "NLR", "PLR", "RDW", "Alb", "Creatinine", "TC", "LDL_C", "HDL_C"]
protein_demo_prs = ['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6','age', 'CD_prs']
protein_serum_prs = ['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6',"LP", "PLC", "S2", "NLR", "PLR", "RDW", "Alb", "Creatinine", "TC", "LDL_C", "HDL_C", 'CD_prs']
combine=['CD274', 'CHI3L1', 'REG1B', 'ITGAV', 'PRSS8', 'ITGA11', 'GDF15', 'DEFA1_DEFA1B', 'IL6','age', "LP", "PLC", "S2", "NLR", "PLR", "RDW", "Alb", "Creatinine", "TC", "LDL_C", "HDL_C",'CD_prs']

models = [
    ('protein', protein),
    ('demo', demo),
    ('serum', serum),
    ('prs', prs),
    ('protein_demo', protein_demo),
    ('protein_serum', protein_serum),
    ('protein_prs', protein_prs),
    ('protein_demo_serum', protein_demo_serum),
    ('protein_demo_prs', protein_demo_prs),
    ('protein_serum_prs', protein_serum_prs),
    ('combine', combine)
]

for lab in label:
    for model_name,model in models:
        for estimator in estimator_list:
            avg_res = Multimachine_with_cv(data, estimator, lab, iter, n_repeats, model_name, model)
            avg_result = pd.concat([avg_result, avg_res], ignore_index=True)

avg_result.to_csv(f'/GDCH/proteinAI/result1/allmodel_CD_average_repeat10_different_years.csv', index=False)

