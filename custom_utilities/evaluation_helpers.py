from typing import Any

import sklearn.metrics as metrics


def get_evaluation(y_true, y_preds, regression:bool=False,classification:bool=False)->dict[str,dict[str,Any]]:
    if (regression + classification)!=1:
        raise Exception(f'Set one and only one of arguments: `regression` or `classification` to `True`.')
    elif regression:
        return get_regression_evaluation(y_true,y_preds)
    else:
        return get_classification_evaluation(y_true,y_preds)

def get_regression_evaluation(y_true, y_preds)->dict[str,dict[str,Any]]:
    eval_results = {'report':{}}
    eval_results['report']["MSE"] = metrics.mean_squared_error(y_true, y_preds)
    eval_results['report']["MAE"] = metrics.mean_absolute_error(y_true, y_preds)
    eval_results['report']["MAPE"] = metrics.mean_absolute_percentage_error(y_true, y_preds)
    eval_results['report']["R^2"] = metrics.r2_score(y_true, y_preds)
    return eval_results

def get_classification_evaluation(y_true, y_preds)->dict[str,dict[str,Any]]:
    eval_results = dict()
    eval_results["report"] = metrics.classification_report(y_true, y_preds,output_dict=True)
    eval_results['report']['confusion_matrix']=metrics.confusion_matrix(y_true, y_preds)
    return eval_results

def get_preds_from_probs(lst)->int:
    return max(range(len(lst)), key=lst.__getitem__)