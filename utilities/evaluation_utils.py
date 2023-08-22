import numbers
import time
from typing import Any, Callable, Union

import numpy as np
import sklearn.metrics as metrics
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.definitions.geometric_models import GraphModel


def _toggle_regression_classification(
    regression=None, classification=None
) -> tuple[bool, bool]:
    """
    Toggle function for regression and classification variables.

    This function allows specifying either `regression` or `classification`, and it automatically sets the other
    variable to the opposite value. Only one of the two variables can be True at a time, and both variables cannot be
    False simultaneously.

    Parameters:
        regression (bool, optional): If True, indicates regression task. If None, the function will set it to the
            opposite value of `classification` (default: None).
        classification (bool, optional): If True, indicates classification task. If None, the function will set it to
            the opposite value of `regression` (default: None).

    Returns:
        tuple: A boolean tuple containing the updated values of `regression` and `classification`.

    Raises:
        ValueError: If both `regression` and `classification` are set to None.
        ValueError: If both `regression` and `classification` are specified, and numerically evaluated they don't add up to 1.
    """
    if regression is None and classification is None:
        raise ValueError(
            "At least one of regression and classification should be specified."
        )

    if (regression is not None and classification is not None) and (
        regression + classification != 1
    ):
        raise ValueError(
            "Only one of regression and classification should be specified."
        )

    if regression is None:
        regression = not classification
    else:
        classification = not regression

    return bool(regression), bool(classification)


def get_json_serializable_dict(d):
    def convert_to_float_or_string(value):
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, numbers.Number):
            return float(value)
        elif isinstance(value, dict):
            return {key: convert_to_float_or_string(val) for key, val in value.items()}
        elif isinstance(value, list):
            return [convert_to_float_or_string(val) for val in value]
        else:
            return str(value)

    if isinstance(d, dict):
        return {key: convert_to_float_or_string(value) for key, value in d.items()}
    else:
        return str(d)


def get_evaluation(
    y_true, y_preds, regression=None, classification=None, time=None
) -> dict[str, dict[str, Any]]:
    regression, classification = _toggle_regression_classification(
        regression, classification
    )
    if regression:
        return get_regression_evaluation(y_true, y_preds, time)
    else:
        return get_classification_evaluation(y_true, y_preds, time)


def get_regression_evaluation(y_true, y_preds, time=None) -> dict[str, dict[str, Any]]:
    eval_results = {"report": {}}
    eval_results["report"]["MSE"] = metrics.mean_squared_error(y_true, y_preds)
    eval_results["report"]["MAE"] = metrics.mean_absolute_error(y_true, y_preds)
    eval_results["report"]["MAPE"] = metrics.mean_absolute_percentage_error(
        y_true, y_preds
    )
    eval_results["report"]["R^2"] = metrics.r2_score(y_true, y_preds)
    if time:
        eval_results["report"]["prediction_time"] = time
    return eval_results


def get_classification_evaluation(
    y_true, y_preds, time=None
) -> dict[str, dict[str, Any]]:
    eval_results = dict()
    eval_results["report"] = metrics.classification_report(
        y_true, y_preds, output_dict=True
    )
    eval_results["report"]["confusion_matrix"] = metrics.confusion_matrix(
        y_true, y_preds
    )
    if time:
        eval_results["report"]["prediction_time"] = time
    return eval_results


def get_preds_from_probs(lst) -> int:
    return max(range(len(lst)), key=lst.__getitem__)


def evaluate_torch_model(
    model: GraphModel,
    dataloader: DataLoader,
    evaluation_reporter: Callable[
        [torch.Tensor, torch.Tensor, bool, bool, float], dict[str, dict[str, Any]]
    ],
    classification: bool,
    regression: bool,
    verbose: bool = False,
    track_time: bool = False,
) -> dict[str, dict[str, Any]]:
    device = torch.device("cpu")
    start_time = time.time()
    elapsed_time = 0

    def _eval_batch(batch, model):
        batch_inputs, batch_adjacency_matrix, batch_labels = (
            batch.x.float(),
            batch.edge_index,
            batch.y.float(),
        )
        return (
            model(batch_inputs, edge_index=batch_adjacency_matrix, batch=batch.batch),
            batch_labels,
        )

    with torch.no_grad():
        model.eval()
        model.train(False)
        model.to(device)
        y_preds = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        for batch in tqdm(dataloader, disable=not (verbose)):
            batch.to(device)
            batch_y_preds, batch_y_true = _eval_batch(batch, model)
            # append current batch prediction
            y_preds = torch.cat((y_preds, batch_y_preds))
            y_true = torch.cat((y_true, batch_y_true))
        y_preds = torch.squeeze(y_preds)
        if track_time:
            elapsed_time = time.time() - start_time
    if classification:
        # yield predicted class probabilities,
        # and convert to hard predictions before passing to the `evaluation_reporter`
        y_probs = y_preds.to(device)
        y_preds = torch.tensor(
            np.apply_along_axis(get_preds_from_probs, axis=1, arr=y_probs)
        )
        return evaluation_reporter(
            y_preds, y_true.to(device), regression, classification, elapsed_time
        )
    else:
        # assuming user wants a regression report
        return evaluation_reporter(
            y_preds.to(device),
            y_true.to(device),
            regression,
            classification,
            elapsed_time,
        )


def get_best_model_evaluation(
    model_state_dict_path: str,
    model: GraphModel,
    evaluation_reporter: Callable[
        [torch.Tensor, torch.Tensor, bool, bool], dict[str, dict[str, Any]]
    ],
    regression: Union[None, bool, int] = None,
    classification: Union[None, bool, int] = None,
    train_loader: Union[DataLoader, None] = None,
    val_loader: Union[DataLoader, None] = None,
    test_loader: Union[DataLoader, None] = None,
    verbose: bool = True,
    track_time: bool = False,
) -> dict[str, dict[str, Any]]:
    regression, classification = _toggle_regression_classification(
        regression, classification
    )  # determine type of evaluation
    best_state_dict = torch.load(model_state_dict_path)  # , map_location=device
    model.load_state_dict(best_state_dict)
    model.eval()
    kwargs = {
        "model": model,
        "evaluation_reporter": evaluation_reporter,
        "classification": classification,
        "regression": regression,
        "verbose": verbose,
        "track_time": track_time,
    }
    evaluation = {}
    if train_loader:
        evaluation |= {
            f"Train": evaluate_torch_model(dataloader=train_loader, **kwargs)
        }
    if val_loader:
        evaluation |= {
            f"Validation": evaluate_torch_model(dataloader=val_loader, **kwargs)
        }
    if test_loader:
        evaluation |= {f"Test": evaluate_torch_model(dataloader=test_loader, **kwargs)}
    return evaluation
