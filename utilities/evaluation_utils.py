import numbers
from typing import Any, Callable

import sklearn.metrics as metrics
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from loan_application_experiment.models.geometric_models import GraphModel


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
    y_true, y_preds, regression: bool = False, classification: bool = False
) -> dict[str, dict[str, Any]]:
    if (regression + classification) != 1:
        raise Exception(
            f"Set one and only one of arguments: `regression` or `classification` to `True`."
        )
    elif regression:
        return get_regression_evaluation(y_true, y_preds)
    else:
        return get_classification_evaluation(y_true, y_preds)


def get_regression_evaluation(y_true, y_preds) -> dict[str, dict[str, Any]]:
    eval_results = {"report": {}}
    eval_results["report"]["MSE"] = metrics.mean_squared_error(y_true, y_preds)
    eval_results["report"]["MAE"] = metrics.mean_absolute_error(y_true, y_preds)
    eval_results["report"]["MAPE"] = metrics.mean_absolute_percentage_error(
        y_true, y_preds
    )
    eval_results["report"]["R^2"] = metrics.r2_score(y_true, y_preds)
    return eval_results


def get_classification_evaluation(y_true, y_preds) -> dict[str, dict[str, Any]]:
    eval_results = dict()
    eval_results["report"] = metrics.classification_report(
        y_true, y_preds, output_dict=True
    )
    eval_results["report"]["confusion_matrix"] = metrics.confusion_matrix(
        y_true, y_preds
    )
    return eval_results


def get_preds_from_probs(lst) -> int:
    return max(range(len(lst)), key=lst.__getitem__)


def evaluate_model(
    model: GraphModel,
    dataloader: DataLoader,
    evaluation_reporter: Callable[
        [torch.Tensor, torch.Tensor, bool, bool], dict[str, dict[str, Any]]
    ],
    regression: bool,
    classification: bool,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    device = torch.device("cpu")
    with torch.no_grad():

        def _eval_batch(batch, model):
            batch_inputs, batch_adjacency_matrix, batch_labels = (
                batch.x.float(),
                batch.edge_index,
                batch.y.float(),
            )
            return (
                model(
                    batch_inputs, edge_index=batch_adjacency_matrix, batch=batch.batch
                ),
                batch_labels,
            )

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
    return evaluation_reporter(
        y_preds.to(device), y_true.to(device), regression, classification
    )


def get_best_model_evaluation(
    model_state_dict_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model: GraphModel,
    evaluation_reporter: Callable[
        [torch.Tensor, torch.Tensor, bool, bool], dict[str, dict[str, Any]]
    ],
    regression: bool,
    classification: bool,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    best_state_dict = torch.load(model_state_dict_path)  # , map_location=device

    model.load_state_dict(best_state_dict)
    model.eval()
    evaluation = {
        f"Train": evaluate_model(
            model=model,
            dataloader=train_loader,
            evaluation_reporter=evaluation_reporter,
            regression=regression,
            classification=classification,
            verbose=verbose,
        ),
        f"Validation": evaluate_model(
            model=model,
            dataloader=val_loader,
            evaluation_reporter=evaluation_reporter,
            regression=regression,
            classification=classification,
            verbose=verbose,
        ),
        f"Test": evaluate_model(
            model=model,
            dataloader=test_loader,
            evaluation_reporter=evaluation_reporter,
            regression=regression,
            classification=classification,
            verbose=verbose,
        ),
    }
    return evaluation
