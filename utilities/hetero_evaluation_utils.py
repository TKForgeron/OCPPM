# Python native
import time
from typing import Any, Callable, Union

import numpy as np

# PyG
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Custom imports
import utilities.evaluation_utils as evaluation_utils
from models.definitions.geometric_models import GraphModel

DEBUG_MODE = False
if DEBUG_MODE:
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="logging/debug.log",
    )
    logging.critical("-" * 32 + " TEST CS HOEG " + "-" * 32)


def evaluate_hetero_model(
    target_node_type: str,
    model: GraphModel,
    dataloader: DataLoader,
    evaluation_reporter: Callable[
        [torch.Tensor, torch.Tensor, bool, bool, float], dict[str, dict[str, Any]]
    ],
    classification: bool,
    regression: bool,
    verbose: bool = False,
    squeeze_required: bool = True,
    track_time: bool = False,
) -> dict[str, dict[str, Any]]:
    device = torch.device("cpu")
    start_time = time.time()
    elapsed_time = 0

    def _eval_batch(batch, model):
        batch_inputs, batch_adjacency_matrix, batch_labels = (
            batch.x_dict,
            batch.edge_index_dict,
            batch[target_node_type].y,
        )
        return (
            model(
                batch_inputs,
                edge_index=batch_adjacency_matrix
                # , batch=batch[target_node_type].batch,
            ),
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
            # append
            y_preds = torch.cat((y_preds, batch_y_preds[target_node_type]))
            y_true = torch.cat((y_true, batch_y_true))
        if DEBUG_MODE:
            logging.debug(y_preds.shape)
            logging.debug(y_true.shape)
            print("y_preds.shape: ", y_preds.shape)
            print("y_true.shape: ", y_true.shape)
            print("*" * 32)
        if squeeze_required:
            y_preds = torch.squeeze(y_preds)
        if track_time:
            elapsed_time = time.time() - start_time
    if classification:
        # yield predicted class probabilities,
        # and convert to hard predictions before passing to the `evaluation_reporter`
        y_probs = y_preds.to(device)
        y_preds = torch.tensor(
            np.apply_along_axis(
                evaluation_utils.get_preds_from_probs, axis=1, arr=y_probs
            )
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
    target_node_type: str,
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
    squeeze_required: bool = True,
    track_time: bool = False,
) -> dict[str, torch.Tensor]:
    regression, classification = evaluation_utils._toggle_regression_classification(
        regression, classification
    )  # determine type of evaluation
    best_state_dict = torch.load(model_state_dict_path)
    model.load_state_dict(best_state_dict)
    kwargs = {
        "model": model,
        "evaluation_reporter": evaluation_reporter,
        "classification": classification,
        "regression": regression,
        "verbose": verbose,
        "track_time": track_time,
        "target_node_type": target_node_type,
        "squeeze_required": squeeze_required,
    }
    evaluation = {}
    if train_loader:
        evaluation |= {
            f"Train": evaluate_hetero_model(dataloader=train_loader, **kwargs)
        }
    if val_loader:
        evaluation |= {
            f"Validation": evaluate_hetero_model(dataloader=val_loader, **kwargs)
        }
    if test_loader:
        evaluation |= {f"Test": evaluate_hetero_model(dataloader=test_loader, **kwargs)}
    return evaluation
