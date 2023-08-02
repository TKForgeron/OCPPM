# Python native
# import os
# os.chdir("/home/tim/Development/OCPPM/")
import logging
import pickle
from typing import Any, Callable, Union

# PyG
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Custom imports
from models.definitions.geometric_models import GraphModel

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logging/debug.log",
)
logging.critical("-" * 32 + ' TEST CS HOEG ' + "-" * 32)

def evaluate_hetero_model(
    target_node_type: str,
    model: GraphModel,
    dataloader: DataLoader,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
    squeeze_required: bool = True,
) -> torch.Tensor:
    with torch.no_grad():

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
        logging.debug(y_preds.shape)
        logging.debug(y_true.shape)
        print('y_preds.shape: ', y_preds.shape)
        print('y_true.shape: ', y_true.shape)
        print('*'*32)
        if squeeze_required:
            y_preds = torch.squeeze(y_preds)
    return metric(y_preds.to(device), y_true.to(device))


def evaluate_best_model(
    target_node_type: str,
    model_state_dict_path: str,
    model: GraphModel,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    train_loader: Union[DataLoader, None] = None,
    val_loader: Union[DataLoader, None] = None,
    test_loader: Union[DataLoader, None] = None,
    verbose: bool = True,
    squeeze_required: bool = True,
) -> dict[str, torch.Tensor]:
    best_state_dict = torch.load(model_state_dict_path, map_location=device)

    model.load_state_dict(best_state_dict)
    model.eval()
    kwargs = {
        "target_node_type": target_node_type,
        "model": model,
        "metric": metric,
        "device": device,
        "verbose": verbose,
        "squeeze_required": squeeze_required,
    }
    evaluation = {}
    if train_loader:
        evaluation |= {
            f"Train {metric}": evaluate_hetero_model(dataloader=train_loader, **kwargs)
        }
    if val_loader:
        evaluation |= {
            f"Val {metric}": evaluate_hetero_model(dataloader=val_loader, **kwargs)
        }
    if test_loader:
        evaluation |= {
            f"Test {metric}": evaluate_hetero_model(dataloader=test_loader, **kwargs)
        }
    return evaluation
