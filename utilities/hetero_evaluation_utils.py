# Python native
from typing import Any, Callable

# PyG
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Custom imports
from loan_application_experiment.models.geometric_models import GraphModel

# import os
# os.chdir("/home/tim/Development/OCPPM/")


def evaluate_hetero_model(
    target_node_type: str,
    model: GraphModel,
    dataloader: DataLoader,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
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
        y_preds = torch.squeeze(y_preds)
    return metric(y_preds.to(device), y_true.to(device))


def evaluate_best_model(
    target_node_type: str,
    model_state_dict_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model: GraphModel,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    best_state_dict = torch.load(model_state_dict_path, map_location=device)

    model.load_state_dict(best_state_dict)
    model.eval()
    evaluation = {
        f"Train {metric}": evaluate_hetero_model(
            target_node_type,
            model=model,
            dataloader=train_loader,
            metric=metric,
            device=device,
            verbose=verbose,
        ),
        f"Val {metric}": evaluate_hetero_model(
            target_node_type,
            model=model,
            dataloader=val_loader,
            metric=metric,
            device=device,
            verbose=verbose,
        ),
        f"Test {metric}": evaluate_hetero_model(
            target_node_type,
            model=model,
            dataloader=test_loader,
            metric=metric,
            device=device,
            verbose=verbose,
        ),
    }
    return evaluation
