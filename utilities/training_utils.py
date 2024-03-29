# Python native
import os
from typing import Callable, Union

# PyG
import torch

# PyTorch TensorBoard support
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Custom imports
from models.definitions.geometric_models import GraphModel


def _custom_verbosity_enumerate(
    iterable, verbose: bool, miniters: Union[int, None] = None
):
    """Returns either just the enumerated iterable, or one with the progress tracked."""
    if verbose:
        return tqdm(enumerate(iterable), miniters=miniters)
    else:
        return enumerate(iterable)


def train_one_epoch(
    epoch_index: int,
    model: GraphModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tb_writer: SummaryWriter,
    x_dtype: torch.dtype,
    y_dtype: torch.dtype,
    device: torch.device,
    verbose: bool = True,
    squeeze: bool = True,
) -> float:
    if verbose:
        print(f"EPOCH {epoch_index}:")

    # Enumerate over the data
    running_loss = 0.0
    last_loss = 0
    for i, batch in _custom_verbosity_enumerate(train_loader, verbose, miniters=None):
        # Use GPU
        batch.to(device)
        # Every data instance is an input + label pair
        inputs, adjacency_matrix, labels = (
            batch.x.to(x_dtype),
            batch.edge_index,
            batch.y.to(y_dtype),
        )
        # Reset gradients (set_to_none is faster than to zero)
        optimizer.zero_grad(set_to_none=True)
        # Passing the node features and connection and batch info
        outputs = model(inputs, edge_index=adjacency_matrix, batch=batch.batch)
        # Compute loss and gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learnable weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 25 == 24:
            last_loss = running_loss / 25  # loss per batch
            if verbose:
                print(f"  batch {i+1} loss: {last_loss}")
            tb_x = epoch_index * len(train_loader) + i
            tb_writer.add_scalar("Mini-batch training loss", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def run_training(
    num_epochs: int,
    model: GraphModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    early_stopping_criterion: int,
    model_path_base: str,
    x_dtype: torch.dtype,
    y_dtype: torch.dtype,
    device: torch.device,
    verbose: bool = True,
    squeeze: bool = True,
) -> str:
    if not os.path.exists(model_path_base):
        os.makedirs(model_path_base)
    best_model_state_dict_path = model_path_base
    writer = SummaryWriter(f"{model_path_base}/run")
    best_vloss = 1_000_000_000_000_000.0
    epochs_without_improvement = 0
    model.to(x_dtype)
    model.to(device)
    for epoch in range(num_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch_index=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            tb_writer=writer,
            x_dtype=x_dtype,
            y_dtype=y_dtype,
            device=device,
            verbose=verbose,
            squeeze=squeeze,
        )
        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader, start=1):
            vdata.to(device)
            vinputs, vadjacency_matrix, vlabels = (
                vdata.x.to(x_dtype),
                vdata.edge_index,
                vdata.y.to(y_dtype),
            )
            voutputs = model(vinputs, edge_index=vadjacency_matrix, batch=vdata.batch)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / i
        if verbose:
            print(f"Epoch loss -> train: {avg_loss} valid: {avg_vloss}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Epoch loss",
            {"train": avg_loss, "valid": avg_vloss},
            epoch,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            epochs_without_improvement = 0
            best_model_state_dict_path = f"{model_path_base}/state_dict_epoch{epoch}.pt"
            torch.save(model.state_dict(), best_model_state_dict_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_criterion:
            if verbose:
                print(f"Early stopping after {epoch+1} epochs.")
            break
    return best_model_state_dict_path
