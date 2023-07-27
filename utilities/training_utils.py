# Python native
from typing import Any, Callable

# PyG
import torch

# PyTorch TensorBoard support
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Custom imports
from loan_application_experiment.models.geometric_models import GraphModel


def train_one_epoch(
    epoch_index: int,
    model: GraphModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tb_writer: SummaryWriter,
    device: torch.device,
    verbose: bool = True,
) -> float:
    if verbose:
        print(f"EPOCH {epoch_index}:")

    # Enumerate over the data
    running_loss = 0.0
    last_loss = 0
    for i, batch in enumerate(tqdm(train_loader, miniters=25)):
        # Use GPU
        batch.to(device)
        # Every data instance is an input + label pair
        inputs, adjacency_matrix, labels = (
            batch.x.float(),  # k times the batch_size, where k is the subgraph size
            batch.edge_index,
            batch.y.float(),
        )
        # Reset gradients (set_to_none is faster than to zero)
        optimizer.zero_grad(set_to_none=True)
        # Passing the node features and the connection info
        outputs = model(inputs, edge_index=adjacency_matrix, batch=batch.batch)
        # Compute loss and gradients
        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()
        # Adjust learnable weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
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
    device: torch.device,
    verbose: bool = True,
) -> str:
    if not os.path.exists(model_path_base):
        os.makedirs(model_path_base)
    best_model_state_dict_path = model_path_base
    writer = SummaryWriter(f"{model_path_base}/run")
    best_vloss = 1_000_000_000_000_000.0
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, writer, device, verbose
        )

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader, start=1):
            vdata.to(device)
            vinputs, vadjacency_matrix, vlabels = (
                vdata.x.float(),
                vdata.edge_index,
                vdata.y.float(),
            )
            voutputs = model(vinputs, edge_index=vadjacency_matrix, batch=vdata.batch)
            vloss = loss_fn(voutputs.squeeze(), vlabels)
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
            print(f"Early stopping after {epoch+1} epochs.")
            break
    return best_model_state_dict_path