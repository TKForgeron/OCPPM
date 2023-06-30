# Python native
import os

os.chdir("/home/tim/Development/OCPPM/")
print(os.getcwd())
import logging
import pickle
from datetime import datetime
from statistics import median as median
from typing import Any, Callable

# Data handling
import numpy as np
import ocpa.algo.predictive_monitoring.factory as feature_factory

# PyG
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as O

# PyTorch TensorBoard support
import torch.utils.tensorboard
import torch_geometric.transforms as T

# Object centric process mining
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage

# # Simple machine learning models, procedure tools, and evaluation metrics
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GeneralConv,
    Linear,
    SAGEConv,
    to_hetero,
)

# from torch_geometric.sampler import HeteroSamplerOutput, HGTSampler
from tqdm import tqdm

# Custom imports
# from config.files.bpi17 import bpi17_config
from loan_application_experiment.feature_encodings.hoeg.hoeg import HOEG

# from experiment.feature_encodings.hoeg.hoeg import HOEG

# from experiment.models.heterogeneous_models import GAT, HCGNN

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logging/debug.log",
)
logging.critical(f"{'-' * 32} NEW RUN {'-' * 32}")

# Config
storage_path = "data/BPI17/feature_encodings/HOEG/hoeg"
split_feature_storage_file = "BPI2017-feature_storage-split-[C1-3,C5,P1-6,O2,O3,O5].fs"
objects_data_file = "bpi17_ofg+oi_graph+app_node_map+off_node_map.pkl"
target_label = (feature_factory.EVENT_REMAINING_TIME, ())
skip_cache = False


def count_parameters(model: torch.nn.Module) -> int:
    # with torch.no_grad():  # Initialize lazy modules.
    #     out = model(data.x_dict, data.edge_index_dict)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_dataloaders(
    batch_size: int,
    ds_train: HOEG = None,
    ds_val: HOEG = None,
    ds_test: HOEG = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    num_workers: int = 4,
    seed_worker: Callable[[int], None] = None,
    generator: torch.Generator = None,
) -> list[DataLoader]:
    dataloaders = []
    if ds_train:
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        dataloaders.append(train_loader)
    if ds_val:
        val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        dataloaders.append(val_loader)
    if ds_test:
        test_loader = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        dataloaders.append(test_loader)
    return dataloaders


def train_one_epoch(
    epoch_index: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tb_writer: SummaryWriter,
    device: torch.device,
    verbose: bool = True,
) -> float:
    if verbose:
        print(f"EPOCH {epoch_index + 1}:")

    # Enumerate over the data
    running_loss = 0.0
    last_loss = 0
    for i, batch in enumerate(tqdm(train_loader, miniters=15)):
        # Use GPU
        batch.to(device)

        # Every data instance is an input + label pair
        inputs, adjacency_matrix, labels = (
            batch.x_dict,  # k times the batch_size, where k is the subgraph size
            batch.edge_index_dict,
            batch["application"].y,
        )
        # Reset gradients (set_to_none is faster than to zero)
        optimizer.zero_grad(set_to_none=True)
        # Passing the node features and the connection info
        outputs = model(inputs, adjacency_matrix)
        # Compute loss and gradients
        loss = loss_fn(torch.squeeze(outputs["application"]), labels)
        loss.backward()
        # Adjust learnable weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.detach().item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            if verbose:
                print(f"  batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def run_training(
    num_epochs: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    verbose: bool = True,
) -> str:
    model_path = f"models/runs/{str(model).split('(')[0]}_{datetime.now().strftime('%Y%m%d_%Hh%Mm')}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter(f"{model_path}/run")
    best_vloss = 1_000_000_000_000_000_000.0
    model.to(device)

    for epoch in range(num_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, writer, device
        )

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        num_batches = 0  # this will count up the number of batches
        for num_batches, vbatch in enumerate(validation_loader, start=1):
            vbatch.to(device)
            vinputs, vadjacency_matrix, vlabels = (
                vbatch.x_dict,
                vbatch.edge_index_dict,
                vbatch["application"].y,
            )
            voutputs = model(vinputs, vadjacency_matrix)
            vloss = (
                loss_fn(torch.squeeze(voutputs["application"]), vlabels).detach().item()
            )
            running_vloss += vloss

        avg_vloss = running_vloss / num_batches
        if verbose:
            print(f"LOSS train {avg_loss} valid {avg_vloss}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f"{model_path}/state_dict_epoch{epoch}.pt")
    return model_path


# def evaluate_model(
#     model: torch.nn.Module,
#     dataloader: DataLoader,
#     metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#     device: torch.device = torch.device("cpu"),
#     verbose: bool = False,
# ) -> torch.Tensor:
#     with torch.no_grad():

#         def _eval_batch(batch, model):
#             batch_inputs, batch_adjacency_matrix, batch_labels = (
#                 batch.x_dict,  # k times the batch_size, where k is the subgraph size
#                 batch.edge_index_dict,
#                 batch["application"].y,
#             )
#             return model(batch_inputs, batch_adjacency_matrix), batch_labels

#         model.eval()
#         model.train(False)
#         model.to(device)
#         y_preds = torch.tensor([]).to(device)
#         y_true = torch.tensor([]).to(device)
#         for batch in tqdm(dataloader, disable=not (verbose)):
#             batch.to(device)
#             batch_y_preds, batch_y_true = _eval_batch(batch, model)
#             y_preds = torch.cat((y_preds, batch_y_preds))
#             y_true = torch.cat((y_true, batch_y_true))
#         y_preds = torch.squeeze(y_preds)
#     return metric(y_preds.to(device), y_true.to(device))


ds_train = HOEG(
    train=True,
    root=storage_path,
    events_filename=split_feature_storage_file,
    objects_filename=objects_data_file,
    label_key=target_label,
    verbosity=51,
    transform=T.ToUndirected(),
    skip_cache=skip_cache,
)

ds_val = HOEG(
    validation=True,
    root=storage_path,
    events_filename=split_feature_storage_file,
    objects_filename=objects_data_file,
    label_key=target_label,
    verbosity=51,
    transform=T.ToUndirected(),
    skip_cache=skip_cache,
)


# ds_test = HOEG(
#     test=True,
#     root=storage_path,
#     events_filename=split_feature_storage_file,
#     objects_filename=objects_data_file,
#     label_key=target_label,
#     verbosity=51,
#     transform=T.ToUndirected()
# )

train_loader, val_loader = prepare_dataloaders(
    batch_size=128, ds_train=ds_train, ds_val=ds_val
)

# FIND OUT WHY GETTING ERROR '...negative dimension...'
# bc all is equal to the GAT in run_ofg.ipynb
# --> I think bc

meta_data = (
    ["event", "application", "offer"],
    [
        ("event", "follows", "event"),
        ("event", "interacts", "application"),
        ("event", "interacts", "offer"),
        ("application", "interacts", "application"),
        ("application", "rev_interacts", "event"),
        ("offer", "rev_interacts", "event"),
    ],
)


# model = HCGNN(hidden_channels=64, out_channels=1, num_heads=2, num_layers=2)
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        # x = self.lin1(x)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        # x = self.lin2(x)
        return x


model = GAT(hidden_channels=128, out_channels=1)
model = to_hetero(model, meta_data, aggr="sum")
# model.double()


optimizer = O.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = run_training(
    30,
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    optimizer=optimizer,
    loss_fn=F.mse_loss,
    device=device,
)
