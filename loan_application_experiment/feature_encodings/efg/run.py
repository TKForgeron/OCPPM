# Python native
import os
import pickle
import random
from copy import copy
from datetime import datetime
from statistics import median as median
from sys import platform
from typing import Any, Callable

# Data handling
import numpy as np
import ocpa.algo.predictive_monitoring.factory as feature_factory

# PyG
import torch

# PyTorch TensorBoard support
import torch.utils.tensorboard

# Object centric process mining
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage

# # Simple machine learning models, procedure tools, and evaluation metrics
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import loan_application_experiment.feature_encodings.ofg.hetero_graph_encoding_utils as hetero_graph_encoding_utils

# Custom imports
from config.files.bpi17 import bpi17_config
from loan_application_experiment.feature_encodings.efg.run import EFG

# from importing_ocel import build_feature_storage, load_ocel, pickle_feature_storage
from loan_application_experiment.models.geometric_models import AdamsGCN, GraphModel

# from dotenv import load_dotenv


def get_experiment_configuration(
    parameters: dict, verbose: bool = False
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
    str,
    tuple[str, tuple],
    str,
    dict[str, Any],
    int,
    int,
    int,
    int,
    torch.Generator,
    Callable[[int], None],
    torch.device,
    bool,
]:
    load_dotenv(".env_secrets")
    if platform == "linux" or platform == "linux2":
        LOCAL_ROOT_DIR = os.getenv("lin_root_dir")
    elif platform == "win32":
        LOCAL_ROOT_DIR = os.getenv("win_root_dir")
    else:
        # We don't do apple
        raise RuntimeError("Cannot determine operating system")

    # Configure whether to run the training loop again
    # Define where to find previously trained best model
    LOAD_PREVIOUS_RUN = {
        "run_train_loop_again": False,
        "state_dict_dir": f"{LOCAL_ROOT_DIR}/models/AdamsGCN_20230321_12h34m",
    }

    (
        STORAGE_PATH,
        SPLIT_FEATURE_STORAGE_FILE,
        RAW_FEATURE_STORAGE_FILE,
        TARGET_LABEL,
        SUBGRAPH_SIZE,
        EPOCHS,
        BATCH_SIZE,
        RANDOM_SEED,
        ORIGINAL_LOG_FILE,
        FEATURE_SET,
        *rest,
    ) = parameters.values()
    LOG_PARAMETERS = None
    if rest:
        if type(rest[0]) == dict:
            LOG_PARAMETERS = rest[0]

    # Initializing random seeds for maximizing reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    def seed_worker(worker_id: int) -> None:
        # worker_seed = torch.initial_seed() % RANDOM_SEED
        worker_seed = RANDOM_SEED
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    # torch.use_deterministic_algorithms(True) # incompatible with GCN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if LOG_PARAMETERS:
        print(f"LOG_PARAMETERS: {LOG_PARAMETERS}")
    print(f"ORIGINAL_LOG_FILE: {ORIGINAL_LOG_FILE}")
    print(f"STORAGE_PATH: {STORAGE_PATH}")
    print(f"SPLIT_FEATURE_STORAGE_FILE: {SPLIT_FEATURE_STORAGE_FILE}")
    print(f"RAW_FEATURE_STORAGE_FILE: {RAW_FEATURE_STORAGE_FILE}")
    print(f"TARGET_LABEL: {TARGET_LABEL}")
    print(f"LOCAL_ROOT_DIR: {LOCAL_ROOT_DIR}")
    print(f"SUBGRAPH_SIZE: {SUBGRAPH_SIZE}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"Device: {device}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    # Verbose determines whether to print anything when running the full script (this config is always printed)
    print(f"verbose: {verbose}")
    print()

    return (
        FEATURE_SET,
        LOG_PARAMETERS,
        ORIGINAL_LOG_FILE,
        STORAGE_PATH,
        SPLIT_FEATURE_STORAGE_FILE,
        RAW_FEATURE_STORAGE_FILE,
        TARGET_LABEL,
        LOCAL_ROOT_DIR,
        LOAD_PREVIOUS_RUN,
        RANDOM_SEED,
        SUBGRAPH_SIZE,
        EPOCHS,
        BATCH_SIZE,
        generator,
        seed_worker,
        device,
        verbose,
    )


def prepare_feature_storage(
    original_log_file: str,
    log_parameters: dict[str, Any],
    feature_set: dict[str, Any],
    storage_path: str,
    split_feature_storage_file: str,
    raw_feature_storage_file: str,
    seed: int,
) -> None:
    # If SPLIT_FEATURE_STORAGE_FILE not cached, generate it
    if not os.path.exists(f"{storage_path}/raw/{split_feature_storage_file}"):
        if not os.path.exists(f"{storage_path}/raw/{raw_feature_storage_file}"):
            ocel = load_ocel(original_log_file, log_parameters)
            act_features = []
            if "C2" in feature_set.keys():
                activities = ocel.log.log["event_activity"].unique().tolist()
                del feature_set["C2"]
                act_features = [
                    (feature_factory.EVENT_PRECEDING_ACTIVITIES, (act,))
                    for act in activities
                ]
            feature_list = act_features + list(feature_set.values())
            feature_storage = build_feature_storage(ocel, feature_list)
            pickle_feature_storage(
                feature_storage, f"{storage_path}/raw/{raw_feature_storage_file}"
            )
        else:
            with open(f"{storage_path}/raw/{raw_feature_storage_file}", "rb") as file:
                feature_storage: FeatureStorage = pickle.load(file)

        # Adams didn't give this split a random seed,
        # thus we can split the validation set in this arbitrary manner
        feature_storage.extract_normalized_train_test_split(
            test_size=0.3,
            validation_size=0.7 * 0.2,
            scaler=StandardScaler,
            scaling_exempt_features=[],
            state=seed,
        )

        with open(
            f"{storage_path}/raw/{split_feature_storage_file}",
            "wb",
        ) as file:
            pickle.dump(feature_storage, file)


def load_datasets(
    storage_path: str,
    split_feature_storage_file: str,
    target_label: tuple[str, tuple],
    train: bool = True,
    val: bool = True,
    test: bool = True,
) -> list[EventSubGraphDataset]:
    datasets = []
    if train:
        ds_train = EventSubGraphDataset(
            train=True,
            root=storage_path,
            filename=split_feature_storage_file,
            label_key=target_label,
            size_subgraph_samples=SUBGRAPH_SIZE,
            verbosity=51,
        )
        datasets.append(ds_train)
    if val:
        ds_val = EventSubGraphDataset(
            validation=True,
            root=storage_path,
            filename=split_feature_storage_file,
            label_key=target_label,
            size_subgraph_samples=SUBGRAPH_SIZE,
            verbosity=51,
        )
        datasets.append(ds_val)
    if test:
        ds_test = EventSubGraphDataset(
            test=True,
            root=storage_path,
            filename=split_feature_storage_file,
            label_key=target_label,
            size_subgraph_samples=SUBGRAPH_SIZE,
            verbosity=51,
        )
        datasets.append(ds_test)
    return datasets


def print_dataset_summaries(
    ds_train: EventSubGraphDataset,
    ds_val: EventSubGraphDataset,
    ds_test: EventSubGraphDataset,
) -> None:
    print("Train set")
    print(ds_train.get_summary(), "\n")
    print("Validation set")
    print(ds_val.get_summary(), "\n")
    print("Test set")
    print(ds_test.get_summary(), "\n")


def configure_model(
    num_node_features: int,
    num_hidden_features: int,
    size_subgraph_samples: int,
    device: torch.device,
) -> GraphModel:
    # Initialize model
    model = AdamsGCN(
        num_node_features=num_node_features,
        hyperparams={
            "num_hidden_features": num_hidden_features,
            "size_subgraph_samples": size_subgraph_samples,
        },
    )

    model = model.to(device)
    # data = ds_train.to(device)
    return model


def count_parameters(model: GraphModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_dataloaders(
    batch_size: int,
    ds_train: EventSubGraphDataset = None,
    ds_val: EventSubGraphDataset = None,
    ds_test: EventSubGraphDataset = None,
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
            batch_size=128,
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
    model: GraphModel,
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
    for i, batch in enumerate(tqdm(train_loader)):
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
        outputs = model(inputs, adjacency_matrix)
        # Compute loss and gradients
        loss = loss_fn(torch.squeeze(outputs), labels)
        loss.backward()
        # Adjust learnable weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
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
    model: GraphModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    timestamp: str,
    device: torch.device,
    verbose: bool = True,
) -> str:
    model_path = f"models/{model.get_class_name()}_{timestamp}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter(f"{model_path}/run")
    best_vloss = 1_000_000_000_000_000.0

    for epoch in range(num_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, writer, device
        )

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vdata.to(device)
            vinputs, vadjacency_matrix, vlabels = (
                vdata.x.float(),
                vdata.edge_index,
                vdata.y.float(),
            )
            voutputs = model(vinputs, vadjacency_matrix)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
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


def evaluate_model(
    model: GraphModel,
    dataloader: DataLoader,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> torch.Tensor:
    with torch.no_grad():

        def _eval_batch(batch, model):
            batch_inputs, batch_adjacency_matrix, batch_labels = (
                batch.x.float(),
                batch.edge_index,
                batch.y.float(),
            )
            return model(batch_inputs, batch_adjacency_matrix), batch_labels

        model.eval()
        model.train(False)
        model.to(device)
        y_preds = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        for batch in tqdm(dataloader, disable=not (verbose)):
            batch.to(device)
            batch_y_preds, batch_y_true = _eval_batch(batch, model)
            y_preds = torch.cat((y_preds, batch_y_preds))
            y_true = torch.cat((y_true, batch_y_true))
        y_preds = torch.squeeze(y_preds)
    return metric(y_preds.to(device), y_true.to(device))


def evaluate_best_model(
    model_state_dir: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    def find_latest_state_dict(dir: str) -> str:
        latest_state_dict_path = sorted(
            [
                item
                for item in os.listdir(dir)
                if len(item.split("state_dict_epoch")) == 2
            ]
        )[-1]
        return os.path.join(dir, latest_state_dict_path)

    best_state_dict = torch.load(
        find_latest_state_dict(model_state_dir), map_location=device
    )
    model.load_state_dict(best_state_dict)
    evaluation = {
        f"Train {metric}": evaluate_model(
            model=model,
            dataloader=train_loader,
            metric=metric,
            device=device,
            verbose=verbose,
        ),
        f"Val {metric}": evaluate_model(
            model=model,
            dataloader=val_loader,
            metric=metric,
            device=device,
            verbose=verbose,
        ),
        f"Test {metric}": evaluate_model(
            model=model,
            dataloader=test_loader,
            metric=metric,
            device=device,
            verbose=verbose,
        ),
    }
    return evaluation


def denormalize_evaluation(
    evaluation: dict[str, torch.Tensor],
    storage_path: str,
    split_feature_storage_file: str,
) -> dict[str, torch.Tensor]:
    with open(
        f"{storage_path}/raw/{split_feature_storage_file}",
        "rb",
    ) as file:
        fs: FeatureStorage = pickle.load(file)
    evaluation_dict = copy(evaluation)
    # Get normalized scores (MAE/L1 loss), assuming train comes 1st, val 2nd, and test 3rd in the dict
    keys = list(evaluation_dict.keys())
    train_key = keys[0]
    val_key = keys[1]
    test_key = keys[2]
    normed_train_score = evaluation_dict[train_key]
    normed_val_score = evaluation_dict[val_key]
    normed_test_score = evaluation_dict[test_key]

    evaluation_dict[train_key] = fs.scaler.inverse_transform([normed_train_score] * 25)[
        -2
    ]
    evaluation_dict[val_key] = fs.scaler.inverse_transform([normed_val_score] * 25)[-2]
    evaluation_dict[test_key] = fs.scaler.inverse_transform([normed_test_score] * 25)[
        -2
    ]

    return evaluation_dict


if __name__ == "__main__":
    # Get experiment configuration variables
    (
        FEATURE_SET,
        LOG_PARAMETERS,
        ORIGINAL_LOG_FILE,
        STORAGE_PATH,
        SPLIT_FEATURE_STORAGE_FILE,
        RAW_FEATURE_STORAGE_FILE,
        TARGET_LABEL,
        LOCAL_ROOT_DIR,
        LOAD_PREVIOUS_RUN,
        RANDOM_SEED,
        SUBGRAPH_SIZE,
        EPOCHS,
        BATCH_SIZE,
        generator,
        seed_worker,
        device,
        verbose,
    ) = get_experiment_configuration(adams_config, verbose=True)

    # Get data and dataloaders
    prepare_feature_storage(
        ORIGINAL_LOG_FILE,
        LOG_PARAMETERS,
        FEATURE_SET,
        STORAGE_PATH,
        SPLIT_FEATURE_STORAGE_FILE,
        RAW_FEATURE_STORAGE_FILE,
        RANDOM_SEED,
    )
    ds_train, ds_val, ds_test = load_datasets(
        STORAGE_PATH,
        SPLIT_FEATURE_STORAGE_FILE,
        TARGET_LABEL,
        train=True,
        val=True,
        test=True,
    )
    train_loader, val_loader, test_loader = prepare_dataloaders(
        batch_size=BATCH_SIZE,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        seed_worker=seed_worker,
        generator=generator,
    )

    # Set model configuration
    model = configure_model(
        ds_train.num_node_features, ds_train.num_node_features, SUBGRAPH_SIZE, device
    )

    # Print summary of data and model
    if verbose:
        print_dataset_summaries(ds_train, ds_val, ds_test)
        print(model)
        print(f"Number of parameters: {count_parameters(model)}")

    # Load best model of completed training (if directory given)
    model_path = LOAD_PREVIOUS_RUN["state_dict_dir"]

    # Run training
    if LOAD_PREVIOUS_RUN["run_train_loop_again"]:
        print("Training started, progress available in Tensorboard")
        model_path = run_training(
            num_epochs=EPOCHS,
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=0.01,
            ),
            loss_fn=torch.nn.L1Loss(),
            timestamp=datetime.now().strftime("%Y%m%d_%Hh%Mm"),
            device=device,
            verbose=verbose,
        )

    # Get MAE results
    normalized_evaluation_dict = evaluate_best_model(
        model_state_dir=model_path,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metric=torch.nn.L1Loss(),
        device=torch.device("cpu"),
        verbose=verbose,
    )
    denormalized_evaluation_dict = denormalize_evaluation(
        normalized_evaluation_dict,
        STORAGE_PATH,
        SPLIT_FEATURE_STORAGE_FILE,
    )

    # Print experiment results
    print("Normalized:  ", normalized_evaluation_dict)
    print("Denormalized:", denormalized_evaluation_dict)
