import json
import os
import time
from datetime import datetime

import torch
import torch_geometric.nn as pygnn
from torch_geometric.loader import DataLoader

import utilities.torch_utils
from utilities import (
    evaluation_utils,
    hetero_data_utils,
    hetero_evaluation_utils,
    hetero_training_utils,
)


def run_hoeg_experiment_configuration(
    model_class,
    lr: float,
    hidden_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hoeg_config,
) -> None:
    # HYPERPARAMETER INITIALIZATION (those that we tune)
    print()
    print(f"lr={lr}, hidden_dim={hidden_dim}:")
    hoeg_config["hidden_dim"] = hidden_dim
    hoeg_config["optimizer_settings"]["lr"] = lr

    # MODEL INITIATION
    model = model_class(
        hidden_channels=hidden_dim,
        out_channels=1,
        squeeze=hoeg_config["squeeze"],
        graph_level_prediction=hoeg_config["graph_level_target"],
    )
    model = pygnn.to_hetero(model, hoeg_config["meta_data"])

    # MODEL TRAINING
    if hoeg_config["verbose"]:
        print("Training started, progress available in Tensorboard")
    torch.cuda.empty_cache()

    start_train_time = datetime.now()
    timestamp = start_train_time.strftime("%Y%m%d_%Hh%Mm")
    model_path_base = f"{hoeg_config['model_output_path']}/lr={hoeg_config['optimizer_settings']['lr']}_hidden_dim={hoeg_config['hidden_dim']}/{str(model).split('(')[0]}_{timestamp}"

    best_state_dict_path = hetero_training_utils.run_training_hetero(
        target_node_type=hoeg_config["target_node_type"],
        num_epochs=hoeg_config["EPOCHS"],
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=hoeg_config["optimizer"](
            model.parameters(), **hoeg_config["optimizer_settings"]
        ),
        loss_fn=hoeg_config["loss_fn"],
        early_stopping_criterion=hoeg_config["early_stopping"],
        model_path_base=model_path_base,
        device=hoeg_config["device"],
        verbose=hoeg_config["verbose"],
        squeeze_required=hoeg_config["squeeze"],
    )
    total_train_time = datetime.now() - start_train_time

    # Write experiment settings as JSON into model path (of the model we've just trained)
    with open(
        os.path.join(model_path_base, "experiment_settings.json"), "w"
    ) as file_path:
        json.dump(
            evaluation_utils.get_json_serializable_dict(hoeg_config),
            file_path,
            indent=2,
        )

    # MODEL EVALUATION
    # Get model evaluation report
    evaluation_report = hetero_evaluation_utils.get_best_model_evaluation(
        model_state_dict_path=best_state_dict_path,
        target_node_type=hoeg_config["target_node_type"],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        regression=True,
        evaluation_reporter=evaluation_utils.get_evaluation,
        verbose=hoeg_config["verbose"],
        track_time=hoeg_config["track_time"],
    )

    evaluation_report["Train"]["report"]["training_time"] = total_train_time
    # Store model results as JSON into model path
    model_architecture = utilities.torch_utils.parse_model_string(model)
    model_architecture["Number of parameters"] = utilities.torch_utils.count_parameters(
        model
    )
    with open(
        os.path.join(model_path_base, "model_architecture.json"), "w"
    ) as file_path:
        json.dump(model_architecture, file_path, indent=2)

    with open(
        os.path.join(model_path_base, "evaluation_report.json"), "w"
    ) as file_path:
        json.dump(
            evaluation_utils.get_json_serializable_dict(evaluation_report),
            file_path,
            indent=2,
        )

    # Print evaluation report
    if hoeg_config["verbose"]:
        print(
            f"lr={hoeg_config['optimizer_settings']['lr']}, hidden_dim={hoeg_config['hidden_dim']}:"
        )
    print(f"    {model_architecture['Number of parameters']} parameters")
    print(f"    {evaluation_report['Train']['report']['training_time']} H:m:s")
    print(f"    {evaluation_report['Test']['report']['MAE']:.4f}")
