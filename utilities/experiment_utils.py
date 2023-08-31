import json
import os
import time
from datetime import datetime

import torch
import torch_geometric.nn as pygnn
from torch_geometric.loader import DataLoader

import utilities.torch_utils
from utilities import data_utils, evaluation_utils, training_utils


def run_efg_experiment_configuration(
    model_class,
    lr: float,
    hidden_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    efg_config: dict,
) -> None:
    # HYPERPARAMETER INITIALIZATION (those that we tune)
    print()
    print(f"lr={lr}, hidden_dim={hidden_dim}:")
    efg_config["hidden_dim"] = hidden_dim
    efg_config["optimizer_settings"]["lr"] = lr
    # MODEL INITIALIZATION
    model = model_class(
        hidden_channels=hidden_dim,
        out_channels=1,
        squeeze=efg_config["squeeze"],
        graph_level_prediction=efg_config["graph_level_prediction"],
    )

    # pretrained_state_dict = torch.load("models/runs/GraphConvNet_20230718_13h59m/state_dict_epoch6.pt")
    # model.load_state_dict(pretrained_state_dict)

    # TRAINING
    if efg_config["verbose"]:
        print("Training started, progress available in Tensorboard")
    torch.cuda.empty_cache()

    start_train_time = datetime.now()
    timestamp = start_train_time.strftime("%Y%m%d_%Hh%Mm")
    model_path_base = f"{efg_config['model_output_path']}/lr={lr}_hidden_dim={hidden_dim}/{str(model).split('(')[0]}_{timestamp}"

    best_state_dict_path = training_utils.run_training(
        num_epochs=efg_config["EPOCHS"],
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=efg_config["optimizer"](
            model.parameters(), **efg_config["optimizer_settings"]
        ),
        loss_fn=efg_config["loss_fn"],
        early_stopping_criterion=efg_config["early_stopping"],
        model_path_base=model_path_base,
        x_dtype=efg_config["features_dtype"],
        y_dtype=efg_config["target_dtype"],
        device=efg_config["device"],
        verbose=efg_config["verbose"],
        squeeze=efg_config["squeeze"],
    )

    total_train_time = datetime.now() - start_train_time
    # Write experiment settings as JSON into model path (of the model we've just trained)
    with open(
        os.path.join(model_path_base, "experiment_settings.json"), "w"
    ) as file_path:
        json.dump(
            evaluation_utils.get_json_serializable_dict(efg_config),
            file_path,
            indent=2,
        )

    # EVALUATION
    # Get model evaluation report
    evaluation_report = evaluation_utils.get_best_model_evaluation(
        model_state_dict_path=best_state_dict_path,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        evaluation_reporter=evaluation_utils.get_evaluation,
        regression=True,
        verbose=efg_config["verbose"],
        track_time=efg_config["track_time"],
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
    if efg_config["verbose"]:
        print(f"lr={lr}, hidden_dim={hidden_dim}:")
    print(f"    {model_architecture['Number of parameters']} parameters")
    print(f"    {evaluation_report['Train']['report']['training_time']} H:m:s")
    print(f"    {evaluation_report['Test']['report']['MAE']:.4f}")
