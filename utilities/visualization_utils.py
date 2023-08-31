import json
import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define Utrecht University Colors
YELLOW = "#FFCD00"
RED = "#C00A35"
BLACK = "#000000"
WHITE = "#FFFFFF"
CREAM = "#FFE6AB"
ORANGE = "#F3965E"
BURGUNDY = "#AA1555"
BROWN = "#6E3B23"
GREEN = "#24A793"
BLUE = "#5287C6"
DARKBLUE = "#001240"
PURPLE = "#5B2182"


# DATA VISUALIZATION FUNCTIONS
def _np_sort_by_column(x: np.ndarray, column_index: int) -> np.ndarray:
    return x[x[:, column_index].argsort()]


def _get_lr_visual_values(lrs_dict: dict, lr_key: str) -> tuple[list, list, list]:
    model_config_results = np.array(lrs_dict[lr_key])
    model_config_results = _np_sort_by_column(model_config_results, column_index=0)
    no_hds, no_params, mae_scores = [list(x) for x in model_config_results.T]
    return [int(hd) for hd in no_hds], [int(p) for p in no_params], mae_scores


def create_exp1_encoding_plot(
    encoding: str,
    exp1_results: dict,
    save: bool,
    figsize: tuple[int, int] = (20, 5),
    fontsize: Union[float, int] = 15,
    dpi: int = 128,
) -> None:
    # CAPTION: test scores only
    datasets = exp1_results[encoding]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=figsize, dpi=dpi
    )  # 1 row and 3 columns of subplots

    for axis, (dataset, lrs) in zip(axes, datasets.items()):
        if dataset == "BPI17":
            axis.set_ylabel("Mean Absolute Error", fontsize=fontsize)
        no_hds, no_params, mae_scores = _get_lr_visual_values(lrs, "learning_rate=0.01")
        axis.semilogx(no_hds, mae_scores, marker="o", label="lr=0.01", color=YELLOW)

        no_hds, no_params, mae_scores = _get_lr_visual_values(
            lrs, "learning_rate=0.001"
        )
        axis.semilogx(no_hds, mae_scores, marker="o", label="lr=0.001", color=RED)

        axis.set_title(f"{encoding.upper()} on {dataset} OCEL", size=fontsize * 1.1)
        axis.set_xlabel("Hidden Dimensions", fontsize=fontsize)

        # Create formatted tick labels
        tick_labels = [f"{hds}" for hds, params in zip(no_hds, no_params)]
        
        axis.set_xticks(no_hds)
        axis.set_xticklabels(
            tick_labels, fontsize=fontsize * 0.9
        )  # Set the formatted tick labels
        axis.tick_params(axis="y", labelsize=fontsize * 0.9)

        axis.legend(fontsize=fontsize*0.85)
        axis.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(f"visualizations/plots/exp1_hp_tuning/{encoding}.pdf")
    plt.show()


def create_exp2a_plot(
    encoding_performances: dict,
    dataset: str,
    encoding_types: list[str] = ["efg", "hoeg"],
    figsize: tuple[int, int] = (20, 5),
    fontsize: Union[float, int] = 15,
    legend_location:Optional[str]=None,
    save: bool = True,
) -> None:
    enc_perf_on_dataset = encoding_performances[dataset]
    efg_pos = []
    efg_data = []
    splits = enc_perf_on_dataset["efg"]
    for i, split in enumerate(splits, start=1):
        efg_pos.append(i)
        split_scores = (
            np.array(splits[split]).T[-1].tolist()
        )  # of the split HP tuning results take the MAE scores
        efg_data += [split_scores]

    hoeg_pos = []
    hoeg_data = []
    splits = enc_perf_on_dataset["hoeg"]
    for i, split in enumerate(splits, start=4):
        hoeg_pos.append(i)
        split_scores = (
            np.array(splits[split]).T[-1].tolist()
        )  # of the split HP tuning results take the MAE scores
        hoeg_data += [split_scores]

    data = efg_data + hoeg_data
    pos = efg_pos + hoeg_pos

    violin_split_colors = [YELLOW, ORANGE, RED] * 2
    violinplot = plt.violinplot(
        dataset=data, positions=pos, showextrema=True, showmedians=True, widths=0.25
    )

    for color, v in zip(violin_split_colors, violinplot["bodies"]):
        v.set_facecolor(color)
        v.set_edgecolor(color)
        v.set_alpha(0.5)

    for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
        vp = violinplot[partname]
        vp.set_edgecolor("black")
        vp.set_alpha(0.45)
        # vp.set_linewidth(1)

    # Customize plot
    title_fontsize = (
        fontsize * 1.1 if type(fontsize) == int or type(fontsize) == float else fontsize
    )
    legend_fontsize = (
        fontsize * 0.85
        if type(fontsize) == int or type(fontsize) == float
        else fontsize
    )
    ticks_fontsize = (
        fontsize * 0.9 if type(fontsize) == int or type(fontsize) == float else fontsize
    )
    plt.title(f"Encoding Performance\nDistribution on {dataset if dataset!='Financial Institution' else 'FI'} OCEL", fontsize=title_fontsize)
    plt.xticks(
        [1, 2, 3, 4, 5, 6],
        ["", encoding_types[0].upper(), "", "", encoding_types[1].upper(), ""],
        fontsize=ticks_fontsize,
    )
    plt.ylabel("Mean Absolute Error", fontsize=fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.tight_layout()
    plt.legend(list(splits.keys()), fontsize=legend_fontsize,loc=legend_location)
    if save:
        plt.savefig(
            f"visualizations/plots/exp2_encoding_type/performance/{dataset}.pdf"
        )
    plt.show()


def get_exp2b_plot(
    dataset: str,
    data_base_path: str = "visualizations/data/learning_curves",
    encoding_colors: dict[str, tuple] = {
        "efg": (BURGUNDY, ORANGE),
        "hoeg": (DARKBLUE, BLUE),
    },
    figsize: tuple[int, int] = (10, 6),
    fontsize: Optional[Union[float, int]] = None,
    save: bool = True,
) -> None:
    # Read data from CSV files
    encodings = list(encoding_colors.keys())
    train_data_efg = (
        train_data_hoeg
    ) = validation_data_efg = validation_data_hoeg = pd.DataFrame()
    for encoding in encodings:
        for root, _, files in os.walk(f"{data_base_path}/{dataset}/{encoding}"):
            for file in files:
                if "valid" in file and encoding == "efg":
                    validation_data_efg = pd.read_csv(os.path.join(root, file))
                elif "train" in file and encoding == "efg":
                    train_data_efg = pd.read_csv(os.path.join(root, file))
                elif "valid" in file and encoding == "hoeg":
                    validation_data_hoeg = pd.read_csv(os.path.join(root, file))
                elif "train" in file and encoding == "hoeg":
                    train_data_hoeg = pd.read_csv(os.path.join(root, file))

    plt.figure(figsize=figsize)  # Set the figure size

    # Iterate over encoding types
    for encoding_type, train_data, validation_data, line_style, colors in zip(
        encodings,
        [train_data_efg, train_data_hoeg],
        [validation_data_efg, validation_data_hoeg],
        [(0, (3, 1, 1, 1)), "-"],  # Line styles for efg and hoeg curves
        [
            encoding_colors[encodings[0]],
            encoding_colors[encodings[1]],
        ],  # Line colors for efg and hoeg curves
    ):
        # Plot Train Curve
        plt.plot(
            train_data["Step"],
            train_data["Value"],
            label=f"{encoding_type.upper()} Train Loss",
            linestyle=line_style,
            color=colors[0],
        )

        # Plot Validation Curve
        plt.plot(
            validation_data["Step"],
            validation_data["Value"],
            label=f"{encoding_type.upper()} Validation Loss",
            linestyle=line_style,
            color=colors[1],
        )

    # Customize plot
    title_fontsize = (
        fontsize * 1.1 if type(fontsize) == int or type(fontsize) == float else fontsize
    )
    legend_fontsize = (
        fontsize * 0.75
        if type(fontsize) == int or type(fontsize) == float
        else fontsize
    )
    ticks_fontsize = (
        fontsize * 0.9 if type(fontsize) == int or type(fontsize) == float else fontsize
    )
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Mean Absolute Error", fontsize=fontsize)
    plt.title(
        f"Learning Curves of Best Performing Models on {dataset} OCEL",
        fontsize=title_fontsize,
    )
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)
    if save:
        plt.savefig(
            f"visualizations/plots/exp2_encoding_type/learning_curve/{dataset}.pdf"
        )
    # Show the plot
    plt.show()


# DATA EXTRACTION FUNCTIONS
def load_experiment_results(
    base_dir,
    hp_key: str = "lr",
    hyperparameters: list[str] = ["learning_rate", "hidden_dimensions"],
) -> dict:
    """
    indicate base directory (e.g.: .../exp_v1)
    list all folders
      extract configurations from them
      create dict based on these configs
      (e.g.: {
              'lr=0.01': {
                  'hidden_dim=8': {
                      'evaluation_report': <the JSON file should be loaded here>,
                      'experiment_settings': experiment_settings,
                      'model_architecture': model_architecture,
                      },
                  'hidden_dim=16': {
                      'evaluation_report': <the JSON file should be loaded here>,
                      'experiment_settings': experiment_settings,
                      'model_architecture': model_architecture,
                      }
              },
              'lr=0.001': {
                  'hidden_dim=8': {
                      'evaluation_report': <the JSON file should be loaded here>,
                      'experiment_settings': experiment_settings,
                      'model_architecture': model_architecture,
                      },
              },
            }
      )
    """

    experiment_results = {}
    # Iterate through all subdirectories and files in the base directory
    for root, dirs, files in os.walk(base_dir):
        if "run" in dirs:
            hp_tuning_runs = [
                part for part in root.split("/") if part.startswith(hp_key)
            ]
            if len(hp_tuning_runs):
                # retrieve hyperparameter values from folder name
                hp_strings = hp_tuning_runs[0].split("_")
                param_values = [p.split("=")[1] for p in [hp_strings[0], hp_strings[2]]]
                hyper_params = [
                    f"{hp}={val}" for hp, val in zip(hyperparameters, param_values)
                ]
                # create nested structure of output dictionary
                if hyper_params[0] not in experiment_results:
                    experiment_results[hyper_params[0]] = {}
                    experiment_results[hyper_params[0]][hyper_params[1]] = {}
                else:
                    if hyper_params[1] not in experiment_results[hyper_params[0]]:
                        experiment_results[hyper_params[0]][hyper_params[1]] = {}
                # append relevant JSON files
                for file in files:
                    if file.endswith(".json"):
                        file_name = file.split(".json")[0]
                        json_file_path = os.path.join(root, file)
                        with open(json_file_path, "r") as f:
                            file_as_dict = json.load(f)
                        experiment_results[hyper_params[0]][hyper_params[1]][
                            file_name
                        ] = file_as_dict

    return experiment_results


def _get_run_summary(run: dict) -> dict[str, list[float]]:
    summary_per_split = {k: [] for k in ["Train", "Validation", "Test"]}
    for split in summary_per_split:
        mae = run["evaluation_report"][split]["report"]["MAE"]
        hd = run["experiment_settings"]["hidden_dim"]
        n_params = run["model_architecture"]["Number of parameters"]
        summary_per_split[split] = [hd, n_params, mae]
    return summary_per_split


def get_exp1_data(run_results: dict) -> dict:
    exp1_data = {"learning_rate=0.01": [], "learning_rate=0.001": []}
    for lr, hidden_dims in run_results.items():
        for hidden_dim, run in hidden_dims.items():
            if "evaluation_report" in run:
                exp1_data[lr].append(_get_run_summary(run)["Test"])
    return exp1_data


def get_exp2_data(run_results: dict, return_encoding_comparison: bool = False) -> dict:
    exp2_data = {
        lr_key: {split: [] for split in ["Train", "Validation", "Test"]}
        for lr_key in ["learning_rate=0.01", "learning_rate=0.001"]
    }  # {"learning_rate=0.01": {"Train": [], ...}, ...}
    for lr, hidden_dims in run_results.items():
        for hidden_dim, run in hidden_dims.items():
            if "evaluation_report" in run:
                for split in exp2_data[lr]:
                    exp2_data[lr][split].append(_get_run_summary(run)[split])
    if return_encoding_comparison:
        encoding_comparison = dict()
        for lr, splits in exp2_data.items():
            for split, mae_scores in splits.items():
                if split not in encoding_comparison:
                    encoding_comparison[split] = mae_scores
                encoding_comparison[split] += mae_scores
        for split in encoding_comparison:
            scores = np.unique(encoding_comparison[split], axis=0).tolist()
            scores = [[int(xs[0]), int(xs[1]), xs[2]] for xs in scores]
            encoding_comparison[split] = scores
        exp2_data = encoding_comparison
    return exp2_data
