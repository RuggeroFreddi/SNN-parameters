import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

TASK = "MNIST"  # possible values: "MNIST", "TRAJECTORY"
OUTPUT_FEATURES = "statistics"  # possible values: "statistics", "trace"
PARAM_NAME = "beta"  # possible value: "beta", "membrane_threshold", "current_amplitude"
NUM_WEIGHT_STEPS = 51
DATE = "2025_11_10"

RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"
CSV_NAME = os.path.join(RESULTS_DIR, f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv")
YAML_NAME = os.path.join(RESULTS_DIR, "experiment_metadata.yaml")


def load_metadata(yaml_path: str):
    with open(yaml_path, "r") as file:
        metadata = yaml.safe_load(file)
    return metadata


def plot_accuracy_model(results_df: pd.DataFrame, metadata: dict,
                        acc_col: str, std_col: str, model_name: str, filename: str):
    """
    Disegna accuracy vs peso con area (std) per tutti i valori del parametro testato.
    acc_col e std_col sono i nomi delle colonne, es. 'accuracy_rf', 'std_accuracy_rf'
    """
    param_values = metadata["tested_parameter"]["values"]
    accuracy_threshold = metadata["global_parameters"]["accuracy_threshold"]
    I = metadata["experiment"]["mean_I"]
    membrane_threshold = metadata["global_parameters"]["membrane_threshold"]
    refractory_period = metadata["global_parameters"]["refractory_period"]
    small_world_graph_k = metadata["global_parameters"]["small_world_graph_k"]
    num_neurons = metadata["global_parameters"]["num_neurons"]

    # stessa formula che avevi tu
    w_critical = (membrane_threshold - 2 * (I / num_neurons) * refractory_period) / (small_world_graph_k / 2)

    plt.figure()

    for value in param_values:
        parameter_df = results_df[results_df["param_value"] == float(value)].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        line, = plt.plot(
            parameter_df["weight"],
            parameter_df[acc_col],
            marker="o",
            label=f"{PARAM_NAME}={value}",
        )

        # shaded area con la std
        if std_col in parameter_df.columns:
            lower = parameter_df[acc_col] - parameter_df[std_col]
            upper = parameter_df[acc_col] + parameter_df[std_col]
            plt.fill_between(
                parameter_df["weight"],
                lower,
                upper,
                color=line.get_color(),
                alpha=0.2,
            )

        # segmento sopra la soglia relativa (rispetto al max di questo parametro)
        max_accuracy = parameter_df[acc_col].max()
        threshold = accuracy_threshold * max_accuracy
        eligible = parameter_df[parameter_df[acc_col] >= threshold]
        if not eligible.empty:
            w1 = eligible["weight"].min()
            w2 = eligible["weight"].max()
            plt.hlines(
                y=threshold,
                xmin=w1,
                xmax=w2,
                colors="black",
                linestyles="dashed",
            )

    # linea del peso critico
    plt.axvline(
        x=w_critical,
        color="red",
        linestyle="--",
        label="critical weight",
    )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean CV accuracy")
    plt.title(f"{model_name}: accuracy vs weight for different {PARAM_NAME} values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(plot_path)
    print(f"saved {plot_path}")


def plot_spike_count(results_df: pd.DataFrame, metadata: dict):
    param_values = metadata["tested_parameter"]["values"]

    plt.figure()

    for value in param_values:
        parameter_df = results_df[results_df["param_value"] == float(value)].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        plt.plot(
            parameter_df["weight"],
            parameter_df["spike_count"],
            marker="o",
            label=f"{PARAM_NAME}={value}",
        )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean spike count")
    plt.title(f"Spike count vs weight for different {PARAM_NAME} values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "plot_spike_count.png")
    plt.savefig(plot_path)
    print(f"saved {plot_path}")


def main():
    results_df = pd.read_csv(CSV_NAME)
    metadata = load_metadata(YAML_NAME)

    # grafico per random forest
    plot_accuracy_model(
        results_df,
        metadata,
        acc_col="accuracy_rf",
        std_col="std_accuracy_rf",
        model_name="Random Forest",
        filename="plot_accuracy_rf.png",
    )

    # grafico per single-layer perceptron
    plot_accuracy_model(
        results_df,
        metadata,
        acc_col="accuracy_slp",
        std_col="std_accuracy_slp",
        model_name="Single-layer perceptron",
        filename="plot_accuracy_slp.png",
    )

    # opzionale: spike
    plot_spike_count(results_df, metadata)

    plt.show()


if __name__ == "__main__":
    main()
