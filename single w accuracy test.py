import os
import numpy as np
import pandas as pd

from snnpy.snn import SimulationParams
from utils.simulates import simulate_trace, simulate_statistic_features
from utils.cross_validations import cross_validation_rf


DATASET_PATH = "dati/mnist_rate_encoded.npz"
CV_NUM_SPLITS = 10

NUM_NEURONS = 1000
MEMBRANE_THRESHOLD = 2
REFRACTORY_PERIOD = 2
NUM_OUTPUT_NEURONS = 35
LEAK_COEFFICIENT = 0
CURRENT_AMPLITUDE = MEMBRANE_THRESHOLD
PRESYNAPTIC_DEGREE = 0.2
SMALL_WORLD_GRAPH_P = 0.2

TRACE_TAU = 60


def load_dataset(filename: str):
    """
    Return (data, labels) from a .npz file.
    """
    npz_data = np.load(filename)
    return npz_data["X"], npz_data["y"]


def compute_critical_weight(inputs: np.ndarray):
    """
    Estimate the average input current and the critical synaptic weight.
    """
    # average current per synapse / tick
    avg_input_current = (
        np.sum(inputs)
        / (NUM_NEURONS * inputs.shape[0] * inputs.shape[2])
    )

    critical_weight = (
        MEMBRANE_THRESHOLD
        - 2 * avg_input_current * CURRENT_AMPLITUDE * (REFRACTORY_PERIOD + 1)
    ) / (PRESYNAPTIC_DEGREE * NUM_NEURONS)

    return avg_input_current, critical_weight

def main():
    os.makedirs("dati", exist_ok=True)

    data, labels = load_dataset(DATASET_PATH)
    print(f"Loaded data: {data.shape}, labels: {labels.shape}")

    _, critical_weight = compute_critical_weight(data)
    print(critical_weight)
    # critical_weight = 0.0038 

    small_world_graph_k = int(PRESYNAPTIC_DEGREE * NUM_NEURONS * 2)

    sim_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=critical_weight,
        weight_variance=critical_weight * 5,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_GRAPH_P,
        small_world_graph_k=small_world_graph_k,
        input_spike_times=np.zeros(
            (data.shape[1], data.shape[2]),
            dtype=np.uint8,
        ),
    )
    
    """trace_dataset, _ = simulate_trace(
            data=data,
            labels=labels,
            parameters=sim_params,
            trace_tau=TRACE_TAU,
    )"""

    trace_df, avg_spike_count = simulate_statistic_features(
        data=data,
        labels=labels,
        parameters=sim_params,
    )

    print("avg spike count:", avg_spike_count)

    mean_accuracy = cross_validation_rf(trace_df, CV_NUM_SPLITS)
    print("mean accuracy:", mean_accuracy)

if __name__ == "__main__":
    main()
