from snnpy.snn import SNN
import numpy as np
import pandas as pd


def simulate_trace(data, labels, parameters, trace_tau):
    # configuration constants
    START_INDEX = 700              # first neuron to keep
    EXCLUDED_TAIL_NEURONS = 200    # how many neurons to drop from the very end

    snn = SNN(parameters)
    initial_membrane_potentials = snn.get_membrane_potentials()

    end_index = parameters.num_neurons - EXCLUDED_TAIL_NEURONS
    kept_indices = list(range(START_INDEX, end_index))
    rows = []

    for i in range(len(data)):
        if i % 100 == 0:
            print(f"processed {i} of {len(data)} samples")

        sample = data[i]
        label = labels[i]

        snn.set_input_spike_times(sample)
        snn.set_membrane_potentials(initial_membrane_potentials)
        snn.simulate(trace_tau=trace_tau, reset_trace=True)

        trace = np.asarray(snn.get_trace()).reshape(-1)

        # select only the neurons we want to export
        selected_trace_values = trace[kept_indices].tolist()

        row = selected_trace_values + [label]
        rows.append(row)

    # column names must match kept indices
    trace_columns = [f"neuron_{idx}_trace" for idx in kept_indices]
    columns = trace_columns + ["label"]

    df = pd.DataFrame(rows, columns=columns)
    return df
