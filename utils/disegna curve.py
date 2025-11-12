import numpy as np
import matplotlib.pyplot as plt

# parametri fissi
N = 1000
T_ref = 2
a = 2
I = (26.203294166666666 / N) * 2   # = 0.05240658833333333...

# valori di b per le tre curve
b_values = [0.2, 0.3, 0.4]

# asse dei pesi sinaptici
# scegliamo un intervallo ragionevole
w = np.linspace(0, 0.2, 500)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, b in zip(axes, b_values):
    # formula dell'ISI
    # isi = 1/(2I) * ((a - w*b*N) + sqrt((a - w*b*N)**2 + 4*I*b*N*w*T_ref))
    term = a - w * b * N
    isi = (term + np.sqrt(term**2 + 4 * I * b * N * w * T_ref)) / (2 * I)

    # spike rate = 1 / isi
    spike = 1.0 / isi

    ax.plot(w, spike)
    ax.set_title(f"b = {b}")
    ax.set_xlabel("w")
    ax.grid(True)

axes[0].set_ylabel("spike (1 / ISI)")

plt.tight_layout()
plt.show()
