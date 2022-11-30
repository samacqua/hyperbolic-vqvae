import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import wishart


plt.style.use('seaborn-darkgrid')


def plot_initializations(model, N=40, conv_n=2, n_codebooks=100):
    """Plots the input distribution, embedding, and quantized embedding for different initializations."""

    N_BINS = 14

    # Set up plots.
    fig, axes = plt.subplots(3, 2)
    (ax0, ax1), (ax2, ax3), (ax4, ax5) = axes
    for i, ax in enumerate((ax0, ax1, ax2, ax3, ax4, ax5)):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # if i == 0:
        #     ax.set_xlim([-3.4, 3.4])
        # else:
        #     ax.set_xlim([-2.4, 2.4])

    # Generate input data.
    in_data = np.random.multivariate_normal(np.zeros(N), np.eye(N))
    ax0.hist(in_data, bins=N_BINS, density=True)
    ax0.set(title='input $x$')

    # Generate simple embedding.
    # conv = np.random.multivariate_normal(np.zeros(conv_n), np.eye(conv_n))
    conv = (np.random.uniform(size=conv_n) - 0.0) * 1
    z_e = np.maximum(0, np.convolve(in_data, conv))
    ax1.hist(z_e, bins=N_BINS, density=True)
    ax1.set(title='embedding $z_e$')

    def quantize(data, cb):
        z_q = np.zeros_like(data)
        for i in range(N):
            cbi = np.argmin(np.abs(cb - data[i]) ** 2)
            z_q[i] = cb[cbi]

        return z_q

    # Generate uniform quantization [-1, 1].
    codebooks_uniform = (np.random.uniform(size=n_codebooks) - 0.5) * 2
    z_q_uniform = quantize(z_e, codebooks_uniform)

    ax2.hist(z_q_uniform, bins=N_BINS, density=True)
    ax2.set(title='$z_q \sim C_{uniform}$')

    # Generate normal quantization N(0, I).
    codebooks_normal = np.random.multivariate_normal(np.zeros(n_codebooks), np.eye(n_codebooks))
    z_q_normal = quantize(z_e, codebooks_normal)

    ax3.hist(z_q_normal, bins=N_BINS, density=True)
    ax3.set(title='$z_q \sim C_{normal}$')

    # Generate wishart quantization.
    codebooks_wishart = None
    # z_q_wishart = quantize(z_e, codebooks_wishart)
    #
    # ax4.hist(z_q_wishart, bins=N_BINS, density=True)
    ax4.set(title='$z_q \sim C_{wishart}$')

    # Generate data quantization.
    codebooks_data = np.random.choice(z_e, size=n_codebooks)
    z_q_data = quantize(z_e, codebooks_data)

    ax5.hist(z_q_data, bins=N_BINS, density=True)
    ax5.set(title='$z_q \sim C_{data}$')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_initializations(None, 1000)
