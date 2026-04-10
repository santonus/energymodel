"""
Experiment 03: Toy Example for Quantile Transfer Learning

Generates a three-panel figure demonstrating why quantile-based transfer
learning works for cross-architecture power prediction.

Panel (a): Power curves showing different scaling behaviors
Panel (b): Raw pairwise scatter showing non-linear relationships
Panel (c): Quantile pairwise scatter showing rank preservation (diagonal collapse)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# Quantile Encoding (matches src/quant_classifier.py)
# =============================================================================

def encode_to_quantiles(values):
    """Convert values to quantiles using Weibull plotting positions.

    Uses Hyndman-Fan Type 6 (Weibull) formula: q_k = k / (n + 1)
    where k is the rank (1-indexed) and n is the sample size.

    This matches the implementation in src/quant_classifier.py.
    """
    n = len(values)
    # Compute 1-indexed ranks: rank 1 = smallest, rank n = largest
    ranks = np.argsort(np.argsort(values)) + 1
    # Weibull plotting position: k / (n + 1)
    return ranks / (n + 1)


# =============================================================================
# Synthetic Power Functions
# =============================================================================

def power_arch_a(x):
    """Architecture A: Linear scaling (older GPU, modest range)."""
    return 50 + 100 * x


def power_arch_b(x):
    """Architecture B: Quadratic scaling (high-end GPU, power explodes at high intensity)."""
    return 100 + 200 * x**2


def power_arch_c(x):
    """Architecture C: Square root scaling (efficient GPU, diminishing returns)."""
    return 80 + 120 * np.sqrt(x)


# =============================================================================
# Data Generation
# =============================================================================

def generate_synthetic_data(n_samples=50, seed=42):
    """Generate synthetic power data for three architectures.

    Args:
        n_samples: Number of kernel workloads to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'x' (workload intensity) and power values for each architecture.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n_samples)

    return {
        'x': x,
        'P_A': power_arch_a(x),
        'P_B': power_arch_b(x),
        'P_C': power_arch_c(x),
    }


# =============================================================================
# Plotting
# =============================================================================

# Style configuration: Okabe-Ito colorblind-friendly palette
# Reference: https://jfly.uni-koeln.de/color/
COLORS = {
    'A': '#0072B2',  # blue
    'B': '#D55E00',  # vermillion
    'C': '#009E73',  # bluish green
}

PAIR_STYLES = {
    'A-B': {'color': '#0072B2', 'marker': 'o', 'label': 'A–B'},
    'A-C': {'color': '#D55E00', 'marker': 's', 'label': 'A–C'},
    'B-C': {'color': '#009E73', 'marker': '^', 'label': 'B–C'},
}


def plot_power_curves(ax, data):
    """Panel (a): Power curves showing different scaling behaviors."""
    # Sort by x for smooth curves
    x_sorted = np.sort(data['x'])

    ax.plot(x_sorted, power_arch_a(x_sorted), color=COLORS['A'],
            linewidth=2, label='Arch A')
    ax.plot(x_sorted, power_arch_b(x_sorted), color=COLORS['B'],
            linewidth=2, label='Arch B')
    ax.plot(x_sorted, power_arch_c(x_sorted), color=COLORS['C'],
            linewidth=2, label='Arch C')

    # Also plot the actual data points
    ax.scatter(data['x'], data['P_A'], color=COLORS['A'], s=20, alpha=0.5)
    ax.scatter(data['x'], data['P_B'], color=COLORS['B'], s=20, alpha=0.5)
    ax.scatter(data['x'], data['P_C'], color=COLORS['C'], s=20, alpha=0.5)

    ax.set_xlabel('Workload Intensity ($x$)')
    ax.set_ylabel('Power (W)')
    ax.set_title('(a) Power Curves')
    ax.legend(loc='upper left')
    ax.set_xlim(-0.05, 1.05)


def plot_raw_pairwise(ax, data):
    """Panel (b): Raw pairwise scatter showing non-linear relationships."""
    alpha = 0.7
    s = 40

    for pair_key, (p_i, p_j) in [('A-B', ('P_A', 'P_B')),
                                   ('A-C', ('P_A', 'P_C')),
                                   ('B-C', ('P_B', 'P_C'))]:
        style = PAIR_STYLES[pair_key]
        ax.scatter(data[p_i], data[p_j],
                   color=style['color'], marker=style['marker'],
                   s=s, alpha=alpha, label=style['label'])

    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Power (W)')
    ax.set_title('(b) Raw Power: Pairwise')
    ax.legend(loc='upper left')


def plot_quantile_pairwise(ax, data, quantiles):
    """Panel (c): Quantile pairwise scatter showing rank preservation."""
    alpha = 0.7
    s = 40

    for pair_key, (q_i, q_j) in [('A-B', ('Q_A', 'Q_B')),
                                   ('A-C', ('Q_A', 'Q_C')),
                                   ('B-C', ('Q_B', 'Q_C'))]:
        style = PAIR_STYLES[pair_key]
        ax.scatter(quantiles[q_i], quantiles[q_j],
                   color=style['color'], marker=style['marker'],
                   s=s, alpha=alpha, label=style['label'])

    # Add y=x reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='$y = x$')

    ax.set_xlabel('Quantile')
    ax.set_ylabel('Quantile')
    ax.set_title('(c) Quantile: Pairwise')
    ax.legend(loc='upper left')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


def generate_figure(output_dir='results/exp03', show=False):
    """Generate the three-panel figure.

    Args:
        output_dir: Directory to save the figure.
        show: If True, display the figure interactively.

    Returns:
        Path to the saved PDF figure.
    """
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=50, seed=42)

    # Compute quantiles for each architecture
    quantiles = {
        'Q_A': encode_to_quantiles(data['P_A']),
        'Q_B': encode_to_quantiles(data['P_B']),
        'Q_C': encode_to_quantiles(data['P_C']),
    }

    # Create figure with three panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot each panel
    plot_power_curves(axes[0], data)
    plot_raw_pairwise(axes[1], data)
    plot_quantile_pairwise(axes[2], data, quantiles)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / 'toy_quantile_demo.pdf'
    png_path = output_path / 'toy_quantile_demo.png'

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"Figure saved to: {pdf_path}")
    print(f"Figure saved to: {png_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_path


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate toy example figure for quantile transfer learning.'
    )
    parser.add_argument('--output-dir', type=str, default='results/exp03',
                        help='Directory to save the figure.')
    parser.add_argument('--show', action='store_true',
                        help='Display the figure interactively.')

    args = parser.parse_args()

    generate_figure(output_dir=args.output_dir, show=args.show)
