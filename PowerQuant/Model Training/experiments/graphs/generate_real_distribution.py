"""
Experiment 03: Real Data Distribution Figures

Generates two figures showing power distributions from real GPU data:
1. Raw power distribution - KDE curves showing different ranges per architecture
2. Quantile distribution - KDE curves showing uniform collapse after transformation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
    ranks = np.argsort(np.argsort(values)) + 1
    return ranks / (n + 1)


# =============================================================================
# Style Configuration: Okabe-Ito colorblind-friendly palette
# =============================================================================

ARCH_COLORS = {
    'adalovelace': '#0072B2',  # blue
    'ampere': '#D55E00',       # vermillion
    'k80': '#009E73',          # bluish green
    'tesla': '#E69F00',        # orange
}

# Display names for legend (optional: can capitalize or rename)
ARCH_LABELS = {
    'adalovelace': 'Ada Lovelace',
    'ampere': 'Ampere',
    'k80': 'K80',
    'tesla': 'Tesla',
}


# =============================================================================
# Data Loading
# =============================================================================

def load_power_data(data_path='data/combined_df.csv'):
    """Load power data from CSV file.

    Args:
        data_path: Path to the combined_df.csv file.

    Returns:
        Dictionary mapping architecture name to power values array.
    """
    df = pd.read_csv(data_path)

    power_data = {}
    for arch in df['Architecture'].unique():
        power_data[arch] = df[df['Architecture'] == arch]['Avg'].values

    return power_data


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_kde(ax, values, color, label, fill_alpha=0.3, line_alpha=0.9):
    """Plot a KDE curve with filled area.

    Args:
        ax: Matplotlib axes object.
        values: Array of values to compute KDE for.
        color: Color for the curve and fill.
        label: Legend label.
        fill_alpha: Transparency for filled area.
        line_alpha: Transparency for line.
    """
    kde = gaussian_kde(values)

    # Create x range for plotting
    x_min, x_max = values.min(), values.max()
    padding = (x_max - x_min) * 0.1
    x = np.linspace(x_min - padding, x_max + padding, 500)

    y = kde(x)

    # ax.fill_between(x, y, alpha=fill_alpha, color=color)
    ax.plot(x, y, color=color, linewidth=2, alpha=line_alpha, label=label)


def generate_raw_power_figure(power_data, output_dir='results/exp03'):
    """Generate Figure 1: Raw power distribution.

    Args:
        power_data: Dictionary mapping architecture to power values.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PDF figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot KDE for each architecture
    for arch in ['adalovelace', 'ampere', 'k80', 'tesla']:
        if arch in power_data:
            plot_kde(ax, power_data[arch], ARCH_COLORS[arch], ARCH_LABELS[arch])

    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

    # Set x-axis to show full range
    ax.set_xlim(0, 280)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / 'real_power_distribution.pdf'
    png_path = output_path / 'real_power_distribution.png'

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"Figure saved to: {pdf_path}")
    print(f"Figure saved to: {png_path}")

    plt.close(fig)
    return pdf_path


def generate_quantile_figure(power_data, output_dir='results/exp03'):
    """Generate Figure 2: Quantile distribution (uniform collapse).

    Args:
        power_data: Dictionary mapping architecture to power values.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PDF figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Compute quantiles and plot KDE for each architecture
    for arch in ['adalovelace', 'ampere', 'k80', 'tesla']:
        if arch in power_data:
            quantiles = encode_to_quantiles(power_data[arch])
            plot_kde(ax, quantiles, ARCH_COLORS[arch], ARCH_LABELS[arch])

    ax.set_xlabel('Quantile')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

    # Set axes for [0, 1] range
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)

    # Add reference line at density = 1 (uniform distribution)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='Uniform')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / 'real_quantile_distribution.pdf'
    png_path = output_path / 'real_quantile_distribution.png'

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"Figure saved to: {pdf_path}")
    print(f"Figure saved to: {png_path}")

    plt.close(fig)
    return pdf_path


def generate_figures(data_path='data/combined_df.csv', output_dir='results/exp03',
                     show=False):
    """Generate both distribution figures.

    Args:
        data_path: Path to the combined_df.csv file.
        output_dir: Directory to save the figures.
        show: If True, display the figures interactively.

    Returns:
        Tuple of paths to the saved PDF figures.
    """
    # Load data
    power_data = load_power_data(data_path)

    print(f"Loaded power data for {len(power_data)} architectures:")
    for arch, values in power_data.items():
        print(f"  {arch}: {len(values)} samples, range [{values.min():.0f}, {values.max():.0f}] W")

    # Generate figures
    raw_path = generate_raw_power_figure(power_data, output_dir)
    quantile_path = generate_quantile_figure(power_data, output_dir)

    if show:
        plt.show()

    return raw_path, quantile_path


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate real data distribution figures for quantile transfer learning.'
    )
    parser.add_argument('--data-path', type=str, default='data/combined_df.csv',
                        help='Path to the combined_df.csv file.')
    parser.add_argument('--output-dir', type=str, default='results/exp03',
                        help='Directory to save the figures.')
    parser.add_argument('--show', action='store_true',
                        help='Display the figures interactively.')

    args = parser.parse_args()

    generate_figures(data_path=args.data_path, output_dir=args.output_dir,
                     show=args.show)
