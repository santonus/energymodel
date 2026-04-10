#!/usr/bin/env python3
"""
PowerQuant Build System

Usage:
    python build.py collect-dataset [--dataset DATASET]
    python build.py build-model <exp> [--file FILE] [--dataset DATASET]
    python build.py list-experiments
    python build.py --help
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Setup
console = Console()
ROOT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = str(ROOT_DIR)

# Load environment variables
load_dotenv(ROOT_DIR / ".env")
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# Directories
DATASET_2_DIR = ROOT_DIR / "Dataset Collection" / "Dataset-2(works for Python PyTorch Models)" / "Benchmark Suite" / "KernelBench"
MODEL_TRAINING_DIR = ROOT_DIR / "Model Training"
DATA_DIR = MODEL_TRAINING_DIR / "data"
KERNELBENCH_SCRIPTS_DIR = DATASET_2_DIR / "scripts"


def setup_data_symlink():
    """Create symlink to dataset if it doesn't exist."""
    dataset_source = ROOT_DIR / "Dataset Collection" / "Dataset-2(works for Python PyTorch Models)" / "Dataset"
    if dataset_source.exists() and not DATA_DIR.exists():
        os.makedirs(DATA_DIR.parent, exist_ok=True)
        try:
            os.symlink(dataset_source, DATA_DIR)
            console.print(f"[green]✓[/green] Created symlink: {DATA_DIR} -> {dataset_source}")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Could not create symlink: {e}")


def run_command(cmd: List[str], cwd: Optional[Path] = None, description: str = "") -> int:
    """Run a shell command and return exit code."""
    if description:
        console.print(f"\n[cyan]→[/cyan] {description}")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    
    try:
        # Set PYTHONPATH and PROJECT_ROOT for Model Training directory
        env = os.environ.copy()
        if cwd:
            env["PYTHONPATH"] = str(cwd)
            # Set PROJECT_ROOT to Model Training directory for training scripts
            env["PROJECT_ROOT"] = str(cwd)
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        return result.returncode
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        return 1


@click.group()
@click.version_option()
def cli():
    """PowerQuant Build System - Manage data collection and model training."""
    pass


@cli.command()
@click.option(
    "--dataset",
    type=str,
    default="data/combined_static_20260127_092347.csv",
    help="Dataset CSV file (relative to Model Training/)"
)
def collect_dataset(dataset: str):
    """Collect baseline timing data from KernelBench models."""
    setup_data_symlink()
    
    script_path = KERNELBENCH_SCRIPTS_DIR / "generate_baseline_time.py"
    if not script_path.exists():
        console.print(f"[red]✗ Script not found:[/red] {script_path}")
        sys.exit(1)
    
    # Change to KernelBench directory for proper imports
    console.print(f"\n[bold cyan]Collecting Dataset[/bold cyan]")
    console.print(f"Script: {script_path}")
    console.print(f"Working directory: {DATASET_2_DIR}")
    
    exit_code = run_command(
        ["python", str(script_path)],
        cwd=DATASET_2_DIR,
        description="Running generate_baseline_time.py"
    )
    
    if exit_code == 0:
        console.print(f"[green]✓[/green] Dataset collection completed successfully")
    else:
        console.print(f"[red]✗[/red] Dataset collection failed with exit code {exit_code}")
    
    sys.exit(exit_code)


@cli.command()
@click.argument("exp", type=click.Choice(["in-distribution", "out-of-distribution"]))
@click.option(
    "--file",
    type=str,
    default=None,
    help="Specific file to run (e.g., run_catboost.py). If not specified, runs all."
)
@click.option(
    "--dataset",
    type=click.Choice(["dataset-1", "dataset-2"]),
    required=True,
    help="Dataset type: 'dataset-1' for C++ CUDA or 'dataset-2' for PyTorch"
)
@click.option(
    "--test-arch",
    type=int,
    default=None,
    help="(Out-of-distribution only) Architecture index to use as test set (0-3)"
)
def build_model(exp: str, file: Optional[str], dataset: str, test_arch: Optional[int]):
    """Build/train model for a specific experiment.
    
    Examples:
        # In-distribution training with Dataset-1
        python build.py build-model in-distribution --dataset dataset-1
        
        # Out-of-distribution training with Dataset-2, test on architecture 0
        python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0
    """
    setup_data_symlink()
    
    # Map dataset choice to file path
    dataset_paths = {
        "dataset-1": "data/combined_df.csv",
        "dataset-2": "data/combined_static_20260127_092347.csv"
    }
    dataset_path = dataset_paths[dataset]
    
    # Validate experiment - convert hyphenated name to underscored folder name
    folder_name = exp.replace("-", "_")
    exp_dir = MODEL_TRAINING_DIR / "experiments" / f"exp_{folder_name}"
    if not exp_dir.exists():
        console.print(f"[red]✗ Experiment not found:[/red] {exp_dir}")
        console.print(f"[dim]Available experiments:[/dim]")
        console.print(f"  - in-distribution")
        console.print(f"  - out-of-distribution")
        sys.exit(1)
    
    # Validate test-arch for out-of-distribution
    if exp == "out-of-distribution":
        if test_arch is None:
            test_arch = 0
            console.print(f"[yellow]⚠[/yellow] No test-arch specified, using default: 0")
    elif test_arch is not None:
        console.print(f"[yellow]⚠[/yellow] test-arch is only used for out-of-distribution experiments")
    
    console.print(f"\n[bold cyan]Building Model[/bold cyan]")
    console.print(f"Experiment: {exp}")
    console.print(f"Directory: {exp_dir}")
    console.print(f"Dataset: {dataset} ({dataset_path})")
    if exp == "out-of-distribution":
        console.print(f"Test Architecture: {test_arch}")
    
    # Find files to run
    if file:
        # Run specific file
        script_path = exp_dir / file
        if not script_path.exists():
            console.print(f"[red]✗ File not found:[/red] {script_path}")
            console.print(f"[dim]Available files in exp_{exp}:[/dim]")
            for f in sorted(exp_dir.glob("run_*.py")):
                console.print(f"  - {f.name}")
            sys.exit(1)
        files_to_run = [script_path]
    else:
        # Run all run_*.py files
        files_to_run = sorted(exp_dir.glob("run_*.py"))
        if not files_to_run:
            console.print(f"[red]✗ No run_*.py files found in[/red] {exp_dir}")
            sys.exit(1)
    
    # Run each file
    failed = []
    succeeded = []
    
    for script_path in files_to_run:
        console.print(f"\n[cyan]→ Running:[/cyan] {script_path.name}")
        
        # Build command
        cmd = ["python", str(script_path), "--dataset", dataset_path]
        if exp == "out-of-distribution":
            cmd.extend(["--test-arch", str(test_arch)])
        
        exit_code = run_command(
            cmd,
            cwd=MODEL_TRAINING_DIR,
            description=f"Training {script_path.stem}"
        )
        
        if exit_code == 0:
            console.print(f"[green]✓[/green] {script_path.name} completed")
            succeeded.append(script_path.name)
        else:
            console.print(f"[red]✗[/red] {script_path.name} failed with exit code {exit_code}")
            failed.append(script_path.name)
    
    # Summary
    console.print(f"\n[bold cyan]Summary[/bold cyan]")
    console.print(f"[green]Succeeded:[/green] {len(succeeded)}/{len(files_to_run)}")
    for name in succeeded:
        console.print(f"  [green]✓[/green] {name}")
    
    if failed:
        console.print(f"[red]Failed:[/red] {len(failed)}/{len(files_to_run)}")
        for name in failed:
            console.print(f"  [red]✗[/red] {name}")
        sys.exit(1)
    else:
        console.print(f"[green]✓ All files completed successfully[/green]")


@cli.command()
def list_experiments():
    """List available experiments."""
    exp_dir = MODEL_TRAINING_DIR / "experiments"
    
    if not exp_dir.exists():
        console.print(f"[red]✗ Experiments directory not found:[/red] {exp_dir}")
        sys.exit(1)
    
    console.print(f"\n[bold cyan]Available Experiments[/bold cyan]\n")
    
    experiments = ["in-distribution", "out-of-distribution"]
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Experiment", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Files", style="green")
    
    for exp_name in experiments:
        # Convert hyphenated name to underscored folder name
        folder_name = exp_name.replace("-", "_")
        exp_path = exp_dir / f"exp_{folder_name}"
        if exp_path.exists():
            files = list(exp_path.glob("run_*.py"))
            desc = "Train and test on same architecture distribution" if exp_name == "in-distribution" else "Test on unseen architecture"
            table.add_row(
                exp_name,
                desc,
                str(len(files))
            )
    
    console.print(table)
    
    console.print(f"\n[dim]Usage examples:[/dim]")
    console.print(f"  python build.py build-model in-distribution --dataset dataset-1")
    console.print(f"  python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0")


@cli.command()
def status():
    """Check build system status."""
    console.print(f"\n[bold cyan]PowerQuant Build System Status[/bold cyan]\n")
    
    checks = {
        "Root Directory": (ROOT_DIR, ROOT_DIR.exists()),
        "Dataset-2": (DATASET_2_DIR, DATASET_2_DIR.exists()),
        "KernelBench Scripts": (KERNELBENCH_SCRIPTS_DIR, KERNELBENCH_SCRIPTS_DIR.exists()),
        "Model Training": (MODEL_TRAINING_DIR, MODEL_TRAINING_DIR.exists()),
        "Data Symlink": (DATA_DIR, DATA_DIR.exists()),
        "In-Distribution Exp": (MODEL_TRAINING_DIR / "experiments" / "exp_in_distribution", (MODEL_TRAINING_DIR / "experiments" / "exp_in_distribution").exists()),
        "Out-Of-Distribution Exp": (MODEL_TRAINING_DIR / "experiments" / "exp_out_of_distribution", (MODEL_TRAINING_DIR / "experiments" / "exp_out_of_distribution").exists()),
    }
    
    for name, (path, exists) in checks.items():
        status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
        console.print(f"{status_icon} {name}")
    
    console.print(f"\n[cyan]Use 'python build.py list-experiments' to see available experiments[/cyan]")


if __name__ == "__main__":
    # Ensure PROJECT_ROOT is set
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
