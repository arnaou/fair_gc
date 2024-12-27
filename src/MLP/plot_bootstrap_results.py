import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_bootstrap_statistics(bootstrap_metrics, results_dir, dataset_name):
    """Plot bootstrap statistics for each metric and dataset"""
    metrics = list(bootstrap_metrics['train'].keys())
    datasets = ['train', 'val', 'test']
    
    # Create box plots for all metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        data = [bootstrap_metrics[dataset][metric] for dataset in datasets]
        axes[idx].boxplot(data, labels=datasets)
        axes[idx].set_title(f'Bootstrap Distribution of {metric.upper()} for {dataset_name}')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True)
        
        # Add individual points for better visualization
        for i, d in enumerate(data, 1):
            axes[idx].scatter([i] * len(d), d, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{dataset_name}_bootstrap_statistics_boxplot.png'
    plt.savefig(os.path.join(results_dir, filename))
    print(f"Bootstrap statistics boxplot saved to {os.path.join(results_dir, filename)}")
    plt.close()

def moment_analysis_plot(bootstrap_metrics, results_dir, dataset_name):
    """Perform and plot moment analysis on test MAE"""
    test_mae = np.array(bootstrap_metrics['test']['mae'])
    
    def ms_calculator(input_array):
        p = 4  # 1st, 2nd, 3rd and 4th moments
        input_array = input_array.flatten()
        R_r = np.zeros((len(input_array), p))
        R_l = np.zeros((len(input_array), p))

        for i in range(p):
            x = np.abs(input_array) ** (i + 1)
            S = np.cumsum(x)
            M_r = np.maximum.accumulate(x)
            M_l = np.minimum.accumulate(x)
            R_r[:, i] = M_r / S
            R_l[:, i] = M_l / S

        return {'r': R_r, 'l': R_l}

    results = ms_calculator(test_mae)
    x = np.arange(1, len(test_mae) + 1)

    # Plot minimum ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(4):
        ax.plot(x, results['l'][:, i], label=f'R_{i+1}')
    ax.set_xlim(left=1)
    ax.set_title(f'Minimum Ratios for Different Moments - MAE of {dataset_name}')
    ax.set_xlabel('Ensemble size')
    ax.set_ylabel('Rk')
    ax.grid(True)
    ax.legend()
    ax.text(0.5, 0.5, f'MAE of {dataset_name}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    plt.tight_layout()
    filename = f'{dataset_name}_moment_analysis_min.png'
    plt.savefig(os.path.join(results_dir, filename))
    print(f"Moment analysis minimum ratios plot saved to {os.path.join(results_dir, filename)}")
    plt.close()

    # Plot maximum ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(4):
        ax.plot(x, results['r'][:, i], label=f'R{i+1}')
    ax.set_xlim(left=1)
    ax.set_title(f'Maximum Ratios for Different Moments - MAE of {dataset_name}')
    ax.set_xlabel('Ensemble size')
    ax.set_ylabel('Rk')
    ax.grid(True)
    ax.legend()
    ax.text(0.5, 0.5, f'MAE of {dataset_name}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    plt.tight_layout()
    filename = f'{dataset_name}_moment_analysis_max.png'
    plt.savefig(os.path.join(results_dir, filename))
    print(f"Moment analysis maximum ratios plot saved to {os.path.join(results_dir, filename)}")
    plt.close()

def plot_metrics_distribution(bootstrap_metrics, results_dir, dataset_name):
    """Plot distribution of metrics across datasets"""
    metrics = list(bootstrap_metrics['train'].keys())
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric_name in enumerate(metrics):
        for dataset in ['train', 'val', 'test']:
            sns.kdeplot(data=bootstrap_metrics[dataset][metric_name], 
                       label=dataset, ax=axes[idx])
        axes[idx].set_xlabel(metric_name.upper())
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'Bootstrap {metric_name.upper()} Distributions - {dataset_name}')
        axes[idx].legend()
    
    plt.tight_layout()
    filename = f'{dataset_name}_bootstrap_metrics_dist.png'
    plt.savefig(os.path.join(results_dir, filename))
    print(f"Bootstrap metrics distribution plot saved to {os.path.join(results_dir, filename)}")
    plt.close()

def print_statistics(bootstrap_results):
    """Print summary statistics from the bootstrap results"""
    print("\nBootstrap Statistics Summary:")
    for dataset in ['train', 'val', 'test']:
        print(f"\n{dataset.upper()} Dataset:")
        for metric, stats in bootstrap_results['statistics'][dataset].items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")

def main():
    parser = argparse.ArgumentParser(description='Plot bootstrap results from saved JSON file')
    parser.add_argument('--dataset_name', type=str, default='Vc', help='Name of the dataset')
    args = parser.parse_args()

    # Define paths
    base_dir = '/zhome/32/3/215274/jinlia/project/Adem/data/results'
    results_dir = os.path.join(base_dir, args.dataset_name+'_processed_bootstrap')
    json_path = os.path.join(results_dir, 'bootstrap_results.json')

    # Load bootstrap results
    print(f"Loading bootstrap results from {json_path}")
    with open(json_path, 'r') as f:
        bootstrap_results = json.load(f)

    # Extract metrics
    bootstrap_metrics = bootstrap_results['metrics']

    # Create plots
    print("Creating plots...")
    plot_bootstrap_statistics(bootstrap_metrics, results_dir, args.dataset_name)
    plot_metrics_distribution(bootstrap_metrics, results_dir, args.dataset_name)
    moment_analysis_plot(bootstrap_metrics, results_dir, args.dataset_name)

    # Print statistics
    print_statistics(bootstrap_results)

if __name__ == "__main__":
    print("Starting plot_bootstrap_results.py")
    main() 
    print("Finished plot_bootstrap_results.py")