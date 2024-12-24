import os
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Plot residuals from a random bootstrap model')
    parser.add_argument('--dataset_name', type=str, default='Omega_processed', help='Name of the dataset')
    parser.add_argument('--y_name', type=str, default='Omega', help='Target variable column name')
    return parser.parse_args()

def plot_residual_distribution(predictions_df, results_dir, dataset_name, model_num, args):
    """Plot residual distribution for train/val/test sets"""
    
    # Calculate residuals
    predictions_df['residuals'] = predictions_df[args.y_name] - predictions_df['pred']
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot distribution for each set
    for label in ['train', 'val', 'test']:
        subset = predictions_df[predictions_df['label'] == label]
        sns.kdeplot(data=subset['residuals'], 
                   label=label, 
                   fill=True)
    
    plt.title(f'Residual Distribution of Bootstrap Model {model_num} - {dataset_name}')
    plt.xlabel('Residuals (True - Predicted)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    
    # Add statistics as text
    stats_text = []
    for label in ['train', 'val', 'test']:
        subset = predictions_df[predictions_df['label'] == label]
        mean = subset['residuals'].mean()
        std = subset['residuals'].std()
        stats_text.append(f'{label}: μ={mean:.4f}, σ={std:.4f}')
    
    plt.text(0.02, 0.98, '\n'.join(stats_text),
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{dataset_name}_bootstrap_{model_num}_residuals.png'
    plt.savefig(os.path.join(results_dir, filename))
    print(f"Residual distribution plot saved to {os.path.join(results_dir, filename)}")
    plt.close()
    
    # Print statistics
    print("\nResidual Statistics:")
    for label in ['train', 'val', 'test']:
        subset = predictions_df[predictions_df['label'] == label]
        print(f"\n{label.upper()}:")
        print(f"Mean: {subset['residuals'].mean():.4f}")
        print(f"Std:  {subset['residuals'].std():.4f}")
        print(f"Min:  {subset['residuals'].min():.4f}")
        print(f"Max:  {subset['residuals'].max():.4f}")

def main():
    args = parse_args()
    
    # Define paths
    base_dir = '/zhome/32/3/215274/jinlia/project/Adem/bootstrap_results_fromRef'
    predictions_dir = os.path.join(base_dir, args.dataset_name, 'predictions')
    results_dir = os.path.join(base_dir, args.dataset_name)
    
    # Get list of prediction files
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.xlsx')]
    
    if not prediction_files:
        print(f"No prediction files found in {predictions_dir}")
        return
    
    # Randomly select one file
    random_file = random.choice(prediction_files)
    # Extract model number from filename (assuming format 'bootstrap_X.xlsx')
    model_num = random_file.split('_')[1].split('.')[0]
    print(f"\nSelected bootstrap model {model_num}: {random_file}")
    
    # Read predictions from Excel file with specific sheet
    predictions_df = pd.read_excel(
        os.path.join(predictions_dir, random_file),
        sheet_name='Predictions'
    )
    
    # Plot residual distribution
    plot_residual_distribution(predictions_df, results_dir, args.dataset_name, model_num, args)

if __name__ == "__main__":
    print("Starting random bootstrap residual analysis...")
    main()
    print("Analysis completed.") 