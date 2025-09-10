import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

def plot_treatment_means(csv_file, output_dir="plots", figsize=(10, 6)):
    """
    Read concatenated CSV file and create separate plots for area, perimeter, 
    and circularity means by treatment condition.
    
    Args:
        csv_file (str): Path to the concatenated CSV file
        output_dir (str): Directory to save the plots
        figsize (tuple): Figure size for plots
    """
    
    # Read the data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Check required columns
    required_columns = ['treatment', 'area', 'perimeter', 'circularity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculate means by treatment
    treatment_means = df.groupby('treatment')[['area', 'perimeter', 'circularity']].agg(['mean', 'std', 'count']).reset_index()
    
    print(f"\nTreatment conditions found: {df['treatment'].unique()}")
    print(f"Sample sizes: {df['treatment'].value_counts().to_dict()}")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Metrics to plot
    metrics = ['area', 'perimeter', 'circularity']
    
    for metric in metrics:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data for this metric
        means = treatment_means['treatment']
        values = treatment_means[(metric, 'mean')]
        errors = treatment_means[(metric, 'std')]
        
        # Create bar plot
        bars = ax.bar(means, values, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add error bars
        ax.errorbar(means, values, yerr=errors, fmt='none', color='black', capsize=5)
        
        # Customize plot
        ax.set_xlabel('Treatment Condition', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Mean {metric.title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Mean {metric.title()} by Treatment Condition', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels if there are many treatments
        if len(means) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add sample size annotations
        for i, (bar, treatment) in enumerate(zip(bars, means)):
            sample_size = treatment_means[treatment_means['treatment'] == treatment][('area', 'count')].iloc[0]
            ax.text(bar.get_x() + bar.get_width()/2., -max(values)*0.05,
                   f'n={sample_size}', ha='center', va='top', fontsize=9, style='italic')
        
        # Improve layout
        plt.tight_layout()
        
        # Save figure
        filename = f'{metric}_by_treatment.png'
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        
        # Also save as PDF for publications
        pdf_filepath = output_path / f'{metric}_by_treatment.pdf'
        plt.savefig(pdf_filepath, bbox_inches='tight')
        
        plt.close()
    
    # Create a summary table and save it
    create_summary_table(treatment_means, output_path)
    
    print(f"\nAll plots saved to: {output_path.absolute()}")

def create_summary_table(treatment_means, output_path):
    """Create and save a summary table of the means and standard deviations."""
    
    # Reshape the data for a cleaner summary table
    summary_data = []
    
    for _, row in treatment_means.iterrows():
        treatment = row['treatment']
        
        for metric in ['area', 'perimeter', 'circularity']:
            mean_val = row[(metric, 'mean')]
            std_val = row[(metric, 'std')]
            count_val = row[(metric, 'count')]
            
            summary_data.append({
                'Treatment': treatment,
                'Metric': metric.title(),
                'Mean': f"{mean_val:.3f}",
                'Std Dev': f"{std_val:.3f}",
                'Sample Size': int(count_val),
                'Mean ± SD': f"{mean_val:.3f} ± {std_val:.3f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_file = output_path / 'treatment_summary_stats.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary statistics: {summary_file}")
    
    # Print summary to console
    print("\nSummary Statistics:")
    print("=" * 60)
    for treatment in treatment_means['treatment'].unique():
        print(f"\nTreatment: {treatment}")
        treatment_data = summary_df[summary_df['Treatment'] == treatment]
        for _, row in treatment_data.iterrows():
            print(f"  {row['Metric']:12}: {row['Mean ± SD']:15} (n={row['Sample Size']})")

def plot_combined_comparison(csv_file, output_dir="plots", figsize=(15, 5)):
    """
    Create a combined plot showing all three metrics side by side.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculate means
    treatment_means = df.groupby('treatment')[['area', 'perimeter', 'circularity']].mean()
    treatment_stds = df.groupby('treatment')[['area', 'perimeter', 'circularity']].std()
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    metrics = ['area', 'perimeter', 'circularity']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        means = treatment_means[metric]
        stds = treatment_stds[metric]
        
        bars = ax.bar(means.index, means.values, capsize=5, alpha=0.8, edgecolor='black')
        ax.errorbar(means.index, means.values, yerr=stds.values, fmt='none', color='black', capsize=5)
        
        ax.set_title(f'Mean {metric.title()}', fontweight='bold')
        ax.set_xlabel('Treatment')
        ax.set_ylabel(f'{metric.title()}')
        
        # Rotate labels if needed
        if len(means.index) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_file = output_path / 'combined_metrics_comparison.png'
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'combined_metrics_comparison.pdf', bbox_inches='tight')
    print(f"Saved combined plot: {combined_file}")
    
    plt.close()

# Example usage
if __name__ == "__main__":
    # Path to your concatenated CSV file
    csv_file = "/Users/nelsschimek/Documents/nancelab/software_packages/TURMorIC/concatenated_data.csv"  # Change this to your file path
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        print("Please update the csv_file variable with the correct path to your data.")
    else:
        # Create individual plots for each metric
        plot_treatment_means(csv_file, output_dir="treatment_plots")
        
        # Create combined comparison plot
        plot_combined_comparison(csv_file, output_dir="treatment_plots")
        
        print("\nPlotting complete! Check the 'treatment_plots' directory for your figures.")