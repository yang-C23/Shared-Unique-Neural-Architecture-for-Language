"""
Visualize Neural Encoding Results

Creates scatter plots and bar charts to visualize the encoding results.

Usage:
    python visualize_encoding_results.py --input encoding_results_csv
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def create_scatter_plot(df_detailed, output_dir):
    """Create scatter plot comparing normal vs break models"""
    
    # Prepare data for comparison
    df_normal = df_detailed[df_detailed['model_type'] == 'normal'].copy()
    df_break = df_detailed[df_detailed['model_type'] == 'break'].copy()
    
    # Merge to compare same subjects
    df_normal.rename(columns={'mean_r': 'mean_r_normal'}, inplace=True)
    df_break.rename(columns={'mean_r': 'mean_r_break'}, inplace=True)
    
    df_compare = pd.merge(
        df_normal[['model_name', 'language', 'subject_id', 'mean_r_normal']],
        df_break[['model_name', 'language', 'subject_id', 'mean_r_break']],
        on=['model_name', 'language', 'subject_id']
    )
    
    # Create scatter plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    models = df_compare['model_name'].unique()
    
    for idx, model in enumerate(sorted(models)):
        ax = axes[idx]
        model_data = df_compare[df_compare['model_name'] == model]
        
        # Plot by language
        for lang in ['EN', 'CN', 'FR']:
            lang_data = model_data[model_data['language'] == lang]
            ax.scatter(
                lang_data['mean_r_normal'], 
                lang_data['mean_r_break'],
                label=lang,
                alpha=0.6,
                s=50
            )
        
        # Add diagonal line (y=x)
        max_val = max(model_data['mean_r_normal'].max(), model_data['mean_r_break'].max())
        min_val = min(model_data['mean_r_normal'].min(), model_data['mean_r_break'].min())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Normal Model (mean r)', fontsize=10)
        ax.set_ylabel('Break Model (mean r)', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(model_data['mean_r_normal'], model_data['mean_r_break'])[0, 1]
        ax.text(0.95, 0.05, f'r={corr:.3f}', transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Normal vs Break Models: Subject-level Correlations', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'scatter_normal_vs_break.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Scatter plot saved: {output_path}")
    return output_path


def create_bar_plot(df_summary, df_detailed, output_dir):
    """Create bar plot showing mean correlations with individual subject data points"""
    
    # Set random seed for reproducible jitter
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # First pass: collect all y values to determine shared y-axis range
    all_y_values = []
    for lang in ['EN', 'CN', 'FR']:
        lang_summary = df_summary[df_summary['language'] == lang]
        lang_detail = df_detailed[df_detailed['language'] == lang]
        all_y_values.extend(lang_summary['mean_r_avg'].values)
        all_y_values.extend(lang_detail['mean_r'].values)
    
    y_min = min(all_y_values) * 0.95
    y_max = max(all_y_values) * 1.05
    
    for idx, lang in enumerate(['EN', 'CN', 'FR']):
        ax = axes[idx]
        
        lang_data = df_summary[df_summary['language'] == lang].copy()
        unique_models = sorted(lang_data['model_name'].unique())
        
        # Prepare data containers
        normal_vals = []
        random_vals = []
        break_vals = []
        
        for m in unique_models:
            n = lang_data[(lang_data['model_name'] == m) & (lang_data['model_type'] == 'normal')]
            r = lang_data[(lang_data['model_name'] == m) & (lang_data['model_type'] == 'random_break')]
            b = lang_data[(lang_data['model_name'] == m) & (lang_data['model_type'] == 'break')]
            
            normal_vals.append(n['mean_r_avg'].iloc[0] if not n.empty else 0)
            random_vals.append(r['mean_r_avg'].iloc[0] if not r.empty else 0)
            break_vals.append(b['mean_r_avg'].iloc[0] if not b.empty else 0)
            
        x_pos = np.arange(len(unique_models))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x_pos - width, normal_vals, width, 
                      label='Intact', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x_pos, random_vals, width,
                      label='random-ablation', color='#9368AB', alpha=0.8)
        bars3 = ax.bar(x_pos + width, break_vals, width,
                      label='core-language-ablation', color='coral', alpha=0.8)
        
        # Add individual subject data points with jitter
        jitter_width = width * 2/3
        for i, model in enumerate(unique_models):
            # Normal subjects
            normal_subjects = df_detailed[
                (df_detailed['model_name'] == model) & 
                (df_detailed['model_type'] == 'normal') &
                (df_detailed['language'] == lang)
            ]['mean_r'].values
            
            # Random subjects
            random_subjects = df_detailed[
                (df_detailed['model_name'] == model) & 
                (df_detailed['model_type'] == 'random_break') &
                (df_detailed['language'] == lang)
            ]['mean_r'].values
            
            # Break subjects
            break_subjects = df_detailed[
                (df_detailed['model_name'] == model) & 
                (df_detailed['model_type'] == 'break') &
                (df_detailed['language'] == lang)
            ]['mean_r'].values
            
            # Add jitter
            normal_jitter = np.random.uniform(-jitter_width/2, jitter_width/2, len(normal_subjects))
            random_jitter = np.random.uniform(-jitter_width/2, jitter_width/2, len(random_subjects))
            break_jitter = np.random.uniform(-jitter_width/2, jitter_width/2, len(break_subjects))
            
            ax.scatter(x_pos[i] - width + normal_jitter, normal_subjects, color='darkgray', alpha=0.4, s=20, zorder=3)
            ax.scatter(x_pos[i] + random_jitter, random_subjects, color='darkgray', alpha=0.4, s=20, zorder=3)
            ax.scatter(x_pos[i] + width + break_jitter, break_subjects, color='darkgray', alpha=0.4, s=20, zorder=3)
        
        # Customize
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Mean Correlation (r)', fontsize=12)
        n_subs = len(df_detailed[(df_detailed['language'] == lang) & (df_detailed['model_type'] == 'normal')]['subject_id'].unique())
        ax.set_title(f'Language: {lang} (n={n_subs})', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique_models, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set shared y-axis range
        ax.set_ylim(y_min, y_max)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only show label if bar exists
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=5)
    
    plt.suptitle('Mean Correlations by Model, Language, and Type', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'bar_mean_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Bar plot saved: {output_path}")
    return output_path


def create_heatmap(df_summary, output_dir):
    """Create heatmap of model performance"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, model_type in enumerate(['normal', 'break']):
        ax = axes[idx]
        
        # Pivot data for heatmap
        type_data = df_summary[df_summary['model_type'] == model_type].copy()
        pivot_data = type_data.pivot(index='model_name', columns='language', values='mean_r_avg')
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Mean Correlation (r)'},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'{model_type.capitalize()} Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
    
    plt.suptitle('Model Performance Heatmap by Language', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'heatmap_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved: {output_path}")
    return output_path


def create_effect_size_plot(df_summary, output_dir):
    """Create plot showing effect of breaking models"""
    
    # Calculate effect size (difference between normal and break)
    df_normal = df_summary[df_summary['model_type'] == 'normal'].set_index(['model_name', 'language'])
    df_break = df_summary[df_summary['model_type'] == 'break'].set_index(['model_name', 'language'])
    
    effect_size = df_normal['mean_r_avg'] - df_break['mean_r_avg']
    effect_size = effect_size.reset_index()
    effect_size.columns = ['model_name', 'language', 'effect_size']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    models = sorted(effect_size['model_name'].unique())
    languages = ['EN', 'CN', 'FR']
    x = np.arange(len(models))
    width = 0.25
    
    for i, lang in enumerate(languages):
        lang_data = effect_size[effect_size['language'] == lang].sort_values('model_name')
        offset = (i - 1) * width
        bars = ax.bar(x + offset, lang_data['effect_size'], width, 
                     label=lang, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Effect Size (Normal - Break)', fontsize=12)
    ax.set_title('Effect of Model Breaking on Neural Encoding Performance', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Language', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'bar_effect_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Effect size plot saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize neural encoding results")
    parser.add_argument(
        '--input',
        type=str,
        default='encoding_results_csv',
        help='Directory containing CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for plots (default: same as input)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df_detailed = pd.read_csv(input_dir / 'encoding_results_detailed.csv')
    df_summary = pd.read_csv(input_dir / 'encoding_results_summary.csv')
    
    print(f"Loaded {len(df_detailed)} detailed records and {len(df_summary)} summary records\n")
    
    # Create visualizations
    print("Creating visualizations...\n")
    


    scatter_path = create_scatter_plot(df_detailed, output_dir)
    bar_path = create_bar_plot(df_summary, df_detailed, output_dir)
    heatmap_path = create_heatmap(df_summary, output_dir)
    effect_path = create_effect_size_plot(df_summary, output_dir)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated plots in {output_dir}:")
    print(f"  1. scatter_normal_vs_break.png - Scatter plot comparing normal vs break")
    print(f"  2. bar_mean_correlations.png - Bar plot of mean correlations")
    print(f"  3. heatmap_performance.png - Heatmap of performance by language")
    print(f"  4. bar_effect_size.png - Effect size of model breaking")
    

if __name__ == "__main__":
    main()