import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_matplotlib_style():
    """English"""
    plt.style.use('seaborn-v0_8-paper')

    # English
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

    # English
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # English
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 6

    # English
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    # English
    return {
        'teacher': '#1F77B4',      # Blue
        'teacher_light': '#AEC7E8',
        'dino': '#D62728',         # Red
        'dino_light': '#FF9896',
        'vit': '#2CA02C',          # Green
        'vit_light': '#98DF8A',
        'clip': '#FF7F0E',         # Orange
        'clip_light': '#FFBB78'
    }

def visualize_artifacts(csv_dino, csv_vit, csv_clip, output_dir):
    """English CSV English"""
    for path in [csv_dino, csv_vit, csv_clip]:
        if not os.path.exists(path):
            print(f"Error: CSV file not found at {path}")
            return

    os.makedirs(output_dir, exist_ok=True)

    # English
    df_dino = pd.read_csv(csv_dino)
    df_vit = pd.read_csv(csv_vit)
    df_clip = pd.read_csv(csv_clip)
    print(f"Loaded data points: DINO ({len(df_dino)}), ViT ({len(df_vit)}), CLIP ({len(df_clip)})")

    # English
    df_teacher = df_dino[['t_mean', 't_max']].copy()
    df_teacher['t_ratio'] = df_teacher['t_max'] / (df_teacher['t_mean'] + 1e-8)
    df_teacher['global_step'] = df_teacher.index

    # English
    for df in [df_dino, df_vit, df_clip]:
        df['global_step'] = df.index
        df['s_ratio'] = df['s_max'] / (df['s_mean'] + 1e-8)

    colors = setup_matplotlib_style()

    # English
    models = {
        'Teacher (CNN)': {'df': df_teacher, 'mean_col': 't_mean', 'max_col': 't_max', 'ratio_col': 't_ratio', 'color': colors['teacher'], 'color_light': colors['teacher_light'], 'cmap': 'Blues', 'marker': 'o'},
        'DINOv2': {'df': df_dino, 'mean_col': 's_mean', 'max_col': 's_max', 'ratio_col': 's_ratio', 'color': colors['dino'], 'color_light': colors['dino_light'], 'cmap': 'Reds', 'marker': 'x'},
        'ViT': {'df': df_vit, 'mean_col': 's_mean', 'max_col': 's_max', 'ratio_col': 's_ratio', 'color': colors['vit'], 'color_light': colors['vit_light'], 'cmap': 'Greens', 'marker': '^'},
        'CLIP': {'df': df_clip, 'mean_col': 's_mean', 'max_col': 's_max', 'ratio_col': 's_ratio', 'color': colors['clip'], 'color_light': colors['clip_light'], 'cmap': 'Oranges', 'marker': 's'}
    }

    # --- 1. English: English vs English ---
    fig, ax = plt.subplots(figsize=(10, 8))

    sample_size = min(2000, len(df_teacher))
    for name, config in models.items():
        df_sample = config['df'].sample(min(sample_size, len(config['df'])), random_state=42)
        ax.scatter(df_sample[config['mean_col']], df_sample[config['max_col']],
                   color=config['color'], alpha=0.5, label=name, s=20, marker=config['marker'])

    # English (y=x)
    max_val = max([config['df'][config['max_col']].max() for config in models.values()])
    min_val = min([config['df'][config['mean_col']].min() for config in models.values()])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x (Ideal)')

    ax.set_xlabel('Mean Activation Norm')
    ax.set_ylabel('Max Activation Norm')
    ax.set_title('Activation Artifacts: Mean vs. Max Norm')
    ax.legend(loc='upper left')

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_scatter_mean_vs_max.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_scatter_mean_vs_max.png'), dpi=300)
    plt.close()

    # --- 2. English: English ---
    fig, ax = plt.subplots(figsize=(12, 6))

    window = max(1, len(df_teacher) // 100)

    for name, config in models.items():
        df = config['df']
        mean_trend = df[config['ratio_col']].rolling(window=window, min_periods=1).mean()
        std_trend = df[config['ratio_col']].rolling(window=window, min_periods=1).std()

        ax.plot(df['global_step'], mean_trend, color=config['color'], label=name)
        ax.fill_between(df['global_step'], mean_trend - std_trend, mean_trend + std_trend,
                        color=config['color_light'], alpha=0.2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Artifact Ratio (Max / Mean Norm)')
    ax.set_title('Evolution of Activation Artifacts during Training')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_ratio_trend.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_ratio_trend.png'), dpi=300)
    plt.close()

    # --- 3. English: English ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_data_list = []
    for name, config in models.items():
        plot_data_list.append(pd.DataFrame({
            'Model': [name] * len(config['df']),
            'Ratio (Max/Mean)': config['df'][config['ratio_col']]
        }))
    plot_data = pd.concat(plot_data_list, ignore_index=True)

    palette = {name: config['color'] for name, config in models.items()}
    palette_light = {name: config['color_light'] for name, config in models.items()}

    sns.boxplot(x='Model', y='Ratio (Max/Mean)', data=plot_data, palette=palette, ax=ax1, width=0.4)
    ax1.set_yscale('log')
    ax1.set_title('Distribution of Artifact Ratios (Boxplot)')

    sns.violinplot(x='Model', y='Ratio (Max/Mean)', data=plot_data, palette=palette_light, ax=ax2, inner="quartile")
    ax2.set_yscale('log')
    ax2.set_title('Density of Artifact Ratios (Violinplot)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_distribution.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_distribution.png'), dpi=300)
    plt.close()

    # --- 4. 2D English (KDE Contour) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, config in models.items():
        sns.kdeplot(data=config['df'], x=config['mean_col'], y=config['max_col'],
                    cmap=config['cmap'], fill=True, alpha=0.4, label=name, ax=ax, log_scale=(True, True))

    ax.set_xlabel('Mean Activation Norm')
    ax.set_ylabel('Max Activation Norm')
    ax.set_title('2D Density Regions of Activation Norms')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=config['color_light'], edgecolor=config['color'], label=name)
                       for name, config in models.items()]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_density_2d.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_density_2d.png'), dpi=300)
    plt.close()

    # --- 5. English (Joint Plot) ---
    joint_data_list = []
    for name, config in models.items():
        df_temp = pd.DataFrame({
            'Mean Norm': config['df'][config['mean_col']],
            'Max Norm': config['df'][config['max_col']],
            'Model': name
        })
        # English
        if len(df_temp) > 1000:
            df_temp = df_temp.sample(1000, random_state=42)
        joint_data_list.append(df_temp)

    df_joint = pd.concat(joint_data_list, ignore_index=True)

    g = sns.JointGrid(data=df_joint, x="Mean Norm", y="Max Norm", hue="Model",
                      palette=palette, height=10, ratio=4)

    g.plot_joint(sns.scatterplot, s=15, alpha=0.6, edgecolor=None)
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.5, linewidth=1.5)

    g.ax_joint.set_xscale('log')
    g.ax_joint.set_yscale('log')

    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x (Ideal)')

    g.fig.suptitle('Joint Distribution of Activation Norms', y=1.02, fontsize=16)

    plt.savefig(os.path.join(output_dir, 'artifact_joint_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'artifact_joint_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 6. English: English ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for name, config in models.items():
        df = config['df']

        # Max Norm
        max_mean = df[config['max_col']].rolling(window=window, min_periods=1).mean()
        max_std = df[config['max_col']].rolling(window=window, min_periods=1).std()
        ax1.plot(df['global_step'], max_mean, color=config['color'], label=name)
        ax1.fill_between(df['global_step'], max_mean - max_std, max_mean + max_std, color=config['color_light'], alpha=0.2)

        # Mean Norm
        mean_mean = df[config['mean_col']].rolling(window=window, min_periods=1).mean()
        mean_std = df[config['mean_col']].rolling(window=window, min_periods=1).std()
        ax2.plot(df['global_step'], mean_mean, color=config['color'], label=name)
        ax2.fill_between(df['global_step'], mean_mean - mean_std, mean_mean + mean_std, color=config['color_light'], alpha=0.2)

    ax1.set_ylabel('Max Activation Norm')
    ax1.set_title('Evolution of Max Norm')
    ax1.legend()
    ax1.set_yscale('log')

    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Mean Activation Norm')
    ax2.set_title('Evolution of Mean Norm')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_separate_trends.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_separate_trends.png'), dpi=300)
    plt.close()

    # --- 7. English: English ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plot_data_max = []
    plot_data_mean = []

    for name, config in models.items():
        plot_data_max.append(pd.DataFrame({'Model': [name] * len(config['df']), 'Max Norm': config['df'][config['max_col']]}))
        plot_data_mean.append(pd.DataFrame({'Model': [name] * len(config['df']), 'Mean Norm': config['df'][config['mean_col']]}))

    df_max_all = pd.concat(plot_data_max, ignore_index=True)
    df_mean_all = pd.concat(plot_data_mean, ignore_index=True)

    sns.violinplot(x='Model', y='Max Norm', data=df_max_all, palette=palette_light, ax=ax1, inner="quartile")
    ax1.set_yscale('log')
    ax1.set_title('Density of Max Activation Norm')

    sns.violinplot(x='Model', y='Mean Norm', data=df_mean_all, palette=palette_light, ax=ax2, inner="quartile")
    ax2.set_yscale('log')
    ax2.set_title('Density of Mean Activation Norm')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_separate_distributions.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_separate_distributions.png'), dpi=300)
    plt.close()

    # --- 8. English: English ---
    fig, ax = plt.subplots(figsize=(12, 6))

    combined_data_list = []
    for name, config in models.items():
        combined_data_list.append(pd.DataFrame({
            'Model': [name] * len(config['df']),
            'Metric': ['Mean Norm'] * len(config['df']),
            'Value': config['df'][config['mean_col']]
        }))
        combined_data_list.append(pd.DataFrame({
            'Model': [name] * len(config['df']),
            'Metric': ['Max Norm'] * len(config['df']),
            'Value': config['df'][config['max_col']]
        }))

    plot_data_combined = pd.concat(combined_data_list, ignore_index=True)

    sns.boxplot(x='Metric', y='Value', hue='Model', data=plot_data_combined, palette=palette, ax=ax, width=0.6)

    ax.set_yscale('log')
    ax.set_title('Comparison of Mean and Max Activation Norms')
    ax.set_ylabel('Activation Norm (Log Scale)')
    ax.set_xlabel('')
    ax.legend(title='Model')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_mean_max_comparison.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_mean_max_comparison.png'), dpi=300)
    plt.close()

    # --- 9. English (Slope/Paired Plot) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, config in models.items():
        df_paired = config['df'].sample(min(300, len(config['df'])), random_state=42)
        for _, row in df_paired.iterrows():
            ax.plot([1, 2], [row[config['mean_col']], row[config['max_col']]],
                    color=config['color'], alpha=0.1, linewidth=1)

        mean_avg = config['df'][config['mean_col']].mean()
        max_avg = config['df'][config['max_col']].mean()

        ax.plot([1, 2], [mean_avg, max_avg], color=config['color'], alpha=1.0,
                linewidth=4, marker=config['marker'], markersize=10, label=f'{name} (Avg Trend)')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Mean Norm', 'Max Norm'], fontsize=14)
    ax.set_xlim(0.5, 2.5)
    ax.set_yscale('log')
    ax.set_ylabel('Activation Norm (Log Scale)')
    ax.set_title('Paired Shift: Mean to Max Activation Norm')
    ax.legend()
    ax.grid(axis='x', alpha=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'artifact_paired_shift.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'artifact_paired_shift.png'), dpi=300)
    plt.close()

    print(f"Visualization complete! Figures saved to {output_dir}")
    print("-" * 50)
    print(f"Statistical Summary:")
    for name, config in models.items():
        print(f"{name} - Mean Artifact Ratio: {config['df'][config['ratio_col']].mean():.2f} ± {config['df'][config['ratio_col']].std():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Activation Artifacts for Paper')
    parser.add_argument('--csv_dino', type=str, default='artifact_stats_dino.csv', help='Path to DINO artifact stats')
    parser.add_argument('--csv_vit', type=str, default='artifact_stats_vit.csv', help='Path to ViT artifact stats')
    parser.add_argument('--csv_clip', type=str, default='artifact_stats_clip.csv', help='Path to CLIP artifact stats')
    parser.add_argument('--out', type=str, default='./artifact_plots', help='Output directory for plots')

    args = parser.parse_args()
    visualize_artifacts(args.csv_dino, args.csv_vit, args.csv_clip, args.out)
