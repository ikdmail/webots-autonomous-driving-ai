import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings (★★Change if necessary★★) ---

# Directory path where log files are stored
LOGS_DIR = r"C:\Users\User\dev\city\controllers\autonomous_car\logs"

# Directory path to save analysis results (graphs)
OUTPUT_DIR = r"C:\Users\User\dev\city\analysis_results"

# --- Main Program (Usually no changes needed below) ---

def load_all_data(logs_path: str) -> pd.DataFrame:
    """
    Loads the 90 newest CSV files starting with 'log_', filters for valid target speeds,
    and returns a combined DataFrame.
    """
    candidate_files = glob.glob(os.path.join(logs_path, "log_*.csv"))

    if not candidate_files:
        print(f"Error: No log files starting with 'log_' found in '{logs_path}'. Please check the path.")
        return pd.DataFrame()

    try:
        candidate_files.sort(key=os.path.getmtime, reverse=True)
    except FileNotFoundError:
        print("Warning: Some files were moved or deleted during the process. Continuing with available files.")
        candidate_files = [f for f in candidate_files if os.path.exists(f)]
        candidate_files.sort(key=os.path.getmtime, reverse=True)

    num_files_to_load = 90
    if len(candidate_files) < num_files_to_load:
        print(f"Warning: Found only {len(candidate_files)} log files (less than 90). Loading all of them.")
        files_to_load = candidate_files
    else:
        files_to_load = candidate_files[:num_files_to_load]

    if not files_to_load:
        print("No files were selected for loading.")
        return pd.DataFrame()

    df_list = [pd.read_csv(file) for file in files_to_load]
    combined_df = pd.concat(df_list, ignore_index=True)

    initial_rows = len(combined_df)
    valid_speeds = [30, 45, 60]
    filtered_df = combined_df[combined_df['target_speed_kmh'].isin(valid_speeds)]
    removed_rows = initial_rows - len(filtered_df)

    if removed_rows > 0:
        print(f"⚠️ Warning: Removed {removed_rows} rows with unexpected target speeds (e.g., 0).")

    print(f"✅ Loaded {len(files_to_load)} newest log files. Analyzing {len(filtered_df)} valid rows.")
    return filtered_df

def analyze_lap_results(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts results for each run and summarizes them by mode and target speed."""
    if df.empty:
        return pd.DataFrame()

    results = []
    # Group by mode, target speed, and run ID for detailed analysis
    for (mode, target_speed, run_id), group in df.groupby(['mode_name', 'target_speed_kmh', 'run_id']):
        goal_event = group[group['is_goal'] == 1].iloc[-1] if 1 in group['is_goal'].values else None

        is_success = goal_event is not None
        lap_time = goal_event['lap_time'] if is_success else None

        active_log = group[group['is_logging_active'] == 1]

        if not active_log.empty:
            avg_speed = active_log['speed_kmh'].mean()
            steering_stability = active_log['steering_angle'].std()
        else:
            avg_speed, steering_stability = None, None

        results.append({
            'mode_name': mode,
            'target_speed_kmh': target_speed,
            'run_id': run_id,
            'is_goal': is_success,
            'lap_time': lap_time,
            'avg_speed_kmh': avg_speed,
            'steering_stability': steering_stability,
        })

    run_summary_df = pd.DataFrame(results)

    # Create the final summary, grouping by mode and target speed
    final_summary = run_summary_df.groupby(['mode_name', 'target_speed_kmh']).agg(
        success_rate=('is_goal', lambda x: x.mean() * 100),
        avg_lap_time=('lap_time', 'mean'),
        avg_speed=('avg_speed_kmh', 'mean'),
        avg_steering_stability=('steering_stability', 'mean')
    ).reset_index()

    # Calculate speed accuracy: how close was the avg speed to the target speed
    final_summary['speed_error_percent'] = ((final_summary['avg_speed'] - final_summary['target_speed_kmh']) / final_summary['target_speed_kmh']) * 100

    return final_summary.round(2)

def create_and_save_plots(df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str):
    """Creates and saves graphs from the analysis results to the specified path."""
    if df.empty or summary_df.empty:
        print("Data is empty, no graphs will be created.")
        return

    os.makedirs(output_path, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Grouped Bar Plots for Summary Metrics
    print("Creating grouped bar plots for performance metrics...")
    # ★★★ 'success_rate' has been added to the metrics dictionary ★★★
    metrics = {
        'success_rate': 'Success Rate (%)',
        'avg_lap_time': 'Average Lap Time (s)',
        'avg_steering_stability': 'Steering Stability (Lower is Better)',
        'speed_error_percent': 'Speed Accuracy Error (%)'
    }
    for key, title in metrics.items():
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=summary_df, x='mode_name', y=key, hue='target_speed_kmh', palette='viridis')
        # Add labels to each bar in the container
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.title(f'{title} by Mode and Target Speed')
        plt.ylabel(title)
        plt.xlabel('Driving Mode')
        plt.legend(title='Target Speed (km/h)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'1_summary_{key}.png'))
        plt.close()

    # 2. Trajectory Plotting Section
    print("Creating trajectory scatter plot for each mode with start and goal...")
    unique_modes = df['mode_name'].unique()
    for mode in unique_modes:
        plt.figure(figsize=(12, 9))

        mode_df = df[df['mode_name'] == mode]

        # Plot trajectory with hue for run_id and style for target_speed
        ax = sns.scatterplot(
            data=mode_df.iloc[::20, :],  # Downsample data for performance
            x='pos_x',
            y='pos_y',
            hue='run_id',
            style='target_speed_kmh',
            palette='viridis',
            s=25,
            alpha=0.8
        )
        
        # This part is for plotting Start/Goal markers
        if not mode_df.empty:
            start_points_df = mode_df.loc[mode_df.groupby('run_id')['timestamp'].idxmin()]
            goal_points_df = mode_df.loc[mode_df[mode_df['is_goal'] == 1].groupby('run_id')['timestamp'].idxmax()]
            
            if not start_points_df.empty:
                 ax.scatter(start_points_df['pos_x'], start_points_df['pos_y'], color='lime', marker='o', s=100, label='Start', zorder=5)
            if not goal_points_df.empty:
                 ax.scatter(goal_points_df['pos_x'], goal_points_df['pos_y'], color='red', marker='*', s=200, label='Goal', zorder=5)

        plt.title(f'Trajectory for {mode} Mode')
        plt.xlabel('X-coordinate (m)')
        plt.ylabel('Y-coordinate (m)')
        
        # Handle combined legends
        handles, labels = ax.get_legend_handles_labels()
        marker_labels = ['Start', 'Goal']
        trajectory_handles, trajectory_labels = [], []
        marker_handles, new_marker_labels = [], []

        for handle, label in zip(handles, labels):
            if label in marker_labels:
                if label not in new_marker_labels: # Avoid duplicate marker labels
                    marker_handles.append(handle)
                    new_marker_labels.append(label)
            else:
                trajectory_handles.append(handle)
                trajectory_labels.append(label)
        
        # Create two separate legends
        legend1 = plt.legend(trajectory_handles, trajectory_labels, title='Run / Target Speed', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.add_artist(legend1) # Add the first legend to the plot
        if marker_handles:
            plt.legend(marker_handles, new_marker_labels, title='Markers', bbox_to_anchor=(1.05, 0), loc='lower left')

        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legends
        plt.savefig(os.path.join(output_path, f'2_trajectory_scatter_{mode}.png'))
        plt.close()

    # 3. Grouped Box Plots for Distributions
    print("Creating grouped box plots for distributions...")
    dist_metrics = {
        'speed_kmh': 'Speed Distribution (km/h)',
        'steering_angle': 'Steering Angle Distribution (rad)'
    }
    active_df = df[df['is_logging_active'] == 1].copy()
    for key, title in dist_metrics.items():
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=active_df, x='mode_name', y=key, hue='target_speed_kmh', palette='coolwarm')
        plt.title(title)
        plt.ylabel(key)
        plt.xlabel('Driving Mode')
        plt.legend(title='Target Speed (km/h)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'3_distribution_{key}.png'))
        plt.close()
    
    print(f"✅ Graphs saved to '{output_path}'.")

def main():
    """Main execution function"""
    full_df = load_all_data(LOGS_DIR)
    if full_df.empty:
        return
    summary_table = analyze_lap_results(full_df)
    print("\n--- Analysis Summary ---")
    print(summary_table.to_string())
    print("----------------------\n")
    create_and_save_plots(full_df, summary_table, OUTPUT_DIR)
    print("✨ Analysis complete.")

if __name__ == '__main__':
    main()