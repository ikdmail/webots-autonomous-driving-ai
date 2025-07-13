import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings (★★Change if necessary★★) ---

# Directory path where log files are stored
LOGS_DIR = r"C:\Users\User\dev\city\controllers\autonomous_car\logs"

# Directory path to save analysis results (graphs)
OUTPUT_DIR = r"C:\Users\User\dev\city\analysis_results_60kmh" # Saved to a new folder

# ★★★ Total number of log files from your 60km/h experiment ★★★
# (3 modes * 10 trials = 30)
NUM_FILES_TO_ANALYZE = 30

# --- Main Program (Usually no changes needed below) ---

def load_all_data(logs_path: str) -> pd.DataFrame:
    """
    Loads the newest CSV files for the 60km/h experiment,
    sorted by modification time.
    """
    candidate_files = glob.glob(os.path.join(logs_path, "log_*.csv"))
    
    if not candidate_files:
        print(f"Error: No log files starting with 'log_' found in '{logs_path}'.")
        return pd.DataFrame()

    try:
        candidate_files.sort(key=os.path.getmtime, reverse=True)
    except FileNotFoundError:
        print("Warning: Some files were moved or deleted. Continuing with available files.")
        candidate_files = [f for f in candidate_files if os.path.exists(f)]
        candidate_files.sort(key=os.path.getmtime, reverse=True)

    if len(candidate_files) < NUM_FILES_TO_ANALYZE:
        print(f"Warning: Found only {len(candidate_files)} log files (less than {NUM_FILES_TO_ANALYZE}). Loading all.")
        files_to_load = candidate_files
    else:
        files_to_load = candidate_files[:NUM_FILES_TO_ANALYZE]

    if not files_to_load:
        print("No files were selected for loading.")
        return pd.DataFrame()
        
    df_list = [pd.read_csv(file) for file in files_to_load]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Filter strictly for 60km/h data to be safe
    initial_rows = len(combined_df)
    filtered_df = combined_df[combined_df['target_speed_kmh'] == 60].copy()
    removed_rows = initial_rows - len(filtered_df)
    if removed_rows > 0:
        print(f"⚠️ Warning: Removed {removed_rows} rows that were not for the 60km/h target speed.")

    print(f"✅ Loaded {len(files_to_load)} newest log files. Analyzing {len(filtered_df)} valid rows for 60km/h.")
    return filtered_df

def analyze_lap_results(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts results for each run and summarizes them by mode."""
    if df.empty:
        return pd.DataFrame()

    results = []
    # Group by mode and run ID
    for (mode, run_id), group in df.groupby(['mode_name', 'run_id']):
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
            'run_id': run_id,
            'is_goal': is_success,
            'lap_time': lap_time,
            'avg_speed_kmh': avg_speed,
            'steering_stability': steering_stability,
        })
    
    run_summary_df = pd.DataFrame(results)

    # Create the final summary, grouping by mode only
    final_summary = run_summary_df.groupby(['mode_name']).agg(
        success_rate=('is_goal', lambda x: x.mean() * 100),
        avg_lap_time=('lap_time', 'mean'),
        avg_speed=('avg_speed_kmh', 'mean'),
        avg_steering_stability=('steering_stability', 'mean')
    ).reset_index()

    # Calculate speed accuracy against the 60km/h target
    final_summary['speed_error_percent'] = ((final_summary['avg_speed'] - 60) / 60) * 100

    return final_summary.round(2)

def create_and_save_plots(df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str):
    """Creates and saves graphs from the analysis results."""
    if df.empty or summary_df.empty:
        print("Data is empty, no graphs will be created.")
        return

    os.makedirs(output_path, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Simple Bar Plots for Summary Metrics
    print("Creating bar plots for performance metrics...")
    metrics = {
        'success_rate': 'Success Rate (%)',
        'avg_lap_time': 'Average Lap Time (s)',
        'avg_steering_stability': 'Steering Stability (Lower is Better)',
        'speed_error_percent': 'Speed Accuracy Error (%)'
    }
    for key, title in metrics.items():
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=summary_df, x='mode_name', y=key, palette='viridis')
        ax.bar_label(ax.containers[0], fmt='%.2f')
        plt.title(f'{title} at 60km/h Target Speed')
        plt.ylabel(title)
        plt.xlabel('Driving Mode')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'1_summary_{key}.png'))
        plt.close()

    # 2. Trajectory Plotting Section
    print("Creating trajectory scatter plot for each mode...")
    unique_modes = df['mode_name'].unique()
    for mode in unique_modes:
        plt.figure(figsize=(12, 9))
        mode_df = df[df['mode_name'] == mode]
        ax = sns.scatterplot(
            data=mode_df.iloc[::20, :],
            x='pos_x', y='pos_y',
            hue='run_id', palette='viridis',
            s=25, alpha=0.8
        )
        if not mode_df.empty:
            start_points_df = mode_df.loc[mode_df.groupby('run_id')['timestamp'].idxmin()]
            goal_points_df = mode_df.loc[mode_df[mode_df['is_goal'] == 1].groupby('run_id')['timestamp'].idxmax()]
            
            if not start_points_df.empty:
                 ax.scatter(start_points_df['pos_x'], start_points_df['pos_y'], color='lime', marker='o', s=100, label='Start', zorder=5)
            if not goal_points_df.empty:
                 ax.scatter(goal_points_df['pos_x'], goal_points_df['pos_y'], color='red', marker='*', s=200, label='Goal', zorder=5)
        
        plt.title(f'Trajectory for {mode} Mode at 60km/h')
        plt.xlabel('X-coordinate (m)')
        plt.ylabel('Y-coordinate (m)')
        plt.legend(title='Run ID / Markers', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(output_path, f'2_trajectory_scatter_{mode}.png'))
        plt.close()

    # 3. Simple Box Plots for Distributions
    print("Creating box plots for distributions...")
    dist_metrics = {
        'speed_kmh': 'Speed Distribution (km/h)',
        'steering_angle': 'Steering Angle Distribution (rad)'
    }
    active_df = df[df['is_logging_active'] == 1].copy()
    for key, title in dist_metrics.items():
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=active_df, x='mode_name', y=key, palette='coolwarm')
        plt.title(f'{title} at 60km/h')
        plt.ylabel(key)
        plt.xlabel('Driving Mode')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'3_distribution_{key}.png'))
        plt.close()
    
    print(f"✅ Graphs saved to '{output_path}'.")

def main():
    """Main execution function"""
    full_df = load_all_data(LOGS_DIR)
    if full_df.empty: return
    summary_table = analyze_lap_results(full_df)
    print("\n--- 60km/h Experiment Analysis Summary ---")
    print(summary_table.to_string())
    print("------------------------------------------\n")
    create_and_save_plots(full_df, summary_table, OUTPUT_DIR)
    print("✨ Analysis complete.")

if __name__ == '__main__':
    main()