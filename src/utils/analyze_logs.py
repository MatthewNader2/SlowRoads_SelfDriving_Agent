import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib


matplotlib.use('Agg')
# Suppress the specific warning from RANSAC on small samples
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

LOG_FILE = 'agent.log'


def parse_perception_log(log_path):
    """
    Parses the agent log file to extract detailed perception data from each frame.
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at '{log_path}'")
        return None

    # Regex to capture key=value pairs
    kv_pattern = re.compile(r"(\w+)=([^,]+)")

    parsed_data = []

    with open(log_path, 'r') as f:
        for line in f:
            if "PERCEPTION_FRAME:" in line:
                # Extract the key-value part of the string
                data_str = line.split("PERCEPTION_FRAME:")[1].strip()

                # Find all key=value pairs
                matches = kv_pattern.findall(data_str)

                frame_data = {}
                for key, value in matches:
                    # Clean and convert value to appropriate type
                    val_str = value.strip()
                    if val_str == 'True':
                        frame_data[key] = True
                    elif val_str == 'False':
                        frame_data[key] = False
                    elif val_str == 'None':
                        frame_data[key] = None
                    else:
                        try:
                            # Attempt to convert to float, which handles ints and floats
                            frame_data[key] = float(val_str)
                        except ValueError:
                            # If it fails, keep it as a string (for line_params)
                            frame_data[key] = val_str

                parsed_data.append(frame_data)

    if not parsed_data:
        print("No 'PERCEPTION_FRAME' data found in log.")
        return None

    print(f"Successfully parsed {len(parsed_data)} perception frames from the log.")
    return pd.DataFrame(parsed_data)


def perform_analysis(df):
    """
    Performs statistical analysis and generates plots from the parsed data.
    """
    if df is None or df.empty:
        print("DataFrame is empty. No analysis to perform.")
        return

    # --- 1. Core Statistics ---
    print("\n--- Core Statistics ---")
    print(df[['angle', 'initial_points', 'dbscan_points', 'inliers', 'target_x']].describe())

    print("\n--- Boolean Flag Analysis ---")
    print(f"Persistence Check Forced: {df['force_accept'].mean():.2%} of frames")
    print(f"Considered Stable (Jitter): {df['is_stable'].mean():.2%} of frames")
    print(f"Final Line Validated:     {df['validated'].mean():.2%} of frames")

    # --- 2. Visualizations ---
    print("\nGenerating plots...")

    # Plot 1: Angle Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['angle'], kde=True, bins=50)
    plt.title('Distribution of RANSAC Angles')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('angle_distribution.png')
    plt.close()

    # Plot 2: Point Filtering Funnel
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['initial_points'], label='Initial Points', alpha=0.8)
    plt.plot(df.index, df['dbscan_points'], label='After DBSCAN Filter', alpha=0.8)
    plt.plot(df.index, df['inliers'], label='RANSAC Inliers', color='green', linewidth=2)
    plt.title('Number of Points at Each Filtering Stage')
    plt.xlabel('Frame Number')
    plt.ylabel('Point Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('point_filter_funnel.png')
    plt.close()

    # Plot 3: Angle vs. Inlier Count
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='inliers', y='angle', data=df, alpha=0.5, hue='validated')
    plt.title('RANSAC Angle vs. Number of Inlier Points')
    plt.xlabel('Number of Inliers')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.savefig('angle_vs_inliers.png')
    plt.close()

    # Plot 4: Target X Position over Time
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['target_x'], label='Calculated Steering Target (target_x)')
    plt.title('Steering Target Position over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Target X-Coordinate (pixels)')
    plt.legend()
    plt.grid(True)
    plt.savefig('target_x_timeseries.png')
    plt.close()

    print("All plots saved successfully.")

    # --- 3. Automated Analysis ---
    print("\n--- Automated Analysis Insights ---")

    # Inlier Ratio
    # Calculate ratio only where dbscan_points is not zero to avoid division by zero
    df['inlier_ratio'] = df.apply(lambda row: row['inliers'] / row['dbscan_points'] if row['dbscan_points'] > 0 else 0,
                                  axis=1)
    avg_inlier_ratio = df['inlier_ratio'].mean()
    print(f"\nOn average, RANSAC considers {avg_inlier_ratio:.2%} of the clustered points to be inliers.")
    if avg_inlier_ratio < 0.75:
        print(
            "-> INSIGHT: This ratio is low. It confirms RANSAC is being too strict and rejecting many points from the clean cluster. The `residual_threshold` is likely still too small.")
    else:
        print(
            "-> INSIGHT: This ratio is high, which is good. It means RANSAC generally agrees with the DBSCAN cluster.")

    # Frames with low inlier count but high angle deviation
    low_inlier_frames = df[(df['inliers'] > 0) & (df['inliers'] < 10)]
    if not low_inlier_frames.empty:
        wild_angle_on_low_inliers = low_inlier_frames[low_inlier_frames['angle'].abs() > 60]
        if not wild_angle_on_low_inliers.empty:
            print(
                f"\nFound {len(wild_angle_on_low_inliers)} frames with very few inliers (<10) that resulted in a wild angle (>60°).")
            print(
                "-> INSIGHT: This is the '3 red dots, 1 green dot' problem. RANSAC is fitting a bad line to a tiny number of points.")
            print("Example frames:")
            print(wild_angle_on_low_inliers[['inliers', 'angle', 'target_x']].head())


if __name__ == '__main__':
    data_df = parse_perception_log(LOG_FILE)
    if data_df is not None:
        perform_analysis(data_df)