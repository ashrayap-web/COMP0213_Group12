"""
Visualisation script for grasp data from CSV files.
Plots x, y, z coordinates with color-coded results (green for success, red for failure)
and arrows pointing towards the origin.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_csv_data(ax, csv_file):
    """
    Plot data from a single CSV file in 3D space on a given axis.
    
    Args:
        ax: Matplotlib 3D axis to plot on
        csv_file: Path to the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Separate successful and failed grasps
    success = df[df['Result'] == 1.0]
    failure = df[df['Result'] == 0.0]
    
    # Plot successful grasps (green dots)
    if not success.empty:
        ax.scatter(success['x'], success['y'], success['z'], 
                  c='green', marker='o', s=50, alpha=0.6, label='Success')
        
        # Add arrows pointing to origin for successful grasps
        for _, row in success.iterrows():
            x, y, z = row['x'], row['y'], row['z']
            # Calculate direction towards origin
            length = np.sqrt(x**2 + y**2 + z**2)
            if length > 0:  # Avoid division by zero
                # Scale arrow to be 10% of distance to origin
                scale = 0.1
                dx, dy, dz = -x * scale, -y * scale, -z * scale
                ax.quiver(x, y, z, dx, dy, dz, 
                         color='darkgreen', alpha=0.3, arrow_length_ratio=0.3)
    
    # Plot failed grasps (red dots)
    if not failure.empty:
        ax.scatter(failure['x'], failure['y'], failure['z'], 
                  c='red', marker='o', s=50, alpha=0.6, label='Failure')
        
        # Add arrows pointing to origin for failed grasps
        for _, row in failure.iterrows():
            x, y, z = row['x'], row['y'], row['z']
            # Calculate direction towards origin
            length = np.sqrt(x**2 + y**2 + z**2)
            if length > 0:  # Avoid division by zero
                # Scale arrow to be 10% of distance to origin
                scale = 0.1
                dx, dy, dz = -x * scale, -y * scale, -z * scale
                ax.quiver(x, y, z, dx, dy, dz, 
                         color='darkred', alpha=0.3, arrow_length_ratio=0.3)
    
    # Plot origin point (object)
    ax.scatter([0], [0], [0], c='blue', marker='s', s=20, label='Object')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    filename = os.path.basename(csv_file)
    ax.set_title(f'Grasp Visualization: {filename}\n'
                f'Success: {len(success)} | Failure: {len(failure)}')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([df['x'].max()-df['x'].min(), 
                         df['y'].max()-df['y'].min(), 
                         df['z'].max()-df['z'].min()]).max() / 2.0
    
    mid_x = (df['x'].max() + df['x'].min()) * 0.5
    mid_y = (df['y'].max() + df['y'].min()) * 0.5
    mid_z = (df['z'].max() + df['z'].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_all_csv_files():
    """
    Visualize all CSV files in the data directory in a single figure with subplots.
    """
    data_dir = 'data'
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    if not csv_files:
        print("No CSV files found in the data directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Create a single figure with subplots (2x2 grid)
    fig = plt.figure(figsize=(16, 14))
    
    # Create plots for each CSV file
    for idx, csv_file in enumerate(csv_files):
        csv_path = os.path.join(data_dir, csv_file)
        print(f"\nProcessing {csv_file}...")
        
        try:
            # Create subplot (2 rows, 2 columns)
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            plot_csv_data(ax, csv_path)
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
    
    plt.tight_layout()
    
    # Save the combined figure
    output_filename = "visualisation_all_data.png"
    fig.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined visualization to {output_filename}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    visualize_all_csv_files()
