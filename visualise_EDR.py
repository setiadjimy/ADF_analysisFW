import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# ======================================================
# CONFIGURATION
# ======================================================
INPUT_FOLDER = "edr_csv"   # Adjust if your CSVs are in a subfolder
OUTPUT_FOLDER = "./edr_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================
# Helper Functions
# ======================================================
def compute_delta_v(df):
    """Compute absolute delta-V using the velocity vector components."""
    # Ensure we use the correct column names from the Integrated Monitor
    v_cols = ['velocity_x', 'velocity_y', 'velocity_z']
    
    # Calculate magnitude: sqrt(vx^2 + vy^2 + vz^2)
    v_mag = np.sqrt((df[v_cols]**2).sum(axis=1))
    
    # Delta-V is typically the change in speed during the event buffer
    v0 = v_mag.iloc[0]
    v1 = v_mag.iloc[-1]
    return abs(v0 - v1)

# ======================================================
# Load All CSVs
# ======================================================
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
all_data = {}

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Check if it's a valid EDR file by checking a known column
        if 'speed_kmh' in df.columns:
            all_data[os.path.basename(file)] = df
    except Exception as e:
        print(f"Skipping {file}: {e}")

print(f"Loaded {len(all_data)} EDR CSV files.")

# ======================================================
# Visualization per file
# ======================================================
for fname, df in all_data.items():
    # Normalize Time using sim_time (float) instead of timestamp (string)
    time_s = df['sim_time'] - df['sim_time'].iloc[0]

    # 1. Speed vs Time
    plt.figure(figsize=(10, 4))
    plt.plot(time_s, df['speed_kmh'], color='blue', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Relative Time (s)")
    plt.ylabel("Speed (km/h)")
    plt.title(f"Speed Profile - {fname}")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{fname}_speed.png"))
    plt.close()

    # 2. IMU: Acceleration (G-Forces) and Gyro
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Accel (Linear Momentum variation)
    ax1.plot(time_s, df['accel_x'], label='Accel X (Forward)')
    ax1.plot(time_s, df['accel_y'], label='Accel Y (Lateral)')
    ax1.plot(time_s, df['accel_z'], label='Accel Z (Vertical)')
    ax1.set_ylabel("m/s²")
    ax1.legend()
    ax1.set_title(f"IMU - Linear Acceleration - {fname}")
    ax1.grid(True, alpha=0.3)

    # Gyro (Angular Momentum variation)
    ax2.plot(time_s, df['gyro_yaw'], label='Yaw Rate', color='red')
    ax2.plot(time_s, df['gyro_roll'], label='Roll Rate', alpha=0.5)
    ax2.set_ylabel("deg/s")
    ax2.set_xlabel("Relative Time (s)")
    ax2.legend()
    ax2.set_title("IMU - Angular Velocity")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{fname}_imu.png"))
    plt.close()

    # 3. Controls vs Time
    plt.figure(figsize=(10, 4))
    plt.step(time_s, df['throttle'], label="Throttle", where='post')
    plt.step(time_s, df['brake'], label="Brake", where='post')
    plt.plot(time_s, df['steer'], label="Steer", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Input (0-1)")
    plt.title(f"Driver Inputs - {fname}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{fname}_controls.png"))
    plt.close()

    # 4. Trajectory plot (using recorded x,y)
    plt.figure(figsize=(6, 6))
    plt.plot(df['lat'], df['lon'], label="GPS Path", color='green')
    plt.scatter(df['lat'].iloc[-1], df['lon'].iloc[-1], c='r', marker='X', s=100, label="Impact")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("GPS Trajectory")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{fname}_trajectory.png"))
    plt.close()

# ======================================================
# Comparative Summary
# ======================================================
summary = []
for fname, df in all_data.items():
    dv = compute_delta_v(df)
    summary.append({
        'file': fname,
        'delta_v_mps': dv,
        'max_speed_kmh': df['speed_kmh'].max(),
        'peak_g': (np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2).max()) / 9.81
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "crash_summary.csv"), index=False)

# Bar chart: Severity Comparison
if not summary_df.empty:
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df['file'], summary_df['peak_g'], color='orange')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Peak G-Force")
    plt.title("Crash Severity Comparison (IMU Peak G)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "_summary_severity.png"))
    plt.close()

print(f"Finished! Plots and summary.csv are in: {OUTPUT_FOLDER}")
