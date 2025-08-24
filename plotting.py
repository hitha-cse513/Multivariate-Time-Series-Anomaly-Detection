import matplotlib.pyplot as plt
import pandas as pd

def plot_anomaly_scores(output_csv_path: str, output_image_path: str = "anomaly_plot.png") -> None:
    """
    Plot the anomaly score vs timestamp and save it as an image.
    
    Args:
        output_csv_path (str): Path to the CSV file containing anomaly scores.
        output_image_path (str): Path where the image should be saved.
    """
    # Load the output CSV
    df = pd.read_csv(output_csv_path, parse_dates=True)

    # Try to find the timestamp column (assumes it's the first one or named 'timestamp')
    timestamp_col = df.columns[0] if 'timestamp' not in df.columns else 'timestamp'

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception:
            print("[WARN] Could not parse timestamps. Plot may not look right.")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df[timestamp_col], df['Abnormality_score'], label='Abnormality Score', color='red')
    plt.axhline(y=10, color='green', linestyle='--', label='Normal Threshold')
    plt.axhline(y=30, color='orange', linestyle='--', label='Moderate Abnormality')
    plt.axhline(y=60, color='purple', linestyle='--', label='Significant Abnormality')
    plt.axhline(y=90, color='black', linestyle='--', label='Severe Abnormality')

    plt.xlabel("Time")
    plt.ylabel("Abnormality Score (0-100)")
    plt.title("Time Series Abnormality Score")
    plt.legend()
    plt.tight_layout()

    # Save image
    plt.savefig(output_image_path)
    print(f"[INFO] Abnormality score plot saved as: {output_image_path}")

plot_anomaly_scores('output.csv')