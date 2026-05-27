import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    # python compute_metrics.py  --model yolov12x_patch --parking_metrics /home/gustavo-luz/code/discovery/smart_parking_reitoria_unicamp/parking_reitoria/results_cam4_mai_yolo12x_patch/parking_results.csv  --labels /home/gustavo-luz/code/discovery/smart_parking_reitoria_unicamp/parking_reitoria/test_set/cam_4/labels.csv --output_dir /home/gustavo-luz/code/discovery/smart_parking_reitoria_unicamp/parking_reitoria/results_cam4_mai_yolo12x_patch/ --max_spots 50
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for parking predictions."
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--parking_metrics", required=True, help="Path to parking_metrics.csv"
    )
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument(
        "--max_spots", type=int, required=True, help="Maximum number of parking spots"
    )
    return parser.parse_args()


def load_data(parking_metrics_path, labels_path):
    """Load parking metrics and labeled data."""
    metrics_df = pd.read_csv(parking_metrics_path)
    if "sahi" in parking_metrics_path:
        # Convert 'image_name' to string if it's not already
        metrics_df["image_name"] = metrics_df["image_name"] + ".jpg"
    labels_df = pd.read_csv(labels_path)
    return metrics_df, labels_df


def process_data(metrics_df, labels_df, max_spots):
    """Process data and compute evaluation metrics."""
    # Merge the dataframes

    merged_df = pd.merge(metrics_df, labels_df, on="image_name", how="inner")

    # Calculate background spots
    merged_df["predicted_background"] = max_spots - merged_df["predicted_cars"]
    merged_df["real_background"] = max_spots - merged_df["real_cars"]

    # Calculate absolute error for each image
    merged_df["absolute_error"] = abs(
        merged_df["predicted_cars"] - merged_df["real_cars"]
    )

    # Calculate metrics
    merged_df = calculate_metrics(merged_df)

    return merged_df


def calculate_metrics(df):
    """Calculate classification metrics."""
    # Calculate confusion matrix components
    df["TP"] = df.apply(
        lambda row: min(row["predicted_background"], row["real_background"]), axis=1
    )
    df["FN"] = df.apply(
        lambda row: max(row["predicted_cars"] - row["real_cars"], 0), axis=1
    )
    df["FP"] = df.apply(
        lambda row: abs(row["predicted_background"] - row["real_background"]), axis=1
    )
    df["TN"] = df.apply(
        lambda row: min(row["predicted_cars"], row["real_cars"]), axis=1
    )

    # Calculate performance metrics
    df["accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"])
    df["recall"] = df["TP"] / (df["TP"] + df["FN"])
    df["precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["f1_score"] = 2 * (
        (df["precision"] * df["recall"]) / (df["precision"] + df["recall"])
    )

    # Sensitivity and specificity
    df["sensitivity"] = df["TP"] / (df["TP"] + df["FN"])
    df["specificity"] = df["TN"] / (df["TN"] + df["FP"])

    # Balanced accuracy
    df["bal_acc"] = (df["sensitivity"] + df["specificity"]) / 2

    return df


def summarize_metrics(df, output_dir, model):
    """Generate summary metrics and save as CSV."""
    # Calculate processing time metrics (skip first 2 if needed)
    processing_times = (
        df["processing_time"].iloc[2:] if len(df) > 2 else df["processing_time"]
    )
    avg_processing_time = processing_times.mean()
    std_processing_time = processing_times.std()

    # Calculate MAE
    mae = df["absolute_error"].mean()

    # Create summary dataframe
    summary_data = {
        "metric": [
            "Number of images",
            "Average accuracy",
            "Balanced accuracy",
            "Average precision",
            "Average recall",
            "Average F1 score",
            "Mean Absolute Error (MAE)",
            "Average processing time (s)",
            "Std processing time (s)",
        ],
        "value": [
            len(df),
            df["accuracy"].mean(),
            df["bal_acc"].mean(),
            df["precision"].mean(),
            df["recall"].mean(),
            df["f1_score"].mean(),
            mae,
            avg_processing_time,
            std_processing_time,
        ],
    }
    # print(mae.round(2))
    print(f'\n\dir {output_dir}\n {model}  \n MAE: {mae:.3f}\n{len(df)} images processed.\n\n\n')
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(output_dir, f"summary_metrics_{model}.csv"), index=False
    )

    return summary_df


def save_confusion_matrix(df, output_dir, model):
    """Generate and save confusion matrix."""
    total_TP = df["TP"].sum().astype(int)
    total_TN = df["TN"].sum().astype(int)
    total_FP = df["FP"].sum().astype(int)
    total_FN = df["FN"].sum().astype(int)

    confusion_matrix = np.array([[total_TP, total_FP], [total_FN, total_TN]])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Background", "Vehicles"],
        yticklabels=["Background", "Vehicles"],
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Confusion Matrix - {model}")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model}.png"))
    plt.close()


def parse_image_time(image_name):
    """Parse datetime from image name in format 'cam_reitoria_4-YYYY-MM-DD-HH-MM-SS.jpg'"""
    try:
        # Remove file extension first
        base_name = os.path.splitext(image_name)[0]

        # Split into parts and get the datetime portion
        parts = base_name.split("-")
        if len(parts) >= 7:  # cam_reitoria_4 + 6 datetime parts
            dt_str = "-".join(parts[1:7])  # Get YYYY-MM-DD-HH-MM-SS
            return datetime.strptime(dt_str, "%Y-%m-%d-%H-%M-%S")
        return None
    except Exception as e:
        print(f"Error parsing time from image {image_name}: {e}")
        return None


def classify_period(hour):
    """Classify hour into specific time periods."""
    if 5 <= hour < 8:
        return "early_morning"
    elif 8 <= hour < 12:
        return "middle_morning"
    elif 12 <= hour < 15:
        return "lunch"
    elif 15 <= hour < 18:
        return "late_afternoon"
    else:
        return "night"


def analyze_by_time_of_day(df, output_dir, model):
    """Analyze metrics by time of day using custom periods."""
    # Extract time from image name
    df["datetime"] = df["image_name"].apply(parse_image_time)

    # Remove rows where datetime couldn't be parsed
    df = df.dropna(subset=["datetime"])

    # Ensure the datetime column is properly converted to datetime type
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Now we can safely use .dt accessor
    df["hour"] = df["datetime"].dt.hour

    # Apply custom time period classification
    df["time_period"] = df["hour"].apply(classify_period)

    # Define the order of periods for consistent plotting
    period_order = [
        "early_morning",
        "middle_morning",
        "lunch",
        "late_afternoon",
        "night",
    ]

    # Group by time period and calculate metrics
    time_metrics = (
        df.groupby("time_period")
        .agg(
            {
                "accuracy": "mean",
                "bal_acc": "mean",
                "precision": "mean",
                "recall": "mean",
                "f1_score": "mean",
                "absolute_error": ["mean", "std"],  # MAE and its std by time period
                "processing_time": "mean",
            }
        )
        .reindex(period_order)
        .reset_index()
    )

    time_metrics.to_csv(
        os.path.join(output_dir, f"metrics_by_time_period_{model}.csv"), index=False
    )

    # Plot MAE by time period
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="time_period",
        y=("absolute_error", "mean"),
        data=time_metrics,
        order=period_order,
    )
    plt.title(f"MAE by Time Period - {model}")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Time Period")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mae_by_time_period_{model}.png"))
    plt.close()

    return time_metrics


def plot_actual_vs_predicted(df, output_dir, model):
    """Plot actual vs predicted number of cars."""
    # Sort by datetime for chronological order
    df_sorted = df.sort_values("datetime")

    plt.figure(figsize=(15, 6))
    plt.plot(
        df_sorted["datetime"],
        df_sorted["real_cars"],
        label="Actual Cars",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        df_sorted["datetime"],
        df_sorted["predicted_cars"],
        label="Predicted Cars",
        marker="o",
        linestyle="-",
        color="orange",
    )
    plt.title(f"Actual vs Predicted Cars - {model}")
    plt.ylabel("Number of Cars")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"actual_vs_predicted_{model}.png"))
    plt.close()


def plot_ordered_mae(df, output_dir, model):
    """Plot ordered image names and their MAE values."""
    # Sort by datetime for proper ordering
    df_sorted = df.sort_values("datetime")

    plt.figure(figsize=(15, 6))
    plt.plot(
        df_sorted["image_name"],
        df_sorted["absolute_error"],
        marker="o",
        linestyle="-",
        markersize=3,
    )
    plt.axhline(
        y=df_sorted["absolute_error"].mean(),
        color="r",
        linestyle="--",
        label=f'Mean MAE: {df_sorted["absolute_error"].mean():.2f}',
    )
    plt.title(f"Absolute Error by Image - {model}")
    plt.ylabel("Absolute Error (cars)")
    plt.xlabel("Image Name")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"absolute_error_by_image_{model}.png"))
    plt.close()


def analyze_by_patching(df, output_dir, model):
    """Analyze metrics by patching status."""
    if "patching" not in df.columns:
        return

    grouped = df.groupby("patching").agg(
        {
            "accuracy": "mean",
            "bal_acc": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1_score": "mean",
            "absolute_error": "mean",  # Add MAE to patching analysis
            "processing_time": "mean",
        }
    )

    grouped.to_csv(os.path.join(output_dir, f"metrics_by_patching_{model}.csv"))


def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process data
    metrics_df, labels_df = load_data(args.parking_metrics, args.labels)
    merged_df = process_data(metrics_df, labels_df, args.max_spots)

    # Save merged data for inspection
    merged_df.to_csv(os.path.join(args.output_dir, "merged_metrics.csv"), index=False)

    # Generate and save metrics
    summary_df = summarize_metrics(merged_df, args.output_dir, args.model)
    save_confusion_matrix(merged_df, args.output_dir, args.model)

    # Generate MAE plots and time of day analysis
    analyze_by_time_of_day(merged_df, args.output_dir, args.model)
    plot_ordered_mae(merged_df, args.output_dir, args.model)
    plot_actual_vs_predicted(merged_df, args.output_dir, args.model)

    # Additional analysis by patching status if available
    analyze_by_patching(merged_df, args.output_dir, args.model)

    print(f"Metrics computed and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
