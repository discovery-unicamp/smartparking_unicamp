import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for model predictions.")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--base_dir_labeled_data", required=True, help="Base directory for labeled data")
    parser.add_argument("--base_dir_inference_csvs", required=True, help="Base directory for inference CSV files")
    parser.add_argument("--base_dir_results", required=True, help="Base directory to save results")
    parser.add_argument("--dataset", required=True, help="Base directory to save results")
    return parser.parse_args()

def load_data(base_dir_labeled_data, base_dir_inference_csvs,dataset):
    """Load labeled data and model predictions."""
    if dataset == 'cnrpark':
        base_labeled_df = pd.read_csv(f'{base_dir_labeled_data}/labels_cnrpark.csv')
    elif dataset == 'ic2':
        base_labeled_df = pd.read_csv(f'{base_dir_labeled_data}/labels_ic2.csv')
    else:
        try:
            base_labeled_df = pd.read_csv(f'{base_dir_labeled_data}/labels_{dataset}.csv')
        except:
            print('Dataset not found')
    
    dfs = []
    # print(base_dir_inference_csvs)
    csv_files = glob.glob(os.path.join(base_dir_inference_csvs, '**', 'df_individual_metrics*.csv'), recursive=True)
    # print(csv_files)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return base_labeled_df, combined_df

def process_data(combined_df, base_labeled_df,dataset):
    """Process data and compute evaluation metrics."""
    print(base_labeled_df['image_name'].iloc[0])
    if dataset == 'cnrpark':
        max_spots = 35
    elif dataset == 'ic2':
        max_spots = 16
    else:
        max_spots = input('Enter the maximum number of spots in the dataset: ')
    print(f'max spots is set to {dataset} parking: {max_spots}')
    if 'croped_' in base_labeled_df['image_name'].iloc[0]:
        input_continue = input('Do you want to continue? (y/n): ')
        if input_continue.lower() != 'y':
            exit()
        combined_df['image_name'] = 'croped_' + combined_df['image_name']
    else:
        input_continue = input('Do you want to continue? (y/n): ')
        if input_continue.lower() != 'y':
            exit()
    combined_df.rename(columns={'predicted_persons': 'predicted_cars', 'inference_time':'processing_time'}, inplace=True)
    print(combined_df,combined_df.columns)
    print(base_labeled_df,base_labeled_df.columns)
    combined_df['image_name'] = combined_df['image_name'].str.replace('.jpg', '')
    merged_df = pd.merge(combined_df, base_labeled_df[['image_name', 'real_cars']], on='image_name', how='inner')
    
    merged_df.to_csv(os.path.join('./', 'merged_metrics.csv'), index=False)
    merged_df['predicted_background'] = max_spots - merged_df['predicted_cars']
    if 'real_cars_x' in merged_df.columns or 'real_cars_y' in merged_df.columns:
        merged_df['real_cars'] = merged_df['real_cars_y']

    merged_df['real_background'] = max_spots - merged_df['real_cars']
    merged_df.to_csv(os.path.join('./', 'merged_metrics.csv'), index=False)
    print(merged_df)
    merged_df = calculate_metrics(merged_df)


    return merged_df

def calculate_metrics(df):
    """Calculate classification metrics."""

    df['TP'] = df.apply(lambda row: min(row['predicted_background'], row['real_background']), axis=1)
    df['FN'] = df.apply(lambda row: max(row['predicted_cars'] - row['real_cars'], 0), axis=1)
    df['FP'] = df.apply(lambda row: abs(row['predicted_background'] - row['real_background']), axis=1)
    df['TN'] = df.apply(lambda row: min(row['predicted_cars'], row['real_cars']), axis=1)

    # Accuracy, recall, precision, F1 score
    df['accuracy'] = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])
    df['recall'] = df['TP'] / (df['TP'] + df['FN'])
    df['precision'] = df['TP'] / (df['TP'] + df['FP'])
    df['f1_score'] = 2 * ((df['precision'] * df['recall']) / (df['precision'] + df['recall']))
    
    # Sensitivity and specificity
    df['sensitivity'] = df['TP'] / (df['TP'] + df['FN'])
    df['specificity'] = df['TN'] / (df['TN'] + df['FP'])
    
    # Balanced accuracy
    df['bal_acc'] = (df['sensitivity'] + df['specificity']) / 2
    
    return df

def summarize_metrics(df, base_dir_results, model, suffix=""):
    """Generate summary metrics and save as CSV."""
    filename_suffix = f"_{suffix}" if suffix else ""
    
    # Calculate processing time metrics while discarding first 2 times
    processing_times = df['processing_time'].iloc[2:]  # Skip first 2 entries
    avg_processing_time = processing_times.mean()
    std_processing_time = processing_times.std()
    
    # Calculate metrics
    avg_bal_acc = df['bal_acc'].mean()
    
    # Create formatted strings
    bal_acc_percent = f"{avg_bal_acc * 100:.2f}%"
    
    # Milliseconds formatting
    avg_processing_ms = avg_processing_time * 1000
    std_processing_ms = std_processing_time * 1000
    processing_time_ms_str = f"{avg_processing_ms:.0f} ± {std_processing_ms:.1f} ms"
    
    # Seconds formatting
    processing_time_sec_str = f"{avg_processing_time:.3f} ± {std_processing_time:.3f} s"

    summary_data = {
        'number_images': [len(df)],
        'average_accuracy': [df['accuracy'].mean()],
        'average_accuracy_balanced': [df['bal_acc'].mean()],
        'average_precision': [df['precision'].mean()],
        'average_recall': [df['recall'].mean()],
        'average_f1': [df['f1_score'].mean()],
        'average_processing_time_seconds': [avg_processing_time],
        'std_processing_time_seconds': [std_processing_time],
        'balanced_accuracy_percentage': [bal_acc_percent],
        'processing_time_ms_formatted': [processing_time_ms_str],
        'processing_time_sec_formatted': [processing_time_sec_str]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)

    print(f"\n Processing time: {processing_time_sec_str}\n")
    print(f"\n Balanced accuracy: {bal_acc_percent}\n")
    summary_df.to_csv(os.path.join(base_dir_results, f'summary_metrics{filename_suffix}_{model}.csv'), index=False)

def save_confusion_matrix(df, base_dir_results, model, suffix=""):
    """Generate and save confusion matrix."""
    filename_suffix = f"_{suffix}" if suffix else ""
    total_TP = df['TP'].sum().astype(int)
    total_TN = df['TN'].sum().astype(int)
    total_FP = df['FP'].sum().astype(int)
    total_FN = df['FN'].sum().astype(int)
    total_samples = len(df)
    
    confusion_matrix = np.array([[total_TP, total_FP], [total_FN, total_TN]])
    print(confusion_matrix)
    group_names = ['TP', 'FP', 'FN', 'TN']
    group_counts = [f"{value}" for value in confusion_matrix.flatten()]
    group_percentages = [f"{value:.2%}" for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    labels = np.asarray([f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]).reshape(2, 2)
    
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=['Background', 'Vehicles'], yticklabels=['Background', 'Vehicles'], annot_kws={"fontsize": 40})
    plt.xlabel('Actual', fontsize=40)
    plt.ylabel('Predicted', fontsize=40)
    plt.title(f'Confusion Matrix - {model} (Samples: {total_samples})', fontsize=42)
    
    heatmap.xaxis.set_tick_params(labelsize=38)
    heatmap.yaxis.set_tick_params(labelsize=38)
    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    
    plt.savefig(os.path.join(base_dir_results, f'confusion_matrix{filename_suffix}_{model}.png'), bbox_inches='tight')

def main():
    args = parse_arguments()
    base_labeled_df, combined_df = load_data(args.base_dir_labeled_data, args.base_dir_inference_csvs,args.dataset)
    merged_df = process_data(combined_df, base_labeled_df,args.dataset)
    
    # Save results for all images
    summarize_metrics(merged_df, args.base_dir_results, args.model, suffix="all")
    save_confusion_matrix(merged_df, args.base_dir_results, args.model, suffix="all")
    
    # Filter and save results for images with at least one car
    filtered_df = merged_df[merged_df['real_cars'] > 0]
    summarize_metrics(filtered_df, args.base_dir_results, args.model, suffix="cars_only")
    save_confusion_matrix(filtered_df, args.base_dir_results, args.model, suffix="cars_only")
    
    print("Evaluation metrics computed and saved successfully.")

if __name__ == "__main__":
    main()
