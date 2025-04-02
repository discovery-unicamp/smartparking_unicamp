import pandas as pd

# Step 1: Read Data from original csv file
df = pd.read_csv('CNRPark+EXT.csv')

#  Extract the common part of the URL to group by (up to 'camera1')
df['url_prefix'] = df['image_url'].str.split('/').str[:6].str.join('/')

#  Filter DataFrame for specific image URL prefix (up to 'camera1')
filtered_df = df[df['url_prefix'].str.contains('CNR-EXT/PATCHES/SUNNY/2015-11-12/camera1')]

#  Convert 'occupancy' to numeric, and ensure there are no invalid values (fill NaN with 0)
filtered_df['occupancy'] = pd.to_numeric(filtered_df['occupancy'], errors='coerce').fillna(0).astype(int)

# Group by 'datetime' and aggregate the relevant columns
aggregated_df = filtered_df.groupby('datetime').agg(
    total_occupancy=('occupancy', 'sum'),  # Sum of occupancy (cars present)
    first_datetime=('datetime', 'first'),  # First datetime for the group
    last_datetime=('datetime', 'last')    # Last datetime for the group
).reset_index()

#Check the aggregation result (you can print it to inspect)
print(aggregated_df['total_occupancy'].value_counts())

filtered_df.to_csv('filtered_df.csv', index=False)

# Add new columns to match your required output format
aggregated_df['Unnamed: 0'] = range(len(aggregated_df))  # Add row numbers (index)

# Extract the date and time portions separately and combine them into the desired format
filtered_df[['date', 'hour', 'minute']] = filtered_df['image_url'].str.split('/').str[-1].str.extract(r'(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})')

# Create the 'image_name' in the 'YYYY-MM-DD_HHMM' format
filtered_df['image_name'] = filtered_df['date'] + '_' + filtered_df['hour'] + filtered_df['minute']

# Now, perform the aggregation
aggregated_df = filtered_df.groupby('datetime').agg(
    total_occupancy=('occupancy', 'sum'),
    first_datetime=('datetime', 'first'),
    last_datetime=('datetime', 'last')
).reset_index()

# Add the formatted 'image_name' to the aggregated DataFrame
aggregated_df['image_name'] = filtered_df.groupby('datetime')['image_name'].first().reset_index(drop=True)

# Save the aggregated DataFrame to a CSV file (optional)
# aggregated_df.to_csv('combined_metrics.csv', index=False)

# Display the result
print(aggregated_df)

aggregated_df['image_name'] = aggregated_df['datetime'].str.replace('.', '').str.replace(':', '')


aggregated_df['timestamp'] = aggregated_df['first_datetime']  # Use the first datetime as timestamp
aggregated_df['predicted_cars'] = ''  # Placeholder for predicted cars
aggregated_df['predicted_cars_parking'] = ''  # Placeholder for predicted cars in parking
aggregated_df['real_cars'] = aggregated_df['total_occupancy']  # Real cars count (aggregated occupancy)
aggregated_df['start_time'] = ''  # Placeholder for start time
aggregated_df['end_time'] = ''  # Placeholder for end time
aggregated_df['processing_time'] = ''  # Placeholder for processing time
aggregated_df['Timestamp'] = ''  # Placeholder for a custom Timestamp
aggregated_df['predicted_background'] = ''  # Placeholder predicted background
aggregated_df['predicted_background_parking'] = ''  # Placeholder predicted background in parking
aggregated_df['real_background'] = ''  # Placeholder real background
aggregated_df['TP'] = ''  # Placeholder True Positives
aggregated_df['TN'] = ''  # Placeholder True Negatives
aggregated_df['FP'] = ''  # Placeholder False Positives
aggregated_df['FN'] = ''  # Placeholder False Negatives
aggregated_df['accuracy'] = ''  # Placeholder accuracy

# Save the aggregated DataFrame to a CSV file (optional)
aggregated_df.to_csv('labels_cnrpark.csv', index=False)

# Display the aggregated DataFrame (optional)
print(aggregated_df)
