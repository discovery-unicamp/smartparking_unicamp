import os
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob
import pandas as pd
from statistics import mean
import gc
import psutil
from datetime import datetime
import warnings
import argparse
warnings.filterwarnings("ignore", message="Setting an item of incompatible dtype is deprecated.*")

import sys
sys.path.append('..') 
from utils import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main(model,input_path, output_path,num_splits,mask_type, mask_file, savefigs='no', remove_intermediate_txts=False, generate_txt=False):
    
    if mask_type == 'post':
        mask = iio.imread(mask_file)

    try:
        model = YOLO(f'{model}.pt')
    except Exception as e:
        print(f"Error loading model: {e}, trying to load tflite")
        model = YOLO(f'{model}.tflite')
    # model.to('cpu')
    os.makedirs(output_path, exist_ok=True)

    savefigs = savefigs
    show_processed_images = False

    filter_processed_images = 'no' # 'csv' or 'images' or anything else to dont filter and always process the same images
    split_lists = True
    num_splits = num_splits

    image_list = list_images_subdirectories(input_path)
    print(f'images list: {image_list}')

    interactive_mode = True

    if filter_processed_images == 'images':

        processed_image_list = list_images_subdirectories(output_path)
        print(f'processed images list: {processed_image_list}')

        for processed_image in processed_image_list:
            modified_filename = processed_image.split('/')[-1][7:]

            # for png images
            # modified_filename = modified_filename[:-3]+'png'
            # for jpg images
            modified_filename = modified_filename[:-3]+'jpg'

            modified_filename = f'{input_path}/{modified_filename}'
            # print(modified_filename)
            if modified_filename  in image_list:
                image_list.remove(modified_filename)

    elif filter_processed_images == 'csv':
        csv_files = glob.glob(os.path.join(output_path, '**', 'df_individual_metrics_*.csv'), recursive=True)

        if len(csv_files) >= 1:
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                
                timestamp = os.path.splitext(os.path.basename(csv_file))[0].split('_')[-1]
                
                df['Timestamp'] = timestamp
                
                dfs.append(df)

            # Concatenate all DataFrames into a single DataFrame
            combined_df = pd.concat(dfs, ignore_index=True)        
            processed_image_list = combined_df['image_name'].to_list()
            print(f'\n processed images list csv: {processed_image_list}')

            for processed_image in processed_image_list:
                modified_filename = processed_image.split('/')[-1]#[7:]

                # for png images
                # modified_filename = modified_filename+'.png'
                # for jpg images
                modified_filename = modified_filename+'.jpg'

                modified_filename = f'{input_path}/{modified_filename}'
                # print(modified_filename)
                if modified_filename  in image_list:
                    image_list.remove(modified_filename)
            
        else:
            print('\n no csv files yet, when running new batch will search for it')


    total_unprocessed = len(image_list)
    print(f'\n\n processing {total_unprocessed} images')

    if split_lists == True:
        divided_image_list = divide_list_into_parts(image_list, num_splits)
    else:
        divided_image_list = image_list

    print(divided_image_list)

    columns = ['image_name','timestamp', 'predicted_cars','predicted_cars_parking', 'real_cars', 'accuracy','precision','recall','f1_score','start_time','end_time','processing_time']




    for idx,lst in enumerate(divided_image_list):


        image_list = divided_image_list[idx]
        print(f'\n\n processing {len(image_list)} images from {total_unprocessed} unprocessed')

        if interactive_mode == True:
            memory_now = psutil.virtual_memory().used / (1024 ** 2)
            print(f'memory: {memory_now}')
            # input_continue = input('do you want to process batch? (y/n)')
            input_continue = 'y'

            if input_continue == 'y':
                print("starting processing")
            else:
                print("stopping")
                break

        df = pd.DataFrame(columns=columns)
        now = datetime.now()
        filename_timestamp = now.strftime("%Y%m%dT%H%M%S")
        output_path = f'{output_path}/batch_{filename_timestamp}'
        os.makedirs(output_path, exist_ok=True)

        if image_list:

            for img_ in image_list:
                gc.collect()

                first_image_timestamp = extract_timestamp(f'{input_path}/{img_}',mode='filename_ic')
                print(first_image_timestamp)
                if 'results' not in img_:
                    print(f"{img_}\n\n\n")
                    with Image.open(img_) as img_object:
                        # img_object = iio.imread(img_)
                        # img_object = cv2.resize(img_object, (640, 480))
                        # img_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
                        if mask_type == 'pre':
                            cars1,img1,df,elapsed_time,start_time,end_time = perform_inference(input_path,output_path,img_,img_object,model,df,save=savefigs,remove_txt=remove_intermediate_txts,generate_txt=generate_txt)
                        elif mask_type == 'post':
                            cars1,img1,df,elapsed_time,start_time,end_time = perform_inference_post(input_path,output_path,img_,img_object,model,df,mask,save=savefigs,remove_txt=remove_intermediate_txts,generate_txt=generate_txt)

                        if show_processed_images == True:
                            plt.figure(figsize=(14, 10))
                            plt.imshow(img1)
                            plt.axis('off')  # Turn off axis
                            plt.show()

                        df.at[df.index[-1], 'processing_time'] = elapsed_time
                        
                        df.at[df.index[-1], 'timestamp']= first_image_timestamp

                        df.at[df.index[-1], 'start_time']= start_time
                        df.at[df.index[-1], 'end_time']= end_time



                else:
                    print("image already processed")



            df.to_csv(f'{output_path}/df_individual_metrics_{idx+1}_{filename_timestamp}.csv')

            df['processing_time'] = df['processing_time'].astype(float)

            average_processing_time = "{:.1f}".format(df['processing_time'].mean())


            data = {
                'number_images': [len(df)],
                'average_processing_time': [f'{average_processing_time}']

            }

            df_full_metrics = pd.DataFrame(data)
            df_full_metrics.to_csv(f'{output_path}/df_full_metrics.csv')
            print(df_full_metrics)

            break



        else:
            print("No images found in the specified directory.")

if __name__ == '__main__':
    """
    example usage:
    python3 inference_yolo_ultralytics.py --input_path ../../assets/demo_images --output_path ../../assets/results/results_yolo_ultralytics/yolov8n --savefigs debug --model yolov8n
    """
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input directory containing images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory where results will be saved')
    parser.add_argument('--savefigs', type=str, default='no', choices=['no', 'partial', 'debug'], help='Save figures options')
    parser.add_argument('--num_splits', type=int, default=1, help='Select number of splits in data')
    parser.add_argument('--mask_type', type=str, default="post", help='Select pre or post to masking method')
    parser.add_argument('--mask_file', type=str, default="all_black_mask", help='Select name ofthe used mask')

    args = parser.parse_args()

    main(args.model,args.input_path, args.output_path, num_splits=args.num_splits, mask_type=args.mask_type,mask_file=args.mask_file,savefigs=args.savefigs)