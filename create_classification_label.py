import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# --- User Configuration ---
# IMPORTANT: Update these paths to point to your official TASK 3 data
OFFICIAL_TRAIN_GT_CSV_PATH = 'data/classification/ISIC2018_Task3_Training_GroundTruth.csv'
OFFICIAL_VAL_GT_CSV_PATH = 'data/classification/ISIC2018_Task3_Validation_GroundTruth.csv'
OFFICIAL_TEST_GT_CSV_PATH = 'data/classification/ISIC2018_Task3_Test_GroundTruth.csv'

# IMPORTANT: Update these paths to point to your downloaded TASK 3 images
TRAIN_IMG_DIR = 'data/classification/ISIC2018_Task3_Training_Input'  # <-- Use your new Task 3 images here
VAL_IMG_DIR = 'data/classification/ISIC2018_Task3_Validation_Input/'    # <-- Use your new Task 3 images here
TEST_IMG_DIR = 'data/classification/ISIC2018_Task3_Test_Input/'       # <-- Use your new Task 3 images here

# Define the names for the final output CSV files
TRAIN_CSV_PATH = 'train_labels.csv'
VAL_CSV_PATH = 'val_labels.csv'
TEST_CSV_PATH = 'test_labels.csv'
# --------------------------

def process_single_csv(official_csv_path, image_dir, output_csv_path, label_encoder=None):
    """
    Reads one official ISIC2018 Ground Truth CSV, converts its labels,
    matches them with images in a directory, and saves the final formatted CSV.
    """
    try:
        print(f"--- Processing {os.path.basename(output_csv_path)} ---")

        # 1. Read the official ground truth data
        print(f"Reading official ground truth data from: {official_csv_path}")
        gt_df = pd.read_csv(official_csv_path)
        print(f"Found {len(gt_df)} labels in the CSV.")

        # 2. Convert one-hot labels to integer labels
        class_columns = gt_df.columns[1:]
        gt_df['class_name'] = gt_df[class_columns].idxmax(axis=1)

        if label_encoder is None:
            label_encoder = LabelEncoder()
            gt_df['label'] = label_encoder.fit_transform(gt_df['class_name'])
            print("\nDetected Classes and their assigned integer labels:")
            for i, class_name in enumerate(label_encoder.classes_):
                print(f"- {class_name}: {i}")
        else:
            gt_df['label'] = label_encoder.transform(gt_df['class_name'])

        gt_df['image_name'] = gt_df['image'] + '.jpg'
        final_df = gt_df[['image_name', 'label']]

        # 3. Find matching image files
        print(f"Scanning for image files in: {image_dir}")
        image_filenames = set(os.listdir(image_dir))
        print(f"Found {len(image_filenames)} image files in the directory.")
        
        # 4. Filter the dataframe to only include images that actually exist
        filtered_df = final_df[final_df['image_name'].isin(image_filenames)]
        
        # 5. Save the final CSV
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"Successfully created {output_csv_path} with {len(filtered_df)} matching labels.\n")
        
        if len(filtered_df) == 0:
            print("WARNING: No matching images were found between the CSV and the image directory.")
            print("Please ensure the paths at the top of the script are correct and that you are using Task 3 images with Task 3 labels.")

        return label_encoder

    except FileNotFoundError:
        print(f"\nCRITICAL ERROR: Could not find a required file or directory.")
        print(f"Please check this path: '{official_csv_path}' or '{image_dir}'")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return label_encoder

if __name__ == '__main__':
    # Process the training data first to create the label encoder
    le = process_single_csv(OFFICIAL_TRAIN_GT_CSV_PATH, TRAIN_IMG_DIR, TRAIN_CSV_PATH)
    
    if le:
        # Reuse the encoder for validation and test sets for consistent labels
        process_single_csv(OFFICIAL_VAL_GT_CSV_PATH, VAL_IMG_DIR, VAL_CSV_PATH, label_encoder=le)
        process_single_csv(OFFICIAL_TEST_GT_CSV_PATH, TEST_IMG_DIR, TEST_CSV_PATH, label_encoder=le)
