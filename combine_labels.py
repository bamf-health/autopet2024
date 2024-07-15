import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
from pathlib import Path
import subprocess


def create_dataset_json(
    script_path, dataset_json_path, images_tr_path, institution, modalities, labels
):
    """
    Runs the brain tumor dataset creation script with the given parameters.

    Parameters:
    - script_path (str): Path to the brain tumor dataset creation script.
    - dataset_json_path (str): Path to the dataset JSON file.
    - images_tr_path (str): Path to the training images directory.
    - institution (str): Name of the institution.
    - modalities (list of str): List of modalities.
    - labels (list of str): List of labels.
    """
    # Validate input paths
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"The script path does not exist: {script_path}")
    if not os.path.isdir(images_tr_path):
        raise NotADirectoryError(
            f"The images training path does not exist: {images_tr_path}"
        )

    # Construct the command
    command = (
        [
            "python",
            script_path,
            dataset_json_path,
            images_tr_path,
            institution,
            "--modalities",
        ]
        + modalities
        + ["--labels"]
        + labels
    )
    print(command)
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Handle the result
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: {result.stderr}"
        )
    else:
        print(f"Command executed successfully: {result.stdout}")


# Directories
labelsTr_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/labelsTr_original"
labelsTr_Totalsegmentator_v1_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/labelsTr_Totalsegmentator_v1"
labelsTr_combined_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/labelsTr"

# # Create the output directory if it does not exist
# os.makedirs(labelsTr_combined_dir, exist_ok=True)

# # Load TotalSegmentator labels to JSON mapping
# with open("ts_labels.json") as f:
#     totalsegmentator_labels = json.load(f)

# # Variable to select labels of interest (example: ['spleen', 'liver'])
# labels_of_interest = [
#     "spleen",
#     "kidney_right",
#     "kidney_left",
#     "gallbladder",
#     "liver",
#     "stomach",
#     "pancreas",
#     "adrenal_gland_right",
#     "adrenal_gland_left",
#     "esophagus",
#     "thyroid_gland",
#     "small_bowel",
#     "duodenum",
#     "colon",
#     "urinary_bladder",
#     "prostate",
#     "heart",
#     "brain",
#     "skull",
# ]  # Adjust this list as needed

# # Get the list of files in the directories
# labelsTr_files = [f for f in os.listdir(labelsTr_dir) if f.endswith(".nii.gz")]

# # Process each file
# tumors_present = []
# for filename in tqdm(
#     labelsTr_files, desc="Combining labels", total=len(labelsTr_files)
# ):
#     labelsTr_path = os.path.join(labelsTr_dir, filename)
#     labelsTr_Totalsegmentator_v1_path = os.path.join(
#         labelsTr_Totalsegmentator_v1_dir, filename
#     )
#     combined_output_path = os.path.join(labelsTr_combined_dir, filename)
#     # Read the image from labelsTr
#     labelsTr_image = sitk.ReadImage(labelsTr_path)
#     labelsTr_array = sitk.GetArrayFromImage(labelsTr_image)

#     # Check if the image has only values 0 and 1
#     unique_values = np.unique(labelsTr_array)
#     tumors_present.append({"Subject": filename, "Tumors_Present": unique_values.max()})
#     if os.path.exists(combined_output_path):
#         continue
#     if set(unique_values) <= {0, 1}:
#         # Read the image from labelsTr_Totalsegmentator_v1
#         labelsTr_Totalsegmentator_v1_image = sitk.ReadImage(
#             labelsTr_Totalsegmentator_v1_path
#         )
#         labelsTr_Totalsegmentator_v1_array = sitk.GetArrayFromImage(
#             labelsTr_Totalsegmentator_v1_image
#         )

#         if labels_of_interest:
#             combined_array = np.zeros(labelsTr_Totalsegmentator_v1_array.shape)
#             # Re-map the selected labels of interest to values 1 to len(labels_of_interest)
#             label_mapping = {
#                 int(key): idx + 1
#                 for idx, label in enumerate(labels_of_interest)
#                 for key, value in totalsegmentator_labels.items()
#                 if value == label
#             }
#             # print(label_mapping)
#             for label, new_value in label_mapping.items():
#                 combined_array[labelsTr_Totalsegmentator_v1_array == label] = new_value

#             # Set the values == 1 in labelsTr_array to the combined_array with value len(labels_of_interest) + 1
#             combined_array[labelsTr_array == 1] = len(labels_of_interest) + 1
#         else:
#             # Set the values == 1 in labelsTr_array to the combined_array with value 118
#             combined_array = np.copy(labelsTr_Totalsegmentator_v1_array)
#             combined_array[labelsTr_array == 1] = 118

#         # Convert the combined array back to an image
#         combined_image = sitk.GetImageFromArray(combined_array)
#         combined_image.CopyInformation(labelsTr_Totalsegmentator_v1_image)

#         # Save the combined image to the new directory
#         sitk.WriteImage(combined_image, combined_output_path)
#         # print(f"Processed and saved: {filename}")
#     else:
#         print(f"Skipping {filename}: contains values other than 0 and 1")

# # Save the tumor presence report
# df = pd.DataFrame(tumors_present)
# df.to_csv("healthy_tumor_patients_report.csv")


dest_dir = Path(
    "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/"
)
script_path = "create_dataset_json.py"
dataset_json_path = Path(dest_dir, "dataset.json").resolve()
images_tr_path = Path(dest_dir, "imagesTr").resolve()
institution = "BAMF"
modalities = ["CT"]
labels = [
    "background",
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "esophagus",
    "thyroid_gland",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "heart",
    "brain",
    "skull",
]

create_dataset_json(
    script_path, dataset_json_path, images_tr_path, institution, modalities, labels
)
