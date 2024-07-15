import os
import argparse
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles

"""
python /home/gmurugesan/projects/experimental_projects/AIMI/aimiv2/brain_tumor/create_dataset_brain_tumor.py /mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset001_BRATS19/dataset.json /mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset001_BRATS19/imagesTr "BAMF" --modalities "t1" "t1ce" "t2" "flair" --labels "background" "edema" "nonenhancing" "enhancing"

"""


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique(
        [i.rsplit("_", 1)[0] for i in subfiles(folder, suffix=".nii.gz", join=False)]
    )
    return uniques


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: str,
    modalities: list,
    labels: list,
    dataset_name: str,
    license: str = "hands off!",
    dataset_description: str = "",
    dataset_reference: str = "",
    dataset_release: str = "0.0",
):
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {
        "name": dataset_name,
        "description": dataset_description,
        "tensorImageSize": "3D",
        "reference": dataset_reference,
        "licence": license,
        "release": dataset_release,
        "channel_names": {str(i): modality for i, modality in enumerate(modalities)},
        "labels": {label: idx for idx, label in enumerate(labels)},
        "file_ending": ".nii.gz",
        "numTraining": len(train_identifiers),
        "numTest": len(test_identifiers),
        "training": [
            {"image": f"./imagesTr/{i}.nii.gz", "label": f"./labelsTr/{i}.nii.gz"}
            for i in train_identifiers
        ],
        "test": [f"./imagesTs/{i}.nii.gz" for i in test_identifiers],
    }

    if not output_file.endswith("dataset.json"):
        print("WARNING: Output file name is not dataset.json! Proceeding anyway...")

    save_json(json_dict, os.path.join(output_file))
    print("saved dataset.json")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Create dataset.json file for nnunet training"
    )
    argparser.add_argument(
        "output_file", type=str, help="Path to json file containing experiment details"
    )
    argparser.add_argument(
        "imagesTr_dir", type=str, help="Directory containing training inputs"
    )
    argparser.add_argument(
        "dataset_name", type=str, help="Dataset name", default="BAMF"
    )
    argparser.add_argument(
        "--dataset_description", type=str, help="Dataset description", default=""
    )
    argparser.add_argument(
        "--dataset_reference",
        type=str,
        help="Website of the dataset, if available",
        default="",
    )
    argparser.add_argument("--license", type=str, default="hands_off")
    argparser.add_argument("--dataset_release", type=str, default="0.0")
    argparser.add_argument(
        "--imagesTs_dir", type=str, help="Directory containing testing labels"
    )
    argparser.add_argument(
        "--modalities", nargs="+", type=str, help="List of modalities"
    )
    argparser.add_argument("--labels", nargs="+", type=str, help="List of label names")
    args = argparser.parse_args()

    generate_dataset_json(
        args.output_file,
        args.imagesTr_dir,
        args.imagesTs_dir,
        args.modalities,
        args.labels,
        args.dataset_name,
        args.license,
        args.dataset_description,
        args.dataset_reference,
        args.dataset_release,
    )
