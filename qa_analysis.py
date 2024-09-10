import os
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, find_objects
from sklearn.metrics import jaccard_score
from collections import defaultdict
from tqdm.auto import tqdm
import skimage
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pandas as pd


def remove_small_blobs(image, min_size=20):
    """
    Removes blobs (connected components) smaller than a specified size in a binary image.

    Parameters:
    - image (numpy array): The input binary image (2D array).
    - min_size (int): The minimum size of blobs to keep (default is 20 pixels).

    Returns:
    - filtered_image (numpy array): The filtered binary image with small blobs removed.
    """
    # Label connected components
    labeled_image, num_features = label(image)
    pixel_counts = np.bincount(labeled_image.ravel())[
        1:
    ]  # Exclude the background (label 0)
    return pixel_counts


def zero_boundary_slices(arr):
    if arr.ndim != 3:
        raise ValueError("Input array must be 3-dimensional.")
    arr[:5, :, :] = 0
    arr[-5:, :, :] = 0
    arr[:, :5, :] = 0
    arr[:, -5:, :] = 0
    arr[:, :, :5] = 0
    arr[:, :, -5:] = 0
    return arr


def load_nifti(file_path):
    # Read the image
    image = sitk.ReadImage(file_path)

    # Cast the image to the desired integer type
    int_image = sitk.Cast(image, sitk.sitkInt32)

    return int_image


def get_connected_components(array):
    # array = sitk.GetArrayFromImage(image)
    array[array > 0] = 1  # Binarize the image
    labeled_array, num_features = label(array)
    return labeled_array, num_features


def compute_overlap_metrics(gt_label, pred_label, min_size=20):

    gt_max = gt_label.max()
    pred_max = pred_label.max()
    if gt_max == 0 and pred_max == 0:
        return 1, 0
    elif gt_max == 1 and pred_max == 0:
        return 0, 1
    elif gt_max == 1 and pred_max == 1:
        intersection = np.logical_and(gt_label, pred_label)
        union = np.logical_or(gt_label, pred_label)

        dice = (
            2.0 * np.sum(intersection) / (np.sum(gt_label) + np.sum(pred_label) + 1e-6)
        )
        jaccard = np.sum(intersection) / np.sum(union + 1e-6)

        return dice, jaccard
    else:
        return 0, 1


def compute_all_metrics(
    img_pred: sitk.Image,
    img_gt: sitk.Image,
    fname: str,
    pred_lesion: int,
    gt_lesion=int,
):
    """Get the false discovery rate for two images, predicted (image_file_pred) and ground truth (image_file_gt)."""
    gt = sitk.GetArrayFromImage(img_gt)
    pred = sitk.GetArrayFromImage(img_pred)
    print(f"GT: {np.unique(gt)}")
    print(f"Pred: {np.unique(pred)}")
    pred = (pred == pred_lesion).astype("int")
    gt = (gt == gt_lesion).astype("int")
    pred = zero_boundary_slices(pred)
    if pred.max() == 0:
        pixel_count = np.array([0])
    else:
        pixel_count = remove_small_blobs(pred, 20)
    print(pixel_count)
    if pixel_count.max() < 10:
        pred[pred > 0] = 0
    vox_ml = np.prod(img_pred.GetSpacing()) / 1000

    pd_arr = sitk.GetArrayFromImage(img_pred)
    gt_arr = sitk.GetArrayFromImage(img_gt)

    pd_cc = skimage.measure.label(pd_arr)
    gt_cc = skimage.measure.label(gt_arr)
    true_pd_cc = pd_cc.copy()
    true_pd_cc[gt_cc == 0] = 0
    overlap_gt_cc = gt_cc.copy()  # gt labels that overlap with predicted
    overlap_gt_cc[pd_cc == 0] = 0

    pd_labels = np.dstack(np.unique(pd_cc, return_counts=True))[0, 1:]
    gt_labels = np.dstack(np.unique(gt_cc, return_counts=True))[0, 1:]
    true_pd_labels = np.dstack(np.unique(true_pd_cc, return_counts=True))[0, 1:]
    overlap_gt_labels = np.dstack(np.unique(overlap_gt_cc, return_counts=True))[0, 1:]

    true_vol_overlap_ml = true_pd_labels[..., 1] * vox_ml

    # get false positives
    false_pos_vols_ml = []
    for pd_label, pd_size in pd_labels:
        if pd_label not in true_pd_labels[..., 0]:
            false_pos_vols_ml.append(pd_size * vox_ml)

    # get false negatives
    false_neg_vols_ml = []
    for gt_label, gt_size in gt_labels:
        if gt_label not in overlap_gt_labels[..., 0]:
            false_neg_vols_ml.append(gt_size * vox_ml)
    dice, jaccard = compute_overlap_metrics(gt, pred)
    print(gt.max())
    print(f"pred max: {pred.max()}")
    if gt.max() == pred.max() == 0:
        cm_val = "tn"
    elif gt.max() == 0 and pred.max() == 1:
        cm_val = "fp"
    elif gt.max() == 1 and pred.max() == 1:
        cm_val = "tp"
    elif gt.max() == 1 and pred.max() == 0:
        cm_val = "fn"
    else:
        cm_val = "unknown"
    print(f"Dice: {dice}")
    print(f"pixel_count: {pixel_count}")
    print(f"cm_val:{cm_val}")
    return {
        "Name": fname,
        "label": pred_lesion,
        "dice": dice,
        "jaccard": jaccard,
        "cm_val": cm_val,
        # "true_vol_overlap_ml_total": sum(true_vol_overlap_ml),
        # "false_pos_vol_ml_total": sum(false_pos_vols_ml),
        # "false_neg_vol_ml_total": sum(false_neg_vols_ml),
        # "true_vol_overlap_cnt": len(true_vol_overlap_ml),
        # "false_pos_cnt": len(false_pos_vols_ml),
        # "false_neg_cnt": len(false_neg_vols_ml),
        # "Sensitivity": len(true_vol_overlap_ml)
        # / (len(true_vol_overlap_ml) + len(false_pos_vols_ml)),
        # "Precision": len(true_vol_overlap_ml)
        # / (len(true_vol_overlap_ml) + len(false_neg_vols_ml)),
    }, pixel_count


def process_files(gt_dir, pred_dir, output_file, pred_lesion, gt_lesion):
    results = defaultdict(list)
    fnames = os.listdir(gt_dir)
    pixel_list = []
    for gt_filename in tqdm(fnames, desc="Calculating metrics", total=len(fnames)):
        if gt_filename.endswith(".nii.gz"):
            gt_path = os.path.join(gt_dir, gt_filename)
            pred_path = os.path.join(pred_dir, gt_filename)

            print(f"GT: {gt_path}")
            print(f"Pred: {pred_path}")
            if not os.path.exists(pred_path):
                continue

            gt_image = load_nifti(gt_path)
            print(f"GT:{np.unique(sitk.GetArrayFromImage(gt_image))}")
            pred_image = load_nifti(pred_path)
            print(f"pred:{np.unique(sitk.GetArrayFromImage(pred_image))}")

            metrics, pixel_count = compute_all_metrics(
                pred_image, gt_image, gt_filename, pred_lesion, gt_lesion
            )
            pixel_list.append({"Subject": metrics["Name"], "Counts": pixel_count})
            print(80 * "*~")
            # Convert ndarray objects to lists for JSON serialization
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()

            results[gt_filename] = metrics
    df = pd.DataFrame(pixel_list)
    df.to_csv("pixel_counts_old.csv")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def plot_box_plots(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)

    dice_scores = []
    sensitivities = []
    precisions = []

    for _, metrics in results.items():
        dice_scores.append(metrics["dice"])
        sensitivities.append(metrics["Sensitivity"])
        precisions.append(metrics["Precision"])

    data = [dice_scores, sensitivities, precisions]
    labels = ["Dice", "Sensitivity", "Precision"]
    print(f"Dice: {np.mean(dice_scores)}+/-{np.std(dice_scores)}")
    plt.boxplot(data, labels=labels)
    plt.title("Metrics Box Plot")
    plt.ylabel("Score")
    plt.show()


def main(gt_dir, pred_dir, output_file, pred_lesion, gt_lesion):
    if os.path.exists(output_file):
        print(f"Results already exist in {output_file}. Skipping computation.")
    else:
        print(f"Processing files...")
        process_files(gt_dir, pred_dir, output_file, pred_lesion, gt_lesion)
        print(f"Results saved to {output_file}.")

    # plot_box_plots(output_file)


if __name__ == "__main__":
    gt_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/labelsTs_combined/"
    pred_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset020_AutoPET2024_Regions/imagesTs_resL_pred"
    tr_gt_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/labelsTr/"
    pred_dir_19_XL = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/imagesTs_res_pred"
    pred_cl = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/confidence_estimates_pred"
    pred_cl = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/confidence_estimates"
    pred_cl_tr = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/imagesTr_resXL_pred_proba_confidence_labels"
    gt_dir_23 = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Task762_AutoPET2023/labelsTr"
    output_file = "tr_xl_cl_prd_zero_all_boundary.json"
    main(tr_gt_dir, pred_cl_tr, output_file, 20, 20)
