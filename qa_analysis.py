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


def compute_overlap_metrics(gt_label, pred_label):
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
    img_pred: sitk.Image, img_gt: sitk.Image, fname: str, lesion: list
):
    """Get the false discovery rate for two images, predicted (image_file_pred) and ground truth (image_file_gt)."""
    gt = sitk.GetArrayFromImage(img_gt)
    pred = sitk.GetArrayFromImage(img_pred)
    print(np.unique(gt))
    print(np.unique(pred))
    for unique_label in lesion:
        if unique_label == 0:
            continue
        elif unique_label == 11:
            pred = pred == 20
        else:
            pred = pred == unique_label
        gt = gt == unique_label

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
    print(dice)
    return {
        "Name": fname,
        "label": unique_label,
        "dice": dice,
        "jaccard": jaccard,
        "true_vol_overlap_ml_total": sum(true_vol_overlap_ml),
        "false_pos_vol_ml_total": sum(false_pos_vols_ml),
        "false_neg_vol_ml_total": sum(false_neg_vols_ml),
        "true_vol_overlap_cnt": len(true_vol_overlap_ml),
        "false_pos_cnt": len(false_pos_vols_ml),
        "false_neg_cnt": len(false_neg_vols_ml),
        "Sensitivity": len(true_vol_overlap_ml)
        / (len(true_vol_overlap_ml) + len(false_pos_vols_ml)),
        "Precision": len(true_vol_overlap_ml)
        / (len(true_vol_overlap_ml) + len(false_neg_vols_ml)),
    }


def process_files(gt_dir, pred_dir, output_file, lesion=[20]):
    results = defaultdict(list)
    fnames = os.listdir(gt_dir)
    for gt_filename in tqdm(fnames, desc="Calculating metrics", total=len(fnames)):
        if gt_filename.endswith(".nii.gz"):
            gt_path = os.path.join(gt_dir, gt_filename)
            print(gt_path)
            pred_path = os.path.join(pred_dir, gt_filename)

            if not os.path.exists(pred_path):
                continue

            gt_image = load_nifti(gt_path)
            pred_image = load_nifti(pred_path)

            metrics = compute_all_metrics(
                gt_image, pred_image, gt_filename, lesion=lesion
            )
            # Convert ndarray objects to lists for JSON serialization
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()

            results[gt_filename] = metrics
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


def main(gt_dir, pred_dir, output_file, lesion):
    if os.path.exists(output_file):
        print(f"Results already exist in {output_file}. Skipping computation.")
    else:
        print(f"Processing files...")
        process_files(gt_dir, pred_dir, output_file, lesion=lesion)
        print(f"Results saved to {output_file}.")

    plot_box_plots(output_file)


if __name__ == "__main__":
    gt_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/labelsTs_combined/"
    pred_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/imagesTs_pred_res/"
    output_file = "Dataset019_AutoPET2024_res.json"
    main(gt_dir, pred_dir, output_file, lesion=[20])
