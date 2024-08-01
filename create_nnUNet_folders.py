import os
import glob
from pathlib import Path
from tqdm.auto import tqdm
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp


def force_symlink(file1, file2):
    if file2.exists():
        os.remove(file2)
    os.symlink(file1, file2)


def resample_image(image_path, reference_path, output_path, modality="ct"):
    if modality == "ct":
        filter = sitk.sitkLinear
    else:
        filter = sitk.sitkNearestNeighbor
    image = sitk.ReadImage(str(image_path))
    reference = sitk.ReadImage(str(reference_path))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(filter)
    resampled_image = resampler.Execute(image)
    sitk.WriteImage(resampled_image, str(output_path))


def process_subject(args):
    row, data_dir, dest_dir, task, gpu_id = args
    task_id = "imagesTr" if task == "train" else "imagesTs"
    lbl_id = "labelsTr" if task == "train" else "labelsTs"
    subject = row["Subjects"]
    ct = data_dir / f"imagesTr" / f"{subject}_0000.nii.gz"
    pt = data_dir / f"imagesTr" / f"{subject}_0001.nii.gz"
    lbl = data_dir / "labelsTr" / f"{subject}.nii.gz"
    assert ct.exists()
    assert pt.exists()
    assert lbl.exists()
    ct_image = sitk.ReadImage(str(ct))
    pt_image = sitk.ReadImage(str(pt))
    lbl_image = sitk.ReadImage(str(lbl))
    subject = subject.replace(" ", "_")
    if ct_image.GetSize() != pt_image.GetSize():
        print(f"Resampling CT image {ct} to match size of PT image {pt}")
        resampled_ct_path = (
            dest_dir / f"{task_id}_resampled" / f"{subject}_0000_resampled.nii.gz"
        )
        resample_image(ct, pt, resampled_ct_path, "ct")
        ct = resampled_ct_path

    if lbl_image.GetSize() != pt_image.GetSize():
        print(f"Resampling seg image {lbl} to match size of PT image {pt}")
        resampled_lbl_path = (
            dest_dir / f"{lbl_id}_resampled" / f"{subject}_0000_resampled.nii.gz"
        )
        resample_image(lbl, pt, resampled_lbl_path, "lbl")
        lbl = resampled_lbl_path

    force_symlink(lbl, dest_dir / f"{lbl_id}" / f"{subject}.nii.gz")
    force_symlink(ct, dest_dir / f"{task_id}" / f"{subject}_0000.nii.gz")
    force_symlink(pt, dest_dir / f"{task_id}" / f"{subject}_0001.nii.gz")

    op_name = dest_dir / f"{lbl_id}_Totalsegmentator_v1" / f"{subject}.nii.gz"
    ip_name = dest_dir / f"{task_id}" / f"{subject}_0000.nii.gz"
    # ip_name = str(ip_name).replace(" ", "\\ ")
    # op_name = str(op_name).replace(" ", "_")
    if Path(op_name).exists():
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.system(f"TotalSegmentator -i {ip_name} -o {op_name} --device gpu --ml")


def create_data_folders(
    data_dir: Path, dest_dir: Path, train_df: pd.DataFrame, task: str
):
    task_id = "imagesTr" if task == "train" else "imagesTs"
    lbl_id = "labelsTr" if task == "train" else "labelsTs"
    os.makedirs(dest_dir / f"{task_id}", exist_ok=True)
    os.makedirs(dest_dir / f"{lbl_id}", exist_ok=True)
    os.makedirs(dest_dir / f"{lbl_id}_Totalsegmentator", exist_ok=True)
    os.makedirs(dest_dir / f"{task_id}_resampled", exist_ok=True)
    os.makedirs(dest_dir / f"{task_id}_resampled", exist_ok=True)

    tasks = [
        (row, data_dir, dest_dir, task, idx % 5) for idx, row in train_df.iterrows()
    ]

    with mp.Pool(processes=5) as pool:
        list(
            tqdm(
                pool.imap(process_subject, tasks),
                total=len(tasks),
                desc="Processing subjects",
            )
        )


data_dir = Path("/mnt/nfs/open_datasets/autopet2024/")
dest_dir = Path(
    "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/"
)
train_df = pd.read_csv("data/train.csv")
# create_data_folders(data_dir, dest_dir, train_df, "train")
create_data_folders(data_dir, dest_dir, train_df, "test")
