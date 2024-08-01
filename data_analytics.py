import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import SimpleITK as sitk


class PETScanAnalysis:
    def __init__(self, data_file, src_dir):
        self.df = pd.read_csv(data_file)
        self.src_dir = Path(src_dir)
        self.pet_list = []
        self.lbl_list = []
        self.health_status = []
        self.label_dict = self.load_label_dict()
        self.label_dict["non_specific_uptake"] = 100

        self.load_data()

    def normalize_pet_data(self, pet_img):
        pet_img_array = sitk.GetArrayFromImage(pet_img)
        mean = np.mean(pet_img_array)
        std = np.std(pet_img_array)
        normalized_array = (pet_img_array - mean) / std
        normalized_array[normalized_array < 3] = 0
        return sitk.GetImageFromArray(normalized_array)

    def load_label_dict(self):
        with open(self.src_dir / "dataset.json") as f:
            dataset_dict = json.load(f)
        return dataset_dict["labels"]

    def load_data(self):
        for idx, row in self.df.iterrows():
            pet = (
                self.src_dir
                / "imagesTr"
                / row["Subject"].replace(".nii.gz", "_0001.nii.gz")
            )
            lbl = self.src_dir / "labelsTr" / row["Subject"]
            assert pet.exists()
            assert lbl.exists()
            self.health_status.append(row["Tumors_Present"])
            self.pet_list.append(pet)
            self.lbl_list.append(lbl)

    def filter_pet_data(
        self, pet_scan, labels_image, selected_label_value, excluded_label_values
    ):
        pet_img = self.normalize_pet_data(sitk.ReadImage(pet_scan))
        labels_image = sitk.ReadImage(labels_image)
        pet_img_array = sitk.GetArrayFromImage(pet_img)
        labels_array = sitk.GetArrayFromImage(labels_image)

        if selected_label_value != 100:
            pet_img_array[labels_array != selected_label_value] = 0

        for value in excluded_label_values:
            pet_img_array[labels_array == value] = 0

        non_zero_mean_of_suv = np.mean(pet_img_array[pet_img_array != 0])
        return non_zero_mean_of_suv

    def get_mean_suv(
        self, selected_label_value, excluded_label_values, progress_bar, label
    ):
        mean_suv = []
        total_scans = len(self.pet_list)
        for idx, (pet_scan, labels_image) in enumerate(
            zip(self.pet_list, self.lbl_list)
        ):
            mean_suv.append(
                self.filter_pet_data(
                    pet_scan, labels_image, selected_label_value, excluded_label_values
                )
            )
            progress_bar.progress((idx + 1) / total_scans)
        return mean_suv

    def save_results(self, mean_suv, selected_label_name):
        result_df = pd.DataFrame(
            {
                "Subject": self.df["Subject"],
                f"Mean_SUV_{selected_label_name}": mean_suv,
                "Health_Status": self.health_status,
            }
        )
        result_df.to_csv(self.result_file, index=False)

    def plot_mean_suv(
        self, mean_suv_x, mean_suv_y, label_x, label_y, filter_subjects=None
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        unique_health_status = set(self.health_status)
        palette = sns.color_palette("hsv", len(unique_health_status))

        for idx, status in enumerate(unique_health_status):
            subset_x = [
                suv
                for suv, hs, subj in zip(
                    mean_suv_x, self.health_status, self.df["Subject"]
                )
                if hs == status
                and (
                    filter_subjects is None
                    or any(subj.lower().startswith(fs) for fs in filter_subjects)
                )
            ]
            subset_y = [
                suv
                for suv, hs, subj in zip(
                    mean_suv_y, self.health_status, self.df["Subject"]
                )
                if hs == status
                and (
                    filter_subjects is None
                    or any(subj.lower().startswith(fs) for fs in filter_subjects)
                )
            ]
            if subset_x and subset_y:  # Ensure lists are not empty
                if subset_x == subset_y:
                    # If subset_x and subset_y are the same, plot only subset_y
                    marker_style = (
                        "o"
                        if any(
                            subj.lower().startswith("psma")
                            for subj in self.df["Subject"]
                        )
                        else "s"
                    )
                    ax.scatter(
                        subset_y,
                        range(len(subset_y)),  # Dummy y-values
                        label=status,
                        color=palette[idx],
                        alpha=0.5,
                        marker=marker_style,
                    )
                    # sns.kdeplot(
                    #     subset_y, ax=ax, label=f"Status {status}", color=palette[idx]
                    # )

                else:
                    marker_style = (
                        "o"
                        if any(
                            subj.lower().startswith("psma")
                            for subj in self.df["Subject"]
                        )
                        else "s"
                    )
                    ax.scatter(
                        subset_x,
                        subset_y,
                        label=status,
                        color=palette[idx],
                        alpha=0.5,
                        marker=marker_style,
                    )

        ax.set_xlabel(f"Mean SUV {label_x}", fontsize=12)
        ax.set_ylabel(f"Mean SUV {label_y}", fontsize=12)
        ax.set_title(f"Mean SUV Scatter Plot for {label_x} vs {label_y}", fontsize=15)
        ax.legend(title="Health Status")
        sns.despine()
        st.pyplot(fig)

    def run_app(self):
        st.title("PET Scan Analysis")

        st.markdown(
            """
        <style>
            .stProgress > div > div > div > div {
                background-color: #4CAF50;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        ## Overview
        This application allows for the analysis of PET scans by filtering data based on selected labels and visualizing mean SUV values. 
        The PET scans are color-coded based on their health status.
        """
        )

        st.markdown("### Single Label Analysis")
        label_names = list(self.label_dict.keys())
        selected_label_name = st.selectbox(
            "Select Label Name for Single Label Analysis",
            ["Select a Label"] + label_names,
        )

        if selected_label_name != "Select a Label":
            self.result_file = Path(f"pet_metrics_{selected_label_name}.csv")

            if self.result_file.exists():
                result_df = pd.read_csv(self.result_file)
                mean_suv = result_df[f"Mean_SUV_{selected_label_name}"].tolist()
            else:
                selected_label_value = self.label_dict[selected_label_name]
                progress_bar = st.progress(0)
                mean_suv = self.get_mean_suv(
                    selected_label_value, [], progress_bar, selected_label_name
                )
                self.save_results(mean_suv, selected_label_name)

            self.plot_mean_suv(
                mean_suv, mean_suv, selected_label_name, selected_label_name
            )

        st.markdown("### Dual Label Analysis")
        selected_label_name_x = st.selectbox(
            "Select Label Name for X-axis", ["Select a Label"] + label_names
        )
        selected_label_name_y = st.selectbox(
            "Select Label Name for Y-axis", ["Select a Label"] + label_names
        )
        filter_psma = st.checkbox("Filter Subjects Starting with 'psma'")
        filter_fdg = st.checkbox("Filter Subjects Starting with 'fdg'")

        filter_subjects = []
        if filter_psma:
            filter_subjects.append("psma")
        if filter_fdg:
            filter_subjects.append("fdg")

        if (
            selected_label_name_x != "Select a Label"
            and selected_label_name_y != "Select a Label"
        ):
            self.result_file_x = Path(f"pet_metrics_{selected_label_name_x}.csv")
            self.result_file_y = Path(f"pet_metrics_{selected_label_name_y}.csv")

            if self.result_file_x.exists() and self.result_file_y.exists():
                result_df_x = pd.read_csv(self.result_file_x)
                result_df_y = pd.read_csv(self.result_file_y)
                mean_suv_x = result_df_x[f"Mean_SUV_{selected_label_name_x}"].tolist()
                mean_suv_y = result_df_y[f"Mean_SUV_{selected_label_name_y}"].tolist()
            else:
                selected_label_value_x = self.label_dict[selected_label_name_x]
                selected_label_value_y = self.label_dict[selected_label_name_y]

                progress_bar_x = st.progress(0)
                progress_bar_y = st.progress(0)

                mean_suv_x = self.get_mean_suv(
                    selected_label_value_x, [], progress_bar_x, selected_label_name_x
                )
                mean_suv_y = self.get_mean_suv(
                    selected_label_value_y, [], progress_bar_y, selected_label_name_y
                )

                self.save_results(mean_suv_x, selected_label_name_x)
                self.save_results(mean_suv_y, selected_label_name_y)

            self.plot_mean_suv(
                mean_suv_x,
                mean_suv_y,
                selected_label_name_x,
                selected_label_name_y,
                filter_subjects,
            )

        st.markdown("### Multiple Organs Analysis")
        selected_labels_to_zero_out = st.multiselect(
            "Select Organs to Zero Out", label_names
        )
        filter_psma_1 = st.checkbox(
            "Filter Subjects Starting with 'psma'", key="filter_psma_checkbox_2"
        )
        filter_fdg_1 = st.checkbox(
            "Filter Subjects Starting with 'fdg'", key="filter_fdg_checkbox_2"
        )
        filter_subjects = []
        if filter_psma_1:
            filter_subjects.append("psma")
        if filter_fdg_1:
            filter_subjects.append("fdg")
        if selected_labels_to_zero_out:
            excluded_label_values = [
                self.label_dict[label] for label in selected_labels_to_zero_out
            ]
            selected_label_name_for_analysis = st.selectbox(
                "Select Label for Mean SUV Calculation",
                ["Select a Label"] + label_names,
            )

            if selected_label_name_for_analysis != "Select a Label":
                self.result_file = Path(
                    f"pet_mean_suv_{selected_label_name_for_analysis}_with_exclusions.csv"
                )

                if self.result_file.exists():
                    result_df = pd.read_csv(self.result_file)
                    mean_suv = result_df[
                        f"Mean_SUV_{selected_label_name_for_analysis}"
                    ].tolist()
                else:
                    selected_label_value = self.label_dict[
                        selected_label_name_for_analysis
                    ]
                    progress_bar = st.progress(0)
                    mean_suv = self.get_mean_suv(
                        selected_label_value,
                        excluded_label_values,
                        progress_bar,
                        selected_label_name_for_analysis,
                    )
                    self.save_results(mean_suv, selected_label_name_for_analysis)

                self.plot_mean_suv(
                    mean_suv,
                    mean_suv,
                    selected_label_name_for_analysis,
                    selected_label_name_for_analysis,
                    filter_subjects,
                )


if __name__ == "__main__":
    data_file = "healthy_tumor_patients_report.csv"
    src_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/"
    pet_scan_analysis = PETScanAnalysis(data_file, src_dir)
    pet_scan_analysis.run_app()
