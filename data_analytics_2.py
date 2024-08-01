import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import glob
from radiomics import featureextractor
import skimage


class PETScanAnalysis:
    def __init__(self, data_file, src_dir):
        self.df = pd.read_csv(data_file)
        self.src_dir = Path(src_dir)
        self.pet_list = []
        self.lbl_list = []
        self.health_status = []
        self.label_dict = self.load_label_dict()
        self.label_dict["non_specific_uptake"] = 100
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

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
        pet_ref = sitk.ReadImage(pet_scan)
        pet_img = self.normalize_pet_data(pet_ref)
        labels_image = sitk.ReadImage(labels_image)
        pet_img_array = sitk.GetArrayFromImage(pet_img)
        labels_array = sitk.GetArrayFromImage(labels_image)

        if selected_label_value != 100:
            pet_img_array[labels_array != selected_label_value] = 0
            labels_array[labels_array != selected_label_value] = 0
        for value in excluded_label_values:
            pet_img_array[labels_array == value] = 0
            labels_array[labels_array == selected_label_value] = 0
        labels_array[labels_array > 0] = 1
        labels_filtered_img = sitk.GetImageFromArray(labels_array)
        labels_filtered_img.CopyInformation(labels_image)
        pet_filtered_img = sitk.GetImageFromArray(pet_img_array)
        pet_filtered_img.CopyInformation(pet_ref)
        return pet_filtered_img, labels_filtered_img

    def cal_suv_vol(self, pet_img):
        pet_img_array = sitk.GetArrayFromImage(pet_img)
        non_zero_mean_of_suv = np.mean(pet_img_array[pet_img_array != 0])
        volume = np.sum(pet_img_array != 0) * np.prod(pet_img.GetSpacing())
        return non_zero_mean_of_suv, volume

    def get_pyradiomics_features(self, pet_img, labels_img, subject):
        # Ensure the label is present in the mask before extracting features
        # labels_array = sitk.GetArrayFromImage(labels_img)
        # if selected_label_value not in np.unique(labels_array):
        #     st.write(
        #         f"Selected label value {selected_label_value} not present in labels."
        #     )
        #     return None  # Skip if the label is not present

        # Extract features
        try:
            features = self.extractor.execute(pet_img, labels_img)
            # st.write(f"Extracted features: {features}")
            features["Subject"] = subject
            return {
                f"{k}": v
                for k, v in features.items()
                if not k.startswith("diagnostics")
            }
        except Exception as e:
            st.write(f"Error extracting features: {e}")
            return None

    def get_mean_suv_and_volume(
        self, selected_label_value, excluded_label_values, progress_bar, label
    ):
        mean_suv = []
        volumes = []
        fname = []
        pyradiomics_features_list = []
        hs_list = []
        total_scans = len(self.pet_list)
        for idx, (pet_scan, labels_image, hs) in enumerate(
            zip(self.pet_list, self.lbl_list, self.health_status)
        ):
            pet_filtered_img, labels_filtered_img = self.filter_pet_data(
                pet_scan, labels_image, selected_label_value, excluded_label_values
            )
            suv, vol = self.cal_suv_vol(pet_filtered_img)
            mean_suv.append(suv)
            volumes.append(vol)
            fname.append(Path(pet_scan).name)
            hs_list.append(hs)
            pyradiomics_features = self.get_pyradiomics_features(
                pet_filtered_img, labels_filtered_img, Path(pet_scan).name
            )
            if pyradiomics_features:
                pyradiomics_features_list.append(pyradiomics_features)
            progress_bar.progress((idx + 1) / total_scans)
            # print(pyradiomics_features_list)
        return fname, mean_suv, volumes, hs, pyradiomics_features_list

    def save_results(
        self, name_, mean_suv, volumes, hs, pyradiomics_features_list, label_name
    ):
        pyradiomics_df = pd.DataFrame(pyradiomics_features_list)
        result_df = pd.DataFrame(
            {
                "Subject": name_,
                f"Mean_SUV_{label_name}": mean_suv,
                f"Volume_{label_name}": volumes,
                "Health_Status": hs,
            }
        )

        # Merge the dataframes on 'Subject'
        print(len(result_df))
        print(len(pyradiomics_df))
        combined_df = pd.merge(result_df, pyradiomics_df, on="Subject", how="outer")
        combined_df.to_csv(f"pet_metrics_{label_name}.csv", index=False)

    def save_pet_metric_combined(self):
        # List all files matching the pattern pet_metrics_*csv
        file_pattern = "pet_metrics_*csv"
        csv_files = glob.glob(file_pattern)

        # Read the first CSV file to initialize the combined dataframe
        combined_df = pd.read_csv(csv_files[0])

        # Merge the rest of the CSV files into the combined dataframe based on the Subjects column
        for file in csv_files[1:]:
            df = pd.read_csv(file)
            existing_columns = set(combined_df.columns)
            columns_to_merge = [
                col
                for col in df.columns
                if col not in existing_columns or col == "Subject"
            ]
            df_to_merge = df[columns_to_merge]
            combined_df = pd.merge(combined_df, df_to_merge, on="Subject", how="outer")
        # Split the combined dataframe into two based on the Subjects column
        fdg_df = combined_df[combined_df["Subject"].str.startswith("fdg")]
        psma_df = combined_df[combined_df["Subject"].str.startswith("psma")]

        # Save each dataframe to a separate CSV file
        fdg_df.to_csv("pet_metrics_fdg.csv", index=False)
        psma_df.to_csv("pet_metrics_psma.csv", index=False)

        print(f"Saved {len(fdg_df)} rows to 'pet_metrics_fdg.csv'")
        print(f"Saved {len(psma_df)} rows to 'pet_metrics_psma.csv'")

    def run_classification(self, label_type):
        if label_type == "fdg":
            label_file = "pet_metrics_fdg.csv"
        elif label_type == "psma":
            label_file = "pet_metrics_psma.csv"
        else:  # Both
            fdg_df = pd.read_csv("pet_metrics_fdg.csv")
            psma_df = pd.read_csv("pet_metrics_psma.csv")
            df = pd.concat([fdg_df, psma_df], ignore_index=True)
            df.to_csv("pet_metrics_both.csv")
            label_file = "pet_metrics_both.csv"
        if Path(label_file).exists():
            result_df = pd.read_csv(label_file)
        else:
            st.error(f"CSV file for {label_type} classification does not exist.")
            return

        feature_columns = [
            col
            for col in result_df.columns
            if col.startswith("Mean_SUV")
            or col.startswith("Volume")
            or "pyradiomics" in col
        ]
        X = result_df[feature_columns].values
        y = result_df["Health_Status"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.write(f"Classification Report for {label_type}:")
        st.text(report)
        st.write(f"Accuracy: {accuracy:.2f}")
        feature_importances = clf.feature_importances_
        feature_names = feature_columns
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        top_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:3]
        st.write(f"Top 3 Important Features for {label_type}:")
        for feature, importance in top_features:
            st.write(f"{feature}: {importance:.4f}")

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

        st.markdown("### Calculate Metrics for All Labels")
        label_names = list(self.label_dict.keys())
        for label_name in label_names:
            if label_name in ["background"]:
                continue
            csv_file = Path(f"pet_metrics_{label_name}.csv")
            if not csv_file.exists():
                st.write(f"Calculating metrics for {label_name}...")
                selected_label_value = self.label_dict[label_name]
                progress_bar = st.progress(0)
                name_, mean_suv, volumes, hs, pyradiomics_features_list = (
                    self.get_mean_suv_and_volume(
                        selected_label_value, [], progress_bar, label_name
                    )
                )
                self.save_results(
                    name_, mean_suv, volumes, hs, pyradiomics_features_list, label_name
                )
                st.write(f"Metrics for {label_name} saved to {csv_file}")
            else:
                st.write(
                    f"CSV file for {label_name} already exists. Skipping calculations."
                )
        self.save_pet_metric_combined()
        st.markdown("### Classification")
        fdg_checkbox = st.checkbox("Include FDG", value=True)
        psma_checkbox = st.checkbox("Include PSMA", value=True)
        both_checkbox = st.checkbox("Include Both", value=True)

        if fdg_checkbox:
            self.run_classification("fdg")
        if psma_checkbox:
            self.run_classification("psma")
        if both_checkbox:
            self.run_classification("both")


if __name__ == "__main__":
    data_file = "healthy_tumor_patients_report.csv"
    src_dir = "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/"
    pet_scan_analysis = PETScanAnalysis(data_file, src_dir)
    pet_scan_analysis.run_app()
