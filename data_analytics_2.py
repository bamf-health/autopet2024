import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
import glob
from radiomics import featureextractor
import skimage
from tqdm.auto import tqdm


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
        file_pattern = "radiomics/pet_metrics_*csv"
        csv_files = glob.glob(file_pattern)

        # Read the first CSV file to initialize the combined dataframe
        combined_df = pd.read_csv(csv_files[0])
        organ_name = (csv_files[0].split("pet_metrics_")[-1]).split(".csv")[0]

        # Get all columns except "Subject"
        columns_except_subject = combined_df.columns.difference(["Subject"])
        # Add prefix to these columns
        combined_df.rename(
            columns={col: f"{organ_name}_{col}" for col in columns_except_subject},
            inplace=True,
        )
        # Merge the rest of the CSV files into the combined dataframe based on the Subjects column
        for file in tqdm(
            csv_files[1:21], total=len(csv_files[1:21]), desc="collecting csv files"
        ):
            df = pd.read_csv(file)
            organ_name = (file.split("pet_metrics_")[-1]).split(".csv")[0]
            # Get all columns except "Subject"
            columns_except_subject = df.columns.difference(["Subject"])
            # Add prefix to these columns
            df.rename(
                columns={col: f"{organ_name}_{col}" for col in columns_except_subject},
                inplace=True,
            )

            # existing_columns = set(combined_df.columns)
            # columns_to_merge = [
            #     col
            #     for col in df.columns
            #     if col not in existing_columns or col == "Subject"
            # ]
            # df_to_merge = df[columns_to_merge]
            combined_df = pd.merge(combined_df, df, on="Subject", how="outer")
        # Split the combined dataframe into two based on the Subjects column
        fdg_df = combined_df[combined_df["Subject"].str.startswith("fdg")]
        psma_df = combined_df[combined_df["Subject"].str.startswith("psma")]

        # Save each dataframe to a separate CSV file
        fdg_df.to_csv("pet_metrics_fdg.csv", index=False)
        psma_df.to_csv("pet_metrics_psma.csv", index=False)

    def remove_correlated_features(self, df, threshold=0.9):
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()

        # Select upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Drop features
        df = df.drop(columns=to_drop)
        return df

    def remove_non_informative_features(self, df):
        # Remove features with low variance
        selector = VarianceThreshold()
        df = df[df.columns[selector.fit(df).get_support()]]
        return df

    def feature_selection(self, X, y):
        # Use SelectKBest for feature selection
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(X, y)
        scores = pd.Series(selector.scores_, index=X.columns)
        top_features = scores.nlargest(10).index
        return top_features

    def run_classification(self, label_type):
        if Path(f"pet_metrics_{label_type}.csv").exists():
            label_file = f"pet_metrics_{label_type}.csv"
            print("file exists")
        elif label_type == "fdg":
            label_file = "pet_metrics_fdg.csv"
            df = pd.read_csv(label_file)
            cols_to_remove = [
                col for col in df.columns if col.startswith("Health_Status")
            ]
            df = df.drop(columns=cols_to_remove)
            df.to_csv(label_file)
        elif label_type == "psma":
            label_file = "pet_metrics_psma.csv"
            df = pd.read_csv(label_file)
            cols_to_remove = [
                col for col in df.columns if col.startswith("Health_Status")
            ]
            df = df.drop(columns=cols_to_remove)
            df.to_csv(label_file)

        else:  # Both
            fdg_df = pd.read_csv("pet_metrics_fdg.csv")
            psma_df = pd.read_csv("pet_metrics_psma.csv")
            df = pd.concat([fdg_df, psma_df], ignore_index=True)
            # Remove columns that start with 'Health_Status'
            cols_to_remove = [
                col for col in df.columns if col.startswith("Health_Status")
            ]
            df = df.drop(columns=cols_to_remove)
            df.to_csv("pet_metrics_both.csv", index=False)
            label_file = "pet_metrics_both.csv"

        if Path(label_file).exists():
            result_df = pd.read_csv(label_file)
            print(f" Final df : {len(result_df)}")
            result_df["Subject"] = result_df["Subject"].apply(
                lambda x: x.split("_00")[0] + ".nii.gz"
            )
            result_df = result_df.drop(columns="Health_Status")
            hs = pd.read_csv("healthy_tumor_patients_report.csv")
            hs.rename(columns={"Tumors_Present": "Health_Status"}, inplace=True)
            print(f" Health df : {len(result_df)}")
            result_df = pd.merge(result_df, hs, on="Subject")
            print(f" Merged df : {len(result_df)}")
        else:
            st.error(f"CSV file for {label_type} classification does not exist.")
            return

        # Drop non-feature columns
        X = result_df.drop(columns=["Health_Status", "Subject"])
        y = result_df["Health_Status"]
        print(len(X))
        st.write("Perfromaing feature selection")
        # Feature Selection
        # X = self.remove_non_informative_features(X)
        # X = self.remove_correlated_features(X)

        top_features = self.feature_selection(X, y)
        X = X[top_features]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        st.write("Perfroming Classification")
        # Initialize classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=10),
            # "Logistic Regression": LogisticRegression(max_iter=1000),
            # "Extra Trees": ExtraTreesClassifier(n_estimators=10),
        }

        # Compare classifiers
        results = {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            results[name] = {"accuracy": accuracy, "report": report}

        # Display results
        for name, result in results.items():
            st.write(f"{name} Classification Report:")
            st.text(result["report"])
            st.write(f"{name} Accuracy: {result['accuracy']:.2f}")

        # Plot the distribution of top features
        self.plot_feature_distribution(result_df, top_features, label_type)

    def plot_feature_distribution(self, df, top_features, label_type):
        # Plot distribution of top features
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(
                df[feature], kde=True, hue=df["Health_Status"], palette="viridis"
            )
            plt.title(f"Distribution of {feature} for {label_type}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.legend(title="Health Status")
            plt.show()

    def run_app(self):
        st.title("PET Radiomics Analysis")

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
                # st.write(
                #     f"CSV file for {label_name} already exists. Skipping calculations."
                # )
                continue
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
