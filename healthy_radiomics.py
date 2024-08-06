import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from tqdm.auto import tqdm
import glob
from pathlib import Path


def is_column_useful(X, y):
    X = X.values.reshape(-1, 1)
    y = y.values.ravel()
    # Normalize the column
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Calculate mutual information
    mi = mutual_info_classif(X, y, discrete_features=False)
    # print(mi)
    # You can set a threshold for usefulness
    return mi  # Adjust this threshold as needed


def correct_hs(df):
    df["Subject"] = df["Subject"].apply(lambda x: x.split("_00")[0] + ".nii.gz")
    df = df.drop(columns="Health_Status")
    hs = pd.read_csv("healthy_tumor_patients_report.csv")
    hs.rename(columns={"Tumors_Present": "Health_Status"}, inplace=True)
    df = pd.merge(df, hs, on="Subject")
    return df


def get_column():
    csv_files = sorted(glob.glob("radiomics/*.csv"))
    for file in tqdm(csv_files, total=len(csv_files), desc="collecting csv files"):
        df = pd.read_csv(file)
        organ_name = (file.split("pet_metrics_")[-1]).split(".csv")[0]
        if Path(f"features_selected_model/Selected_cols_{organ_name}.csv").exists():
            continue
        selected_columns = []
        columns_list = df.columns.difference(["Subject", "Health_Status"])
        X = correct_hs(df)
        y = X["Health_Status"]
        unnamed_columns = [col for col in X.columns if col.startswith("Unnamed:")]
        X = X.drop(columns=["Subject", "Health_Status"])
        X = X.drop(columns=unnamed_columns)
        X = X.fillna(X.median())

        # Use RandomForest model to select top 2 features
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = [columns_list[i] for i in indices[:2]]
        selected_columns.append(
            {
                "organ": organ_name,
                "feature1": top_features[0],
                "feature2": top_features[1],
            }
        )
        cols_ = pd.DataFrame(selected_columns)
        cols_.to_csv(f"features_selected_model/Selected_cols_{organ_name}.csv")


def accumulate_feature(file_pattern):
    # List all CSV files in the directory
    csv_files = glob.glob(file_pattern)

    # Initialize an empty list to store DataFrames
    filtered_dfs = []

    # Loop through each CSV file
    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file)
        # Append the filtered DataFrame to the list
        filtered_dfs.append(df)

    # Concatenate all filtered DataFrames into one
    result_df = pd.concat(filtered_dfs, ignore_index=True)
    return result_df


def classify(cdf):
    y = cdf["Health_Status"]
    X = cdf.drop(columns=["Subject", "Health_Status"])

    # Replace NaN values with the median of each column
    X = X.fillna(X.median())

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Step 2: Feature selection

    # Remove features with low variance
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)

    # Remove correlated features
    def remove_correlated_features(data, threshold=0.9):
        corr_matrix = pd.DataFrame(data).corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return pd.DataFrame(data).drop(columns=to_drop, axis=1).values

    X = remove_correlated_features(X)

    # Step 3: Run Extremely Randomized Trees (ExtraTreesClassifier)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = ExtraTreesClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train, y_train)
    # Step 4: Predict probabilities and adjust threshold
    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Define a custom threshold to maximize accuracy for negatives
    custom_threshold = 0.3  # Adjust this threshold as needed

    y_pred = (y_pred_proba >= custom_threshold).astype(int)

    # Step 5: Evaluate the model
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    # Get feature importances and indices of top features
    importances = clf.feature_importances_
    print(importances)


get_column()
df = accumulate_feature("features_selected_model/*.csv")
for idx, row in df.iterrows():
    organ = row["organ"]
    feature1 = row["feature1"]
    feature2 = row["feature2"]
    if idx == 0:
        combined_df = pd.read_csv(
            f"radiomics/pet_metrics_{organ}.csv",
            usecols=[feature1, feature2, "Subject", "Health_Status"],
        )
        combined_df.rename(
            columns={feature1: f"{organ}_{feature1}", feature2: f"{organ}_{feature2}"},
            inplace=True,
        )
    else:
        df_organ_feature = pd.read_csv(
            f"radiomics/pet_metrics_{organ}.csv",
            usecols=[feature1, feature2, "Subject", "Health_Status"],
        )
        combined_df.rename(
            columns={feature1: f"{organ}_{feature1}", feature2: f"{organ}_{feature2}"},
            inplace=True,
        )
        combined_df = pd.merge(
            combined_df, df_organ_feature, on=["Subject", "Health_Status"], how="outer"
        )
cdf = correct_hs(combined_df)
fdg = cdf[cdf["Subject"].str.startswith("fdg")]
classify(fdg)
psma = cdf[cdf["Subject"].str.startswith("psma")]
classify(psma)
classify(cdf)

# indices = np.argsort(importances)[::-1]  # Sort in descending order

# # Top features
# top_features = [
#     X_train.columns[i] for i in indices[:10]
# ]  # Change 10 to however many top features you want to consider

# # Step 4: Plot box plots for top features comparing both classes
# plt.figure(figsize=(12, 8))
# for i, feature in enumerate(top_features):
#     plt.subplot(2, 5, i + 1)  # Adjust subplot grid as needed
#     plt.boxplot(
#         [X[y == 0][:, indices[i]], X[y == 1][:, indices[i]]],
#         labels=["Negative", "Positive"],
#     )
#     plt.title(feature)
#     plt.tight_layout()

# plt.show()


# # Optionally, write the report to a text file
# with open("classification_report.txt", "w") as f:
#     f.write(report)
