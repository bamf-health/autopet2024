import json

# Data from the table
data = [
    {"Index": 1, "TotalSegmentator name": "spleen", "TA2 name": ""},
    {"Index": 2, "TotalSegmentator name": "kidney_right", "TA2 name": ""},
    {"Index": 3, "TotalSegmentator name": "kidney_left", "TA2 name": ""},
    {"Index": 4, "TotalSegmentator name": "gallbladder", "TA2 name": ""},
    {"Index": 5, "TotalSegmentator name": "liver", "TA2 name": ""},
    {"Index": 6, "TotalSegmentator name": "stomach", "TA2 name": ""},
    {"Index": 7, "TotalSegmentator name": "pancreas", "TA2 name": ""},
    {
        "Index": 8,
        "TotalSegmentator name": "adrenal_gland_right",
        "TA2 name": "suprarenal gland",
    },
    {
        "Index": 9,
        "TotalSegmentator name": "adrenal_gland_left",
        "TA2 name": "suprarenal gland",
    },
    {
        "Index": 10,
        "TotalSegmentator name": "lung_upper_lobe_left",
        "TA2 name": "superior lobe of left lung",
    },
    {
        "Index": 11,
        "TotalSegmentator name": "lung_lower_lobe_left",
        "TA2 name": "inferior lobe of left lung",
    },
    {
        "Index": 12,
        "TotalSegmentator name": "lung_upper_lobe_right",
        "TA2 name": "superior lobe of right lung",
    },
    {
        "Index": 13,
        "TotalSegmentator name": "lung_middle_lobe_right",
        "TA2 name": "middle lobe of right lung",
    },
    {
        "Index": 14,
        "TotalSegmentator name": "lung_lower_lobe_right",
        "TA2 name": "inferior lobe of right lung",
    },
    {"Index": 15, "TotalSegmentator name": "esophagus", "TA2 name": ""},
    {"Index": 16, "TotalSegmentator name": "trachea", "TA2 name": ""},
    {"Index": 17, "TotalSegmentator name": "thyroid_gland", "TA2 name": ""},
    {
        "Index": 18,
        "TotalSegmentator name": "small_bowel",
        "TA2 name": "small intestine",
    },
    {"Index": 19, "TotalSegmentator name": "duodenum", "TA2 name": ""},
    {"Index": 20, "TotalSegmentator name": "colon", "TA2 name": ""},
    {"Index": 21, "TotalSegmentator name": "urinary_bladder", "TA2 name": ""},
    {"Index": 22, "TotalSegmentator name": "prostate", "TA2 name": ""},
    {"Index": 23, "TotalSegmentator name": "kidney_cyst_left", "TA2 name": ""},
    {"Index": 24, "TotalSegmentator name": "kidney_cyst_right", "TA2 name": ""},
    {"Index": 25, "TotalSegmentator name": "sacrum", "TA2 name": ""},
    {"Index": 26, "TotalSegmentator name": "vertebrae_S1", "TA2 name": ""},
    {"Index": 27, "TotalSegmentator name": "vertebrae_L5", "TA2 name": ""},
    {"Index": 28, "TotalSegmentator name": "vertebrae_L4", "TA2 name": ""},
    {"Index": 29, "TotalSegmentator name": "vertebrae_L3", "TA2 name": ""},
    {"Index": 30, "TotalSegmentator name": "vertebrae_L2", "TA2 name": ""},
    {"Index": 31, "TotalSegmentator name": "vertebrae_L1", "TA2 name": ""},
    {"Index": 32, "TotalSegmentator name": "vertebrae_T12", "TA2 name": ""},
    {"Index": 33, "TotalSegmentator name": "vertebrae_T11", "TA2 name": ""},
    {"Index": 34, "TotalSegmentator name": "vertebrae_T10", "TA2 name": ""},
    {"Index": 35, "TotalSegmentator name": "vertebrae_T9", "TA2 name": ""},
    {"Index": 36, "TotalSegmentator name": "vertebrae_T8", "TA2 name": ""},
    {"Index": 37, "TotalSegmentator name": "vertebrae_T7", "TA2 name": ""},
    {"Index": 38, "TotalSegmentator name": "vertebrae_T6", "TA2 name": ""},
    {"Index": 39, "TotalSegmentator name": "vertebrae_T5", "TA2 name": ""},
    {"Index": 40, "TotalSegmentator name": "vertebrae_T4", "TA2 name": ""},
    {"Index": 41, "TotalSegmentator name": "vertebrae_T3", "TA2 name": ""},
    {"Index": 42, "TotalSegmentator name": "vertebrae_T2", "TA2 name": ""},
    {"Index": 43, "TotalSegmentator name": "vertebrae_T1", "TA2 name": ""},
    {"Index": 44, "TotalSegmentator name": "vertebrae_C7", "TA2 name": ""},
    {"Index": 45, "TotalSegmentator name": "vertebrae_C6", "TA2 name": ""},
    {"Index": 46, "TotalSegmentator name": "vertebrae_C5", "TA2 name": ""},
    {"Index": 47, "TotalSegmentator name": "vertebrae_C4", "TA2 name": ""},
    {"Index": 48, "TotalSegmentator name": "vertebrae_C3", "TA2 name": ""},
    {"Index": 49, "TotalSegmentator name": "vertebrae_C2", "TA2 name": ""},
    {"Index": 50, "TotalSegmentator name": "vertebrae_C1", "TA2 name": ""},
    {"Index": 51, "TotalSegmentator name": "heart", "TA2 name": ""},
    {"Index": 52, "TotalSegmentator name": "aorta", "TA2 name": ""},
    {"Index": 53, "TotalSegmentator name": "pulmonary_vein", "TA2 name": ""},
    {"Index": 54, "TotalSegmentator name": "brachiocephalic_trunk", "TA2 name": ""},
    {"Index": 55, "TotalSegmentator name": "subclavian_artery_right", "TA2 name": ""},
    {"Index": 56, "TotalSegmentator name": "subclavian_artery_left", "TA2 name": ""},
    {
        "Index": 57,
        "TotalSegmentator name": "common_carotid_artery_right",
        "TA2 name": "",
    },
    {
        "Index": 58,
        "TotalSegmentator name": "common_carotid_artery_left",
        "TA2 name": "",
    },
    {"Index": 59, "TotalSegmentator name": "brachiocephalic_vein_left", "TA2 name": ""},
    {
        "Index": 60,
        "TotalSegmentator name": "brachiocephalic_vein_right",
        "TA2 name": "",
    },
    {"Index": 61, "TotalSegmentator name": "atrial_appendage_left", "TA2 name": ""},
    {"Index": 62, "TotalSegmentator name": "superior_vena_cava", "TA2 name": ""},
    {"Index": 63, "TotalSegmentator name": "inferior_vena_cava", "TA2 name": ""},
    {
        "Index": 64,
        "TotalSegmentator name": "portal_vein_and_splenic_vein",
        "TA2 name": "hepatic portal vein",
    },
    {
        "Index": 65,
        "TotalSegmentator name": "iliac_artery_left",
        "TA2 name": "common iliac artery",
    },
    {
        "Index": 66,
        "TotalSegmentator name": "iliac_artery_right",
        "TA2 name": "common iliac artery",
    },
    {
        "Index": 67,
        "TotalSegmentator name": "iliac_vena_left",
        "TA2 name": "common iliac vein",
    },
    {
        "Index": 68,
        "TotalSegmentator name": "iliac_vena_right",
        "TA2 name": "common iliac vein",
    },
    {"Index": 69, "TotalSegmentator name": "humerus_left", "TA2 name": ""},
    {"Index": 70, "TotalSegmentator name": "humerus_right", "TA2 name": ""},
    {"Index": 71, "TotalSegmentator name": "scapula_left", "TA2 name": ""},
    {"Index": 72, "TotalSegmentator name": "scapula_right", "TA2 name": ""},
    {"Index": 73, "TotalSegmentator name": "clavicula_left", "TA2 name": "clavicle"},
    {"Index": 74, "TotalSegmentator name": "clavicula_right", "TA2 name": "clavicle"},
    {"Index": 75, "TotalSegmentator name": "femur_left", "TA2 name": ""},
    {"Index": 76, "TotalSegmentator name": "femur_right", "TA2 name": ""},
    {"Index": 77, "TotalSegmentator name": "hip_left", "TA2 name": ""},
    {"Index": 78, "TotalSegmentator name": "hip_right", "TA2 name": ""},
    {"Index": 79, "TotalSegmentator name": "spinal_cord", "TA2 name": ""},
    {
        "Index": 80,
        "TotalSegmentator name": "gluteus_maximus_left",
        "TA2 name": "gluteus maximus muscle",
    },
    {
        "Index": 81,
        "TotalSegmentator name": "gluteus_maximus_right",
        "TA2 name": "gluteus maximus muscle",
    },
    {
        "Index": 82,
        "TotalSegmentator name": "gluteus_medius_left",
        "TA2 name": "gluteus medius muscle",
    },
    {
        "Index": 83,
        "TotalSegmentator name": "gluteus_medius_right",
        "TA2 name": "gluteus medius muscle",
    },
    {
        "Index": 84,
        "TotalSegmentator name": "gluteus_minimus_left",
        "TA2 name": "gluteus minimus muscle",
    },
    {
        "Index": 85,
        "TotalSegmentator name": "gluteus_minimus_right",
        "TA2 name": "gluteus minimus muscle",
    },
    {"Index": 86, "TotalSegmentator name": "autochthon_left", "TA2 name": ""},
    {"Index": 87, "TotalSegmentator name": "autochthon_right", "TA2 name": ""},
    {
        "Index": 88,
        "TotalSegmentator name": "iliopsoas_left",
        "TA2 name": "iliopsoas muscle",
    },
    {
        "Index": 89,
        "TotalSegmentator name": "iliopsoas_right",
        "TA2 name": "iliopsoas muscle",
    },
    {"Index": 90, "TotalSegmentator name": "brain", "TA2 name": ""},
    {"Index": 91, "TotalSegmentator name": "skull", "TA2 name": ""},
    {"Index": 92, "TotalSegmentator name": "rib_left_1", "TA2 name": ""},
    {"Index": 93, "TotalSegmentator name": "rib_left_2", "TA2 name": ""},
    {"Index": 94, "TotalSegmentator name": "rib_left_3", "TA2 name": ""},
    {"Index": 95, "TotalSegmentator name": "rib_left_4", "TA2 name": ""},
    {"Index": 96, "TotalSegmentator name": "rib_left_5", "TA2 name": ""},
    {"Index": 97, "TotalSegmentator name": "rib_left_6", "TA2 name": ""},
    {"Index": 98, "TotalSegmentator name": "rib_left_7", "TA2 name": ""},
    {"Index": 99, "TotalSegmentator name": "rib_left_8", "TA2 name": ""},
    {"Index": 100, "TotalSegmentator name": "rib_left_9", "TA2 name": ""},
    {"Index": 101, "TotalSegmentator name": "rib_left_10", "TA2 name": ""},
    {"Index": 102, "TotalSegmentator name": "rib_left_11", "TA2 name": ""},
    {"Index": 103, "TotalSegmentator name": "rib_left_12", "TA2 name": ""},
    {"Index": 104, "TotalSegmentator name": "rib_right_1", "TA2 name": ""},
    {"Index": 105, "TotalSegmentator name": "rib_right_2", "TA2 name": ""},
    {"Index": 106, "TotalSegmentator name": "rib_right_3", "TA2 name": ""},
    {"Index": 107, "TotalSegmentator name": "rib_right_4", "TA2 name": ""},
    {"Index": 108, "TotalSegmentator name": "rib_right_5", "TA2 name": ""},
    {"Index": 109, "TotalSegmentator name": "rib_right_6", "TA2 name": ""},
    {"Index": 110, "TotalSegmentator name": "rib_right_7", "TA2 name": ""},
    {"Index": 111, "TotalSegmentator name": "rib_right_8", "TA2 name": ""},
    {"Index": 112, "TotalSegmentator name": "rib_right_9", "TA2 name": ""},
    {"Index": 113, "TotalSegmentator name": "rib_right_10", "TA2 name": ""},
    {"Index": 114, "TotalSegmentator name": "rib_right_11", "TA2 name": ""},
    {"Index": 115, "TotalSegmentator name": "rib_right_12", "TA2 name": ""},
    {"Index": 116, "TotalSegmentator name": "sternum", "TA2 name": ""},
    {"Index": 117, "TotalSegmentator name": "costal_cartilages", "TA2 name": ""},
]

# Create a dictionary from the data
labels_dict = {str(item["Index"]): item["TotalSegmentator name"] for item in data}

# Convert the dictionary to a JSON string
json_data = json.dumps(labels_dict, indent=4)

# Write the JSON string to a file
with open("ts_labels.json", "w") as json_file:
    json_file.write(json_data)

print("JSON file has been created successfully.")
