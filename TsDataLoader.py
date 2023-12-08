import os
import pickle
import numpy as np
import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# Forecasting Dataloader
def Window_processing_CNUH(df, window_len, scaler, stride):
    # CNUH features list
    features_list = ['Albumin', 'Hgb', 'BUN', 'Alkaline phosphatase', 'WBC Count',
                 'SBP', 'Gender', 'Total calcium', 'RR', 'Age', 'Total bilirubin',
                 'Creatinin', 'ALT', 'Lactate', 'SaO2', 'AST', 'Glucose', 'Sodium', 'BT',
                 'HR', 'CRP', 'Chloride', 'Potassium', 'platelet', 'Total protein']

    #Normalization
    if scaler is not None:
        scaler = StandardScaler()
        scaler.fit(df[features_list])
        features = scaler.transform(df[features_list])
        for idx, feature_name in enumerate(features_list):
            df[feature_name] = features[:, idx]

    #id_label
    patient_cnts = np.unique(df["Patient"], return_counts=True)
    patient_cnts = dict(zip(patient_cnts[0], patient_cnts[1]))
    patient_ids  = list(patient_cnts.keys())

    patient_abnormal_ids = np.unique(df.query('target==1')['Patient'])
    patient_normal_ids = np.unique(df.query('target==0')['Patient'])

    #Window_setting
    data_info = {}
    idx_debug = 0
    for patient_id in tqdm.tqdm(patient_ids):
        df_patient = df.query(f'Patient=={patient_id}')
        for idx in range(len(df_patient) - window_len + 1 - stride):
            row_info = {}
            from_idx = idx
            to_idx   = idx + window_len - 1
            to_target = to_idx + stride

            ############# row ################
            row_info["pid"] = df_patient["Patient"].iloc[from_idx: to_idx + 1].values
            row_info["x"] = df_patient[features_list].iloc[from_idx: to_idx + 1].values
            row_info["y"] = df_patient["target"].iloc[from_idx: to_target].values
            row_info["seq_y"] = df_patient["target"].iloc[to_target]
            #row_info["seq_y"] = df_patient["target"].iloc[to_idx + 1: to_target + 1] # lay het

            ############ append ##############
            for key in row_info:
                if data_info.get(key) is None: data_info[key] = []
                data_info[key].append(row_info[key])
                pass # key


            # print(f'{idx} - {idx + window_len - 1}')
            # break
            pass # row
        #idx_debug = idx_debug + 1 # Neu debug -> day het tham so ra ngoai
        #if idx_debug>=10: break
        pass # data

    for key in row_info:
        data_info[key] = np.array(data_info[key])
        pass

    x = data_info["x"]
    y = data_info["seq_y"]
    #y = data_info["seq_y"]

    y_onehot = np.zeros((len(data_info["seq_y"]),2), dtype=np.float32)
    for idx in range(len(y)):
        y_onehot[idx, y[idx]] = 1.0

    #if kwargs.get("global_scope") is not None: kwargs["global_scope"].update(**locals())
    return x, y, y_onehot


def Window_processing_UV(df, window_len, scaler, stride):
    # UV features list
    features_list = [x for x in df.columns.tolist() if x not in ["id", "y"]]

    #Normalization
    if scaler is not None:
        scaler = StandardScaler()
        scaler.fit(df[features_list])
        features = scaler.transform(df[features_list])
        for idx, feature_name in enumerate(features_list):
            df[feature_name] = features[:, idx]

    #id_label
    patient_cnts = np.unique(df["id"], return_counts=True)
    patient_cnts = dict(zip(patient_cnts[0], patient_cnts[1]))
    patient_ids  = list(patient_cnts.keys())

    patient_abnormal_ids = np.unique(df.query('y==1')['id'])
    patient_normal_ids = np.unique(df.query('y==0')['id'])

    #Window_setting
    data_info = {}

    for patient_id in tqdm.tqdm(patient_ids):
        df_patient = df.query(f'id=={patient_id}')
        for idx in range(len(df_patient) - window_len + 1 - stride):
            row_info = {}
            from_idx = idx
            to_idx   = idx + window_len - 1
            to_target = to_idx + stride

            ############# row ################
            row_info["pid"] = df_patient["id"].iloc[from_idx: to_idx + 1].values
            row_info["x"] = df_patient[features_list].iloc[from_idx: to_idx + 1].values
            row_info["y"] = df_patient["y"].iloc[from_idx: to_target].values
            row_info["seq_y"] = df_patient["y"].iloc[to_target]
            #row_info["seq_y"] = df_patient["target"].iloc[to_idx + 1: to_target + 1] # lay het

            ############ append ##############
            for key in row_info:
                if data_info.get(key) is None: data_info[key] = []
                data_info[key].append(row_info[key])
                pass # key


            # print(f'{idx} - {idx + window_len - 1}')
            # break
            pass # row
        #idx_debug = idx_debug + 1 # Neu debug -> day het tham so ra ngoai
        #if idx_debug>=10: break
        pass # data

    for key in row_info:
        data_info[key] = np.array(data_info[key])
        pass

    #np.savez(f'{save_dir}/data_info.npz', **data_info)

    x = data_info["x"]
    y = data_info["seq_y"]
    #y = data_info["seq_y"]

    y_onehot = np.zeros((len(data_info["seq_y"]),2), dtype=np.float32)
    for idx in range(len(y)):
        y_onehot[idx, y[idx]] = 1.0

    #if kwargs.get("global_scope") is not None: kwargs["global_scope"].update(**locals())
    return x, y, y_onehot