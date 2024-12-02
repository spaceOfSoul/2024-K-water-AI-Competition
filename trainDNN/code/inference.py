import os

import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from data_loader import DataLoader
from utils import load_config

class Inference():
    def __init__(self, model, threshold, inference_config):
        self.model = model
        self.threshold = threshold
        self.inference_config = inference_config
        
        
    def inference_test_files(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs = batch["input"].to(self.inference_config["DEVICE"])
            original_hidden, reconstructed_hidden = self.model(inputs)
            reconstruction_loss = torch.mean((original_hidden - reconstructed_hidden) ** 2, dim=1).cpu().numpy()
        return reconstruction_loss


    def detect_anomaly(self, test_directory):
        test_files = [f for f in os.listdir(test_directory) if f.startswith("TEST") and f.endswith(".csv")]
        test_datasets = []
        all_test_data = []

        for filename in tqdm(test_files, desc='Processing test files'):
            test_file = os.path.join(test_directory, filename)
            df = pd.read_csv(test_file)
            df['file_id'] = filename.replace('.csv', '')
            individual_df = df[['timestamp', 'file_id'] + df.filter(like='P').columns.tolist()]
            individual_dataset = DataLoader(self.inference_config, individual_df, inference=True)
            test_datasets.append(individual_dataset)
            
            all_test_data.append(df)

        combined_dataset = torch.utils.data.ConcatDataset(test_datasets)

        test_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=256,
            shuffle=False
        )

        reconstruction_errors = []
        for batch in tqdm(test_loader):
            reconstruction_loss = self.inference_test_files(batch)
            
            for i in range(len(reconstruction_loss)):
                reconstruction_errors.append({
                    "ID": batch["file_id"][i],
                    "column_name": batch["column_name"][i],
                    "reconstruction_error": reconstruction_loss[i]
                })
        
        errors_df = pd.DataFrame(reconstruction_errors)
        
        flag_columns = []
        for column in sorted(errors_df['column_name'].unique()):
            flag_column = f'{column}_flag'
            errors_df[flag_column] = (errors_df.loc[errors_df['column_name'] == column, 'reconstruction_error'] > self.threshold).astype(int)
            flag_columns.append(flag_column)

        errors_df_pivot = errors_df.pivot_table(index='ID', 
                                            columns='column_name', 
                                            values=flag_columns, 
                                            aggfunc='first')
        errors_df_pivot.columns = [f'{col[1]}' for col in errors_df_pivot.columns]
        errors_df_flat = errors_df_pivot.reset_index()

        errors_df_flat['flag_list'] = errors_df_flat.loc[:, 'P1':'P' + str(len(flag_columns))].apply(lambda x: x.tolist(), axis=1).apply(lambda x: [int(i) for i in x])
        return errors_df_flat[["ID", "flag_list"]]