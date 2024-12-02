import os
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import DataLoader
from model import Model_Handler
from trainer import Trainer
from set_threshold import SetThreshold
from inference import Inference
from utils import load_config

def main():

    config = load_config()
    
    config["CURRENT_TIME"] = datetime.now().strftime("%Y%m%d_%H%M")

    df_A = pd.read_csv(config["train_data"]["train_A_path"])
    df_B = pd.read_csv(config["train_data"]["train_B_path"])

    # data_loader = DataLoader(data_config=config, df=df_A, stride=1, inference=False)

    # Create Dataset
    train_dataset_A = DataLoader(config, df_A, stride=60)
    train_dataset_B = DataLoader(config, df_B, stride=60)
    train_dataset_A_B = torch.utils.data.ConcatDataset([train_dataset_A, train_dataset_B])

    train_loader = torch.utils.data.DataLoader(train_dataset_A_B, 
                                                batch_size=config["BATCH_SIZE"], 
                                                shuffle=True)

    model = Model_Handler(config).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    train_losses, best_model = Trainer(model=model, 
                                        train_loader=train_loader, 
                                        optimizer=optimizer, 
                                        criterion=criterion,
                                        trainer_config=config
                                        ).train_AE()

    inference_model = model
    inference_model.load_state_dict(best_model["state"])

    THRESHOLD = SetThreshold(config, inference_model, train_loader).calculate_and_save_threshold()

    C_list = Inference(inference_model, THRESHOLD, config).detect_anomaly(test_directory=config["test_data"]["test_C_path"])
    D_list = Inference(inference_model, THRESHOLD, config).detect_anomaly(test_directory=config["test_data"]["test_D_path"])
    C_D_list = pd.concat([C_list, D_list])

    sample_submission = pd.read_csv(config["submission_template"]["submission_template_path"])
    # 매핑된 값으로 업데이트하되, 매핑되지 않은 경우 기존 값 유지
    flag_mapping = C_D_list.set_index("ID")["flag_list"]
    sample_submission["flag_list"] = sample_submission["ID"].map(flag_mapping).fillna(sample_submission["flag_list"])

    # output file 이름 지정
    output_path_and_name = os.path.join(config["output_path"]["output_dir"], config["output_path"]["output_file_name_format"])
    sample_submission.to_csv(output_path_and_name, index=False)
    # curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    # batch_size = config.get("batch_size", None)
    # epochs = config.get("epochs", None)
    # learning_rate = config.get("learning_rate", None)
    
    # fime_name = f"{curr_time}_bs{batch_size}_ep{epochs}_lr{learning_rate}.csv"

if __name__ == "__main__":
    main()