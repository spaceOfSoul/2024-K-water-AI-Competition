import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from model import Model_Handler
from trainer import Trainer
from set_threshold import SetThreshold
from inference import Inference
from utils import load_config

def main():

    config = load_config()
    
    df_A = pd.read_csv(config["train_data"]["train_A_path"])
    df_B = pd.read_csv(config["train_data"]["train_B_path"])

    # data_loader = DataLoader(data_config=config, df=df_A, stride=1, inference=False)

    # Create Dataset
    train_dataset_A = DataLoader(config, df_A, stride=60)
    train_dataset_B = DataLoader(config, df_B, stride=60)
    train_dataset_A_B = torch.utils.data.ConcatDataset([train_dataset_A, train_dataset_B])
    
    train_dataset, val_dataset = train_test_split(train_dataset_A_B, test_size=0.2, random_state=42) # val set split & 반환값은 리스트
    print(f"Total Dataset Length: {len(train_dataset_A_B)}")
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Validation Dataset Length: {len(val_dataset)}")
    
    # torch.utils.data.DataLoader : 배치 처리, 셔플링, 병렬 처리, 데이터 변환(전처리 작업 시) 등을 지원
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=config["BATCH_SIZE"], 
                                                shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config["BATCH_SIZE"],
                                                shuffle=False)
    # (확인용) train_loader, val_loader가 배치사이즈를 기반으로, 제대로 된 DataLoader의 반복횟수가 나오는지 확인
    print(f"Train Loader Length: {len(train_loader)}")
    print(f"Validation Loader Length: {len(val_loader)}")
    for i, batch in enumerate(train_loader):
        print(f"[Train] Batch {i+1} / {len(train_loader)}")
        print(batch["input"].shape)
        break
    for i, batch in enumerate(val_loader):
        print(f"[Validation] Batch {i+1} / {len(val_loader)}")
        print(batch["input"].shape)
        break
                                             
    
    model = Model_Handler(config).cuda()
    print("<Model Summary>")
    print(model)
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    
    # # 테스트용 사이즈 확인
    # sample_batch = next(iter(train_loader))
    # inputs = sample_batch["input"].to(config["DEVICE"])
    # last_hidden, reconstructed_hidden = model(inputs)
    # print(f"Input Shape: {inputs.shape}") # torch.Size([64, 10080, 1]) -> [Batch_size, Sequence_length, Feature_dim]
    # print(f"Last Hidden Shape: {last_hidden.shape}") # torch.Size([64, 128]) -> [Batch_size, Hidden_dim]
    # print(f"Reconstructed Hidden Shape: {reconstructed_hidden.shape}") # torch.Size([64, 128]) -> [Batch_size, Hidden_dim]

    criterion = nn.MSELoss()
    print(criterion)
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    print(optimizer)

    train_losses, val_losses, best_model = Trainer(model=model, 
                                        train_loader=train_loader, 
                                        val_loader=val_loader,
                                        optimizer=optimizer, 
                                        criterion=criterion,
                                        trainer_config=config
                                        ).train_AE()
    print("Final Training Losses:", train_losses)
    print("Final Validation Losses:", val_losses)
    print("Best Model Loss:", best_model["loss"])
    print("Best Model Saved at Epoch:", best_model["epoch"])
    

    inference_model = model
    inference_model.load_state_dict(best_model["state"])
    # load_state_dict : 모델의 파라미터를 저장된 상태 딕셔너리로부터 로드해줌 - best_model["state"]에 저장된 파라미터값들이 model에 로드됨 (nn.Module로부터 상속받음)
    # best_model 제대로 저장되었는지 확인
    torch.save(best_model["state"], "best_model.pth")
    loaded_state = torch.load("best_model.pth")
    
    for name, param in loaded_state.items():
        if not torch.equal(param, best_model["state"][name]):
            raise ValueError("Model Loading Failed")
        else:
            print(f"Parameter {name} Loaded Successfully")

    print("-"*20, "Inference Start", "-"*20)
    THRESHOLD = SetThreshold(config, inference_model, train_loader).calculate_and_save_threshold()
    

    C_list = Inference(inference_model, THRESHOLD, config).detect_anomaly(test_directory=config["test_data"]["test_C_path"])
    print(f"C_list Length: {len(C_list)}") # C에 있는 .csv파일 개수
    print(f"C_list Sample\n: {C_list.head()}")
    D_list = Inference(inference_model, THRESHOLD, config).detect_anomaly(test_directory=config["test_data"]["test_D_path"])
    print(f"D_list Length: {len(D_list)}") # D에 있는 .csv파일 개수
    print(f"D_list Sample\n: {D_list.head()}")
    C_D_list = pd.concat([C_list, D_list])
    print(f"C_D_list Length: {len(C_D_list)}")
    print(f"C_D_list Sample\n: {C_D_list.head()}")

    sample_submission = pd.read_csv(config["submission_template"]["submission_template_path"])
    # 매핑된 값으로 업데이트하되, 매핑되지 않은 경우 기존 값 유지
    flag_mapping = C_D_list.set_index("ID")["flag_list"]
    sample_submission["flag_list"] = sample_submission["ID"].map(flag_mapping).fillna(sample_submission["flag_list"])
    print(f"Sample Submission Length: {len(sample_submission)}")
    print(f"Sample Submission Sample (After Mapping)\n: {sample_submission.head()}")

    # output file 이름 지정 & csv 파일로 저장
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    batch_size = config["BATCH_SIZE"]
    epochs = config["EPOCHS"]
    learning_rate = config["LEARNING_RATE"]
        
    file_name = f"{curr_time}_bs{batch_size}_ep{epochs}_lr{learning_rate}.csv"
    
    output_path_and_name = os.path.join(config["output_path"]["output_dir"], file_name)
    sample_submission.to_csv(output_path_and_name, index=False)
    print(f"Submission FIle Saved at {output_path_and_name}")
    print(f"Sample Rows from Saved Submission File:\n{sample_submission.head()}")
    

if __name__ == "__main__":
    main()