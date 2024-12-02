import torch
import numpy as np

from tqdm import tqdm
from utils import load_config

class SetThreshold():
    def __init__(self, threshold_config, inference_model, train_loader):
        self.threshold_config = threshold_config
        self.inference_model = inference_model
        self.train_loader = train_loader
        
    def calculate_and_save_threshold(self, percentile=98):
        self.inference_model.eval()
        train_errors = []
        with torch.no_grad():
            for batch in tqdm(self.train_loader):
                inputs = batch["input"].to(self.threshold_config["DEVICE"])
                original_hidden, reconstructed_hidden = self.inference_model(inputs)
                mse_errors = torch.mean((original_hidden - reconstructed_hidden) ** 2, dim=1).cpu().numpy()
                train_errors.extend(mse_errors)

        threshold = np.percentile(train_errors, percentile)

        print(f"Threshold calculated and saved: {threshold}")
        return threshold
