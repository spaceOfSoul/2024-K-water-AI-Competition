import torch
import torch.nn as nn

from utils import load_config

class Model_Handler(nn.Module):
    def __init__(self, model_config):
        self.model_config = model_config
        super(Model_Handler, self).__init__()

        # LSTM feature extractor
        self.lstm_feature = nn.LSTM(
            input_size=1, 
            hidden_size=self.model_config["HIDDEN_DIM_LSTM"],
            num_layers=self.model_config["NUM_LAYERS"],
            batch_first=True,
            dropout=self.model_config["DROPOUT"] if self.model_config["NUM_LAYERS"] > 1 else 0
        )

        # Encoder modules
        self.encoder = nn.Sequential(
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"], self.model_config["HIDDEN_DIM_LSTM"]//4),
            nn.ReLU(),
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"]//4, self.model_config["HIDDEN_DIM_LSTM"]//8),
            nn.ReLU(),
        )

        # Decoder modules
        self.decoder = nn.Sequential(
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"]//8, self.model_config["HIDDEN_DIM_LSTM"]//4),
            nn.ReLU(),
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"]//4, self.model_config["HIDDEN_DIM_LSTM"]),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm_feature(x)
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # AE
        latent_z = self.encoder(last_hidden)
        reconstructed_hidden = self.decoder(latent_z)

        return last_hidden, reconstructed_hidden