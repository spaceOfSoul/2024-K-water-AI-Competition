import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, model, train_loader, optimizer, criterion, trainer_config):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainer_config = trainer_config
        
    def train_AE(self):
        train_losses = []
        best_model = {
            "loss": float('inf'),
            "state": None,
            "epoch": 0
        }

        for epoch in range(self.trainer_config["EPOCHS"]):
            self.model.train()
            epoch_loss = 0.0

            with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.trainer_config['EPOCHS']}", unit="batch") as t:
                for batch in t:
                    inputs = batch["input"].to(self.trainer_config["DEVICE"])
                    original_hidden, reconstructed_hidden = self.model(inputs) # [ Batch_size, HIDDEN_DIM_LSTM ]

                    loss = self.criterion(reconstructed_hidden, original_hidden)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(loss=loss.item())

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_epoch_loss)

            print(f"Epoch {epoch + 1}/{self.trainer_config['EPOCHS']}, Average Train Loss: {avg_epoch_loss:.8f}")
            
            if avg_epoch_loss < best_model["loss"]:
                best_model["state"] = self.model.state_dict()
                best_model["loss"] = avg_epoch_loss
                best_model["epoch"] = epoch + 1

        return train_losses, best_model