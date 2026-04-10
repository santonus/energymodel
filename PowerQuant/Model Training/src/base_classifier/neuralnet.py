from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim
from pydantic import BaseModel
from torch import nn
from tqdm import tqdm


class NeuralNetConfig(BaseModel):
    hidden_layer_sizes: tuple[int, ...] = (8, 100, 1)
    activation: str = 'relu'
    optimizer: str = 'adam'
    learning_rate: float = 0.0003
    momentum: float = 0.9
    num_epochs: int = 1000
    batch_size: int = 128
    loss_function: str = 'mse'
    weight_decay: float = 0.0001
    verbose: int = 100

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dictionary."""
        data = self.model_dump()
        # Convert non-serializable objects to their string representation
        if not isinstance(data['activation'], str):
            data['activation'] = data['activation'].__class__.__name__
        return data


class MLP(nn.Module):
    def __init__(self, config: NeuralNetConfig):
        super(MLP, self).__init__()
        self.hidden_layer_sizes = config.hidden_layer_sizes
        self.ffn = self.construct_architecture()

    def construct_architecture(self):
        layers = [nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])]
        for i in range(1, len(self.hidden_layer_sizes) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


def get_optimizer(model: nn.Module, config: NeuralNetConfig):
    if config.optimizer == 'adam':
        return optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not supported')


def get_loss_function(config: NeuralNetConfig):
    if config.loss_function == 'mse':
        return nn.MSELoss()
    elif config.loss_function == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Loss function {config.loss_function} not supported')


def get_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    verbose: int = 100,
):
    model.train()
    pbar = tqdm(total=num_epochs, desc='Training', leave=False)
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_function(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        average_loss = total_loss / num_batches
        pbar.set_postfix({'loss': average_loss})
        pbar.update(1)
    pbar.close()
    return model


class NeuralNet:
    def __init__(self, config: NeuralNetConfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = MLP(config)
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, config)
        self.loss_function = get_loss_function(config)
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.verbose = config.verbose

        self.mean_X = None
        self.std_X = None
        self.mean_y = None
        self.std_y = None

    def preprocess_data(self, X, y):
        self.mean_X = np.mean(X, axis=0, keepdims=True)
        self.std_X = np.std(X, axis=0, keepdims=True)
        X = (X - self.mean_X) / self.std_X
        if self.config.loss_function == 'mse':
            self.mean_y = np.mean(y, axis=0, keepdims=True)
            self.std_y = np.std(y, axis=0, keepdims=True)
            y = (y - self.mean_y) / self.std_y
        return X, y

    def fit(self, X, y):
        X, y = self.preprocess_data(X, y)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        dataloader = get_dataloader(X, y, self.batch_size)
        model = train_model(
            self.model,
            self.optimizer,
            self.loss_function,
            dataloader,
            self.num_epochs,
            self.device,
            self.verbose,
        )
        self.model = model

    def predict_mse(self, X):
        X = (X - self.mean_X) / self.std_X
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_logits = self.model(X)
            y_pred = y_logits.squeeze().cpu().numpy()
            y_pred = y_pred * self.std_y + self.mean_y
        return y_pred

    def predict_bce(self, X):
        X = (X - self.mean_X) / self.std_X
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_logits = self.model(X)
            y_pred = torch.sigmoid(y_logits).squeeze().cpu().numpy()
        return y_pred

    def predict(self, X):
        if self.config.loss_function == 'mse':
            return self.predict_mse(X)
        elif self.config.loss_function == 'bce':
            return self.predict_bce(X)
        else:
            raise ValueError(f'Loss function {self.config.loss_function} not supported')
