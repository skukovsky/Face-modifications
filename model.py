import torch
import torch.nn as nn
from torch import optim

class Autoencoder(nn.Module):
    def __init__(self, dim_code=50):
        super(Autoencoder, self).__init__()
        self.dim_code = dim_code
        self.encoder = nn.Sequential(
            nn.Linear(45 * 45 * 3, 3000), 
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            nn.Linear(3000, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, dim_code),
            nn.BatchNorm1d(dim_code),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_code, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 3000),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            nn.Linear(3000, 45 * 45 * 3)
        )
        
    def forward(self, x):
        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        return reconstruction, latent_code


criterion = nn.MSELoss()
autoencoder = Autoencoder().to(device)
optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4, amsgrad=True, weight_decay=1e-5)