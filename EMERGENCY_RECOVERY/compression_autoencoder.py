
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CompressionAutoencoder(nn.Module):
    '''Custom neural compression architecture'''
    def __init__(self, input_dim=1024, compressed_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, compressed_dim),
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed, compressed
    
    def compress(self, x):
        return self.encoder(x)
    
    def decompress(self, compressed):
        return self.decoder(compressed)

# Test the architecture
model = CompressionAutoencoder()
test_input = torch.randn(32, 1024)
reconstructed, compressed = model(test_input)

compression_ratio = test_input.numel() / compressed.numel()
print(f"Compression Ratio: {compression_ratio:.2f}x")
print(f"Input shape: {test_input.shape}")
print(f"Compressed shape: {compressed.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")

# Calculate reconstruction error
mse = nn.MSELoss()(reconstructed, test_input).item()
print(f"Reconstruction MSE: {mse:.6f}")
