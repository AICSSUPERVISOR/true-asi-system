
# Tiny Recursive Model Architecture (PyTorch)
# Paper: https://arxiv.org/abs/2510.04871

import torch
import torch.nn as nn
import torch.nn.functional as F

class GridEncoder(nn.Module):
    '''Encode ARC grid to embeddings'''
    def __init__(self, config):
        super().__init__()
        self.color_embed = nn.Embedding(config.num_colors, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_grid_size * config.max_grid_size, config.hidden_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.hidden_dim, config.num_heads, config.hidden_dim * 4, config.dropout),
            num_layers=config.num_layers
        )
    
    def forward(self, grid):
        # grid: (batch, height, width)
        B, H, W = grid.shape
        flat = grid.view(B, -1)  # (batch, H*W)
        x = self.color_embed(flat)  # (batch, H*W, hidden)
        x = x + self.pos_embed[:, :H*W, :]
        x = self.transformer(x)
        return x.mean(dim=1)  # (batch, hidden)

class LatentUpdater(nn.Module):
    '''Recursively update latent state'''
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim + config.latent_dim + config.answer_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
    
    def forward(self, x, z, y):
        # x: question embedding, z: latent, y: current answer
        combined = torch.cat([x, z, y], dim=-1)
        return self.mlp(combined)

class AnswerUpdater(nn.Module):
    '''Update answer based on latent'''
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim + config.answer_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.answer_dim)
        )
    
    def forward(self, z, y):
        combined = torch.cat([z, y], dim=-1)
        return self.mlp(combined)

class GridDecoder(nn.Module):
    '''Decode answer embedding to grid'''
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.answer_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.max_grid_size * config.max_grid_size * config.num_colors)
        )
    
    def forward(self, y, target_shape):
        # y: answer embedding
        # target_shape: (height, width)
        H, W = target_shape
        logits = self.mlp(y)  # (batch, max*max*colors)
        logits = logits.view(-1, config.max_grid_size, config.max_grid_size, config.num_colors)
        logits = logits[:, :H, :W, :]  # Crop to target size
        return logits

class TinyRecursiveModel(nn.Module):
    '''Complete TRM for ARC-AGI'''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = GridEncoder(config)
        self.latent_updater = LatentUpdater(config)
        self.answer_updater = AnswerUpdater(config)
        self.decoder = GridDecoder(config)
        
        # Initial embeddings
        self.init_latent = nn.Parameter(torch.randn(1, config.latent_dim))
        self.init_answer = nn.Parameter(torch.randn(1, config.answer_dim))
    
    def forward(self, input_grid, target_shape, num_steps=None):
        '''
        Recursive reasoning forward pass
        
        Args:
            input_grid: (batch, height, width) input grid
            target_shape: (height, width) of expected output
            num_steps: number of improvement steps (default: config.max_improvement_steps)
        
        Returns:
            logits: (batch, height, width, num_colors) output logits
        '''
        if num_steps is None:
            num_steps = self.config.max_improvement_steps
        
        B = input_grid.shape[0]
        
        # Encode input
        x = self.encoder(input_grid)  # (batch, hidden)
        
        # Initialize latent and answer
        z = self.init_latent.expand(B, -1)  # (batch, latent)
        y = self.init_answer.expand(B, -1)  # (batch, answer)
        
        # Recursive improvement loop
        for step in range(num_steps):
            # Recursive latent updates
            for _ in range(self.config.recursive_updates):
                z = z + self.latent_updater(x, z, y)
            
            # Update answer
            y = y + self.answer_updater(z, y)
        
        # Decode to grid
        logits = self.decoder(y, target_shape)
        
        return logits
    
    def predict(self, input_grid, target_shape):
        '''Get predicted grid (argmax of logits)'''
        logits = self.forward(input_grid, target_shape)
        return logits.argmax(dim=-1)

# Training loop
def train_trm(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_grids, target_grids = batch
            target_shape = (target_grids.shape[1], target_grids.shape[2])
            
            optimizer.zero_grad()
            logits = model(input_grids, target_shape)
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, model.config.num_colors),
                target_grids.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

# Inference
def solve_with_trm(model, input_grid, train_examples):
    '''Solve an ARC task with TRM'''
    model.eval()
    
    # Estimate output shape from training examples
    output_shapes = [ex['output'].shape for ex in train_examples]
    avg_h = sum(s[0] for s in output_shapes) // len(output_shapes)
    avg_w = sum(s[1] for s in output_shapes) // len(output_shapes)
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_grid).unsqueeze(0)
        output = model.predict(input_tensor, (avg_h, avg_w))
    
    return output.squeeze(0).tolist()
