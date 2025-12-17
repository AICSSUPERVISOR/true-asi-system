
import torch
import torch.nn as nn

class CrossDomainReasoningNetwork(nn.Module):
    '''Custom architecture for cross-domain reasoning'''
    
    def __init__(self, num_domains=10, domain_dim=256, shared_dim=512):
        super().__init__()
        
        # Domain-specific encoders
        self.domain_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(domain_dim, 256),
                nn.ReLU(),
                nn.Linear(256, shared_dim)
            )
            for _ in range(num_domains)
        ])
        
        # Shared reasoning layer
        self.reasoning = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=shared_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Cross-domain attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projections
        self.domain_decoders = nn.ModuleList([
            nn.Linear(shared_dim, domain_dim)
            for _ in range(num_domains)
        ])
    
    def forward(self, domain_inputs):
        '''
        domain_inputs: list of tensors, one per domain
        '''
        # Encode each domain
        encoded = []
        for i, x in enumerate(domain_inputs):
            enc = self.domain_encoders[i](x)
            encoded.append(enc)
        
        # Stack for reasoning
        stacked = torch.stack(encoded, dim=1)  # [batch, num_domains, shared_dim]
        
        # Shared reasoning
        reasoned = self.reasoning(stacked)
        
        # Cross-domain attention
        attended, _ = self.cross_attention(reasoned, reasoned, reasoned)
        
        # Decode to each domain
        outputs = []
        for i in range(len(domain_inputs)):
            out = self.domain_decoders[i](attended[:, i, :])
            outputs.append(out)
        
        return outputs

# Test the network
model = CrossDomainReasoningNetwork(num_domains=10, domain_dim=256)
test_inputs = [torch.randn(4, 256) for _ in range(10)]
outputs = model(test_inputs)

print(f"Number of domains: 10")
print(f"Input shape per domain: {test_inputs[0].shape}")
print(f"Output shape per domain: {outputs[0].shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("✅ Cross-domain reasoning architecture ready")
