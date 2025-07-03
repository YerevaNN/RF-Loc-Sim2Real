import torch
import torch.nn as nn
import math


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding Module
    
    Converts (x, y) coordinates to 256-dimensional positional embeddings
    using sinusoidal encoding similar to transformers.
    
    Args:
        d_model (int): Dimension of the output embeddings (default: 256)
        max_len (int): Maximum length for positional encoding (default: 10000)
    """
    
    def __init__(self, d_model: int = 256, max_len: int = 10000):
        super(PositionalEncoding2D, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Ensure d_model is even so we can split it equally between x and y
        assert d_model % 2 == 0, "d_model must be even to split between x and y coordinates"
        
        # Half dimensions for x, half for y
        self.d_x = d_model // 2
        self.d_y = d_model // 2
        
        # Create frequency dividers for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, self.d_x, 2).float() * 
                           -(math.log(max_len) / self.d_x))
        self.register_buffer('div_term', div_term)
    
    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            coords (torch.Tensor): Input coordinates of shape [batch_size, num_points, 2]
                                 where coords[:, :, 0] are x coordinates and coords[:, :, 1] are y coordinates
                                 Coordinates should be in range (0, 1)
        
        Returns:
            torch.Tensor: Positional embeddings of shape [batch_size, num_points, d_model]
        """
        batch_size, num_points, _ = coords.shape
        
        # Extract x and y coordinates
        x_coords = coords[:, :, 0]  # Shape: [batch_size, num_points]
        y_coords = coords[:, :, 1]  # Shape: [batch_size, num_points]
        
        # Scale coordinates to a reasonable range for sinusoidal encoding
        # Since coordinates are in (0, 1), we scale them to (0, max_len)
        x_scaled = x_coords * self.max_len
        y_scaled = y_coords * self.max_len
        
        # Create positional encodings for x coordinates
        pe_x = torch.zeros(batch_size, num_points, self.d_x, device=coords.device)
        pe_x[:, :, 0::2] = torch.sin(x_scaled.unsqueeze(-1) * self.div_term)
        pe_x[:, :, 1::2] = torch.cos(x_scaled.unsqueeze(-1) * self.div_term)
        
        # Create positional encodings for y coordinates
        pe_y = torch.zeros(batch_size, num_points, self.d_y, device=coords.device)
        pe_y[:, :, 0::2] = torch.sin(y_scaled.unsqueeze(-1) * self.div_term)
        pe_y[:, :, 1::2] = torch.cos(y_scaled.unsqueeze(-1) * self.div_term)
        
        # Concatenate x and y positional encodings
        pe = torch.cat([pe_x, pe_y], dim=-1)  # Shape: [batch_size, num_points, d_model]
        
        return pe