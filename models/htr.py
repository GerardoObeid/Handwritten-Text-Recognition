import torch.nn as nn
from models.attention import RowColLSTM, IterativeWeightedCollapse

class FullParagraphHTR(nn.Module):
    def __init__(self, num_clases, hidden_size=256, max_lines=8): 
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        
        self.spatial_context = RowColLSTM(in_channels=256, hidden_size=64)
        self.iterative_collapse = IterativeWeightedCollapse(in_channels=256, max_lines=max_lines)
        self.decoder = nn.LSTM(256, hidden_size, bidirectional=True, num_layers=2, dropout=0.4)
        self.fc = nn.Linear(hidden_size * 2, num_clases)

    def forward(self, x):
        features = self.cnn(x)
        spatial = self.spatial_context(features)
        collapsed = self.iterative_collapse(spatial)
        decoded, _ = self.decoder(collapsed)
        return self.fc(decoded).permute(1, 0, 2)