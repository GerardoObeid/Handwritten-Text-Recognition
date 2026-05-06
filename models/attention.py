import torch
import torch.nn as nn
import torch.nn.functional as F

class RowColLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, p_dropout=0.2):
        super().__init__()
        self.spatial_dropout = nn.Dropout2d(p=p_dropout)
        self.row_lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True)
        self.col_lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True)
        self.dropout_fusion = nn.Dropout(p=p_dropout)
        self.fusion = nn.Conv2d(hidden_size * 4, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.spatial_dropout(x)
        b, c, h, w = x.size()
        out_rows, _ = self.row_lstm(x.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        out_cols, _ = self.col_lstm(x.permute(0, 3, 2, 1).contiguous().view(b * w, h, c))
        
        tensor_fusion = torch.cat([
            out_rows.view(b, h, w, -1).permute(0, 3, 1, 2), 
            out_cols.view(b, w, h, -1).permute(0, 3, 2, 1)
        ], dim=1)
        
        tensor_fusion = self.dropout_fusion(tensor_fusion)
        return F.relu(self.fusion(tensor_fusion))

class IterativeWeightedCollapse(nn.Module):
    def __init__(self, in_channels, max_lines=3): 
        super().__init__()
        self.max_lines = max_lines
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.newline_marker = nn.Parameter(torch.randn(1, in_channels, 8))

    def forward(self, x):
        b, c, h, w = x.size()
        prev_attention = torch.zeros(b, 1, h, w, device=x.device)
        line_features = []
        
        for _ in range(self.max_lines):
            raw_scores = self.attention_conv(torch.cat([x, prev_attention], dim=1))
            sink_row = torch.full((b, 1, 1, w), -5.0, device=x.device)
            scores_with_sink = torch.cat([raw_scores, sink_row], dim=2)
            
            attention_weights = F.softmax(scores_with_sink, dim=2)[:, :, :-1, :] 
            prev_attention = prev_attention + attention_weights
            linea_colapsada = torch.sum(x * attention_weights, dim=2)
            espaciador = self.newline_marker.expand(b, -1, -1) 
            
            line_features.append(linea_colapsada)
            line_features.append(espaciador)

        return torch.cat(line_features, dim=2).permute(2, 0, 1)