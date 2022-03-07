import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder:
    def __init__(self, number_of_filter, length_of_filter):
        self.number_of_filter = number_of_filter
        self.length = length_of_filter

        self.conv1d_encoder = nn.Conv1d(1, self.number_of_filter, length_of_filter, length_of_filter // 2, bias=False)

    def forward(self, mix_data):
        # 在索引1的位置插入1维  [batch_size, samples] => [batch_size, 1, samples]
        mix_data = torch.unsqueeze(mix_data, 1)
        mix_data_encoded = self.conv1d_encoder(mix_data)
        mix_data_encoded = F.relu(mix_data_encoded)
        return mix_data_encoded  # [batch_size, number_of_filter, 2*samples/length_of_filter-1]
