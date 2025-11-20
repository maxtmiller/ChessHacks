import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from chess import Board


def board_to_matrix(board: Board):
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1
    return matrix


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=4, num_moves=4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32*8*8, num_moves)
        
    def forward(self, x):
        """
        x: [batch, 18, 8, 8] board input
        legal_moves_mask: optional [batch, num_moves] mask to zero illegal moves
        """
        out = self.stem(x)
        out = self.res_blocks(out)
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        return policy
    

class ChessResNetPa(nn.Module):
    def __init__(self, num_res_blocks=4, num_moves=4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32*8*8, num_moves)

        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32*8*8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        """
        x: [batch, 18, 8, 8] board input
        legal_moves_mask: optional [batch, num_moves] mask to zero illegal moves
        """
        out = self.stem(x)
        out = self.res_blocks(out)
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # output in [-1, 1]
        
        return policy, value
