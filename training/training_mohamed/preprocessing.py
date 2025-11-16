import chess
import torch
import numpy as np
from chess import Board
import tqdm.auto as tqdm
import chess.pgn
import pickle

def is_valid_game(game):
    board = game.board()
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            return False  # invalid game
        board.push(move)
    return True

def load_pgn(file_path):
    games = []
    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            if not is_valid_game(game):
                continue
            games.append(game)
    return games

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

def create_input_for_nn(games):
    X = []
    y_policy = []
    y_value = []
    for game in tqdm.tqdm(games):
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y_policy.append(move.uci())
            y_value.append(result_to_value(game, board))
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y_policy), np.array(y_value, dtype=np.float32)

def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int

def result_to_value(game, board):
    result = game.headers["Result"]
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        return 0.0  # draw

    # value should be from POV of current player
    return 1.0 if board.turn == winner else -1.0

def preprocess_data(dataset):
    games = []
    for pgn_file in tqdm.tqdm(dataset):
        games.extend(load_pgn(pgn_file))
    X, y_moves_policy, y_value = create_input_for_nn(games)
    y, move_to_int = encode_moves(y_moves_policy)
    num_classes = len(move_to_int)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y_value = torch.tensor(y_value, dtype=torch.float32)
    return X, y, y_value, num_classes, move_to_int

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PreProcess:
    def __init__(self, x, y, num_classes, move_to_int):
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.move_to_int = move_to_int

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

