import torch
from .utils import chess_manager, GameContext
import chess
import chess.engine
import torch
import numpy as np
import torch.nn.functional as F
import random
import time
import pickle # <-- NEW: Import the pickle module
import os
from .model import ChessResNet # adjust path if needed

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# !!! IMPORTANT: Update these paths
MODEL_PATH = "/Users/maxmiller/Documents/GitHub/ChessHacks/chess_resnet.pth"
# CRITICAL FIX: SET THIS PATH TO YOUR PICKLE FILE (.pkl or similar)
MOVE_DICT_PATH = "/Users/maxmiller/Documents/GitHub/ChessHacks/src/move_to_int.pkl" 

NUM_MOVES = 1917 # Must match your model's output size (ChessResNet init)
# ---------------------

print("Loading state_dict model...")

# 1. MODEL LOADING
model = ChessResNet(num_res_blocks=12, num_moves=NUM_MOVES).to(DEVICE)

# Load the weights
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Check your path!")
    model = None
    pass


# ==============================================================
# 2. BOARD → MODEL INPUT (13-CHANNEL IMPLEMENTATION)
# ==============================================================

PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING
]

def board_to_tensor(board):
    """
    Convert python-chess board into the required 13x8x8 model input tensor (1, 13, 8, 8).
    """
    planes = np.zeros((13, 8, 8), dtype=np.float32)

    for i, piece_type in enumerate(PIECE_TYPES):
        # White pieces (channels 0-5)
        for square in board.pieces(piece_type, chess.WHITE):
            rank, file = chess.square_rank(square), chess.square_file(square)
            planes[i, rank, file] = 1.0

        # Black pieces (channels 6-11)
        for square in board.pieces(piece_type, chess.BLACK):
            rank, file = chess.square_rank(square), chess.square_file(square)
            planes[i + 6, rank, file] = 1.0

    # Channel 12: Player to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    x = torch.tensor(planes).unsqueeze(0)
    return x.to(DEVICE)


# ==============================================================
# 3. MOVE INDEXING (Loading the pickled dictionary)
# ==============================================================

MOVE_DICT = {}

def build_move_dict():
    """Loads the EXACT move dictionary used for training from the pickle file."""
    global MOVE_DICT
    MOVE_DICT = {}

    if os.path.exists(MOVE_DICT_PATH):
        try:
            # Open the file in 'read binary' ('rb') mode
            with open(MOVE_DICT_PATH, 'rb') as f:
                # Use pickle.load to deserialize the object
                MOVE_DICT = pickle.load(f)
            
            print(f"Loaded move dict from pickle: {len(MOVE_DICT)} entries.")
            return

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load move dict from pickle file. Error: {e}")
    else:
        print(f"CRITICAL ERROR: Move dict file not found at {MOVE_DICT_PATH}. Cannot map moves!")
    
    # Fallback/Safety Check
    if len(MOVE_DICT) != NUM_MOVES:
        print(f"CRITICAL WARNING: Dictionary has {len(MOVE_DICT)} entries, expected {NUM_MOVES}.")
        print("This will likely cause the 'No move matched' warning.")


build_move_dict()


def move_to_index(move: chess.Move):
    """Convert python-chess Move to the model's policy index."""
    uci = move.uci()
    if uci not in MOVE_DICT:
        raise ValueError(f"Move not in training move dictionary: {uci}")
    return MOVE_DICT[uci]

# ==============================================================
# 4. GET POLICY AND MOVE SELECTION
# ==============================================================

def get_policy(board):
    """Feeds board to model and gets policy probabilities."""
    if model is None:
        raise RuntimeError("Model is not loaded.")
    
    x = board_to_tensor(board)

    with torch.no_grad():
        logits = model(x)

    probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy().flatten()


def choose_move_from_model(board):
    """Selects the legal move with the highest probability from the model."""
    policy = get_policy(board)
    legal_moves = list(board.legal_moves)

    best_move = None
    best_score = -float("inf")
    legal_move_scores = {}

    for move in legal_moves:
        try:
            idx = move_to_index(move)
            score = policy[idx]
            legal_move_scores[move] = score
            
            if score > best_score:
                best_score = score
                best_move = move
                
        except ValueError:
            continue
        except IndexError:
            print(f"ERROR: Index {idx} out of bounds for policy vector.")
            continue


    if best_move is None:
        print("WARNING: No move matched model indexing — using random move")
        best_move = random.choice(legal_moves)
        
    return best_move, legal_move_scores


model = ChessResNet(num_res_blocks=16, num_moves=1917)
model.load_state_dict(torch.load("/Users/maxmiller/Documents/GitHub/ChessHacks/src/chess_resnet.pth"))
model.eval()
move_to_index = pickle.load(open("/Users/maxmiller/Documents/GitHub/ChessHacks/src/move_to_index.pkl", "rb"))

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board
    
    try:
        best_move, move_scores = choose_move_from_model(board)
        
        # Normalize the scores for GameContext logging
        total_score = sum(move_scores.values())
        if total_score > 0:
            move_probs = {move: score / total_score for move, score in move_scores.items()}
        else:
            legal_moves = list(board.legal_moves)
            move_probs = {move: (1.0 / len(legal_moves)) for move in legal_moves}


    except Exception as e:
        print("Model failed:", e)
        legal_moves = list(board.legal_moves)
        best_move = random.choice(legal_moves)
        move_probs = {move: (1.0 / len(legal_moves)) for move in legal_moves}


    ctx.logProbabilities(move_probs)
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    pass