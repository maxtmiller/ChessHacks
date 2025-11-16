import torch
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import random
import time
from .model import ChessResNet, board_to_matrix
import pickle
import math # Needed for math.exp or using torch.exp

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessResNet(num_res_blocks=40, num_moves=1917)

state_dict = torch.load(
    "./models/chess_resnet_40.pth",
    map_location=torch.device('cpu')
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

move_to_index = pickle.load(open("./move_to_int.pkl", "rb"))

# --- New Constant for Temperature Sampling ---
TEMPERATURE = 1.75
# ---

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board

    print("Cooking move...main_temp")
    print(board.move_stack)

    # 1. Get legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (probably lost).")
  
    # 2. Convert board to tensor
    input_matrix = board_to_matrix(board)
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # add batch dim

    # 3. Run model
    with torch.no_grad():
        # Get raw logits for temperature application
        logits = model(input_tensor).cpu().numpy().flatten()
    
    # --- Temperature Sampling Logic (Replaces steps 4, 5, 6 with modification) ---

    # 4. Map legal moves to **logits**
    legal_move_logits = {}
    legal_moves_filtered = []
    
    for move in legal_moves:
        idx = move_to_index.get(move.uci(), None)
        # Check if move is in the dictionary AND is a valid index for the logits array
        if idx is not None and 0 <= idx < len(logits): 
            legal_move_logits[move] = logits[idx]
            legal_moves_filtered.append(move)

    if not legal_moves_filtered:
        print("No legal moves found in move dictionary. Picking randomly.")
        best_move = random.choice(legal_moves)
        ctx.logProbabilities({m.uci(): 1.0 if m == best_move else 1e-6 for m in legal_moves})
        return best_move

    # 5. Apply Temperature and Softmax to legal move logits
    
    # Get a list of the logits
    logits_list = list(legal_move_logits.values())
    
    if TEMPERATURE <= 0:
        # Handle zero or negative temperature (equivalent to argmax/greedy selection)
        # Set a very small positive temperature to avoid division by zero if necessary, 
        # or implement greedy selection explicitly.
        print("Temperature is <= 0. Performing greedy selection (argmax).")
        
        # Find the index of the max logit
        max_logit = max(logits_list)
        best_move_index = logits_list.index(max_logit)
        best_move = legal_moves_filtered[best_move_index]

        # Log probabilities for greedy choice
        normalized_probs = {
            m.uci(): 1.0 if m == best_move else 1e-6 
            for m in legal_moves_filtered
        }
        ctx.logProbabilities(normalized_probs)
        return best_move

    # Apply temperature: z_i' = z_i / T
    scaled_logits = torch.tensor(logits_list, dtype=torch.float32) / TEMPERATURE
    
    # Apply softmax: P_i = exp(z_i / T) / sum_j exp(z_j / T)
    # Using torch.softmax is numerically stable.
    temperature_probs_tensor = torch.softmax(scaled_logits, dim=0)
    temperature_probs = temperature_probs_tensor.cpu().numpy().tolist()

    # 6. Log probabilities and choose move
    
    normalized_probs = {
        move.uci(): float(prob)
        for move, prob in zip(legal_moves_filtered, temperature_probs)
    }

    normalized_probs = {chess.Move.from_uci(k): float(v) for k, v in normalized_probs.items()}

    ctx.logProbabilities(normalized_probs)

    # Sample the move based on the calculated probabilities
    best_move = random.choices(
        legal_moves_filtered, 
        weights=temperature_probs, 
        k=1
    )[0]

    return best_move


# --- Top-K function is now redundant but kept for completeness, 
# --- though it is not used in the final version of test_func
def top_k_moves(normalized_probs: dict, k: int):
    """
    Return the top-K moves and renormalized probabilities.
    normalized_probs: dict {Move: probability}
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    # Sort moves by probability (descending)
    sorted_moves = sorted(
        normalized_probs.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Take first k moves
    top_k = sorted_moves[:k]

    # Unpack moves + probabilities
    moves, probs = zip(*top_k)

    # Re-normalize probabilities to sum to 1
    total_prob = sum(probs)
    renormalized = [p / total_prob for p in probs]

    return list(moves), renormalized
# --- End redundant Top-K function ---


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
