import torch
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import random
import time
from .model_pa import ChessResNet, board_to_matrix
import pickle
import math
# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessResNet(num_res_blocks=1, num_moves=1917)

state_dict = torch.load(
    "./models/chess_resnet_small.pth",
    map_location=torch.device('cpu')
)
model.load_state_dict(state_dict)



model.to(DEVICE)
# model.half()
model.eval()

move_to_index = pickle.load(open("./move_to_int.pkl", "rb"))

TEMPERATURE = 1

def evaluate_board(board, model):
    """
    Returns the model's value prediction for WHITE in [-1, 1].
    (Assuming your model is trained to always predict White's score)
    """
    mat = board_to_matrix(board)
    inp = torch.tensor(mat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Your model returns TWO things. We only need the second one.
        policy_logits, value = model(inp) 
    
    return float(value.item())


def shallow_search(board, model):
    """
    1-ply search using the value head.
    This is a "minimax" search at depth 1.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    best_move = None
    best_score = -9999  # We're always trying to maximize this

    # Store whose turn it is *before* we make a move
    is_white_to_move = (board.turn == chess.WHITE)

    for move in legal:
        board.push(move)
        
        # Get the score of the *resulting* board.
        # This score is always from White's perspective.
        white_score = evaluate_board(board, model)
        
        board.pop()

        # This is the "negamax" logic.
        # If we are White, we want to maximize white_score.
        # If we are Black, we want to minimize white_score (i.e., maximize -white_score).
        my_score = white_score if is_white_to_move else -white_score

        if my_score > best_score:
            best_score = my_score
            best_move = move

    return best_move

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    try:
        # --- YOUR ORIGINAL CODE STARTS HERE ---
        board = ctx.board
        print("Cooking move... 1-ply value search!")
        print(board.move_stack)

        # 1. Get all legal moves (for logging)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (probably lost).")
        
        # 2. Run your new search function to get the *single* best move
        print("--- DEBUG: Calling shallow_search ---")
        best_move = shallow_search(board, model)
        print(f"--- DEBUG: shallow_search returned {best_move} ---")

        if best_move is None:
            # Failsafe
            print("Search returned no move. Picking randomly.")
            best_move = random.choice(legal_moves)

        # 3. Create a "fake" probability map for logging
        ctx.logProbabilities({})

        print(f"Chosen move: {best_move.uci()}")
        print(best_move, type(best_move))
        return best_move
    
    except Exception as e:
        # --- THIS IS THE DEBUGGING PART ---
        print("!!!!!!!!!!!!!! ERROR IN TEST_FUNC !!!!!!!!!!!!!!")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Args: {e.args}")
        
        # This prints the full traceback to your console
        import traceback
        traceback.print_exc()
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Re-raise the exception so the server still fails
        raise e

"""
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board

    print("Cooking move...main_temp_50")
    print(board.move_stack)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (probably lost).")
  
    input_matrix = board_to_matrix(board)
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        logits = logits.cpu().numpy().flatten()
    
    legal_move_logits = {}
    legal_moves_filtered = []
    
    for move in legal_moves:
        idx = move_to_index.get(move.uci(), None)
        if idx is not None and 0 <= idx < len(logits): 
            legal_move_logits[move] = logits[idx]
            legal_moves_filtered.append(move)

    if not legal_moves_filtered:
        print("No legal moves found in move dictionary. Picking randomly.")
        best_move = random.choice(legal_moves)
        ctx.logProbabilities({m.uci(): 1.0 if m == best_move else 1e-6 for m in legal_moves})
        return best_move
    
    logits_list = list(legal_move_logits.values())
    
    if TEMPERATURE <= 0:
        print("Temperature is <= 0. Performing greedy selection (argmax).")
        
        max_logit = max(logits_list)
        best_move_index = logits_list.index(max_logit)
        best_move = legal_moves_filtered[best_move_index]

        normalized_probs = {
            m.uci(): 1.0 if m == best_move else 1e-6 
            for m in legal_moves_filtered
        }
        ctx.logProbabilities(normalized_probs)
        return best_move

    scaled_logits = torch.tensor(logits_list, dtype=torch.float32) / TEMPERATURE
    temperature_probs_tensor = torch.softmax(scaled_logits, dim=0)
    temperature_probs = temperature_probs_tensor.cpu().numpy().tolist()
    
    normalized_probs = {
        move.uci(): float(prob)
        for move, prob in zip(legal_moves_filtered, temperature_probs)
    }

    normalized_probs = {chess.Move.from_uci(k): float(v) for k, v in normalized_probs.items()}
    ctx.logProbabilities(normalized_probs)

    best_move = random.choices(
        legal_moves_filtered, 
        weights=temperature_probs, 
        k=1
    )[0]

    return best_move
"""

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


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass