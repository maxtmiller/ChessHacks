import torch
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import random
import time
from .model import ChessResNet, board_to_matrix
import pickle

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessResNet(num_res_blocks=40, num_moves=1917)

state_dict = torch.load(
    "./models/chess_resnet_40.pth",
    map_location=torch.device('cpu')
)

model.load_state_dict(state_dict)
model.eval()

move_to_index = pickle.load(open("./move_to_int.pkl", "rb"))


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board

    print("Cooking move...main2")
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
        logits = model(input_tensor)
        all_move_probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    # 4. Map legal moves to probabilities
    move_weights = []
    legal_moves_filtered = []
    for move in legal_moves:
        idx = move_to_index.get(move.uci(), None)
        if idx is not None:
            prob = all_move_probs[idx]
            move_weights.append(prob)
            legal_moves_filtered.append(move)

    if not legal_moves_filtered:
        print("No legal moves found in move dictionary. Picking randomly.")
        best_move = random.choice(legal_moves)
        ctx.logProbabilities({m.uci(): 1.0 if m == best_move else 1e-6 for m in legal_moves}) # <-- FIX
        return best_move

    # 5. Normalize probabilities
    total_weight = sum(move_weights)
    if total_weight == 0:
        print("Model gave zero probability to all legal moves. Picking randomly.")
        best_move = random.choice(legal_moves_filtered)
        ctx.logProbabilities({m.uci(): 1.0 if m == best_move else 1e-6 for m in legal_moves_filtered}) # <-- FIX
        return best_move

    normalized_probs = {
        move.uci(): weight / total_weight  # <-- FIX
        for move, weight in zip(legal_moves_filtered, move_weights)
    }

    normalized_probs = {chess.Move.from_uci(k): float(v) for k, v in normalized_probs.items()}

    # 6. Log probabilities and choose move
    ctx.logProbabilities(normalized_probs)
    # best_move = random.choices(
    #     legal_moves_filtered, weights=[p for p in normalized_probs.values()], k=1
    # )[0]

    print("top k")

    # Top K
    K = 3
    top_moves, top_probs = top_k_moves(normalized_probs, K)

    # Convert to dict for logging
    topk_dict = {m: p for m, p in zip(top_moves, top_probs)}

    ctx.logProbabilities(topk_dict)  # NOW logging matches the sampled distribution

    best_move = random.choices(top_moves, weights=top_probs, k=1)[0]

    return best_move


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
