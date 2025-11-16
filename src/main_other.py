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

# engine = chess.engine.SimpleEngine.popen_uci("/Users/maxmiller/Documents/GitHub/ChessHacks/model1")  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessResNet(num_res_blocks=40, num_moves=1917)

state_dict = torch.load(
    "/Users/maxmiller/Documents/GitHub/ChessHacks/chess_resnet.pth",
    map_location=torch.device('cpu')
)

model.load_state_dict(state_dict) 
model.eval()

move_to_index = pickle.load(open("/Users/maxmiller/Documents/GitHub/ChessHacks/chess_resnet1.pth", "rb"))


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board

    print("Cooking move...main")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)
 
    print(f"Chosen move: {random.choices(legal_moves, weights=move_weights, k=1)[0]}")

    return random.choices(legal_moves, weights=move_weights, k=1)[0]

    print("Cooking move...MODEL")
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

    # 6. Log probabilities and choose move
    ctx.logProbabilities(normalized_probs)
    print(normalized_probs)
    best_move = random.choices(
        legal_moves_filtered, weights=[p for p in normalized_probs.values()], k=1
    )[0]

    print(f"Chosen move: {type(best_move)}")

    return best_move

"""
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # # This gets called every time the model needs to make a move
    # # Return a python-chess Move object that is a legal move for the current position


    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)
    # Use the
    input_matrix = board_to_matrix(ctx.board)
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    output = model(input_tensor)
    move_probs = torch.softmax(output, dim=1).detach().numpy().flatten()

    

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")
    
    

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    #Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]
"""

@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
