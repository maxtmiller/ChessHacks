import torch
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import random
import time
from model import ChessResNet, board_to_matrix
import pickle
# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

engine = chess.engine.SimpleEngine.popen_uci("/Users/maxmiller/Documents/GitHub/ChessHacks/model1")  

model = ChessResNet(num_res_blocks=16, num_moves=1917)
model.load_state_dict(torch.load("/Users/maxmiller/Documents/GitHub/ChessHacks/src/chess_resnet.pth"))
model.eval()
move_to_index = pickle.load(open("/Users/maxmiller/Documents/GitHub/ChessHacks/src/move_to_index.pkl", "rb"))

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    print("Cooking move...")
    print(ctx.board.move_stack)
    
    # 1. Get legal moves *first*
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # 2. Convert board and get model output (Logits)
    # (Assuming you fixed the softmax bug in your model from our last chat)
    input_matrix = board_to_matrix(ctx.board)
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad(): # Disable gradient calculation for inference
        output_logits = model(input_tensor)

    # 3. Get probabilities for ALL 1917 moves
    # We apply softmax here, on the raw logits, which is correct!
    all_move_probs = torch.softmax(output_logits, dim=1).detach().numpy().flatten()

    # 4. Map legal moves to their probabilities
    move_weights = []
    for move in legal_moves:
        # Use your mapping function to find the index for this move
        index = move_to_index(move) # Or whatever your function is called
        
        # Get the model's probability for THIS specific legal move
        prob = all_move_probs[index]
        move_weights.append(prob)
        
    # 5. Re-normalize the probabilities
    # This ensures the probabilities of *only the legal moves* sum to 1.
    total_weight = sum(move_weights)
    
    if total_weight == 0:
        # Failsafe: If model gave 0 prob to all legal moves, pick randomly
        # This can happen if the model is poorly trained
        print("Model has no preference, picking randomly.")
        return random.choice(legal_moves)

    # Normalize the weights
    normalized_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }

    # 6. Log and return the move
    ctx.logProbabilities(normalized_probs)

    # We use random.choices to *sample* from the distribution.
    # This makes your bot less predictable and often stronger.
    return random.choices(legal_moves, weights=[p for p in normalized_probs.values()], k=1)[0]

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
