import torch
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import random
import time
from model import ChessResNet, board_to_matrix

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

engine = chess.engine.SimpleEngine.popen_uci("/Users/maxmiller/Documents/GitHub/ChessHacks/model1")  

model = ChessResNet(num_res_blocks=12, num_moves=1917)
model.load_state_dict(torch.load("/Users/maxmiller/Documents/GitHub/ChessHacks/chess_resnet.pth"))
model.eval()

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


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
