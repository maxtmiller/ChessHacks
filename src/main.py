from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import random
import time

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

engine = chess.engine.SimpleEngine.popen_uci("/Users/maxmiller/Documents/GitHub/ChessHacks/model1")  

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # Called every time a move is needed
    board = ctx.board

    print("Stockfish is cooking...")
    print(board.move_stack)
    time.sleep(0.1)

    # ---- Get best move from Stockfish ----
    # Limit thinking time or depth
    result = engine.play(board, chess.engine.Limit(depth=15))
    best_move = result.move

    # ---- Log probability distribution ----
    # Create a probability distribution where:
    #   - best move has very high probability
    #   - other legal moves have tiny probabilities

    legal_moves = list(board.generate_legal_moves())
    move_probs = {}

    big = 0.95  # best move confidence
    small = (1.0 - big) / (len(legal_moves) - 1)

    for move in legal_moves:
        if move == best_move:
            move_probs[move] = big
        else:
            move_probs[move] = small

    ctx.logProbabilities(move_probs)

    return best_move

    # # This gets called every time the model needs to make a move
    # # Return a python-chess Move object that is a legal move for the current position


    # print("Cooking move...")
    # print(ctx.board.move_stack)
    # time.sleep(0.1)

    # legal_moves = list(ctx.board.generate_legal_moves())
    # if not legal_moves:
    #     ctx.logProbabilities({})
    #     raise ValueError("No legal moves available (i probably lost didn't i)")

    # move_weights = [random.random() for _ in legal_moves]
    # total_weight = sum(move_weights)
    # # Normalize so probabilities sum to 1
    # move_probs = {
    #     move: weight / total_weight
    #     for move, weight in zip(legal_moves, move_weights)
    # }
    # ctx.logProbabilities(move_probs)

    # return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
