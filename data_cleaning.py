import chess
import chess.pgn
import csv
import os
from tqdm import tqdm

INPUT_FOLDER = "/Users/maxmiller/Documents/GitHub/ChessHacks/training_data" 
OUTPUT_CSV = "training_data.csv"

def extract_from_all_pgns(folder, out_csv):
    # Collect all .pgn files in the folder
    pgn_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pgn")
    ]

    print(f"Found {len(pgn_files)} PGN files.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["fen", "move"])  # header row

        game_total = 0

        # Process each PGN file
        for pgn_path in pgn_files:
            print("\nProcessing:", pgn_path)
            pgn = open(pgn_path, encoding="utf-8")

            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break  # file finished

                game_total += 1
                board = game.board()
                legal_game = True

                # -------------------------------
                # STRONG VALIDATION
                # -------------------------------
                for move in game.mainline_moves():
                    # Check legality
                    if move not in board.legal_moves:
                        legal_game = False
                        break
                    # Try pushing (rare PGN corruption)
                    try:
                        board.push(move)
                    except:
                        legal_game = False
                        break

                # Skip broken game
                if not legal_game:
                    continue

                # -------------------------------
                # EXTRACT TRAINING PAIRS
                # -------------------------------
                board = game.board()
                for move in game.mainline_moves():
                    writer.writerow([board.fen(), move.uci()])
                    try:
                        board.push(move)
                    except:
                        break  # shouldn't ever happen now

                # Progress log
                if game_total % 100 == 0:
                    print(f"{game_total} games processed...")

        print("\nFinished!")
        print(f"Total games processed: {game_total}")
        print(f"Training data saved to {out_csv}")


extract_from_all_pgns(INPUT_FOLDER, OUTPUT_CSV)