import chess.pgn

def extract_n_games(in_path, out_path, n_games):
    """
    Extract the first n_games full games from a PGN file
    and save them into out_path.
    """

    print(f"Extracting first {n_games} games from {in_path}...")

    count = 0

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        while count < n_games:
            game = chess.pgn.read_game(f_in)
            if game is None:  # reached end of file
                break
            
            f_out.write(str(game) + "\n\n")
            count += 1

    print(f"Done! Wrote {count} games to {out_path}")
    if count < n_games:
        print(f"WARNING: Requested {n_games} games but file only contained {count}.")


if __name__ == "__main__":
    # Example:
    extract_n_games("./training/training_data/lichess_elite_2025-01.pgn", "first_100000.pgn", n_games=100000)
