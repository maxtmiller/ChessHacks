from preprocessing import preprocess_data, PreProcess
import os
import torch
import argparse
import pickle

if __name__ == "__main__":
    # Preprocess data
    parser = argparse.ArgumentParser(description="Preprocess chess data")
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing data files")
    args = parser.parse_args()
    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    x, y, num_classes, move_to_int = preprocess_data(files)

    pickle.dump(move_to_int, open("move_to_int.pkl", "wb"))
    print(f"Preprocessed {len(files)} files. Number of classes: {num_classes}")
    torch.save(x, "X.pt")
    torch.save(y, "y.pt")