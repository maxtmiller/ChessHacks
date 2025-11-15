from preprocessing import preprocess_data, PreProcess
import os
import torch
import argparse

if __name__ == "__main__":
    # Preprocess data
    parser = argparse.ArgumentParser(description="Preprocess chess data")
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing data files")
    args = parser.parse_args()
    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    x, y, num_classes, move_to_int = preprocess_data(files)
    torch.save(x, "data/X.pt")
    torch.save(y, "data/y.pt")