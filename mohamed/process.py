from preprocessing import preprocess_data, PreProcess
import os
import torch

if __name__ == "__main__":
    # Preprocess data
    files = [os.path.join("data", f) for f in os.listdir("data/")]
    x, y, num_classes, move_to_int = preprocess_data(files)
    torch.save(x, "data/X.pt")
    torch.save(y, "data/y.pt")