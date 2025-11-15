import torch 
import mlflow
import torch.nn.functional as F
import torch.optim as optim
import time
import tqdm
from model import ChessResNet
import torch.nn as nn
from preprocessing import preprocess_data, ChessDataset
from torch.utils.data import DataLoader
import argparse
from huggingface_hub import hf_hub_download


def topk_accuracy(output, target, k=3):
    """Compute top-k accuracy"""
    with torch.no_grad():
        topk = output.topk(k, dim=1).indices
        target_expanded = target.view(-1, 1).expand_as(topk)
        correct = (topk == target_expanded).any(dim=1).float().sum()
        return 100 * correct / target.size(0)

def train(epochs, model, train_loader, val_loader, device, eval_interval=1, lr=1e-4, grad_clip=1.0):
    """
    MLflow-ready training loop for supervised chess policy network
    """
    model.train()
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True, backend="inductor")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # raw logits
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch+1)
        mlflow.log_metric("grad_norm", total_norm, step=epoch+1)

        # Validation
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss, correct, total, top3_correct = 0.0, 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()

                    # Top-1 accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

                    # Top-3 accuracy
                    top3 = outputs.topk(3, dim=1).indices
                    top3_correct += (top3 == labels.view(-1, 1)).any(dim=1).sum().item()

                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            top3_val_acc = 100 * top3_correct / total

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch+1)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch+1)
            mlflow.log_metric("top3_val_accuracy", top3_val_acc, step=epoch+1)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.2f}% | "
                  f"Top-3 Val Acc: {top3_val_acc:.2f}% | "
                  f"Grad Norm: {total_norm:.4f}")

        # Epoch time
        epoch_time = time.time() - start_time
        minutes, seconds = divmod(int(epoch_time), 60)
        print(f"Epoch {epoch+1} time: {minutes}m {seconds}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chess policy network")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_size", type=int, default=4, help="Number of residual blocks in the model")
    args = parser.parse_args()

    X_path = hf_hub_download(
        repo_id="Maynx/Xy",
        filename="X.pt",
        repo_type="dataset"
    )

    y_path = hf_hub_download(
        repo_id="Maynx/Xy",
        filename="y.pt",
        repo_type="dataset"
    )

    X = torch.load(X_path)
    y = torch.load(y_path)

    train_size = int(0.8 * len(X))
    val_size = len(X) - train_size
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:]
    y_val = y[train_size:]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    train_dataset = ChessDataset(X_train, y_train)
    val_dataset = ChessDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = ChessResNet(num_res_blocks=args.model_size, num_moves=1917)
    print("Model initialized.")

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(epochs=args.epochs, model=model, train_loader=train_loader, val_loader=val_loader, device=device, lr=args.lr)

    torch.save(model.state_dict(), "model/chess_resnet.pth")