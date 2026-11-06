import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import numpy as np
import time


# ---------------------------
# Hyperparameters (Only manual settings)
# ---------------------------
config = {
    "split_ratio": 0.8,  # Ratio to split support and query sets
    "batch_size": 4,  # Number of tasks per outer loop iteration
    "hidden_size": 256,  # Size of hidden layers in MLP
    "activation": "ReLU",  # Activation function type
    "num_layers": 3,  # Number of layers in the MLP
    "n_inner": 5,  # Number of inner-loop adaptation steps
    "inner_lr": 0.01,  # Learning rate for inner loop fintune时调小
    "outer_lr": 0.01,  # Learning rate for outer loop
    "result_path": "./results",
    "save_path": "reptile_model.pth",
    # Early stopping parameters for finetune
    "convergence_threshold": 1e-10,  # Loss improvement threshold for convergence
    "patience": 50,  # Number of epochs to wait before early stopping
    "max_epochs": 2000,  # Maximum number of epochs for finetune
}


# ---------------------------
# Network Definition: MLP
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3, activation="ReLU"):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_name = activation
        self.activation = getattr(nn, activation)()

        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Reptile Meta-Learner Class
# ---------------------------
class Reptile:
    def __init__(self, dataset_dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_dict = dataset_dict
        try:
            from datasets import CURRENT_SYSTEM

            self.system_name = CURRENT_SYSTEM
        except Exception:
            self.system_name = "system"

        # Extract input/output dimensions from dataset info
        self.input_dim = self.dataset_dict["info"]["input_dim"]
        self.output_dim = self.dataset_dict["info"]["output_dim"]

        # Build network using config + dataset-derived dims
        self.network = MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=config["hidden_size"],
            num_layers=config["num_layers"],
            activation=config["activation"],
        ).to(self.device)

        # Optimizers
        self.outer_opt = torch.optim.Adam(self.network.parameters(), lr=config["outer_lr"])

    def _prepare_batch(self, split="train"):
        X_list, y_list = self.dataset_dict["normalized"][split]
        spt_size = min(config["batch_size"], len(X_list))
        spt_indices = np.random.choice(len(X_list), spt_size, replace=False)

        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        for idx in spt_indices:
            X = torch.from_numpy(X_list[idx]).float().to(self.device)
            y = torch.from_numpy(y_list[idx]).float().to(self.device)

            split_idx = int(config["split_ratio"] * len(X))
            x_spt.append(X[:split_idx])
            y_spt.append(y[:split_idx])
            x_qry.append(X[split_idx:])
            y_qry.append(y[split_idx:])

        return (x_spt, y_spt), (x_qry, y_qry)

    def _safe_clone_model(self, model):
        clone = MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=config["hidden_size"],
            num_layers=config["num_layers"],
            activation=config["activation"],
        ).to(self.device)
        clone.load_state_dict(model.state_dict())
        return clone

    def run(self, split="train", is_train=True, return_fast_weights=False):
        self.network.train(is_train)
        (spt_x, spt_y), (qry_x, qry_y) = self._prepare_batch(split)
        losses = []
        fast_weights_list = []

        for inputs, targets, q_inputs, q_targets in zip(spt_x, spt_y, qry_x, qry_y):
            fmodel = self._safe_clone_model(self.network)
            params = list(fmodel.parameters())

            # Inner loop adaptation
            for _ in range(config["n_inner"]):
                logits = fmodel(inputs)
                loss = F.mse_loss(logits, targets)
                grads = torch.autograd.grad(loss, params)
                for p, g in zip(params, grads):
                    p.data -= config["inner_lr"] * g

            # Save fast weights for outer update
            if return_fast_weights:
                fast_weights_list.append([p.data.clone() for p in fmodel.parameters()])

            # Query set evaluation
            with torch.no_grad():
                logits = fmodel(q_inputs)
                loss = F.mse_loss(logits, q_targets).item()
                losses.append(loss)

        avg_loss = sum(losses) / len(losses)

        if is_train:
            # Outer loop update (not used directly in Reptile, but kept for flexibility)
            pass

        if return_fast_weights:
            return avg_loss, fast_weights_list
        else:
            return avg_loss, None

    def train(self, n_epochs=10):
        print("=== Starting Training ===")
        start_time = time.time()
        phase = "pretrain"
        csv_path = os.path.join("final_data", f"{self.system_name}_{phase}_time.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "elapsed_seconds"])
            for epoch in range(n_epochs):
                total_loss = 0
                for _ in range(10):  # 10 batches per epoch
                    loss, fast_weights_list = self.run(
                        split="train", is_train=True, return_fast_weights=True
                    )
                    total_loss += loss
                    # Reptile outer loop update
                    # 1. Get current model parameters
                    main_params = list(self.network.parameters())
                    # 2. For each task, update main_params towards fast_weights
                    for fast_weights in fast_weights_list:
                        for p, fast_p in zip(main_params, fast_weights):
                            p.data += config["outer_lr"] * (fast_p - p.data)
                avg_loss = total_loss / 10
                elapsed_seconds = time.time() - start_time
                writer.writerow([epoch + 1, avg_loss, elapsed_seconds])
                print(f"Epoch [{epoch + 1}/{n_epochs}] Loss: {avg_loss:.6f}")

    def finetune(self, query_split="query"):
        """
        Fine-tune the model with early stopping.

        Args:
            query_split: Split to use for validation (default: 'query')

        Returns:
            dict: Metrics including converged_epoch, total_training_time, final_loss
        """
        print("=== Fine-tuning with the entire support dataset ===")

        # Prepare support data for training
        X_list, y_list = self.dataset_dict["normalized"]["support"]
        all_X = torch.cat([torch.from_numpy(X).float().to(self.device) for X in X_list])
        all_y = torch.cat([torch.from_numpy(y).float().to(self.device) for y in y_list])

        # Prepare query data for validation
        query_X_list, query_y_list = self.dataset_dict["normalized"][query_split]
        query_X = torch.cat([torch.from_numpy(X).float().to(self.device) for X in query_X_list])
        query_y = torch.cat([torch.from_numpy(y).float().to(self.device) for y in query_y_list])

        # Evaluate initial RMSE before finetuning
        self.network.eval()
        with torch.no_grad():
            init_logits = self.network(query_X)
            init_rmse = torch.sqrt(F.mse_loss(init_logits, query_y)).item()
        print(f"Initial RMSE before finetune: {init_rmse:.6f}")

        # Initialize optimizer
        inner_opt = torch.optim.Adam(self.network.parameters(), lr=config["inner_lr"])

        # Early stopping variables
        best_loss = float("inf")
        patience_counter = 0
        converged_epoch = 0
        start_time = time.time()
        phase = "finetune"
        csv_path = os.path.join("final_data", f"{self.system_name}_{phase}_time.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        f = open(csv_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "elapsed_seconds"])

        # Training loop with early stopping
        for epoch in range(config["max_epochs"]):
            # Training step
            self.network.train()
            inner_opt.zero_grad()
            logits = self.network(all_X)
            loss = F.mse_loss(logits, all_y)
            loss.backward()
            inner_opt.step()

            # Validation step
            self.network.eval()
            with torch.no_grad():
                val_logits = self.network(query_X)
                val_loss = F.mse_loss(val_logits, query_y).item()

            elapsed_seconds = time.time() - start_time
            log_loss = val_loss
            writer.writerow([epoch + 1, log_loss, elapsed_seconds])

            # Check for improvement
            improvement = best_loss - val_loss
            if improvement > config["convergence_threshold"]:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= config["patience"]:
                converged_epoch = epoch + 1
                print(f"Early stopping at epoch {converged_epoch}, validation loss: {val_loss:.6f}")
                break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{config['max_epochs']}] Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}"
                )

        # If no early stopping occurred, use max_epochs
        if converged_epoch == 0:
            converged_epoch = config["max_epochs"]
            print(f"Training completed after {converged_epoch} epochs")

        total_training_time = time.time() - start_time
        f.close()

        # Get final loss on query set
        self.network.eval()
        with torch.no_grad():
            final_logits = self.network(query_X)
            final_loss = torch.sqrt(F.mse_loss(final_logits, query_y)).item()

        metrics = {
            "converged_epoch": converged_epoch,
            "total_training_time": total_training_time,
            "final_loss": final_loss,
        }

        return metrics

    def save(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
