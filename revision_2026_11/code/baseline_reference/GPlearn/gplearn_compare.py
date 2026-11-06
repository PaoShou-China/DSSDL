import numpy as np
import csv
import os
import time
from gplearn.genetic import SymbolicRegressor
import pickle
from sklearn.metrics import mean_squared_error

# ---------------------------
# Configuration
# ---------------------------

CONFIG = {
    "population_size": 500,
    "generations": 200,
    "stopping_criteria": 0.01,
    "p_crossover": 0.7,
    "p_subtree_mutation": 0.1,
    "p_hoist_mutation": 0.05,
    "p_point_mutation": 0.1,
    "max_samples": 0.9,
    "verbose": 1,
    "parsimony_coefficient": 0.005,
    "random_state": 0,
    "feature_names": ["x0", "x1", "x2"],  # Will be updated based on data
    "save_path": "gplearn_model.pkl",
}


# ---------------------------
# GPlearn Regressor Class
# ---------------------------


class GPlearnRegressor:
    """
    A wrapper for gplearn.SymbolicRegressor to perform symbolic regression on dynamical systems.

    Features:
        - Trains one regressor per output dimension.
        - Fine-tunes on the support set and evaluates on the query set.
        - Saves/loads a list of trained models.
    """

    def __init__(self, dataset_dict):
        """
        Initialize the GPlearn regressor with configuration and dataset.

        Args:
            dataset_dict (dict): Dataset dictionary containing data arrays.
        """
        self.config = CONFIG.copy()
        self.dataset_dict = dataset_dict

        # Update dimensions and feature names from dataset info
        self.config["input_dim"] = self.dataset_dict["info"]["input_dim"]
        self.config["output_dim"] = self.dataset_dict["info"]["output_dim"]
        self.config["feature_names"] = [f"x{i}" for i in range(self.config["input_dim"])]

        self.models = []

    def prepare_data(self, split="train"):
        """
        Load and concatenate state variables and their derivatives.

        Args:
            split (str): Dataset split ('train', 'support', or 'query')

        Returns:
            X (np.ndarray): State variables (n_samples, n_features)
            y (np.ndarray): Derivatives (n_samples, n_targets)
        """
        data_key = "raw"  # Using raw data as gplearn doesn't require pre-normalization like SINDy
        X_list, y_list = self.dataset_dict[data_key][split]

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        return X, y

    def finetune(self):
        """
        Fine-tune the model using the support dataset and evaluate on the query dataset.
        """

        def _is_valid_number(value):
            try:
                return value is not None and not np.isnan(value)
            except Exception:
                return value is not None

        def _extract_loss(run_detail):
            for key in ("best_oob_fitness", "oob_fitness", "test_fitness", "test_loss"):
                if key in run_detail and _is_valid_number(run_detail[key]):
                    return run_detail[key]
            for key in ("loss", "train_loss", "best_fitness", "fitness"):
                if key in run_detail and _is_valid_number(run_detail[key]):
                    return run_detail[key]
            return None

        def _get_system_name():
            try:
                from datasets import CURRENT_SYSTEM

                return str(CURRENT_SYSTEM).lower()
            except Exception:
                return "system"

        phase = "finetune"
        epoch_rows = []
        epoch_counter = 0
        start_time = time.time()

        # 1. Train on the support set
        X_support, y_support = self.prepare_data(split="support")
        print(f"=== Training GPlearn on the support data ({X_support.shape[0]} samples) ===")

        if X_support.shape[0] == 0:
            raise ValueError("Support set is empty. Check dataset generation and support t_span.")

        max_samples = self.config["max_samples"]
        if int(max_samples * X_support.shape[0]) < 1:
            # Ensure at least one sample is used to avoid zero-weight issues.
            max_samples = 1.0

        self.models = []
        for i in range(self.config["output_dim"]):
            print(f"\n--- Training model for dimension {i + 1}/{self.config['output_dim']} ---")
            regressor = SymbolicRegressor(
                population_size=self.config["population_size"],
                generations=self.config["generations"],
                stopping_criteria=self.config["stopping_criteria"],
                p_crossover=self.config["p_crossover"],
                p_subtree_mutation=self.config["p_subtree_mutation"],
                p_hoist_mutation=self.config["p_hoist_mutation"],
                p_point_mutation=self.config["p_point_mutation"],
                max_samples=max_samples,
                verbose=self.config["verbose"],
                parsimony_coefficient=self.config["parsimony_coefficient"],
                random_state=self.config["random_state"],
                feature_names=self.config["feature_names"],
            )
            reg_start_time = time.time()
            regressor.fit(X_support, y_support[:, i])
            reg_total_time = time.time() - reg_start_time
            elapsed_offset = reg_start_time - start_time
            run_details = getattr(regressor, "run_details_", None)
            if run_details:
                if isinstance(run_details, dict):
                    keys = list(run_details.keys())
                    total_gens = max(
                        (len(run_details[k]) for k in keys if hasattr(run_details[k], "__len__")),
                        default=0,
                    )
                    normalized = []
                    for idx in range(total_gens):
                        detail = {}
                        for k in keys:
                            values = run_details.get(k)
                            if hasattr(values, "__len__") and idx < len(values):
                                detail[k] = values[idx]
                        normalized.append(detail)
                    run_details = normalized

                gen_times = []
                for detail in run_details:
                    gen_time = detail.get("generation_time", None)
                    gen_times.append(gen_time if _is_valid_number(gen_time) else None)
                total_gens = len(run_details)
                cumulative = 0.0
                for gen_index, detail in enumerate(run_details):
                    epoch_counter += 1
                    loss_value = _extract_loss(detail)
                    gen_time = gen_times[gen_index]
                    if gen_time is not None:
                        cumulative += float(gen_time)
                        elapsed_seconds = elapsed_offset + cumulative
                    else:
                        if total_gens > 0:
                            elapsed_seconds = elapsed_offset + reg_total_time * (
                                (gen_index + 1) / total_gens
                            )
                        else:
                            elapsed_seconds = time.time() - start_time
                    epoch_rows.append((epoch_counter, loss_value, elapsed_seconds))
            else:
                epoch_counter += 1
                epoch_rows.append((epoch_counter, None, time.time() - start_time))
            self.models.append(regressor)
            print(f"Discovered equation for dim {i + 1}: {regressor._program}")

        if epoch_rows:
            os.makedirs("final_data", exist_ok=True)
            system_name = _get_system_name()
            csv_path = os.path.join("final_data", f"{system_name}_{phase}_time.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "elapsed_seconds"])
                for epoch, loss_value, elapsed_seconds in epoch_rows:
                    loss_out = "" if loss_value is None else float(loss_value)
                    writer.writerow([epoch, loss_out, float(elapsed_seconds)])

        # 2. Evaluate on the query (test) set
        print("\n=== Evaluating on the query data ===")
        X_query, y_query = self.prepare_data(split="query")

        y_pred = self.predict(X_query)

        # 3. Calculate RMSE as final_loss
        mse = mean_squared_error(y_query, y_pred)
        final_loss = np.sqrt(mse)
        print(f"Final RMSE on query data: {final_loss:.6f}")

        # 4. Return metrics
        return {"final_loss": final_loss}

    def predict(self, X):
        """
        Predict derivatives for a given state X.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.vstack(predictions).T

    def save(self):
        """Save the list of trained models to file using pickle."""
        path = self.config["save_path"]
        with open(path, "wb") as f:
            pickle.dump(self.models, f)
        print(f"Models saved to {path}")

    def load(self, path=None):
        """
        Load a list of pre-trained models from file.
        """
        path = path or self.config["save_path"]
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        print(f"Models loaded from {path}")

    def print_equations(self):
        """Print the discovered equations for each dimension."""
        if not self.models:
            print("No models trained or loaded.")
            return
        print("\n=== GPlearn Discovered Equations ===")
        for i, model in enumerate(self.models):
            print(f"  x{i}' = {model._program}")
