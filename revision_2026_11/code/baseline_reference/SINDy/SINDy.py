import os
import sys
import types
import numpy as np
import pickle
import csv
import time

# Avoid optional JAX-based optimizer import failures (SBR) when JAX is broken/unneeded.
# Set SINDY_BLOCK_JAX=0 to allow importing real JAX.
if os.environ.get("SINDY_BLOCK_JAX", "1") == "1" and "jax" not in sys.modules:
    sys.modules["jax"] = types.ModuleType("jax")

import pysindy as ps


# ---------------------------
# Configuration (Hyperparameters)
# ---------------------------

CONFIG = {
    'poly_degree': 3,             # Degree of polynomial features
    'include_intercepts': True,   # Whether to include bias/intercept terms
    'library_type': 'polynomial', # Type of feature library
    'optimizer': 'STLSQ',         # Sparse optimizer (e.g., STLSQ)
    'threshold': 0.1,             # Threshold for sparsity constraint
    # 'feature_names': ['x0', 'x1', 'x2'],# Feature names for model output
    'feature_names': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'
    , 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18'
    ],
    'save_path': 'sindy_model.pkl' # File path for saving the model
}


# ---------------------------
# SINDy Regressor Class
# ---------------------------

class SINDyRegressor:
    """
    A wrapper around pysindy.SINDy for training and fine-tuning sparse dynamical system models.

    Features:
        - Train on dataset_dict data (always using normalized data)
        - Fine-tune using the entire test data (always using normalized data)
        - Save/load trained models using pickle

    Methods:
        - prepare_data(): Load and concatenate state and derivative data (uses normalized data)
        - train(): Fit the model on training data (uses normalized data)
        - finetune(): Refit the model on test data (uses normalized data)
        - save(): Save the model to disk
        - load(): Load a saved model from disk
    """

    def __init__(self, dataset_dict, system_name=None):
        """
        Initialize the SINDy model with configuration and dataset.

        Args:
            dataset_dict (dict): Dataset dictionary containing normalized/raw arrays.
        """
        self.config = CONFIG.copy()
        self.dataset_dict = dataset_dict
        self.system_name = system_name or "sindy"

        # Update input/output dimensions based on dataset info
        self.config['input_dim'] = self.dataset_dict['info']['input_dim']
        self.config['output_dim'] = self.dataset_dict['info']['output_dim']
        self.config['dt'] = self.dataset_dict['info'].get('dt', 0.01)

        # Build feature library
        if self.config['library_type'] == 'polynomial':
            self.library = ps.PolynomialLibrary(
                degree=self.config['poly_degree'],
                include_bias=self.config['include_intercepts']
            )
        else:
            raise ValueError("Only polynomial library is currently supported.")

        # Set up optimizer
        self.optimizer = getattr(ps.optimizers, self.config['optimizer'])(
            threshold=self.config['threshold']
        )

        # Initialize SINDy model (feature_names not supported in some versions)
        try:
            self.model = ps.SINDy(
                feature_library=self.library,
                optimizer=self.optimizer,
                feature_names=self.config['feature_names']
            )
        except TypeError as exc:
            if "feature_names" in str(exc):
                self.model = ps.SINDy(
                    feature_library=self.library,
                    optimizer=self.optimizer
                )
            else:
                raise

    def prepare_data(self, split='train'):
        """
        Load and concatenate state variables and their derivatives using normalized data.

        Args:
            split (str): Dataset split ('train', 'test', or 'eval')

        Returns:
            X (np.ndarray): State variables (n_samples, n_features)
            y (np.ndarray): Derivatives (n_samples, n_targets)
        """
        # Always use normalized data, remove the use_normalized parameter
        data_key = 'raw'
        X_list, y_list = self.dataset_dict[data_key][split]

        X = np.vstack(X_list)  # Combine all samples into one array
        y = np.vstack(y_list)

        return X, y

    def train(self):
        """
        Train the SINDy model on the specified dataset split using normalized data.
        """
        start_time = time.time()
        # Removed use_normalized parameter, as it's always True
        X, y = self.prepare_data(split='train') # Call without use_normalized
        try:
            self.model.fit(X, t=self.config['dt'], x_dot=y)
        except TypeError as exc:
            if "t" in str(exc):
                self.model.fit(X, x_dot=y)
            else:
                raise
        print("Discovered equations:")
        self.model.print()

        train_rmse = None
        try:
            y_pred = self.model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            train_rmse = float(np.sqrt(mse))
        except Exception as exc:
            print(f"鈿狅笍 Warning during training loss evaluation: {exc}")

        elapsed_seconds = time.time() - start_time
        self._write_time_log(
            phase="pretrain",
            rows=[
                {
                    "epoch": 1,
                    "loss": train_rmse if train_rmse is not None else "",
                    "elapsed_seconds": elapsed_seconds,
                }
            ],
        )

    def finetune(self):
        """
        Fine-tune the model using the support dataset and evaluate on the query dataset.
        """
        start_time = time.time()
        # 1. Finetune on the support set
        X_support, y_support = self.prepare_data(split='support')

        print("=== Fine-tuning with the support data ===")
        try:
            try:
                self.model.fit(X_support, t=self.config['dt'], x_dot=y_support)
            except TypeError as exc:
                if "t" in str(exc):
                    self.model.fit(X_support, x_dot=y_support)
                else:
                    raise
        except Exception as e:
            print(f"⚠️ Warning during fine-tuning: {e}")

        print("Updated equations after fine-tuning:")
        self.model.print()

        # 2. Evaluate on the query (test) set
        final_loss = self.evaluate_on_query()

        elapsed_seconds = time.time() - start_time
        self._write_time_log(
            phase="finetune",
            rows=[
                {
                    "epoch": 1,
                    "loss": float(final_loss),
                    "elapsed_seconds": elapsed_seconds,
                }
            ],
        )

        return {'final_loss': final_loss}

    def evaluate_on_query(self):
        """
        Evaluate the current model on the query dataset and report RMSE.

        Returns:
            float: RMSE on the query split.
        """
        print("\n=== Evaluating on the query data ===")
        X_query, y_query = self.prepare_data(split='query')

        # Predict derivatives on the query set
        y_pred = self.model.predict(X_query)

        # Calculate RMSE
        mse = np.mean((y_query - y_pred) ** 2)
        rmse = np.sqrt(mse)
        print(f"Final RMSE on query data: {rmse:.6f}")

        return rmse

    def save(self):
        """Save the trained model to file using pickle."""
        path = self.config['save_path']
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def _write_time_log(self, phase, rows):
        os.makedirs("final_data", exist_ok=True)
        csv_path = os.path.join("final_data", f"{self.system_name}_{phase}_time.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "elapsed_seconds"])
            writer.writeheader()
            writer.writerows(rows)

    def load(self, path=None):
        """
        Load a pre-trained model from file.

        Args:
            path (str): Optional custom model path; defaults to config['save_path']
        """
        path = path or self.config['save_path']
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")
