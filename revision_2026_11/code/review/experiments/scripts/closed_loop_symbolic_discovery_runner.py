"""Only the unchanged model-construction helpers used by the current closure."""
from __future__ import annotations
from typing import Any, Dict

class ZeroPrior:
    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def __call__(self, x, dataset_dict=None, use_raw_data: bool = False):
        import torch

        return torch.zeros((x.shape[0], self.output_dim), device=x.device, dtype=x.dtype)

    def raw_predict(self, features):
        import numpy as np

        return np.zeros((features.shape[0], self.output_dim), dtype=np.float64)



def build_model(dssdl_module, dataset_dict, cfg: Dict[str, Any], prior) -> Any:
    input_dim = dataset_dict["info"]["input_dim"]
    output_dim = dataset_dict["info"]["output_dim"]
    one_hot_dim = dataset_dict["info"]["file_counts"]["train"]
    model = dssdl_module.MetaNet(
        input_dim=input_dim,
        output_dim=output_dim,
        one_hot_dim=one_hot_dim,
        specific_param_dim=cfg["specific_param_dim"],
        hidden_widths=cfg["hidden_widths"],
        device=cfg["device"],
        dataset_dict=dataset_dict,
        use_raw_data=False,
    )
    model.f_hat_func = prior
    return model

