"""Dataset/model import helper only; no physical-parameter decoder included."""
import sys
import importlib
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = PROJECT_ROOT / "compare2/DSSDL"

def import_dssdl_for_system(system: str):
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    import datasets

    datasets.set_current_system(system)
    dssdl_module = importlib.import_module("DSSDL_")
    pretrain_script = importlib.import_module("pretrain")
    finetune_script = importlib.import_module("finetune")
    dssdl_module.CURRENT_SYSTEM = system
    return datasets.CURRENT_SYSTEM, datasets.dataset_dict, dssdl_module, pretrain_script, finetune_script

