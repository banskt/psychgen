from pathlib import Path

DATA_ROOT = Path(config["paths"]["data_root"])

def get_data_path(relpath):
    return str(DATA_ROOT / relpath)
