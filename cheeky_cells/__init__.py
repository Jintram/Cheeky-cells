import configparser
import os
import pathlib


def _apply_config():
    """Read cheeky_cells.cfg from the package directory (if it exists) and
    apply settings.  Called once at import time, before torch is imported."""
    
    print("Checking for configuration file..")
    cfg_path = pathlib.Path(__file__).parent / "cheeky_cells.cfg"
    
    if not cfg_path.is_file():
        print("Not found, all behavior is now default..")
        return

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    # Allow PyTorch's MPS backend (Apple Silicon) to exceed the default
    # memory limit instead of raising "MPS backend out of memory".
    # Setting the ratio to 0.0 removes the cap entirely so macOS can
    # page to disk under pressure rather than crashing.
    if cfg.getboolean("pytorch", "disable_mps_memory_limit", fallback=False):
        print("Removing pytroch memory limit..")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


_apply_config()
