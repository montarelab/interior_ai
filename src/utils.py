from datetime import datetime
from pathlib import Path

DATASET_PATH = Path("dataset")


def get_prompt_path(task: str, prompt_version: int) -> Path:
    return DATASET_PATH / task / f"prompt_v{prompt_version}.md"


def get_image_path(task: str) -> Path:
    return DATASET_PATH / task / "image.png"


def standardize_name(name: str) -> str:
    if "/" in name:
        return name.replace("/", "__")
    return name
