import base64
import mimetypes
from collections import defaultdict
from pathlib import Path

import aiofiles
from jinja2 import Environment, FileSystemLoader

from src.models import UsageMetadata

PROMPTS_PATH = Path("prompts")
env = Environment(loader=FileSystemLoader("prompts"))


async def render_template_async(template_name: str, **context) -> str:
    """Asynchronously read prompt template and render with 'context' variables."""
    template_path = PROMPTS_PATH / template_name
    async with aiofiles.open(template_path, "r", encoding="utf-8") as f:
        template_str = await f.read()

    template = env.from_string(template_str)
    return template.render(**context)


def standardize_model_name(name: str) -> str:
    """Standardize model name by replacing '/' with '__' to ensure its usage in path won't create folders."""
    if "/" in name:
        return name.replace("/", "__")
    return name


def group_usages(usages: list[UsageMetadata]) -> dict[str, list[UsageMetadata]]:
    """Group usages by model name."""
    groups = defaultdict(list)
    for usage in usages:
        groups[usage.model].append(usage)
    return dict(groups)


def compound_usages(usages: list[UsageMetadata]) -> list[UsageMetadata]:
    """Compund usages by model name."""
    groups = group_usages(usages)
    return [
        UsageMetadata(
            model=model,
            input_tokens=sum([u.input_tokens for u in model_usages]),
            output_tokens=sum([u.output_tokens for u in model_usages]),
            total_tokens=sum([u.total_tokens for u in model_usages]),
        )
        for model, model_usages in groups.items()
    ]


async def img_path_to_data_url(img_path: Path) -> str:
    """Get Base64 from image path by reading the file."""
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type")

    async with aiofiles.open(img_path, "rb") as f:
        content = await f.read()
        encoded = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
