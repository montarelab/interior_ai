import asyncio
import itertools
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import aiofiles
from fastapi import Depends, FastAPI, File, Form, UploadFile

from src.models import EvalConfigPayload
from src.service import ImageGenPipelineClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path("dataset")

app = FastAPI()

image_gen_pipeline_service = ImageGenPipelineClient()


def get_img_pipeline_service():
    """Dependency injection method to retrieve an instance of ImageGenPipelineClient."""
    global image_gen_pipeline_service
    return image_gen_pipeline_service


ImageGenPipelineDep = Annotated[
    "ImageGenPipelineClient", Depends(get_img_pipeline_service)
]


def get_prompt_path(task: str, prompt_version: int) -> Path:
    return DATASET_PATH / task / f"prompt_v{prompt_version}.md"


def get_image_path(task: str) -> Path:
    return DATASET_PATH / task / "image.png"


@app.post("/generate")
async def generate(
    pipeline: ImageGenPipelineDep,
    prompt: str = Form(None),
    image: UploadFile = File(...),
):
    """Generate POST endpoint. Takes string prompt and image and generates an image."""
    try:
        logger.info(f"Geneate request.")
        job_dt = datetime.now().isoformat()
        job_id = uuid4()
        job_path = Path("jobs") / f"{job_dt}_{job_id}"
        job_path.mkdir(parents=True, exist_ok=True)

        contents = await image.read()
        suffix = image.filename.split(".")[-1]
        src_img_path = job_path / f"source_img.{suffix}"
        async with aiofiles.open(src_img_path, "wb") as f:
            await f.write(contents)

        async with aiofiles.open(job_path / "user_prompt.md", "wb") as f:
            await f.write(prompt)

        logger.info(f"Saved request details.")

        result_dir_path = job_path / "result"
        result_dir_path.mkdir(parents=True, exist_ok=True)

        await pipeline.handle_image(
            model=None,
            user_prompt=prompt,
            image_path=src_img_path,
            result_dir_path=result_dir_path,
        )
        return {"status": "success", "job_path": job_path.expanduser().resolve()}
    except Exception:
        return {"status": "failed"}


async def run_eval(job_path: Path, tasks: list[asyncio.Task]):
    results = await asyncio.gather(*tasks)
    exceptions = [r for r in results if isinstance(r, Exception)]
    if not exceptions:
        return

    logger.warning(f"Total exceptions: {len(exceptions)}")
    for idx, e in enumerate(exceptions):
        err_task_path = job_path / f"err_{idx}.txt"
        async with aiofiles.open(err_task_path, "w") as f:
            await f.write(str(e))


@app.post("/evaluate")
async def evaluate(pipeline: ImageGenPipelineDep, payload: EvalConfigPayload):
    """Evaluate POST endpoint. Takes evaluation details and works asynchronously."""
    try:
        logger.info(f"Payload: \n{payload.model_dump_json(indent=4)}")
        product = itertools.product(
            payload.model_names, payload.tasks, payload.versions
        )
        job_dt = datetime.now().isoformat()
        job_id = uuid4()
        job_path = Path("jobs") / f"{job_dt}_{job_id}"
        job_path.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(job_path / "payload.json", "w") as f:
            await f.write(payload.model_dump_json(indent=4))

        product_len: int = 0
        tasks: list[asyncio.Task] = []

        for item in product:
            model, task, version = item[0], item[1], item[2]
            prompt_path = get_prompt_path(task, version)
            user_prompt = prompt_path.read_text(encoding="utf-8")
            result_path = job_path / f"{task}_v{version}"

            tasks.append(
                asyncio.create_task(
                    pipeline.handle_image(
                        model=model,
                        user_prompt=user_prompt,
                        image_path=get_image_path(task),
                        result_path=result_path,
                    )
                )
            )
            product_len += 1

        logger.info(f"Total tasks: {product_len}")
        asyncio.create_task(run_eval(job_path=job_path, tasks=tasks))
        return {
            "status": "started",
            "total_tasks": product_len,
            "job_path": job_path.expanduser().resolve(),
        }
    except Exception:
        return {"status": "failed"}
