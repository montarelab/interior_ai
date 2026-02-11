import asyncio
import itertools
import logging
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import aiofiles
import uvicorn
from fastapi import Body, FastAPI, File, Form, UploadFile

from src.graph import build_graph, init_graph_state
from src.models import EvalConfigPayload
from src.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path("dataset")

app = FastAPI()

bg_tasks_list: list[asyncio.Task] = []
bg_tasks_lock = threading.Lock()


def remove_done_task(task: asyncio.Task):
    """Callback that runs after the background task is done. It is used in order to garbage collector didn't delete background tasks."""
    with bg_tasks_lock:
        bg_tasks_list.remove(task)


def get_prompt_path(task: str, prompt_version: int) -> Path:
    """Get prompt path of dataset for evaluation."""
    return DATASET_PATH / task / f"prompt_v{prompt_version}.md"


def get_image_path(task: str) -> Path:
    """Get image path of dataset for evaluation."""
    return DATASET_PATH / task / "image.png"


def init_job_path() -> Path:
    """Initializes job ID, creates a folder for job results, and returns the path."""
    job_dt = datetime.now().isoformat()
    job_id = uuid4()
    job_path = Path("jobs") / f"{job_dt}_{job_id}"
    job_path.mkdir(parents=True, exist_ok=True)
    return job_path


@app.post("/generate")
async def generate(prompt: str = Form(None), image: UploadFile = File(...)):
    """Generate POST endpoint. Takes string prompt and image and generates an image."""
    try:
        logger.info(f"Geneate request.")
        job_path = init_job_path()

        contents = await image.read()
        suffix = image.filename.split(".")[-1]
        user_image_path = job_path / f"source_img.{suffix}"
        async with aiofiles.open(user_image_path, "wb") as f:
            await f.write(contents)

        async with aiofiles.open(job_path / "user_prompt.md", "w") as f:
            await f.write(prompt)

        logger.info(f"Saved request details.")

        result_path = job_path / "result"
        result_path.mkdir(parents=True, exist_ok=True)

        graph = build_graph()
        init_state = init_graph_state(
            prompt=prompt,
            user_image_path=user_image_path,
            result_path=result_path,
        )
        response = await graph.ainvoke(input=init_state)
        return {"status": "success", "job_path": job_path.expanduser().resolve()}
    except Exception:
        logger.exception(f"Error during generate request.")
        return {"status": "failed"}


async def run_eval(job_path: Path, tasks: list[asyncio.Task]):
    """Run evaluation of asyncio generation tasks."""
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
async def evaluate(
    payload: EvalConfigPayload = Body(default_factory=EvalConfigPayload),
):
    """Evaluate POST endpoint. Takes evaluation details and works asynchronously."""
    try:
        logger.info(
            f"Evaluation request. Payload: \n{payload.model_dump_json(indent=4)}"
        )
        product = itertools.product(
            payload.model_names, payload.tasks, payload.versions
        )
        job_path = init_job_path()

        async with aiofiles.open(job_path / "payload.json", "w") as f:
            await f.write(payload.model_dump_json(indent=4))

        product_len: int = (
            len(payload.model_names) * len(payload.tasks) * len(payload.versions)
        )
        tasks: list[asyncio.Task] = []
        logger.info(f"Total tasks: {product_len}")

        for item in product:
            model, task, version = item[0], item[1], item[2]
            image_path = get_image_path(task)
            prompt_path = get_prompt_path(task, version)
            prompt = prompt_path.read_text(encoding="utf-8")

            result_path = job_path / f"{task}_v{version}"
            result_path.mkdir(exist_ok=True, parents=True)

            graph = build_graph()
            init_state = init_graph_state(
                prompt=prompt,
                result_path=result_path,
                llm_model=model,
                user_image_path=image_path,
            )
            task = asyncio.create_task(graph.ainvoke(input=init_state))
            tasks.append(task)

        bg_task = asyncio.create_task(run_eval(job_path=job_path, tasks=tasks))
        bg_task.add_done_callback(remove_done_task)
        bg_tasks_list.append(bg_task)

        return {
            "status": "started",
            "total_tasks": product_len,
            "job_path": job_path.expanduser().resolve(),
        }
    except Exception:
        logger.exception(f"Error during generate request.")
        return {"status": "failed"}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
