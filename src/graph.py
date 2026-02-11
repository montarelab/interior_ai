import asyncio
import base64
import json
import logging
import mimetypes
import operator
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

import aiofiles
from google import genai
from jinja2 import Environment, FileSystemLoader
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from openai import AsyncClient
from PIL import Image

from src.models import (
    AppResultsModel,
    ImageDescriptor,
    ImageEvalResponse,
    PlanResponse,
    UsageMetadata,
)
from src.settings import settings

logger = logging.getLogger(__name__)

SEMAPHORE_VAL = 5
PROMPTS_PATH = Path("prompts")
EVAL_MODEL = "gpt-5-mini"
PLAN_MODEL = "gpt-5-mini"

env = Environment(loader=FileSystemLoader("prompts"))
openai_semaphore = asyncio.Semaphore(SEMAPHORE_VAL)
google_semaphore = asyncio.Semaphore(SEMAPHORE_VAL)
openai_client = AsyncClient(api_key=settings.OPENAI_API_KEY)
gemini_client = genai.Client()


def standardize_name(name: str) -> str:
    if "/" in name:
        return name.replace("/", "__")
    return name


class GraphState(TypedDict):
    """Graph state of the image generation graph."""

    result_path: Path
    llm_model: str
    user_prompt: str
    gen_mode: Literal["init", "later"] = "init"
    enhanced_prompt: str
    user_image_path: Path
    init_image_path: Path
    image_descriptors: list[ImageDescriptor]
    plan_response: PlanResponse
    plan_gen_duration_sec: float
    img_gen_duration_sec: Annotated[list[float], operator.add]
    img_eval_duration_sec: Annotated[list[float], operator.add]
    token_usages: Annotated[list[UsageMetadata], operator.add]
    eval_responses: Annotated[list[ImageEvalResponse], operator.add]
    current_img_descriptor: ImageDescriptor
    img_data_url: Annotated[list[str], operator.add]


async def img_path_to_data_url(img_path: Path) -> str:
    """Get Base64 from image path by reading the file."""
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type")

    async with aiofiles.open(img_path, "rb") as f:
        content = await f.read()
        encoded = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"


async def render_template_async(template_name: str, **context) -> str:
    """Asynchronously read prompt template and render with 'context' variables."""
    template_path = PROMPTS_PATH / template_name
    async with aiofiles.open(template_path, "r", encoding="utf-8") as f:
        template_str = await f.read()

    template = env.from_string(template_str)
    return template.render(**context)


async def plan(state: GraphState):
    """Plan the further actions."""
    try:
        logger.info(f"Plan generation started.")
        eval_schema = ImageEvalResponse.model_json_schema()
        prompt = await render_template_async(
            "plan.jinja",
            user_prompt=state["user_prompt"],
            eval_schema=json.dumps(eval_schema, indent=4),
        )
        img_data_url = await img_path_to_data_url(state["user_image_path"])
        async with openai_semaphore:
            t0 = time.perf_counter()
            response = await openai_client.responses.parse(
                model=PLAN_MODEL,
                text_format=PlanResponse,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": img_data_url},
                        ],
                    }
                ],
            )
            plan_gen_seconds = time.perf_counter() - t0

        logger.info(
            f"Interrier plan was generated. Requested images: {len(response.output_parsed.images)} Duration: {plan_gen_seconds}s"
        )
        async with aiofiles.open(state["result_path"] / "plan.json", "w") as f:
            await f.write(response.output_parsed.model_dump_json(indent=4))

        usage = UsageMetadata.from_openai_usage(PLAN_MODEL, response.usage)
        return Command(
            goto="init_image_gen",
            update={
                "enhanced_prompt": response.output_parsed.improved_prompt,
                "image_descriptors": response.output_parsed.images,
                "plan_gen_duration_sec": plan_gen_seconds,
                "current_img_descriptor": (response.output_parsed.images[0]),
                "token_usages": [usage],
                "plan_response": response.output_parsed,
            },
        )
    except Exception:
        logger.exception("Error occured during process planning.")
        raise


@dataclass
class ImgGenResponse:
    img_base64: str
    usage: UsageMetadata
    img_gen_seconds: float


async def google_img_gen(
    model: str, prompt: str, result_path: Path, image_path: Path
) -> ImgGenResponse:
    """Generate image with Google model."""
    img = Image.open(image_path)
    async with google_semaphore:
        t0 = time.perf_counter()
        response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=[prompt, img],
        )
        img_gen_seconds = time.perf_counter() - t0
        logger.info(f"Image was generated with '{model}'. Duration: {img_gen_seconds}s")
    usage = UsageMetadata.from_google_usage(model, response.usage_metadata)
    img_base64: str = None
    for part in response.parts:
        if part.text:
            text_path = result_path / f"{model}.txt"
            async with aiofiles.open(text_path, "w") as f:
                await f.write(part.text)
        elif part.inline_data:
            image_bytes = part.as_image().image_bytes
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return ImgGenResponse(
        img_base64=img_base64, usage=usage, img_gen_seconds=img_gen_seconds
    )


async def openai_img_gen(
    model: str, prompt: str, result_path: Path, image_path: Path
) -> ImgGenResponse:
    """Generate image with OpenAI model."""
    async with openai_semaphore:
        t0 = time.perf_counter()
        response = await openai_client.images.edit(
            model=model,
            prompt=prompt,
            image=open(image_path, "rb"),
        )
        img_gen_seconds = time.perf_counter() - t0
        logger.info(f"Image was generated with '{model}'. Duration: {img_gen_seconds}s")
    usage = UsageMetadata.from_openai_usage(model, response.usage)
    img_base64 = response.data[0].b64_json
    return ImgGenResponse(
        img_base64=img_base64, usage=usage, img_gen_seconds=img_gen_seconds
    )


async def image_gen(state: GraphState, gen_mode: Literal["init", "later"]):
    """Geneates an image based on user's prompt and prefernce image."""
    try:
        if gen_mode == "init":
            image_path = state["user_image_path"]
            user_prompt = state["user_prompt"]
        else:
            image_path = state["init_image_path"]
            user_prompt = state["enhanced_prompt"]

        current_img_descriptor = state["current_img_descriptor"]
        prompt = await render_template_async(
            "img_gen.jinja",
            user_prompt=user_prompt,
            specific_details=current_img_descriptor.description,
        )
        model = state["llm_model"]
        logger.info(f"Image generation with model '{model}' was started.")
        if "gemini" in model:
            response = await google_img_gen(
                model=model,
                prompt=prompt,
                result_path=state["result_path"],
                image_path=image_path,
            )
        elif "gpt" in model:
            response = await openai_img_gen(
                model=model,
                prompt=prompt,
                result_path=state["result_path"],
                image_path=image_path,
            )
        else:
            raise NotImplementedError(f"Model '{model}' is not supported.")

        result_img_path = (
            state["result_path"]
            / f"{current_img_descriptor.key}_{standardize_name(model)}.png"
        )
        async with aiofiles.open(result_img_path, "wb") as f:
            img_bytes = base64.b64decode(response.img_base64)
            await f.write(img_bytes)

        update_dict = {
            "img_data_url": [f"data:image/png;base64,{response.img_base64}"],
            "token_usages": [response.usage],
            "img_gen_duration_sec": [response.img_gen_seconds],
        }

        if gen_mode == "init":
            update_dict["init_image_path"] = result_img_path

        return Command(goto="eval_image", update=update_dict)
    except Exception as e:
        logger.exception(f"Error occured during image generation")
        raise


async def eval_image(state: GraphState):
    """Evaluate image."""
    try:
        img_data_url = state["img_data_url"][-1]
        current_img_descriptor = state["current_img_descriptor"]
        user_prompt = f"{state["enhanced_prompt"]}\n{current_img_descriptor}"
        prompt = await render_template_async("img_judge.jinja", user_prompt=user_prompt)
        async with openai_semaphore:
            t0 = time.perf_counter()
            response = await openai_client.responses.parse(
                model=EVAL_MODEL,
                text_format=ImageEvalResponse,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": img_data_url,
                            },
                        ],
                    }
                ],
            )
            img_eval_seconds = time.perf_counter() - t0
            logger.info(f"Image was evaluated. Duration: {img_eval_seconds}s")
            usage = UsageMetadata.from_openai_usage(EVAL_MODEL, response.usage)
        return Command(
            update={
                "eval_responses": [response.output_parsed],
                "img_eval_duration_sec": [img_eval_seconds],
                "token_usages": [usage],
            }
        )
    except Exception:
        logger.exception("Error occured during LLM judgement.")
        raise


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


async def finalize(state: GraphState):
    """Finalize graph execution."""
    usages = compound_usages(state["token_usages"])
    app_response = AppResultsModel(
        usages=usages,
        evals=state["eval_responses"],
        plan_gen_duration_sec=state["plan_gen_duration_sec"],
        img_gen_duration_sec=state["img_gen_duration_sec"],
        img_eval_duration_sec=state["img_eval_duration_sec"],
        plan_response=state["plan_response"],
    )
    result_json_path = state["result_path"] / "results.json"
    async with aiofiles.open(result_json_path, "w") as f:
        await f.write(app_response.model_dump_json(indent=4))
    logger.info(f"The results were stored.")


async def init_image_gen(state: GraphState):
    """Initial reference image generation."""
    return await image_gen(state, "init")


async def later_image_gen(state: GraphState):
    """Image generation node for the rest of image descriptors without the first one.."""
    return await image_gen(state, "later")


async def gate(state: GraphState):
    """Gate node that is executed for every eval image node except for the final one."""
    logger.info(
        f"Length of eval responses: {len(state["eval_responses"])}, length of image descriptor: {len(state["image_descriptors"])}"
    )
    if len(state["eval_responses"]) == len(state["image_descriptors"]):
        return Command(goto="finalize")

    return state


def build_graph():
    """Build graph for interrier generation."""
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("plan", plan)
    graph_builder.add_node("init_image_gen", init_image_gen)
    graph_builder.add_node("later_image_gen", later_image_gen)
    graph_builder.add_node("eval_image", eval_image)
    graph_builder.add_node("gate", gate)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "plan")
    graph_builder.add_edge("plan", "init_image_gen")

    def fanout_after_init(state: GraphState):
        sends = [Send("eval_image", state)]
        if len(state["image_descriptors"]) > 1:
            sends.extend(
                [
                    Send(
                        "later_image_gen",
                        state | {"current_img_descriptor": descriptor},
                    )
                    for descriptor in state["image_descriptors"][1:]
                ]
            )
        return sends

    graph_builder.add_conditional_edges(
        "init_image_gen", fanout_after_init, ["later_image_gen", "eval_image"]
    )
    graph_builder.add_edge("later_image_gen", "eval_image")
    graph_builder.add_edge("eval_image", "gate")
    graph_builder.add_edge("finalize", END)

    graph = graph_builder.compile()
    logger.info(f"Graph was compiled.")
    return graph
