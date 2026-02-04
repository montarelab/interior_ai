import asyncio
import base64
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Literal

import aiofiles
from google import genai
from jinja2 import Environment, FileSystemLoader
from openai import AsyncClient
from PIL import Image

from src.models import (
    AppResultsModel,
    ImageJudgeResponse,
    PromptEnhancementResponse,
    UsageMetadata,
)
from src.settings import settings

SEMAPHORE_VAL = 5
IMG_GEN_PROMPT_PATH = Path("prompts/img_gen.jinja")
JUDGE_PROMPT_PATH = Path("prompts/img_judge.jinja")
PLANNER_PROMPT_PATH = Path("prompts/prompt_enhancer.jinja")

logger = logging.getLogger(__name__)


def standardize_name(name: str) -> str:
    if "/" in name:
        return name.replace("/", "__")
    return name


class ImageGenPipelineClient:
    """Main application service that is dedicated for the main image generation pipeline."""

    def __init__(self):
        """Initializes main service components."""
        self.openai_client = AsyncClient(api_key=settings.OPENAI_API_KEY)
        self.gemini_client = genai.Client()

        self.token_usage_list: list[UsageMetadata] = []
        self.token_usage_lock = asyncio.Lock()

        self.openai_semaphore = asyncio.Semaphore(SEMAPHORE_VAL)
        self.google_semaphore = asyncio.Semaphore(SEMAPHORE_VAL)

        env = Environment(loader=FileSystemLoader("prompts"))

        img_gen_prompt = IMG_GEN_PROMPT_PATH.read_text(encoding="utf-8")
        self.img_gen_template = env.from_string(img_gen_prompt)

        judge_prompt_str = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
        self.judge_template = env.from_string(judge_prompt_str)

        planner_prompt_str = PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
        self.planner_template = env.from_string(planner_prompt_str)

        self.eval_model = "gpt-5-mini"
        self.planner_model = "gpt-5-mini"

    async def eval_image(
        self, user_prompt: str, img_data_url: str
    ) -> tuple[ImageJudgeResponse, UsageMetadata]:
        """
        Evaluates generated image using LLM-as-a-judge based on user's prompt and other metrics.
        Returns LLM's structured output.
        """
        try:
            async with self.openai_semaphore:
                judge_prompt = self.judge_template.render(user_prompt=user_prompt)
                response = await self.openai_client.responses.parse(
                    model=self.eval_model,
                    text_format=ImageJudgeResponse,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": judge_prompt},
                                {"type": "input_image", "image_url": img_data_url},
                            ],
                        }
                    ],
                )
                return response.output_parsed, response.usage
        except Exception:
            logger.exception("Error occured during LLM judgement.")

    async def generate_image(
        self,
        model: str,
        user_prompt: str,
        mode: Literal["inital", "after"],
        specific_details: str,
        image_path: Path,
        result_path: Path,
    ) -> tuple[str, UsageMetadata]:
        """
        Geneates an image based on user's prompt and prefernce image.
        Returns the generated image as Base64 and LLM request's usage metadata.
        """
        try:
            prompt = self.img_gen_template.render(
                user_prompt=user_prompt, mode=mode, specific_details=specific_details
            )
            if "gemini" in model:
                async with self.google_semaphore:
                    img = Image.open(image_path)
                    response = await self.gemini_client.aio.models.generate_content(
                        model=model,
                        contents=[prompt, img],
                    )
                    usage = UsageMetadata.from_google_usage(
                        model, response.usage_metadata
                    )
                    for part in response.parts:
                        if part.text:
                            text_path = result_path / f"{model}.txt"
                            async with aiofiles.open(text_path, "w") as f:
                                await f.write(part.text)
                        elif part.inline_data:
                            image_bytes = part.as_image().image_bytes
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    return image_base64, usage
            elif "gpt" in model:
                async with self.openai_semaphore:
                    response = await self.openai_client.images.edit(
                        model=model,
                        prompt=prompt,
                        image=open(image_path, "rb"),
                    )
                    usage = UsageMetadata.from_openai_usage(model, response.usage)
                    image_base64 = response.data[0].b64_json
                    return image_base64, usage
            else:
                raise NotImplementedError(f"Model '{model}' is not supported.")
        except Exception as e:
            logger.exception(f"Error occured during image generation")
            raise

    def resolve_best_model(self) -> str:
        return "gpt-image-1-mini"

    async def plan_work(
        self, user_prompt: str, img_data_url: str
    ) -> tuple[PromptEnhancementResponse, UsageMetadata]:
        try:
            eval_schema = ImageJudgeResponse.model_json_schema()
            async with self.openai_semaphore:
                planner_prompt = self.planner_template.render(
                    user_prompt=user_prompt,
                    eval_schema=json.dumps(eval_schema, indent=4),
                )
                response = await self.openai_client.responses.parse(
                    model=self.planner_model,
                    text_format=PromptEnhancementResponse,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": planner_prompt},
                                {"type": "input_image", "image_url": img_data_url},
                            ],
                        }
                    ],
                )
                return response.output_parsed, response.usage
        except Exception:
            logger.exception("Error occured during process planning.")

    async def handle_image(
        self,
        model: str | None,
        user_prompt: str,
        image_path: Path,
        result_dir_path: Path,
    ):
        """Handles image: generates a new image, automatically evaluates, and stores the results."""
        try:
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                raise ValueError("Could not determine MIME type")

            if not model:
                model = self.resolve_best_model()

            async with aiofiles.open(image_path, "rb") as f:
                content = await f.read()
                encoded = base64.b64encode(content).decode("utf-8")
                img_data_url = f"data:{mime_type};base64,{encoded}"

            t0 = time.perf_counter()
            plan_response, plan_usage = await self.plan_work(
                user_prompt=user_prompt, img_data_url=img_data_url
            )
            plan_gen_seconds = time.perf_counter() - t0
            logger.info(f"Interrier plan was generated. Duration: {plan_gen_seconds}s")

            async with aiofiles.open(result_dir_path / "plan.json", "w") as f:
                await f.write(plan_response.model_dump_json(indent=4))

            async with self.token_usage_lock:
                self.token_usage_list.append(plan_usage)

            t0 = time.perf_counter()
            # TODO fill mode and specific_details
            # TODO rewrite in langgraph
            image_base64, img_gen_usage = await self.generate_image(
                model=model,
                user_prompt=plan_response.improved_prompt,
                mode=...,
                specific_details=...,
                image_path=image_path,
                result_path=result_dir_path,
            )
            img_gen_seconds = time.perf_counter() - t0
            logger.info(f"Image was generated. Duration: {img_gen_seconds}s")

            async with self.token_usage_lock:
                self.token_usage_list.append(img_gen_usage)

            result_image_path = result_dir_path / f"{standardize_name(model)}.png"
            async with aiofiles.open(result_image_path, "wb") as f:
                img_bytes = base64.b64decode(image_base64)
                await f.write(img_bytes)

            t0 = time.perf_counter()
            data_url = f"data:image/{mime_type};base64,{image_base64}"
            eval_response, eval_usage = await self.eval_image(user_prompt, data_url)
            img_eval_seconds = time.perf_counter() - t0
            logger.info(f"Image was evaluated. Duration: {img_eval_seconds}s")

            async with self.token_usage_lock:
                self.token_usage_list.append(eval_usage)

            # TODO several usages need to be here
            app_response = AppResultsModel(
                usages=[plan_usage, img_gen_usage, eval_usage],
                judge=eval_response,
                plan_gen_duration_sec=plan_gen_seconds,
                img_gen_duration_sec=[img_gen_seconds],
                img_eval_duration_sec=img_eval_seconds,
            )
            result_json_path = result_dir_path / f"{standardize_name(model)}.json"
            async with aiofiles.open(result_json_path, "w") as f:
                await f.write(app_response.model_dump_json(indent=4))
            logger.info(f"The results were stored.")

        except Exception as e:
            logger.exception(f"Error occured during image handling")
            return e
