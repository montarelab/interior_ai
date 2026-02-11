import litellm
from google.genai.types import GenerateContentResponseUsageMetadata
from openai.types.images_response import Usage
from pydantic import BaseModel, Field


class EvalConfigPayload(BaseModel):
    """Evaluation Payload."""

    model_names: list[str] = Field(
        default_factory=lambda: [
            "gpt-image-1-mini",
            "gemini-2.5-flash-image",
            # "gpt-image-1.5",
            # "gemini-3-pro-image-preview",
        ]
    )
    tasks: list[str] = Field(
        default_factory=lambda: [
            "1_modern_minimalist",
            "2_industrial_loft",
            # "3_japandi_zen",
            # "4_tech_savvy",
            # "5_warm_mediterranean",
        ]
    )
    versions: list[int] = Field(default_factory=lambda: [1])


class StructuredMetric(BaseModel):
    """Metrics response that includes float value and its explanation."""

    value: float = Field(
        ge=0.0,
        le=1.0,
        description="Float value between 0.0 and 1.0. The more the better.",
    )
    explanation: str = Field(description="Explanation of your judgement.")


class ImageEvalResponse(BaseModel):
    """Response to automatic judge of a generated image."""

    image_key: str = Field(
        description="Key of the image you evaluated, added inside single quotes in user's prompt."
    )
    reasoning_checklist: str = Field(description="Your reasoning checklist.")
    budget_correspondence: StructuredMetric = Field(
        description="Metric that measures how much the generated image corresponds to the budget given by user."
    )
    general_prompt_correspondence: StructuredMetric = Field(
        description="Metric that measures how much the generated image corresponds to the prompt given by a user."
    )
    realism_value: StructuredMetric = Field(
        description="Metric that measures how realistic doing such interier is."
    )
    satisfaction_value: StructuredMetric = Field(
        description="Metric that measures how much the generated image looks pleasent, aesthetical, and beautiful for living."
    )
    usability_value: StructuredMetric = Field(
        description="Metric that measures how much the generated image looks usable and comfortable for living."
    )


class ImageDescriptor(BaseModel):
    """Image description."""

    key: str = Field(
        "JSON-like short semantic image identifier key 1-3 lowercase joined by '_' words."
    )
    description: str = Field("Additional description for the image.")

    def __str__(self):
        return f"Additional image '{self.key}' description: {self.description}"


class PlanResponse(BaseModel):
    """Prompt enhancement response."""

    improved_prompt: str = Field(
        description="Enhanced user's prompt that will improve image generation."
    )
    images: list[ImageDescriptor] = Field(
        description="A list of image description the model needs to generate for a client.  One of the strategies is an image per a room in appartments. Do not do more than 5 images."
    )


class UsageMetadata(BaseModel):
    """Usage metadata model that holds token usage information from the LLM requests."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost_usd: float = 0.0

    def model_post_init(self, context):
        """Model post initializer method that assigns 'total_cost_usd'."""
        model_cost: dict = litellm.model_cost.get(self.model, None)
        if not model_cost:
            self.total_cost_usd = 0.0
        else:
            input_cost_per_token_key = [
                key for key in model_cost.keys() if "input_cost" in key
            ][0]
            output_cost_per_token_key = [
                key for key in model_cost.keys() if "output_cost" in key
            ][0]

            self.total_cost_usd = 0.0

            self.total_cost_usd = (
                self.input_tokens * model_cost[input_cost_per_token_key]
                + self.output_tokens * model_cost[output_cost_per_token_key]
            )
        return super().model_post_init(context)

    @classmethod
    def from_google_usage(
        cls, model: str, usage: GenerateContentResponseUsageMetadata
    ) -> "UsageMetadata":
        return cls(
            model=model,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
        )

    @classmethod
    def from_openai_usage(cls, model: str, usage: Usage) -> "UsageMetadata":
        return cls(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )


class AppResultsModel(BaseModel):
    """Structured response of the one generate run."""

    usages: list[UsageMetadata]
    plan_gen_duration_sec: float
    img_gen_duration_sec: list[float]
    img_eval_duration_sec: list[float]
    evals: list[ImageEvalResponse]
    plan_response: PlanResponse
