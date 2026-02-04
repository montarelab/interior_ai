from google.genai.types import GenerateContentResponseUsageMetadata
from openai.types.images_response import Usage
from pydantic import BaseModel, Field


class EvalConfigPayload(BaseModel):
    """Evaluation Payload."""

    model_names: list[str] = Field(
        default_factory=lambda: [
            "gpt-image-1.5",
            "gpt-image-1-mini",
            "gemini-2.5-flash-image",
            "gemini-3-pro-image-preview",
        ]
    )
    tasks: list[str] = Field(
        default_factory=lambda: [
            "1_modern_minimalist",
            "2_industrial_loft",
            "3_japandi_zen",
            "4_tech_savvy",
            "5_warm_mediterranean",
        ]
    )
    versions: list[int] = Field(default_factory=lambda: [1, 3])


class StructuredMetric(BaseModel):
    """Metrics response that includes float value and its explanation."""

    value: float = Field(
        ge=0.0,
        le=1.0,
        description="Float value between 0.0 and 1.0. The more the better.",
    )
    explanation: str = Field(description="Explanation of your judgement.")


class ImageJudgeResponse(BaseModel):
    """Response to automatic judge of a generated image."""

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


class UsageMetadata(BaseModel):
    """Usage metadata model that holds token usage information from the LLM requests."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int

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

    usage: UsageMetadata
    img_gen_duration_sec: float
    img_eval_duration_sec: float
    judge: ImageJudgeResponse
