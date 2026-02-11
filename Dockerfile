FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY prompts ./prompts

RUN uv sync --frozen

CMD ["uv","run", "python", "-m", "src.main"]