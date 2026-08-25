"""Central configuration.

Every knob the project has lives here, sourced from environment variables with
sane local defaults. The important idea: `base_url` is what lets the same agent
code run against Ollama on a laptop and vLLM on a Kaggle T4. Nothing downstream
of this module knows or cares which one it is talking to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
CACHE_DIR = DATA_DIR / "cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="AQ_",
        extra="ignore",
    )

    # ── LLM backend ────────────────────────────────────────────────
    # Ollama and vLLM both speak the OpenAI protocol, so switching tiers
    # is a URL change, not a code change.
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # Ollama ignores it; vLLM/OpenAI need something
    backend: Literal["ollama", "vllm", "openai"] = "ollama"

    # ── Model registry, by role ────────────────────────────────────
    # Small models are fine for classification and review. SQL generation
    # is where quality actually matters, so it gets the biggest model we
    # can afford on the current tier.
    model_lead: str = "qwen2.5-coder:3b"
    model_sql: str = "qwen2.5-coder:3b"
    model_critic: str = "qwen2.5-coder:3b"
    model_analyst: str = "qwen2.5-coder:3b"

    temperature: float = 0.0
    request_timeout: float = 180.0

    # ── Database ───────────────────────────────────────────────────
    db_url: str = Field(default_factory=lambda: f"sqlite:///{DB_DIR / 'demo.db'}")

    # ── Safety rails ───────────────────────────────────────────────
    # Enforced by parsing the SQL, not by asking the model nicely.
    max_rows: int = 500
    query_timeout_s: int = 30

    # ── Agent behaviour ────────────────────────────────────────────
    max_repair_attempts: int = 3
    critic_count: int = 3
    critic_votes_needed: int = 2

    # ── Caching ────────────────────────────────────────────────────
    # Makes re-running an evaluation sweep free, which matters a lot
    # when the GPU budget is 30 hours a week.
    cache_enabled: bool = True

    def model_for(self, role: str) -> str:
        """Look up which model a given agent role should use."""
        return {
            "lead": self.model_lead,
            "sql": self.model_sql,
            "critic": self.model_critic,
            "analyst": self.model_analyst,
        }.get(role, self.model_sql)


settings = Settings()

for _d in (DATA_DIR, DB_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
