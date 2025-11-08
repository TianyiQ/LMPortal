import pydantic
from pydantic import SkipValidation

from .inference import LLMParams, LLMResponse, TaggedModeration
from .messages import Prompt


class LLMCache(pydantic.BaseModel):
    params: SkipValidation[LLMParams]
    prompt: SkipValidation[Prompt]
    responses: list[LLMResponse] | None = None


class LLMCacheModeration(pydantic.BaseModel):
    texts: list[str]
    moderation: list[TaggedModeration] | None = None
