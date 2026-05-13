"""Shared strict-model base for the scenario spec.

`StrictModel` is the base every spec dataclass-equivalent inherits from.
`extra="forbid"` makes unknown YAML keys fail loudly (typos die before
any run launches); `frozen=True` keeps spec instances immutable so they
can be passed around safely.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
