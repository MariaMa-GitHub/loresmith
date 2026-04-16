from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class _NoopSpan:
    def set_output(self, output: Any) -> None:
        pass

    def set_metadata(self, metadata: dict) -> None:
        pass


class _NoopTracer:
    enabled = False

    @contextmanager
    def trace(self, name: str, metadata: dict | None = None, **kwargs) -> Iterator[_NoopSpan]:
        yield _NoopSpan()

    def flush(self) -> None:
        pass


def noop_tracer() -> _NoopTracer:
    return _NoopTracer()


class _LangfuseSpan:
    """Adapts a Langfuse v4 observation to the tracer's minimal span interface."""

    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def set_output(self, output: Any) -> None:
        self._obs.update(output=output)

    def set_metadata(self, metadata: dict) -> None:
        self._obs.update(metadata=metadata)


class LangfuseTracer:
    """Thin wrapper around the Langfuse v4 SDK.

    Disabled automatically when public_key or secret_key is empty so tests,
    CI, and local runs without credentials never fail. When enabled, each
    ``.trace(name)`` call opens a v4 observation via
    ``start_as_current_observation(as_type="span")`` and closes it on exit.
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        self.enabled = bool(public_key and secret_key)
        self._client = None
        if self.enabled:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=public_key, secret_key=secret_key, host=host
            )

    @contextmanager
    def trace(self, name: str, metadata: dict | None = None, **kwargs):
        if not self.enabled or self._client is None:
            yield _NoopSpan()
            return
        with self._client.start_as_current_observation(
            name=name, as_type="span", metadata=metadata or {},
        ) as obs:
            yield _LangfuseSpan(obs)

    def flush(self) -> None:
        """Flush any buffered events. Called at FastAPI shutdown."""
        if self._client is not None:
            self._client.flush()
