from unittest.mock import MagicMock, patch

from app.tracing.langfuse import LangfuseTracer, noop_tracer


def test_noop_tracer_does_not_raise():
    tracer = noop_tracer()
    with tracer.trace("test-operation", metadata={"k": "v"}) as span:
        span.set_output("result")
        span.set_metadata({"k": "v"})
    # Should complete without error


def test_langfuse_tracer_disabled_when_no_keys():
    tracer = LangfuseTracer(public_key="", secret_key="", host="https://cloud.langfuse.com")
    assert not tracer.enabled


def test_langfuse_tracer_enabled_when_keys_present():
    # The wrapper imports `Langfuse` from the `langfuse` module at call time,
    # so patch the source module directly.
    with patch("langfuse.Langfuse") as mock_lf:
        mock_lf.return_value = MagicMock()
        tracer = LangfuseTracer(
            public_key="pk-test", secret_key="sk-test", host="https://cloud.langfuse.com"
        )
    assert tracer.enabled
    mock_lf.assert_called_once()


def test_langfuse_tracer_disabled_trace_yields_noop_span():
    """When disabled, .trace() must still be safe to use as a context manager."""
    tracer = LangfuseTracer(public_key="", secret_key="", host="")
    with tracer.trace("op", metadata={"x": 1}) as span:
        span.set_output("anything")
        span.set_metadata({"k": "v"})


def test_noop_span_set_output_is_safe():
    tracer = noop_tracer()
    with tracer.trace("op") as span:
        span.set_output("anything")
        span.set_metadata({"key": "value"})
