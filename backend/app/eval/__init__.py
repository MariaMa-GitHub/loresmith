from app.eval.metrics import exact_match, has_inline_citation, token_f1, token_recall
from app.eval.report import build_report, default_report_path, write_report

__all__ = [
    "build_report",
    "default_report_path",
    "exact_match",
    "has_inline_citation",
    "token_f1",
    "token_recall",
    "write_report",
]


def __getattr__(name: str):
    if name in {"EvalExample", "load_dataset", "run_eval", "run_pipeline_eval"}:
        from app.eval.runner import EvalExample, load_dataset, run_eval, run_pipeline_eval

        exports = {
            "EvalExample": EvalExample,
            "load_dataset": load_dataset,
            "run_eval": run_eval,
            "run_pipeline_eval": run_pipeline_eval,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
