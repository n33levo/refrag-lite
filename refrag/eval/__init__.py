"""Evaluation components."""
from refrag.eval.qa_eval import evaluate_qa
from refrag.eval.speed_eval import evaluate_speed
from refrag.eval.report import generate_report

__all__ = ["evaluate_qa", "evaluate_speed", "generate_report"]
