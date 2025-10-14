"""Test utilities."""
import pytest
from refrag.utils.metrics import compute_em_f1, evidence_prec_recall
from refrag.utils.token_budget import estimate_tokens

def test_em_f1():
    em, f1 = compute_em_f1("Paris", ["Paris", "paris"])
    assert em == 1.0
    assert f1 == 1.0

def test_estimate_tokens():
    tokens = estimate_tokens("Hello world")
    assert tokens > 0