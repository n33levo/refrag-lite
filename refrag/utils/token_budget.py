"""Token budget accounting utilities."""

from typing import List, Union

import tiktoken
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Global tokenizer cache
_TOKENIZER_CACHE = {}


def get_tokenizer(model_name: str = "gpt-4"):
    """Get tiktoken tokenizer (cached).

    Args:
        model_name: Model name for tiktoken

    Returns:
        Tokenizer

    Example:
        >>> tokenizer = get_tokenizer("gpt-4")
        >>> tokens = tokenizer.encode("Hello world")
    """
    if model_name not in _TOKENIZER_CACHE:
        try:
            _TOKENIZER_CACHE[model_name] = tiktoken.encoding_for_model(model_name)
        except Exception:
            # Fallback to cl100k_base
            _TOKENIZER_CACHE[model_name] = tiktoken.get_encoding("cl100k_base")

    return _TOKENIZER_CACHE[model_name]


def estimate_tokens(
    text: str,
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
) -> int:
    """Estimate number of tokens in text.

    Args:
        text: Input text
        tokenizer: Tokenizer (string name, HF tokenizer, or None for default)

    Returns:
        Estimated token count

    Example:
        >>> count = estimate_tokens("Hello world")
        >>> count > 0
        True
    """
    if text is None or text == "":
        return 0

    # If tokenizer is HF tokenizer
    if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        return len(tokenizer.encode(text, add_special_tokens=False))

    # If tokenizer is string or None, use tiktoken
    tokenizer_name = tokenizer if isinstance(tokenizer, str) else "gpt-4"
    enc = get_tokenizer(tokenizer_name)
    return len(enc.encode(text))


def count_expanded(
    chunks: List[str],
    expanded_indices: List[int],
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
) -> int:
    """Count tokens in expanded chunks.

    Args:
        chunks: All chunks
        expanded_indices: Indices of chunks to expand
        tokenizer: Tokenizer

    Returns:
        Total token count for expanded chunks

    Example:
        >>> chunks = ["First chunk", "Second chunk", "Third chunk"]
        >>> count = count_expanded(chunks, [0, 2])
        >>> count > 0
        True
    """
    total = 0
    for idx in expanded_indices:
        if 0 <= idx < len(chunks):
            total += estimate_tokens(chunks[idx], tokenizer)
    return total


def check_budget(
    current_tokens: int,
    budget: int,
    chunk_to_add: str,
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
) -> bool:
    """Check if adding a chunk would exceed budget.

    Args:
        current_tokens: Current token count
        budget: Token budget
        chunk_to_add: Chunk to potentially add
        tokenizer: Tokenizer

    Returns:
        True if chunk can be added without exceeding budget

    Example:
        >>> can_add = check_budget(100, 150, "New chunk")
        >>> isinstance(can_add, bool)
        True
    """
    chunk_tokens = estimate_tokens(chunk_to_add, tokenizer)
    return current_tokens + chunk_tokens <= budget


def select_chunks_greedy(
    chunks: List[str],
    scores: List[float],
    budget: int,
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
) -> List[int]:
    """Greedily select chunks by score until budget exhausted.

    Args:
        chunks: All chunks
        scores: Score for each chunk (higher is better)
        budget: Token budget
        tokenizer: Tokenizer

    Returns:
        Indices of selected chunks

    Example:
        >>> chunks = ["Short", "A longer chunk here", "Medium"]
        >>> scores = [0.9, 0.8, 0.7]
        >>> selected = select_chunks_greedy(chunks, scores, 100)
        >>> len(selected) > 0
        True
    """
    # Sort by score (descending)
    sorted_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    selected = []
    current_tokens = 0

    for idx in sorted_indices:
        chunk_tokens = estimate_tokens(chunks[idx], tokenizer)
        if current_tokens + chunk_tokens <= budget:
            selected.append(idx)
            current_tokens += chunk_tokens

    return sorted(selected)  # Return in original order


def compute_compression_ratio(
    total_tokens: int,
    expanded_tokens: int,
) -> float:
    """Compute compression ratio.

    Args:
        total_tokens: Total tokens if all chunks expanded
        expanded_tokens: Actual tokens used (with compression)

    Returns:
        Compression ratio (0 = full compression, 1 = no compression)

    Example:
        >>> ratio = compute_compression_ratio(1000, 300)
        >>> ratio
        0.3
    """
    if total_tokens <= 0:
        return 0.0
    return expanded_tokens / total_tokens


def format_token_count(count: int) -> str:
    """Format token count for display.

    Args:
        count: Token count

    Returns:
        Formatted string

    Example:
        >>> format_token_count(1500)
        '1.5K'
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"
