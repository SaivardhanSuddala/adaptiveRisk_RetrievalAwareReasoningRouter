import re


NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")
SUBSTRING_TYPES = {"code_output", "short_answer", "algorithm"}
NUMERIC_TYPES = {"numerical", "complexity"}


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s`\"']+", " ", text)
    return text.strip()


def _extract_numbers(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(text)


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(b_tokens)


def score_answer(predicted: str, ground_truth: str, answer_type: str) -> bool:
    predicted_norm = _normalize(predicted)
    truth_norm = _normalize(ground_truth)

    if not truth_norm:
        return False

    if answer_type in NUMERIC_TYPES:
        predicted_numbers = _extract_numbers(predicted)
        truth_numbers = _extract_numbers(ground_truth)

        if truth_numbers and predicted_numbers:
            return predicted_numbers[-1] == truth_numbers[-1] or truth_norm in predicted_norm

        return truth_norm in predicted_norm

    if answer_type in SUBSTRING_TYPES:
        return truth_norm in predicted_norm

    return truth_norm in predicted_norm or _token_overlap(predicted_norm, truth_norm) >= 0.5
