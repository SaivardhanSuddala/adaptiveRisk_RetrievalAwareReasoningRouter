import re


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def split_claims(answer: str) -> list[str]:
    return [claim.strip() for claim in SENTENCE_SPLIT.split(answer) if claim.strip()]


def _claim_overlap(claim: str, context_tokens: set[str]) -> float:
    claim_tokens = _tokenize(claim)

    if not claim_tokens:
        return 1.0

    return len(claim_tokens & context_tokens) / len(claim_tokens)


def check_groundedness(answer: str, contexts: list[str], threshold: float = 0.3) -> dict:
    context_tokens = set()
    for context in contexts:
        context_tokens |= _tokenize(context)

    claims = split_claims(answer)
    flagged = []

    for claim in claims:
        overlap = _claim_overlap(claim, context_tokens)
        if overlap < threshold:
            flagged.append({"claim": claim, "overlap": round(overlap, 4)})

    total = len(claims)
    flagged_count = len(flagged)
    groundedness_rate = round(1 - (flagged_count / total), 4) if total else 1.0

    return {
        "claims": total,
        "flagged": flagged,
        "flagged_count": flagged_count,
        "groundedness_rate": groundedness_rate,
    }
