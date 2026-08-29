from collections import Counter
from typing import Iterable
import re

from groq import BadRequestError

from configs import MODEL_NAME, REACT_MAX_STEPS, SELF_CONSISTENCY_SAMPLES, get_client_for_model
from retrieval import retrieve_documents


SYSTEM_PROMPT = (
    "You prioritize correctness, evidence, and concise technical answers. "
    "Use plain ASCII text only. Do not include hidden reasoning, chain-of-thought, "
    "analysis notes, or think tags."
)

REACT_SYSTEM = (
    "You may reason and take actions. At each step, output either:\n"
    "Thought: <reasoning>\nAction: retrieve[<search query>]\n"
    "or, once ready:\nThought: <reasoning>\nFinal Answer: <answer>\n"
    "Only one action is available: retrieve[<search query>], which looks up "
    "relevant technical context. Use plain ASCII text only."
)


def _context_block(contexts: Iterable[str]) -> str:
    context_text = "\n\n".join(
        f"[{index + 1}] {context}"
        for index, context in enumerate(contexts)
    )

    return context_text or "No retrieved context was available."


def clean_answer(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()

    if text.lstrip().startswith("<think>"):
        answer_index = text.rfind("Answer:")

        if answer_index != -1:
            return text[answer_index:].strip()

    return text.strip()


def _split_final_answer(text: str) -> tuple[str, str]:
    marker = re.search(r"final answer\s*:", text, re.IGNORECASE)

    if marker:
        return text[:marker.start()].strip(), text[marker.end():].strip()

    return text, text


def _split_reflection(text: str) -> tuple[str, str, str]:
    draft_m = re.search(r"draft answer\s*:", text, re.IGNORECASE)
    critique_m = re.search(r"self-critique\s*:", text, re.IGNORECASE)
    revised_m = re.search(r"revised final answer\s*:", text, re.IGNORECASE)

    draft = text[draft_m.end():critique_m.start()].strip() if draft_m and critique_m else ""
    critique = text[critique_m.end():revised_m.start()].strip() if critique_m and revised_m else ""
    revised = text[revised_m.end():].strip() if revised_m else ""

    return draft, critique, revised


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _complete(client, model: str, system: str, user: str, temperature: float, max_tokens: int) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except BadRequestError as error:
        if "tool_use_failed" not in str(error):
            raise

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system + " Never call a tool or function. Respond only in plain text.",
                },
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    usage = getattr(response, "usage", None)

    return {
        "text": clean_answer(response.choices[0].message.content),
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }


def generate_direct(client, model: str, query: str, contexts: list[str]) -> dict:
    prompt = f"""Answer the following question directly and concisely.
Do not show your reasoning steps.

Context:
{_context_block(contexts)}

Question:
{query}

Answer:"""

    result = _complete(client, model, SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=400)

    return {
        "answer": result["text"],
        "trace": None,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


def generate_cot(client, model: str, query: str, contexts: list[str]) -> dict:
    prompt = f"""Solve the following question. Think through the problem step by step
before giving your final answer. Show all intermediate reasoning explicitly.

Context:
{_context_block(contexts)}

Question:
{query}

Let's think step by step:"""

    result = _complete(client, model, SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=800)
    trace, final = _split_final_answer(result["text"])

    return {
        "answer": final,
        "trace": trace,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


def generate_reflection(client, model: str, query: str, contexts: list[str]) -> dict:
    prompt = f"""First, answer the question. Then critique your own answer for
factual errors, logical gaps, or unstated assumptions. Finally, provide a
revised final answer based on your critique.

Context:
{_context_block(contexts)}

Question:
{query}

Draft Answer:
Self-Critique:
Revised Final Answer:"""

    result = _complete(client, model, SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=900)
    draft, critique, revised = _split_reflection(result["text"])

    return {
        "answer": revised or result["text"],
        "trace": {"draft": draft, "critique": critique},
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


def generate_self_consistency(client, model: str, query: str, contexts: list[str], samples: int | None = None) -> dict:
    samples = samples or SELF_CONSISTENCY_SAMPLES
    votes = []
    answers = []
    input_tokens = 0
    output_tokens = 0

    prompt = f"""Solve the following question step by step. Show your reasoning,
then give a clearly marked final answer.

Context:
{_context_block(contexts)}

Question:
{query}"""

    for _ in range(samples):
        result = _complete(client, model, SYSTEM_PROMPT, prompt, temperature=0.7, max_tokens=700)
        input_tokens += result["input_tokens"]
        output_tokens += result["output_tokens"]

        _, final = _split_final_answer(result["text"])
        votes.append(_normalize_answer(final))
        answers.append(final)

    counts = Counter(votes)
    majority_vote, majority_count = counts.most_common(1)[0]
    agreement = round(majority_count / len(votes), 4)
    majority_answer = next(answer for answer, vote in zip(answers, votes) if vote == majority_vote)

    return {
        "answer": majority_answer,
        "trace": answers,
        "answer_stability": agreement,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def generate_react(client, model: str, query: str, contexts: list[str], max_steps: int | None = None) -> dict:
    max_steps = max_steps or REACT_MAX_STEPS
    transcript = f"Question:\n{query}\n\nInitial context:\n{_context_block(contexts)}\n"
    steps = []
    input_tokens = 0
    output_tokens = 0

    for _ in range(max_steps):
        result = _complete(client, model, REACT_SYSTEM, transcript, temperature=0.2, max_tokens=400)
        input_tokens += result["input_tokens"]
        output_tokens += result["output_tokens"]
        raw = result["text"]
        steps.append(raw)

        final_match = re.search(r"final answer\s*:(.*)", raw, re.IGNORECASE | re.DOTALL)
        if final_match:
            return {
                "answer": final_match.group(1).strip(),
                "trace": steps,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        action_match = re.search(r"action\s*:\s*retrieve\[(.*?)\]", raw, re.IGNORECASE)
        if action_match:
            observation_docs = retrieve_documents(action_match.group(1).strip(), top_k=3)
            transcript += f"\n{raw}\nObservation: {_context_block(observation_docs)}\n"
            continue

        return {
            "answer": raw,
            "trace": steps,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    return {
        "answer": steps[-1] if steps else "",
        "trace": steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


STRATEGY_FUNCTIONS = {
    "direct": generate_direct,
    "cot": generate_cot,
    "reflection": generate_reflection,
    "self_consistency": generate_self_consistency,
    "react": generate_react,
}


def generate_answer(query: str, contexts: list[str] | None = None, strategy: str = "cot", model: str | None = None) -> dict:
    contexts = contexts or []
    model = model or MODEL_NAME
    client, native_model = get_client_for_model(model)
    func = STRATEGY_FUNCTIONS.get(strategy, generate_cot)
    result = func(client, native_model, query, contexts)

    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)

    return {
        "model": model,
        "strategy": strategy,
        "answer": result["answer"],
        "trace": result.get("trace"),
        "answer_stability": result.get("answer_stability"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
