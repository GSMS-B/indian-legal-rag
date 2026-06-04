"""
rag/generator.py
----------------
Handles LLM calls via multi-provider cascade:
  - Primary: Groq (llama-3.3-70b-versatile)
  - Fallback: OpenRouter (auto)
  - query expansion (legal terminology rephrasing)
  - answer generation with section citations

Uses OpenAI-compatible SDK for both providers.
"""

import os
from dotenv import load_dotenv
import openai

# ── load API keys from .env ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── Provider cascade configuration ──────────────────────────────────────────
# Tried in order. First success wins, no sleep between retries.
PROVIDERS = [
    {
        "name": "groq-70b",
        "type": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    {
        "name": "groq-8b",
        "type": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant",
    },
    {
        "name": "openrouter-auto",
        "type": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "openrouter/auto",
    },
]

# ── System instruction for answer generation ─────────────────────────────────
ANSWER_SYSTEM_INSTRUCTION = (
    "You are a legal assistant specializing in Indian criminal law "
    "(BNS 2023, BNSS 2023, BSA 2023). Answer the user's question "
    "using ONLY the legal sections provided in the context. "
    "Rules: "
    "(1) For every factual statement you make, cite the exact section "
    "number and Act name in parentheses -- example: (Section 101, BNS 2023). "
    "(2) If the provided sections do not contain enough information, "
    "say exactly what is not covered. "
    "(3) Never use knowledge outside the provided context. "
    "(4) Never speculate or infer beyond what the text states."
)

# ── Client cache (one per unique base_url) ───────────────────────────────────
_clients: dict[str, openai.OpenAI] = {}


def _get_client(base_url: str, api_key: str) -> openai.OpenAI:
    """Get or create an OpenAI-compatible client for a given base_url."""
    if base_url not in _clients:
        _clients[base_url] = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    return _clients[base_url]


def _call_llm(
    messages: list[dict],
    temperature: float = 0.1,
) -> tuple[str, str]:
    """
    Call LLM with provider cascade. Tries each provider in PROVIDERS order.

    Returns:
        (response_text, provider_name) on success.
        ("Service temporarily unavailable. Please try again.", "none") if all fail.
    """
    for provider in PROVIDERS:
        api_key = os.getenv(provider["api_key_env"], "")
        if not api_key:
            print(f"[generator] Skipping {provider['name']}: no API key set for {provider['api_key_env']}")
            continue

        try:
            client = _get_client(provider["base_url"], api_key)
            response = client.chat.completions.create(
                model=provider["model"],
                messages=messages,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            print(f"[generator] Success via {provider['name']}")
            return text, provider["name"]
        except Exception as e:
            print(f"[generator] {provider['name']} failed: {e}")
            continue

    return "Service temporarily unavailable. Please try again.", "none"


def expand_query(query: str) -> str:
    """
    Rephrase the user's plain-English question using formal Indian legal
    terminology from BNS/BNSS/BSA. Returns the expanded query string.

    On any failure, returns the original query unchanged.
    """
    try:
        prompt = (
            "You are a search query optimizer for an Indian criminal law database. "
            "The database contains sections from three documents covering: "
            "criminal offences and punishments, criminal procedure and police powers, "
            "and rules of evidence in court. "
            "\n\n"
            "Your task: take the user's question and produce an enhanced search query "
            "by appending relevant legal terms that actually appear in Indian criminal law text  (BNS 2023, BNSS 2023, BSA 2023). "
            "\n\n"
            "Examples:\n"
            "Question: What is the punishment for murder?\n"
            "Enhanced: What is the punishment for murder, culpable homicide, death sentence, imprisonment for life, offences affecting the human body\n\n"
            "Question: Can police arrest without a warrant?\n"
            "Enhanced: Can police arrest without a warrant, cognizable offence, arrest without warrant, police officer, investigation powers\n\n"
            "Question: Are WhatsApp messages valid as evidence in court?\n"
            "Enhanced: Are WhatsApp messages valid as evidence in court, electronic records, admissibility, digital evidence, documentary evidence\n\n"
            "Rules: "
            "(1) Keep every word from the original question exactly as written. "
            "(2) After the original question, add a comma and then append 3 to 6 "
            "additional legal terms. "
            "(3) Only add terms that are genuinely relevant to the question. "
            "(4) Do not add section numbers, Act names, or Latin terms. "
            "(5) Return only the enhanced query. No explanation. Under 70 words. "
            "\n\n"
            f"Question: {query}"
        )

        messages = [{"role": "user", "content": prompt}]
        expanded, _ = _call_llm(messages, temperature=0.3)
        return expanded if expanded else query
    except Exception as e:
        print(f"[generator] Query expansion failed: {e}")
        return query


def generate_answer(original_query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a cited legal answer using the retrieved chunks as context.

    Args:
        original_query: The user's original question (not the expanded one).
        retrieved_chunks: List of chunk dicts with 'text' and 'source_label'.

    Returns:
        dict with keys:
            - text (str): The LLM's answer string
            - provider_used (str): Name of the provider that answered
    """
    try:
        # Build context block from retrieved chunks
        context_parts = []
        for chunk in retrieved_chunks:
            source = chunk.get("source_label", "Unknown Source")
            text = chunk.get("text", "")
            context_parts.append(f"[Source: {source}]\n{text}")

        context_block = "\n\n".join(context_parts)

        user_message = (
            f"{context_block}\n\n"
            f"Question: {original_query}"
        )

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_message}
        ]

        answer_text, provider_name = _call_llm(messages, temperature=0.1)
        return {"text": answer_text, "provider_used": provider_name}

    except Exception as e:
        print(f"[generator] Answer generation failed: {e}")
        return {
            "text": (
                "I was unable to generate an answer at this time. "
                "This may be due to an API issue. Please try again in a moment.\n\n"
                f"Error details: {str(e)}"
            ),
            "provider_used": "none",
        }
