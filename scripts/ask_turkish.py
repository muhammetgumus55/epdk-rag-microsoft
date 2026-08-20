"""Run a list of Turkish EPDK-domain questions through the chat model for manual fluency review.

This is a language-quality check only — there is no corpus ingested yet, so the model
answers from parametric memory. Judge the Turkish, not the factual accuracy.

Usage:
    .venv\\Scripts\\python.exe scripts\\ask_turkish.py                       # uses questions.txt
    .venv\\Scripts\\python.exe scripts\\ask_turkish.py path\\to\\questions.txt
"""

import sys
from pathlib import Path

from openai import OpenAI

from verify_foundry import CHAT_ALIAS, discover_endpoint, ensure_loaded, resolve_model_id, strip_thinking

DEFAULT_QUESTIONS = Path(__file__).parent / "turkish_questions.txt"

SYSTEM_PROMPT = (
    "Sen Türkiye enerji mevzuatı konusunda uzman bir asistansın. "
    "Sorulara açık, resmi ve akıcı bir Türkçe ile cevap ver. "
    "Cevaplarını kısa tut: en fazla üç cümle."
)


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Question file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    questions = load_questions(path)

    ensure_loaded(CHAT_ALIAS)
    client = OpenAI(base_url=discover_endpoint(), api_key="not-needed")
    model_id = resolve_model_id(client, CHAT_ALIAS)

    print("=" * 70)
    print(f"TURKISH FLUENCY REVIEW - {model_id}")
    print(f"{len(questions)} questions from {path.name}")
    print("=" * 70)

    for i, question in enumerate(questions, 1):
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_completion_tokens=400,
            temperature=0.7,
        )
        answer = strip_thinking(completion.choices[0].message.content or "")
        print(f"\n[{i}] SORU: {question}")
        print(f"    CEVAP: {answer}")
        print("-" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
