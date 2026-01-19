#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional

import requests

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = (
    "Ты профессиональный корректор. Исправь орфографию и пунктуацию, "
    "сохраняя смысл и стиль. Верни только исправленный текст без пояснений."
)


def read_stdin() -> str:
    return sys.stdin.read().strip()


def build_payload(text: str, model: str, temperature: float) -> dict:
    return {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }


def call_openai(text: str, api_key: str, model: str, temperature: float) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = build_payload(text, model, temperature)
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка орфографии и пунктуации через ChatGPT (OpenAI API)."
    )
    parser.add_argument(
        "-t",
        "--text",
        help="Текст для проверки. Если не указан, читается из stdin.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Модель OpenAI (по умолчанию gpt-4.1-mini или OPENAI_MODEL).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Температура генерации (по умолчанию 0.0).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.text or read_stdin()
    if not text:
        print("Ошибка: нет текста для проверки.", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Ошибка: переменная окружения OPENAI_API_KEY не задана.", file=sys.stderr)
        return 1

    try:
        corrected = call_openai(text, api_key, args.model, args.temperature)
    except requests.HTTPError as exc:
        print(f"Ошибка HTTP: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Ошибка запроса: {exc}", file=sys.stderr)
        return 1
    except (KeyError, IndexError) as exc:
        print(f"Ошибка ответа API: {exc}", file=sys.stderr)
        return 1

    print(corrected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
