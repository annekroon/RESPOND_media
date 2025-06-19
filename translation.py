# translate.py

import os
import time
import pandas as pd
import requests
from typing import List
from tqdm import tqdm

# Config / Constants

COUNTRY_TO_LANG = {
    "Bulgaria": "bg",
    "Italy": "it",
    "Netherlands": "nl",
    "United_Kingdom": "en"
}

MAX_CHUNK_SIZE = 1500
MIN_TRANSLATION_RATIO = 0.7

# ========== Utils ==========

def is_probably_same_language(original: str, translation: str, threshold=0.3) -> bool:
    non_ascii_chars = sum(1 for c in translation if ord(c) > 127)
    return non_ascii_chars / max(len(translation), 1) > threshold

def split_text_into_chunks(text: str, max_chunk_size=MAX_CHUNK_SIZE) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            while len(para) > max_chunk_size:
                chunks.append(para[:max_chunk_size].strip())
                para = para[max_chunk_size:]
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def translation_is_too_short(original: str, translated: str, threshold=MIN_TRANSLATION_RATIO) -> bool:
    return len(translated) < threshold * len(original)

# ========== Gemma3 Translation ==========

def translate_chunk_with_gemma3(chunk_text: str, source_lang: str) -> str:
    system_prompt = (
        f"You are a translation assistant. Translate the following {source_lang} text into English. "
        "Translate the entire text exactly. Do NOT shorten, paraphrase, or summarize. "
        "Output ONLY the translated text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk_text}
    ]

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "zongwei/gemma3-translator:4b",
                "messages": messages,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        raw_translation = result.get("message", {}).get("content", "").strip()

        # Clean common leading phrases
        if raw_translation.lower().startswith("here’s the translation"):
            parts = raw_translation.split("\n\n", 1)
            if len(parts) == 2:
                raw_translation = parts[1].strip()

        # Detect untranslated chunk
        if source_lang != "en" and is_probably_same_language(chunk_text, raw_translation):
            print("⚠️ Suspected untranslated chunk — marking empty")
            return ""

        return raw_translation

    except Exception as e:
        print(f"❌ Translation error (Gemma3): {e}")
        return ""

def translate_article_with_chunking(text: str, lang: str) -> str:
    if lang == "en":
        return text

    chunks = split_text_into_chunks(text)
    translated_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)} (chars: {len(chunk)})")
        translated_chunk = translate_chunk_with_gemma3(chunk, source_lang=lang)

        if not translated_chunk or translation_is_too_short(chunk, translated_chunk):
            print(f"❌ Failed to translate chunk {i+1} properly.")
            translated_chunk = "[Translation Failed]"

        translated_chunks.append(translated_chunk)

    return "\n\n".join(translated_chunks)

# ========== Main Translation Function ==========

def translate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🌍 Starting translation for multilingual dataset using 'combined_text'")

    df = df.copy()
    input_column = "combined_text"
    output_column = "translated_text"

    if input_column not in df.columns:
        raise ValueError(f"❌ Expected column '{input_column}' not found in dataframe.")
    if "country" not in df.columns:
        raise ValueError("❌ Expected column 'country' not found in dataframe.")

    def translate_row(row):
        country = row["country"]
        lang = COUNTRY_TO_LANG.get(country, "en")
        text = str(row[input_column])

        if lang == "en":
            return text
        return translate_article_with_chunking(text, lang)

    tqdm.pandas(desc="🔁 Translating")
    df[output_column] = df.progress_apply(translate_row, axis=1)

    return df


