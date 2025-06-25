import requests
import pandas as pd
import time
import os

# ========= Configuration ==========
LLM_ENDPOINT = "http://localhost:11434/api/chat"
LLM_MODEL_NAME = "llama3:70b"
MAX_FRAMES = 7
SLEEP_SECONDS = 1
TEMP_OUTPUT_PATH = 'annotated_temp_output.csv'

# ========= Prompt Builder ==========
def build_multi_frame_prompt(article_text: str) -> str:
    return f"""You are an annotation assistant helping a human coder identify which **corruption narrative frames** are present in a news article.

### Task Definition

Your job is to read the article and identify **all applicable narrative frames** that describe how corruption is being framed.

### Frame Categories

1. "Foreign influence threat" – Corruption is depicted as driven by foreign powers interfering in domestic politics through propaganda, covert funding, or manipulation.
2. "Systemic institutional corruption" – Corruption is described as a deep-rooted, structural problem across political institutions with historical or cultural causes.
3. "Elite collusion" – Focuses on collusive deals between politicians and elites (e.g. business leaders) involving cronyism or insider advantage.
4. "Politicized investigations" – Investigations into corruption are presented as biased, partisan, or politically motivated.
5. "Authoritarian overreach" – Corruption is part of a broader pattern of power consolidation, repression, and dismantling democratic safeguards.
6. "Judicial loopholes enabling corruption" – Legal or institutional loopholes protect corrupt actors or make accountability difficult.
7. "Public outrage and call for reform" – The article highlights mass protests, civil mobilization, or reform efforts sparked by corruption.

---

### Output Format (IMPORTANT)

Return ONLY a JSON list like the following:

[
  {{
    "frame": "Frame Name",
    "highlights": ["Quote 1", "Quote 2"],
    "rationale": "Short explanation",
    "confidence": 87
  }},
  ...
]

- Use only the frame names from the list above.
- Do NOT invent or paraphrase frame titles.
- Only include frames with clear evidence.

---

Article:
{article_text}
"""

# ========= LLM JSON Call ==========
def classify_multiple_frames(article_text: str) -> list:
    prompt = build_multi_frame_prompt(article_text)

    try:
        response = requests.post(
            LLM_ENDPOINT,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        if "message" not in result or result["message"] is None:
            raise ValueError(f"LLM response missing 'message' field. Full response: {result}")

        content = result["message"].get("content", "").strip()
        if not content:
            raise ValueError("Empty LLM output.")

        # Evaluate response safely
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        json_str = content[json_start:json_end]

        import json
        frames = json.loads(json_str)

        return frames

    except Exception as e:
        print(f"❌ LLM classification error: {e}")
        return [{
            "frame": "Error",
            "rationale": str(e),
            "confidence": None,
            "highlights": []
        }]

# ========= Output Formatter ==========
def format_llm_frames_fixed_order(llm_frames: list) -> dict:
    formatted = {}

    fixed_order = [
        "Foreign influence threat",
        "Systemic institutional corruption",
        "Elite collusion",
        "Politicized investigations",
        "Authoritarian overreach",
        "Judicial loopholes enabling corruption",
        "Public outrage and call for reform",
    ]

    def normalize(name: str) -> str:
        return name.lower().strip()

    # Build map from normalized name
    frame_map = {
        normalize(f["frame"]): f for f in llm_frames if f.get("frame") and f["frame"].lower() != "error"
    }

    for i, frame_name in enumerate(fixed_order, 1):
        key = normalize(frame_name)
        matched_frame = frame_map.get(key)

        if not matched_frame:
            print(f"⚠️ Frame '{frame_name}' not found in LLM output.")
        
        fields = {
            "name": matched_frame.get("frame", "") if matched_frame else "",
            "rationale": matched_frame.get("rationale", "") if matched_frame else "",
            "confidence": matched_frame.get("confidence", "") if matched_frame else "",
            "evidence": "\n".join(matched_frame.get("highlights", [])) if matched_frame else ""
        }

        for suffix, val in fields.items():
            formatted[f"frame_{i}_{suffix}"] = val

    return formatted

# ========= Load Data ==========
df = pd.read_csv('~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_translated_2000_with_llm_annotations.csv')
# First 10 articles where llm_label == 'Yes'
df = df[df['llm_label'] == 'Yes'].head(50).reset_index(drop=True)

# Ensure frame columns exist
for i in range(1, 8):
    for suffix in ['name', 'rationale', 'confidence', 'evidence']:
        col = f'frame_{i}_{suffix}'
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")

# ========= Annotate Articles ==========
for idx, article_text in df['translated_text'].items():
    if pd.notna(df.at[idx, 'frame_1_name']) and df.at[idx, 'frame_1_name'] != "":
        print(f"⏩ Skipping already annotated article at index {idx}.")
        continue

    print(f"\n🔍 Processing article at index {idx}...\n")

    try:
        frames = classify_multiple_frames(article_text)
        formatted = format_llm_frames_fixed_order(frames)

        for col, val in formatted.items():
            df.at[idx, col] = val

        print(f"✅ Frames for article {idx} saved.\n")

        # Tussentijdse opslag
        df.to_csv(TEMP_OUTPUT_PATH, index=False)
        print(f"💾 Tussentijds opgeslagen in: {TEMP_OUTPUT_PATH}")

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print(f"❌ Error at index {idx}: {e}")

# ========= Save Final Output ==========
output_path = '~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_with_7_frames.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Final DataFrame saved to: {output_path}")