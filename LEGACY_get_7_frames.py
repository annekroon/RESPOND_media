## FIRST TRY AT CLASSIFYING FRAMES (NOT SUCCESFUL)


import requests
import pandas as pd
import time
import json
import os

# ========== CONFIG ==========
LLM_ENDPOINT = "http://localhost:11434/api/chat"
LLM_MODEL_NAME = "llama3:70b"
TEMP_OUTPUT_PATH = "annotated_temp_output.csv"
FINAL_OUTPUT_PATH = "news_sample_annotated.csv"
SLEEP_BETWEEN_REQUESTS = 1

# ========== FRAMES & PROMPTS ==========
FRAME_ORDER = [
    "Foreign influence threat",
    "Systemic institutional corruption",
    "Elite collusion",
    "Politicized investigations",
    "Authoritarian reformism",
    "Judicial and institutional accountability failures",
    "Mobilizing anti-corruption"
]

FRAME_PROMPT = """
You are an annotation assistant helping a human coder identify which corruption narrative frames are present in a news article. An article may contain multiple frames or none.

### Frame Definitions:

1. **Foreign influence threat**: Identify any sentences or phrases that frame political corruption as an external attack by foreign actors. Look for references to external meddling, covert financing from abroad, secret deals with foreign entities, propaganda or undue influence by external powers.

2. **Systemic institutional corruption**: Find any passages that describe corruption as a deep‐rooted, system‐wide problem—built into institutions, laws, or culture—and not just one‐off wrongdoing. Look for terms like “endemic,” “deep-rooted,” “fragile institutions,” or metaphors like “weed” or “cancer.”

3. **Elite collusion**: Mark any passages that describe secretive alliances among powerful elites (e.g., businessmen and politicians)—such as backroom deals, undisclosed financing, or informal networks rigging policy in favor of those already in power.

4. **Politicized investigations**: Identify any passages that either:
   1. Depict corruption investigations as partisan tools or “witch hunts” (look for claims of bias, factional motives, selective enforcement),  
   OR  
   2. Show accused politicians publicly denying the allegations—portraying themselves as fair, law‐abiding citizens and claiming the probe is politically motivated.

5. **Authoritarian reformism**: Identify any passages that either:
   1. Describe reforms or institutional changes used to consolidate power, weaken democratic checks and balances, or target political opponents,  
   OR  
   2. Show politicians accusing other politicians or institutions of corruption as a campaign strategy or to gain electoral advantage.

6. **Judicial and institutional accountability failures**: Identify any passages that either:
   1. Describe how legal frameworks are manipulated (e.g., ambiguous laws, loopholes, selective enforcement) in ways that let corruption persist,  
   OR  
   2. Criticize anti-corruption laws, promised reforms, or public pledges as failures—empty rhetoric, broken promises, or poorly implemented measures.

7. **Mobilizing anti-corruption**: Identify any passages that either:
   1. Describe grassroots protests or elite demands calling for action against corruption (e.g., demonstrations, petitions, political speeches urging reform),  
   OR  
   2. Describe real institutional responses to corruption (e.g., new anti-corruption laws, court cases against officials, restructuring of oversight bodies).

---

### Output Format:

Return ONLY a JSON list like this:

[
  {{
    "frame": "Frame Name",
    "highlights": ["Exact sentence", "..."],
    "rationale": "Short explanation of why the frame applies",
    "confidence": 85
  }},
  ...
]

Only include frames that are clearly evidenced.
"""

# ========== LLM REQUEST ==========
def build_prompt(article_text: str) -> str:
    return f"{FRAME_PROMPT}\n\n---\n\nArticle:\n{article_text}"

def query_llm(article_text: str) -> list:
    try:
        response = requests.post(
            LLM_ENDPOINT,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": build_prompt(article_text)}],
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "").strip()
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        json_str = content[json_start:json_end]
        return json.loads(json_str)

    except Exception as e:
        print(f"❌ Error querying LLM: {e}")
        return [{
            "frame": "Error",
            "rationale": str(e),
            "confidence": None,
            "highlights": []
        }]

# ========== RESULT FORMAT ==========
def format_llm_output(llm_frames: list) -> dict:
    formatted = {}
    frame_map = {
        frame["frame"].strip().lower(): frame
        for frame in llm_frames if frame.get("frame", "").lower() != "error"
    }

    for i, frame_name in enumerate(FRAME_ORDER, 1):
        key = frame_name.lower()
        match = frame_map.get(key, {})
        formatted[f"frame_{i}_name"] = match.get("frame", "")
        formatted[f"frame_{i}_rationale"] = match.get("rationale", "")
        formatted[f"frame_{i}_confidence"] = match.get("confidence", "")
        formatted[f"frame_{i}_evidence"] = "\n".join(match.get("highlights", []))

    return formatted

# ========== ANNOTATION LOOP ==========
def annotate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(1, len(FRAME_ORDER) + 1):
        for field in ["name", "rationale", "confidence", "evidence"]:
            col = f"frame_{i}_{field}"
            if col not in df.columns:
                df[col] = ""

    for idx, row in df.iterrows():
        if pd.notna(row.get("frame_1_name")) and row.get("frame_1_name") != "":
            print(f"⏭️ Article {idx} already annotated. Skipping.")
            continue

        print(f"\n🔍 Annotating article {idx}...\n")
        article_text = row.get("translated_text", "")
        frames = query_llm(article_text)
        formatted = format_llm_output(frames)

        for col, val in formatted.items():
            df.at[idx, col] = val

        df.to_csv(TEMP_OUTPUT_PATH, index=False)
        print(f"✅ Saved progress after article {idx}.")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return df

# ========== MAIN ==========
if __name__ == "__main__":
    INPUT_PATH = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_translated_10000_with_llm_annotations.csv"
    df = pd.read_csv(INPUT_PATH)

    # Optional filtering
    df = df[df.get("llm_label", "") == "Yes"].sample(n=10, random_state=42).reset_index(drop=True)

    df = annotate_dataframe(df)

    # ========= Save updated DataFrame ==========
    output_path = os.path.expanduser('~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_with_7_frames.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved annotated dataframe with up to 7 frames per article to: {output_path}")

