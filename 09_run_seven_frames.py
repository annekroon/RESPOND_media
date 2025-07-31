import requests
import pandas as pd
import time
import json
import os
import re
from tqdm import tqdm

# ========== CONFIG ==========
LLM_ENDPOINT = "http://localhost:11434/api/chat"
LLM_MODEL_NAME = "llama3:70b"
SLEEP_BETWEEN_REQUESTS = 1
PROMPT_DIR = "prompts"
TEMPERATURE = 0.0

# ========== FRAME ORDER ==========
FRAME_ORDER = [
    "Foreign influence threat",
    "Systemic institutional corruption",
    "Elite collusion",
    "Politicized investigations",
    "Authoritarian reformism",
    "Judicial and institutional accountability failures",
    "Mobilizing anti-corruption"
]

# ========== LOAD PROMPTS PER FRAME ==========
def load_frame_prompt(index: int, frame_name: str) -> str:
    filename = f"frame_{index}_{frame_name.lower().replace(' ', '_').replace('-', '')}.txt"
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ========== PROMPT CONSTRUCTION ==========
def build_prompt(article_text: str, frame_index: int, frame_name: str) -> str:
    frame_prompt = load_frame_prompt(frame_index, frame_name)
    return f"{frame_prompt}\n\n---\n\nArticle:\n{article_text}"

# ========== CLEANING AND PARSING ==========
def sanitize_double_quotes(json_str):
    return re.sub(r'"\s*"\s*([^"]+)"', r'"\1"', json_str)

def clean_llm_response(content):
    content = content.strip()
    content = re.sub(r"```(?:json)?|```", "", content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
    if match:
        json_str = match.group()
        json_str = sanitize_double_quotes(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ Still couldn't parse extracted JSON: {e}")
            print("🔍 JSON candidate (truncated):")
            print(json_str[:500])
            return None
    else:
        print("⚠️ No JSON array found in LLM output.")
        print("🔍 Raw LLM output (truncated):")
        print(content[:1000])
        return None

# ========== LLM QUERY ==========
def query_frame_llm(article_text: str, frame_index: int, frame_name: str) -> dict:
    prompt = build_prompt(article_text, frame_index, frame_name)
    try:
        response = requests.post(
            LLM_ENDPOINT,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": TEMPERATURE
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "").strip()

        if not content:
            raise ValueError("Empty response from LLM")

        parsed = clean_llm_response(content)
        if not parsed or not isinstance(parsed, list):
            raise ValueError("No valid JSON array found or parsed content is not a list")

        return parsed[0] if parsed else {}

    except Exception as e:
        print(f"❌ Error querying frame '{frame_name}': {e}")
        return {
            "frame": frame_name,
            "rationale": f"⚠️ Error: {str(e)}",
            "confidence": None,
            "evidence": ""
        }

# ========== ANNOTATION LOOP ==========
def annotate_dataframe(df: pd.DataFrame, temp_output_path: str) -> pd.DataFrame:
    for i in range(1, len(FRAME_ORDER) + 1):
        for field in ["name", "rationale", "confidence", "evidence"]:
            col = f"frame_{i}_{field}"
            if col not in df.columns:
                df[col] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Articles"):
        if pd.notna(row.get("frame_1_name")) and row.get("frame_1_name") != "":
            print(f"⏭️ Article {idx} already annotated. Skipping.")
            continue

        print(f"\n🔍 Annotating article {idx}...")
        article_text = row.get("translated_text", "")

        for i, frame_name in enumerate(FRAME_ORDER, 1):
            result = query_frame_llm(article_text, i, frame_name)

            if result.get("frame") and result.get("rationale"):
                confidence = result.get("confidence", "")
                rationale = result.get("rationale", "")
                if confidence != "" and isinstance(confidence, (int, float)) and confidence < 80:
                    rationale += f"\n\n⚠️ Model confidence is only {confidence}%. Please verify carefully."

                df.at[idx, f"frame_{i}_name"] = result.get("frame", "")
                df.at[idx, f"frame_{i}_rationale"] = rationale
                df.at[idx, f"frame_{i}_confidence"] = confidence
                df.at[idx, f"frame_{i}_evidence"] = result.get("evidence", "")

        # Try to save progress
        try:
            os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
            df.to_csv(temp_output_path, index=False)
        except Exception as e:
            print(f"⚠️ Failed to save temp file: {e}")
            fallback = f"fallback_{os.path.basename(temp_output_path)}"
            df.to_csv(fallback, index=False)
            print(f"💾 Temp fallback saved locally as {fallback}")

        print(f"✅ Saved progress after article {idx}.")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return df

# ========== MAIN PROCESSING ==========
def process_files(file_paths):
    for file_path in tqdm(file_paths, desc="Files"):
        if not file_path.endswith(".csv"):
            continue

        print(f"\n📄 Processing file: {file_path}")
        df = pd.read_csv(file_path)
        annotated_path = file_path.replace(".csv", "_llm_annotated.csv")
        temp_output_path = file_path.replace(".csv", "_llm_temp.csv")

        if os.path.exists(annotated_path):
            print("✅ Annotated file already exists. Skipping.")
            continue
        elif os.path.exists(temp_output_path):
            print("🔁 Resuming from temporary file.")
            df = pd.read_csv(temp_output_path)

        df = annotate_dataframe(df, temp_output_path)

        try:
            os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
            df.to_csv(annotated_path, index=False)
        except Exception as e:
            print(f"⚠️ Failed to save final file: {e}")
            fallback = f"fallback_{os.path.basename(annotated_path)}"
            df.to_csv(fallback, index=False)
            print(f"💾 Final fallback saved locally as {fallback}")

        print("\n" + "="*60)
        print(f"✅ DONE: {file_path} → saved to {annotated_path}")
        print("="*60 + "\n")

        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

# ========== FILE LIST ==========
csv_files = [
    "/home/akroon/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/data-deductive-analysis/sample-manual-content-analysis/Bulgaria_Alexander_sample_250.csv",
    "/home/akroon/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/data-deductive-analysis/sample-manual-content-analysis/Italy_Luigia_sample_250.csv",
    "/home/akroon/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/data-deductive-analysis/sample-manual-content-analysis/Netherlands_Assia_sample_250.csv",
    "/home/akroon/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/data-deductive-analysis/sample-manual-content-analysis/United_Kingdom_Elisa_sample_250.csv"
]

if __name__ == "__main__":
    process_files(csv_files)
