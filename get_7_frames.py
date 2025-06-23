import requests
import re
import pandas as pd
import difflib

# ========= Configuration ==========
LLM_ENDPOINT = "http://localhost:11434/api/chat"
LLM_MODEL_NAME = "llama3:70b"
MAX_FRAMES = 7
SLEEP_SECONDS = 1  # Can be used if throttling needed

# ========= Prompt Builder ==========
def build_multi_frame_prompt(article_text: str) -> str:
    return f"""You are an annotation assistant helping a human coder identify which **corruption narrative frames** are present in a news article.

### Task Definition

Your job is to read the article and identify **all applicable narrative frames** that describe how corruption is being framed.

### Frame Categories

1. **Foreign influence threat** – Corruption is depicted as driven by foreign powers interfering in domestic politics through propaganda, covert funding, or manipulation.
2. **Systemic institutional corruption** – Corruption is described as a deep-rooted, structural problem across political institutions with historical or cultural causes.
3. **Elite collusion** – Focuses on collusive deals between politicians and elites (e.g. business leaders) involving cronyism or insider advantage.
4. **Politicized investigations** – Investigations into corruption are presented as biased, partisan, or politically motivated.
5. **Authoritarian overreach** – Corruption is part of a broader pattern of power consolidation, repression, and dismantling democratic safeguards.
6. **Judicial loopholes enabling corruption** – Legal or institutional loopholes protect corrupt actors or make accountability difficult.
7. **Public outrage and call for reform** – The article highlights mass protests, civil mobilization, or reform efforts sparked by corruption.

---

### Your Task

1. For each frame that clearly applies, provide:
   - A short list of supporting highlights (quotes from the article)
   - A brief explanation (reasoning)
   - A confidence score from 0–100

2. Only include frames where there is sufficient evidence. If no frames apply, say `None`.

---

Article:
{article_text}

Assistant Output Format:

Frame: [Frame Name]  
Highlights:
- [Key sentence 1]
- [Key sentence 2]
...  
Reasoning: [Short explanation]  
Confidence: [0–100]

(Repeat for each frame that applies.)
"""

# ========= LLM Call & Response Parsing ==========
def classify_multiple_frames(article_text: str) -> list:
    prompt = build_multi_frame_prompt(article_text)
    #print("📤 Prompt sent to LLM:\n", prompt)

    try:
        response = requests.post(
            LLM_ENDPOINT,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=90
        )
        response.raise_for_status()
        result = response.json()

        if "message" not in result or result["message"] is None:
            raise ValueError(f"LLM response missing 'message' field. Full response: {result}")

        message = result["message"]
        answer = message.get("content", "").strip()

        if not answer:
            raise ValueError("LLM response 'message.content' is empty.")

        # Print raw LLM output for debugging
        print("🧠 Raw LLM response:\n", answer)

        frames = []
        current_frame = None
        reading_highlights = False
        reading_rationale = False

        for line in answer.splitlines():
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith("frame:") or re.match(r"^\*{1,3}.+?\*{1,3}$", line):
                if current_frame:
                    frames.append(current_frame)
                frame_name = line.split(":", 1)[1].strip() if ":" in line else re.sub(r"[*_`]", "", line).strip()
                current_frame = {
                    "frame": frame_name,
                    "highlights": [],
                    "rationale": "",
                    "confidence": None
                }
                reading_highlights = False
                reading_rationale = False


            elif line.lower().startswith("highlights:"):
                reading_highlights = True
                reading_rationale = False

            elif line.lower().startswith("reasoning:"):
                reading_rationale = True
                reading_highlights = False
                rationale_text = line.split(":", 1)[1].strip()
                if rationale_text:
                    current_frame["rationale"] += rationale_text + " "

            elif line.lower().startswith("confidence:"):
                reading_highlights = False
                reading_rationale = False
                match = re.search(r"\d{1,3}", line)
                if match and current_frame:
                    current_frame["confidence"] = int(match.group(0))

            else:
                if reading_highlights and line.startswith("- "):
                    current_frame["highlights"].append(line[2:].strip())
                elif reading_rationale:
                    current_frame["rationale"] += line + " "

        if current_frame:
            current_frame["rationale"] = current_frame["rationale"].strip()
            frames.append(current_frame)

        return frames

    except Exception as e:
        print(f"❌ Multi-frame classification error: {e}")
        return [{
            "frame": "Error",
            "rationale": str(e),
            "confidence": None,
            "highlights": []
        }]


# ========= Formatting Output ==========
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

    def normalize_frame_name(name: str) -> str:
        return re.sub(r"[*_`]", "", name).strip().lower()

    # Print raw frames for debugging
    for frame in llm_frames:
        print("🧩 Raw frame:", frame)

    # Normalize LLM frame names and store in map
    frame_map = {
        normalize_frame_name(frame['frame']): frame
        for frame in llm_frames
        if frame['frame'].lower() != "error"
    }

    # Debug: show normalized keys
    print("🎯 Normalized LLM Frames:", list(frame_map.keys()))
    print("🎯 Target Fixed Order:", [normalize_frame_name(f) for f in fixed_order])

    for i, frame_name in enumerate(fixed_order, 1):
        key = normalize_frame_name(frame_name)
        closest_match = difflib.get_close_matches(key, frame_map.keys(), n=1, cutoff=0.7)
        frame = frame_map.get(closest_match[0]) if closest_match else None
    
        fields = {
            'name': frame.get('frame', '') if frame else '',
            'rationale': frame.get('rationale', '') if frame else '',
            'confidence': frame.get('confidence', '') if frame else '',
            'evidence': "\n".join(frame.get('highlights', [])) if frame else ''
        }
    
        for suffix, val in fields.items():
            formatted[f"frame_{i}_{suffix}"] = val

    return formatted



# ========= Load and Prepare Data ==========
df = pd.read_csv('~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_translated_2000_with_llm_annotations.csv')
df = df[df['llm_label'] == 'Yes']

df = df.head(25).reset_index(drop=True)

# Ensure all frame columns exist and are string type
for i in range(1, 8):
    for suffix in ['name', 'rationale', 'confidence', 'evidence']:
        col = f'frame_{i}_{suffix}'
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")

# ========= Process all articles ==========
for idx, article_text in df['translated_text'].items():
    if pd.notna(df.at[idx, 'frame_1_name']) and df.at[idx, 'frame_1_name'] != "":
        print(f"⏩ Skipping already annotated article at index {idx}.")
        continue

    print(f"\n🔍 Processing article at index {idx}...\n")

    try:
        frames = classify_multiple_frames(article_text)
        formatted = format_llm_frames_fixed_order(frames)
        print(f"🧾 Formatted frame data for index {idx}: {formatted}")


        for col, val in formatted.items():
            df.at[idx, col] = val
        print("🧪 Written columns:")
        for col in sorted(formatted):
            print(f"{col}: {df.at[idx, col]}")


        print(f"✅ Frames for article {idx} written to DataFrame.\n")

        print("🔎 Detected Frames:")
        for frame in frames:
            if frame["frame"].lower() == "error":
                print(f"❌ Error: {frame['rationale']}")
                continue

            print(f"\n🧩 Frame: {frame['frame']}")
            print(f"Confidence: {frame['confidence']}")
            print(f"Reasoning: {frame['rationale']}")
            print("Highlights:")
            for h in frame.get("highlights", []):
                print(f"- {h}")

    except Exception as e:
        print(f"❌ Error processing article at index {idx}: {e}\n")

# ========= Save updated DataFrame ==========
output_path = '~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_with_7_frames.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved annotated dataframe with up to 7 frames per article to: {output_path}")
