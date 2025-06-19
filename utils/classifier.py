import requests
import re

# ========= Ollama Configuration ==========
LLM_ENDPOINT = "http://localhost:11434/api/chat"
LLM_MODEL_NAME = "llama3:70b"

# ========= Prompt Builder ==========
def build_detailed_prompt(article_text: str) -> str:
    return f"""You are an annotation assistant helping a human coder classify whether a news article is **primarily about political corruption**.

### Strict Definition

**Political corruption** involves public officials misusing political power for personal or political gain.

**Key criteria:**
- It must involve **public officials** in political decision-making roles:
  - Examples: ministers, members of parliament, presidents, judges, local council members
  - Exclude: police chiefs, military officials, CEOs of state companies (unless also acting politically)

**Common forms of political corruption**:
- Bribery or kickbacks for political influence
- Embezzlement or theft of public funds by officials
- Nepotism and cronyism in public appointments
- Misuse of authority (e.g. election fraud, shielding allies)

**Important:**  
- Articles **should be labeled** as political corruption if they focus on accusations, charges, or suspicions of corruption by political officials—even if not yet proven.
- Do **not** label articles that focus solely on general crime, private sector fraud, or misconduct by non-political actors.

---

### Your Task

1. Identify full sentences that directly point to political corruption.
2. Make a careful judgment on whether this is the central focus.
3. Return your response in this format:

Highlights:
- [Key sentence 1]
- [Key sentence 2]
...

Tentative Label: Yes / Mentioned but not central / No / Unsure  
Reasoning: [Short explanation]  
Confidence: [0–100]

---

Article:
{article_text}

Assistant Output:"""

# ========= LLM Classification Call ==========
def classify_article(article_text: str) -> dict:
    prompt = build_detailed_prompt(article_text)

    try:
        response = requests.post(
            LLM_ENDPOINT,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("message", {}).get("content", "").strip()

        # ========== Parsing ==========
        highlights = []
        tentative_label = "Unclear"
        rationale_lines = []
        confidence = None

        lines = answer.splitlines()
        reading_highlights = False
        reading_rationale = False

        for line in lines:
            line_strip = line.strip()

            if line_strip.lower().startswith("highlights:"):
                reading_highlights = True
                reading_rationale = False
                continue
            elif line_strip.lower().startswith("tentative label:"):
                reading_highlights = False
                reading_rationale = False
                val = line_strip.split(":", 1)[1].strip().capitalize()
                if val in ["Yes", "No", "Unsure", "Mentioned but not central"]:
                    tentative_label = val
                continue
            elif line_strip.lower().startswith("reasoning:"):
                reading_highlights = False
                reading_rationale = True
                rationale_lines.append(line_strip.split(":", 1)[1].strip())
                continue
            elif line_strip.lower().startswith("confidence:"):
                reading_highlights = False
                reading_rationale = False
                match = re.search(r"\d{1,3}", line_strip)
                if match:
                    confidence = int(match.group(0))
                continue

            if reading_highlights and line_strip.startswith("- "):
                highlights.append(line_strip[2:].strip())
            elif reading_rationale and line_strip:
                rationale_lines.append(line_strip)

        rationale = " ".join(rationale_lines).strip()

        return {
            "tentative_label": tentative_label,
            "rationale": rationale,
            "confidence": confidence,
            "highlights": highlights
        }

    except Exception as e:
        print(f"❌ Classification error: {e}")
        return {
            "tentative_label": "Error",
            "rationale": str(e),
            "confidence": None,
            "highlights": []
        }
