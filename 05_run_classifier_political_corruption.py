from utils.classifier import classify_article
import pandas as pd
from tqdm import tqdm
import requests
from translation import translate_dataframe  # assuming you use this elsewhere

# Define input and output file paths
INPUT_FILE = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_translated_10000.csv"
OUTPUT_FILE = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_translated_10000_with_llm_annotations.csv"

# Load the full dataset
df = pd.read_csv(INPUT_FILE)

# Take a random sample of 100 articles
#df_sample = df.sample(n=20, random_state=42).reset_index(drop=True)

# Results containers
results = {
    "llm_evidence": [],
    "llm_rationale": [],
    "llm_confidence": [],
    "llm_label": []
}

print(f"Processing {len(df)} randomly sampled articles...")

# Apply classifier to each sampled row with tqdm progress bar
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying articles"):
    article_text = row.get("translated_text", "")

    if not isinstance(article_text, str) or len(article_text.strip()) == 0:
        results["llm_evidence"].append("")
        results["llm_rationale"].append("No content")
        results["llm_confidence"].append(None)
        results["llm_label"].append("No")
        continue

    output = classify_article(article_text)

    # Join highlights with semicolons to store in one column
    evidence = "; ".join(output.get("highlights", []))

    results["llm_evidence"].append(evidence)
    results["llm_rationale"].append(output.get("rationale", ""))
    results["llm_confidence"].append(output.get("confidence", ""))
    results["llm_label"].append(output.get("tentative_label", ""))

# Merge results into sampled DataFrame
df["llm_evidence"] = results["llm_evidence"]
df["llm_rationale"] = results["llm_rationale"]
df["llm_confidence"] = results["llm_confidence"]
df["llm_label"] = results["llm_label"]

# Define the columns to keep from the original dataset
columns_to_keep = [
    "combined_text", "translated_text", "uri",
    "country", "dateTime", "source.uri",
    "llm_evidence", "llm_rationale", "llm_confidence", "llm_label"
]

# Filter the DataFrame to keep only the specified columns
df_filtered = df[columns_to_keep]

# Save the annotated sample to CSV
df_filtered.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved annotated sample file to: {OUTPUT_FILE}")
