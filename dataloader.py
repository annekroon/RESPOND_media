import os
import pandas as pd
from typing import List, Dict
from config import ANNOTATION_PATH, ANNOTATION_FILE, VALID_CORRUPTION_LABELS, SELECTED_COUNTRIES

def load_selected_outlets(outlet_dir: str, countries: List[str]) -> Dict[str, List[str]]:
    selected_outlets = {}
    for country in countries:
        outlet_file = os.path.join(outlet_dir, f"{country}_outlets_selection.txt")
        if os.path.exists(outlet_file):
            with open(outlet_file, 'r', encoding='utf-8') as f:
                outlets = [line.strip().lower() for line in f if line.strip()]
                selected_outlets[country] = outlets
        else:
            print(f"⚠️ No outlet file for {country}: {outlet_file}")
            selected_outlets[country] = []  # No outlets means exclude all
    return selected_outlets

def load_and_prepare_data(news_folder: str, countries: List[str], outlet_dir: str) -> pd.DataFrame:
    news_folder = os.path.expanduser(news_folder)
    outlet_dir = os.path.expanduser(outlet_dir)
    all_data = []

    selected_outlets = load_selected_outlets(outlet_dir, countries)

    for country in countries:
        file_path = os.path.join(news_folder, f"{country}_news.csv")
        print(f"🔍 Looking for: {file_path}")
        
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
            df = pd.read_csv(file_path)
            df = df.drop_duplicates(subset=["title", "body"])

            # Outlet filtering
            df["source.uri"] = df["source.uri"].astype(str).str.strip().str.lower()
            allowed_outlets = selected_outlets.get(country, [])
            if allowed_outlets:
                df = df[df["source.uri"].isin(allowed_outlets)]
                print(f"📰 {country}: {len(df)} articles after outlet filtering")
            else:
                print(f"⚠️ No selected outlets for {country}, skipping all rows")
                df = df.iloc[0:0]  # Empty DataFrame

            if not df.empty:
                df["country"] = country
                df["combined_text"] = df["title"] + "\n" + df["body"]
                all_data.append(df)
        else:
            print(f"❌ File not found: {file_path}")

    if not all_data:
        raise ValueError(f"❌ No valid data after outlet filtering for countries: {countries}")
    
    return pd.concat(all_data, ignore_index=True)

def balanced_sample(df: pd.DataFrame, total_samples: int, countries: List[str]) -> pd.DataFrame:
    df = df.copy()
    df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')  # convert to datetime, safely
    
    # Extract year and month for stratification
    df['year'] = df['dateTime'].dt.year
    df['month'] = df['dateTime'].dt.month

    per_country = total_samples // len(countries)
    sampled_dfs = []

    for country in countries:
        country_df = df[df['country'] == country]

        # Number of samples to draw for this country
        n_samples = per_country

        # Group by year and month to stratify
        groups = country_df.groupby(['year', 'month'])
        
        # Calculate number of samples per group proportional to group size
        group_sizes = groups.size()
        group_proportions = group_sizes / group_sizes.sum()
        group_samples = (group_proportions * n_samples).round().astype(int)

        sampled_groups = []
        for (year, month), n in group_samples.items():
            group_data = groups.get_group((year, month))
            if n > len(group_data):
                n = len(group_data)  # sample at most group size
            if n > 0:
                sampled_groups.append(group_data.sample(n, random_state=42))

        country_sampled = pd.concat(sampled_groups)
        sampled_dfs.append(country_sampled)

    result = pd.concat(sampled_dfs).reset_index(drop=True)
    
    # Clean up helper columns if you want
    result = result.drop(columns=['year', 'month'], errors='ignore')

    return result


def load_human_annotated_for_translation():
    filepath = os.path.expanduser(os.path.join(ANNOTATION_PATH, ANNOTATION_FILE))
    df = pd.read_csv(filepath, encoding="utf-8", encoding_errors='replace')

    df = df[df["country"].isin(SELECTED_COUNTRIES)]

    # Clean and filter labels
    df['corruption_label_m'] = df['corruption_label_m'].str.strip().str.lower()
    df = df[df['corruption_label_m'].isin([label.lower() for label in VALID_CORRUPTION_LABELS])]

    # Drop rows with missing or empty 'title' or 'body'
    df = df[df['title'].notna() & df['body'].notna()]
    df = df[df['title'].str.strip().ne('') & df['body'].str.strip().ne('')]

    # Create combined text for translation
    df["combined_text"] = df["title"].str.strip() + "\n" + df["body"].str.strip()

    # Ensure 'country' column exists
    if "country" in df.columns:
        df['country'] = df['country'].str.strip()
    else:
        df['country'] = 'manual_annotated'

    return df
