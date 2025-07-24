from translation import translate_dataframe
import pandas as pd

df = pd.read_csv("~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_10000.csv")
print(len(df))
df["text"] = df["combined_text"]
df_translated = translate_dataframe(df)
df_translated.to_csv("~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_translated_10000.csv")
