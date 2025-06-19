from translation import translate_dataframe
import pandas as pd

df = pd.read_csv("~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_2000.csv")
df["text"] = df["combined_text"]
df_translated = translate_dataframe(df)
df_translated.to_csv("~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/output/news_sample_translated.csv")
