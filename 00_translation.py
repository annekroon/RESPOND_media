import os
import logging
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='translation.log', level=logging.INFO)
logger = logging.getLogger("translation")

# File paths
path_to_news = "/home/akroon/data/volume_2/RESPONDE/data/data_conbined/"
path_to_RESPOND_data = '/home/akroon/data/volume_2/RESPONDE/'

# List of countries and their corresponding language codes for MarianMT
country_to_lang = {
    "Sweden": "sv",
    "Netherlands": "nl",
    "United_Kingdom": "en",
    "Hungary": "hu",
    "Italy": "it",
    "France": "fr",
    "Ukraine": "uk",
    "Serbia": "sr",
    "Bulgaria": "bg"
}

# Read the input dataframe
df = pd.read_csv(f'{path_to_RESPOND_data}combined_sample.csv')

# Retrieve 'combined_text'
texts = df['combined_text'].tolist()

# Cache for loaded models
models_cache = {}

# Step 1: Initialize MarianMT translation model (dynamically selecting based on source language)
def get_translation_model(source_lang):
    if source_lang not in models_cache:
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-en'  # Use language to English
        try:
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            models_cache[source_lang] = (model, tokenizer)
        except Exception as e:
            logger.error(f"Failed to load model for {source_lang}: {e}")
            return None, None
    return models_cache[source_lang]

# Function to translate text (from any of the supported languages to English)
def translate_text(text, source_lang='es'):
    try:
        # Load the correct translation model based on source language
        translation_model, translation_tokenizer = get_translation_model(source_lang)
        
        if translation_model is None or translation_tokenizer is None:
            return text  # If model loading failed, return the original text
        
        # Move inputs to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        translation_model.to(device)

        # Split long text into smaller chunks if needed
        max_input_length = translation_tokenizer.model_max_length
        input_chunks = [text[i:i+max_input_length] for i in range(0, len(text), max_input_length)]
        
        translated_chunks = []
        for chunk in input_chunks:
            inputs = translation_tokenizer(chunk, return_tensors='pt', padding=True, truncation=True).to(device)
            translated = translation_model.generate(
                **inputs,
                num_beams=4,  # Use beam search to prevent repetitive output
                no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
                temperature=1.0,  # Control randomness (try values less than 1.0 for less creativity)
                top_k=50,  # Top-k sampling for more diverse translations
                top_p=0.95,  # Top-p sampling for more diverse translations
            )
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_chunks.append(translated_text)
        
        # Combine all chunks into one translated text
        return " ".join(translated_chunks)
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text  # Return original text in case of failure

# Step 2: Translate texts in the 'combined_text' column to English
translated_texts = []
total_texts = len(texts)

# Use tqdm for a nice progress bar
for i, (text, country) in tqdm(enumerate(zip(texts, df['country']), 1), total=total_texts, desc="Translating Texts"):
    if country not in country_to_lang:
        logger.warning(f"Country {country} not found in country_to_lang dictionary.")
        translated_texts.append(text)
    else:
        source_lang = country_to_lang[country]  # Get language code for translation
        translated_text = translate_text(text, source_lang=source_lang)
        translated_texts.append(translated_text)

    # Log progress every 10,000 texts and save intermediate results
    if i % 10000 == 0:
        logger.info(f"{i}/{total_texts} texts translated...")
        partial_output_file = f'{path_to_RESPOND_data}partial_translation_{i}.csv'
        df.iloc[:i].to_csv(partial_output_file, index=False)
        logger.info(f"Intermediate translations saved: {partial_output_file}")

# Add translated texts to the dataframe
df['translated_text'] = translated_texts

# Save the final dataframe with translated texts
output_file = f'{path_to_RESPOND_data}full_data_translation.csv'
df.to_csv(output_file, index=False)

logger.info(f"Translation complete and saved to {output_file}")
