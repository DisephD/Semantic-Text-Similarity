#import the necessary libraries
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# Load and clean dataset
df = pd.read_csv('/content/DataNeuron_Text_Similarity.csv')
df.dropna(inplace=True)

# function to clean input text by removing non-ASCII characters
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

# apply cleaning function to both text columns
df['text1'] = df['text1'].apply(clean_text)
df['text2'] = df['text2'].apply(clean_text)

# load pre-trained transformer model for generating embeddings
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# encode the texts to obtain vector embeddings
embeddings1 = model.encode(df['text1'].tolist(), convert_to_tensor=True, device="cpu")
embeddings2 = model.encode(df['text2'].tolist(), convert_to_tensor=True, device="cpu")

# compute cosine similarity for each embeddings
cosine_scores = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().numpy()

# scale the scores between 0 and 1
scaled_scores = (cosine_scores + 1) / 2

# add to the dataframe
df['Similarity'] = scaled_scores

# save some basic statistics of the similarity scores
stats = {
    "mean": np.mean(cosine_scores),
    "min": np.min(cosine_scores),
    "max": np.max(cosine_scores)
}

# export the final dataframe to a new CSV file
df.to_csv('model_output.csv', index=False)
