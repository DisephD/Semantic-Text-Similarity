import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# Load and clean dataset
df = pd.read_csv('/content/DataNeuron_Text_Similarity.csv')
df.dropna(inplace=True)

def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

df['text1'] = df['text1'].apply(clean_text)
df['text2'] = df['text2'].apply(clean_text)

# Load model and encode text
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings1 = model.encode(df['text1'].tolist(), convert_to_tensor=True, device="cpu")
embeddings2 = model.encode(df['text2'].tolist(), convert_to_tensor=True, device="cpu")

# Compute and scale similarity score
cosine_scores = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().numpy()
scaled_scores = (cosine_scores + 1) / 2

df['Similarity'] = scaled_scores

# save stats
stats = {
    "mean": np.mean(cosine_scores),
    "min": np.min(cosine_scores),
    "max": np.max(cosine_scores)
}

# save results
df.to_csv('model_output.csv', index=False)

