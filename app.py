import gradio as gr
from sentence_transformers import SentenceTransformer, util
import re

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

def get_similarity(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    scaled_score = (score + 1) / 2
    return {"similarity score": scaled_score}

interface = gr.Interface(
    fn=get_similarity,
    inputs=["text", "text"],
    outputs="json",
    title="Text Similarity API",
    description="Enter two texts to get a similarity score (0 to 1)"
)

interface.launch()
