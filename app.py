# Import necessary libraries
import gradio as gr  # Gradio is used to create a web interface for HuggingFace
from sentence_transformers import SentenceTransformer, util 
import re 

# Load a pre-trained SentenceTransformer model for generating sentence embeddings
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Function to clean input text by removing non-ASCII characters and trimming spaces
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

# Function to compute similarity between two input texts
def get_similarity(text1, text2):
    # Check if either input is empty or only whitespace
    if not text1.strip() or not text2.strip():
        return {"error": "Please enter text in both fields."}
    
    # Clean both input texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Generate sentence embeddings for each cleaned text
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    
    # Calculate cosine similarity between the two embeddings
    score = util.cos_sim(emb1, emb2).item()
    
    # Scale the cosine similarity score from range [-1, 1] to [0, 1]
    scaled_score = (score + 1) / 2
    
    # Return the similarity score as JSON
    return {"similarity score": scaled_score}

# Create a Gradio interface for the similarity function
interface = gr.Interface(
    fn=get_similarity,  # Function to call
    inputs=["text", "text"],  # Two text inputs
    outputs="json",  # Output will be shown as JSON
    title="Text Similarity API",  # Interface title
    description="Enter two texts to get a similarity score (0 to 1)",  # Short description
    live=True  # Allows the interface to update results immediately
)

# Launch the Gradio app and enable public sharing
interface.launch(share=True)
