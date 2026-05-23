import torch
import numpy as np
from typing import Tuple, List, Union, Dict
from transformers import CLIPTokenizer, CLIPTextModel
import os

class TextPreprocessor:
    """
    A professional-grade NLP pipeline to clean, tokenize, and encode text descriptions
    into high-dimensional embeddings using Hugging Face Transformers (CLIP).
    Designed to serve as the front-end processor for a text-to-image generation pipeline.
    """
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        print(f"[TextPreprocessor] Initializing tokenizer and text encoder on {self.device}")
        
        # Load pre-trained CLIP tokenizer and text encoder from Hugging Face
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(self.device)
        self.text_encoder.eval() # Set to evaluation mode

    def clean_text(self, text: str) -> str:
        """Cleans and standardizes raw text inputs."""
        if not text:
            return ""
        # Convert to lowercase and strip excess whitespaces
        cleaned = text.strip().lower()
        return cleaned

    def tokenize(self, text: Union[str, List[str]], max_length: int = 77) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input strings, adding padding and truncation up to the maximum 
        sequence length (default 77 for CLIP standard).
        """
        if isinstance(text, str):
            text = [self.clean_text(text)]
        else:
            text = [self.clean_text(t) for t in text]

        # Process inputs using HF Tokenizer
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        # Move tensors to the designated device
        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_embeddings(self, text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts high-dimensional text embeddings and hidden states.
        
        Returns:
            - text_embeddings (pooled_output): [batch_size, embedding_dim] - representation of whole prompt
            - last_hidden_state: [batch_size, sequence_length, embedding_dim] - token-level features for cross-attention
        """
        inputs = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            
        last_hidden_state = outputs.last_hidden_state
        # CLIP's pooled output (the projection of the EOS token embedding)
        text_embeddings = outputs.pooler_output 
        
        return text_embeddings, last_hidden_state

    def visualize_embeddings_pca(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Generates 2D coordinates for a list of text descriptions using Principal Component
        Analysis (PCA) on their extracted CLIP embeddings.
        """
        if len(texts) < 2:
            raise ValueError("Must provide at least 2 text prompts for PCA visualization.")
            
        embeddings, _ = self.get_embeddings(texts)
        embeddings_np = embeddings.cpu().numpy()
        
        # Apply PCA to project 768-dim embeddings down to 2 dimensions
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(embeddings_np)
        
        return coords_2d, texts

if __name__ == "__main__":
    # Test script run
    preprocessor = TextPreprocessor()
    test_prompts = [
        "a photorealistic red rose in full bloom",
        "a vibrant yellow sunflower shining in a sunny garden",
        "a dark gothic painting of a withered black rose",
        "a high-resolution macro shot of a white lily with water droplets"
    ]
    
    print("\n--- Testing Tokenization Map ---")
    inputs = preprocessor.tokenize(test_prompts[0])
    tokens = preprocessor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"Prompt: '{test_prompts[0]}'")
    print(f"Token IDs: {inputs['input_ids'][0][:15].cpu().tolist()} ...")
    print(f"Tokens: {tokens[:15]} ...")
    
    print("\n--- Extracting Embeddings ---")
    embeddings, hidden_states = preprocessor.get_embeddings(test_prompts)
    print(f"Pooled Text Embeddings Shape: {embeddings.shape}")
    print(f"Last Hidden States Shape: {hidden_states.shape}")
    
    print("\n--- Testing 2D PCA Projections ---")
    coords, labels = preprocessor.visualize_embeddings_pca(test_prompts)
    for coord, label in zip(coords, labels):
        print(f"Prompt: '{label}' -> Coordinates: {coord.tolist()}")
