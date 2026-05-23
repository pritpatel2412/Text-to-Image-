import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Any

# Import all modules we created!
from preprocess import TextPreprocessor
from cgan import CGANGenerator
from attention_gan import AttentionCGANGenerator

class UnifiedTextToImagePipeline:
    """
    A comprehensive and unified Text-to-Image generating pipeline (Task 6).
    Orchestrates text preprocessing, embedding creation, conditional shape synthesis
    (CGAN and Attention-Enhanced GAN), and pre-trained Latent Diffusion (Stable Diffusion).
    It simulates a real-world multi-model production pipeline.
    """
    def __init__(self, cgan_weights_path: Optional[str] = None, attn_gan_weights_path: Optional[str] = None, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        print(f"[UnifiedPipeline] Initializing pipeline on {self.device}")
        
        # 1. NLP Preprocessor (Task 4)
        self.preprocessor = TextPreprocessor(device=str(self.device))
        
        # 2. Conditional GAN (Task 2)
        self.cgan = CGANGenerator().to(self.device)
        if cgan_weights_path and os.path.exists(cgan_weights_path):
            self.cgan.load_state_dict(torch.load(cgan_weights_path, map_location=self.device))
            print(f"[UnifiedPipeline] Loaded CGAN weights from {cgan_weights_path}")
        self.cgan.eval()
            
        # 3. Attention-Enhanced GAN (Task 5)
        self.attn_gan = AttentionCGANGenerator().to(self.device)
        if attn_gan_weights_path and os.path.exists(attn_gan_weights_path):
            self.attn_gan.load_state_dict(torch.load(attn_gan_weights_path, map_location=self.device))
            print(f"[UnifiedPipeline] Loaded Attention GAN weights from {attn_gan_weights_path}")
        self.attn_gan.eval()
            
        # 4. Latent Diffusion Generator (SD / Task 1)
        self.sd_generator = None # Loaded dynamically if complex prompts are sent, to conserve VRAM
        
    def initialize_diffusion(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """Loads and initializes Stable Diffusion on demand to manage system resources."""
        if self.sd_generator is None:
            # We import and load the Diffusion pipeline dynamically
            from diffusers import StableDiffusionPipeline
            print(f"[UnifiedPipeline] Loading pre-trained Latent Diffusion '{model_id}' ...")
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.sd_generator = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            self.sd_generator.enable_attention_slicing()
            print("[UnifiedPipeline] Latent Diffusion loaded successfully!")

    def route_prompt(self, prompt: str) -> str:
        """Determines the appropriate generative model based on the semantic properties of the prompt."""
        cleaned = self.preprocessor.clean_text(prompt)
        
        # Identify conditional GAN triggers in text
        if any(shape in cleaned for shape in ["circle", "round", "oval"]):
            return "cgan_circle"
        elif any(shape in cleaned for shape in ["square", "rectangle", "box"]):
            return "cgan_square"
        elif any(shape in cleaned for shape in ["triangle", "pyramid", "cone"]):
            return "cgan_triangle"
            
        # Default route is Latent Diffusion
        return "latent_diffusion"

    def generate(self, prompt: str, mode: str = "Attention GAN", sd_model_id: str = "runwayml/stable-diffusion-v1-5") -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Orchestrates end-to-end generation: tokenizes, embeds, routes,
        runs model inference, and compiles visual metadata.
        """
        # Step 1: Preprocess text and extract token embeddings (Task 4)
        embeddings, hidden_states = self.preprocessor.get_embeddings(prompt)
        tokens_map = self.preprocessor.tokenize(prompt)
        
        # Step 2: Route prompt to the designated module
        route = self.route_prompt(prompt)
        print(f"[UnifiedPipeline] Routing prompt '{prompt}' -> {route.upper()}")
        
        metadata = {
            "prompt": prompt,
            "route": route,
            "device": str(self.device),
            "embeddings_shape": list(embeddings.shape),
            "hidden_states_shape": list(hidden_states.shape)
        }
        
        if "cgan" in route:
            # Map route to specific shape labels (0: circle, 1: square, 2: triangle)
            shape_map = {"cgan_circle": 0, "cgan_square": 1, "cgan_triangle": 2}
            label_idx = shape_map[route]
            
            label_tensor = torch.tensor([label_idx], device=self.device)
            noise = torch.randn(1, 100, device=self.device)
            
            if mode == "Baseline GAN":
                with torch.no_grad():
                    gen_img = self.cgan(noise, label_tensor)
                # Format to PIL Image
                img_np = (gen_img[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                return Image.fromarray(img_np), metadata
                
            else: # Attention-Enhanced GAN
                with torch.no_grad():
                    gen_img, attn_map = self.attn_gan(noise, label_tensor)
                
                # Format spatial image output
                img_np = (gen_img[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                
                # Generate attention heatmap overlay
                attn_np = attn_map[0].cpu().numpy()
                spatial_attn = np.mean(attn_np, axis=0).reshape(32, 32)
                
                metadata["attention_map"] = spatial_attn.tolist()
                return Image.fromarray(img_np), metadata
                
        else: # Latent Diffusion (Stable Diffusion)
            self.initialize_diffusion(sd_model_id)
            print(f"[UnifiedPipeline] Generating image using Stable Diffusion...")
            
            # Stable Diffusion Inference using our device and torch float precision
            with torch.inference_mode():
                result = self.sd_generator(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5
                )
            return result.images[0], metadata

if __name__ == "__main__":
    # Test script run
    pipeline = UnifiedTextToImagePipeline()
    
    print("\n--- Testing CGAN routing ---")
    circle_img, meta_circle = pipeline.generate("draw a glowing neon red circle", mode="Attention GAN")
    print(f"Metadata Circle: {meta_circle}")
    
    print("\n--- Testing SD routing (dynamic load check) ---")
    # We do not run the full SD load in quick tests to avoid large VRAM/download delays on test runs
    route = pipeline.route_prompt("a beautiful garden with colorful roses")
    print(f"Prompt: 'a beautiful garden with colorful roses' -> Route: {route}")
