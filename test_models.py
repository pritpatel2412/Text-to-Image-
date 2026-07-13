import unittest
import torch
from cgan import CGANGenerator
from attention_gan import AttentionCGANGenerator
from pipeline import UnifiedTextToImagePipeline
from preprocess import TextPreprocessor

class TestModels(unittest.TestCase):
    def test_cgan_initialization(self):
        model = CGANGenerator()
        self.assertIsInstance(model, torch.nn.Module)
        noise = torch.randn(2, 100)
        labels = torch.tensor([0, 1])
        output = model(noise, labels)
        self.assertEqual(output.shape, (2, 3, 64, 64))

    def test_attention_cgan_initialization(self):
        model = AttentionCGANGenerator()
        self.assertIsInstance(model, torch.nn.Module)
        noise = torch.randn(2, 100)
        labels = torch.tensor([0, 1])
        output, attn = model(noise, labels)
        self.assertEqual(output.shape, (2, 3, 64, 64))
        self.assertEqual(attn.shape, (2, 1024, 1024))

    def test_pipeline_routing(self):
        pipeline = UnifiedTextToImagePipeline(device="cpu")
        self.assertEqual(pipeline.route_prompt("draw a red circle"), "cgan_circle")
        self.assertEqual(pipeline.route_prompt("draw a blue square"), "cgan_square")
        self.assertEqual(pipeline.route_prompt("draw a green triangle"), "cgan_triangle")
        self.assertEqual(pipeline.route_prompt("a beautiful garden"), "latent_diffusion")

if __name__ == "__main__":
    unittest.main()
