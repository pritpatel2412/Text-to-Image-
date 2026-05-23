import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from cgan import ShapeDataset # Import synthetic dataset loader

class SelfAttention(nn.Module):
    """
    Self-Attention module for Generative Adversarial Networks (SAGAN).
    Enables spatial feature maps to capture long-range and multi-scale dependencies.
    """
    def __init__(self, in_channels: int):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Projection layers
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scale parameter (initialized to 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature maps of shape [batch_size, in_channels, height, width]
        Returns:
            - out: Attention-enhanced feature map of shape [batch_size, in_channels, height, width]
            - attention: Spatial attention map of shape [batch_size, height*width, height*width]
        """
        batch_size, channels, height, width = x.size()
        N = height * width
        
        # Project queries: [B, C//8, N] -> transpose to [B, N, C']
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)
        # Project keys: [B, C//8, N]
        proj_key = self.key_conv(x).view(batch_size, -1, N)
        
        # Calculate raw attention map: [B, N, N]
        energy = torch.bmm(proj_query, proj_key) # Matrix dot-product of query & key
        attention = self.softmax(energy) # Spatial attention distribution
        
        # Project values: [B, C, N]
        proj_value = self.value_conv(x).view(batch_size, -1, N)
        # Multiply values by spatial attention: [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        # Reshape to spatial format, apply learnable scale gamma, and add residual link
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        
        return out, attention

class AttentionCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 3, embed_dim: int = 10, img_channels: int = 3):
        super(AttentionCGANGenerator, self).__init__()
        
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.ReLU(True)
        )
        
        # Upsampling layer: [128, 16, 16] -> [64, 32, 32]
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        # Self-Attention Layer placed at the 32x32 feature map scale
        self.attention = SelfAttention(in_channels=64)
        
        # Final upsampling layer: [64, 32, 32] -> [3, 64, 64]
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        label_embed = self.label_embed(labels)
        x = torch.cat([noise, label_embed], dim=-1)
        x = self.fc(x).view(-1, 128, 16, 16)
        x = self.up_conv1(x)
        
        # Apply self-attention
        x, attn_map = self.attention(x)
        
        x = self.up_conv2(x)
        return x, attn_map

class AttentionCGANDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 3, embed_dim: int = 64, img_channels: int = 3):
        super(AttentionCGANDiscriminator, self).__init__()
        
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        
        # Conv block 1: [3 + 64, 64, 64] -> [64, 32, 32]
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels + embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self-Attention layer inside the discriminator pipeline at 32x32 resolution
        self.attention = SelfAttention(in_channels=64)
        
        # Conv block 2: [64, 32, 32] -> [128, 16, 16]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, h, w = img.size()
        embed_dim = self.label_embed.embedding_dim
        label_embed = self.label_embed(labels).view(batch_size, embed_dim, 1, 1)
        label_map = label_embed.expand(batch_size, embed_dim, h, w)
        
        x = torch.cat([img, label_map], dim=1)
        x = self.conv1(x)
        
        # Apply self-attention
        x, attn_map = self.attention(x)
        
        out = self.conv2(x)
        return out, attn_map

def train_attention_cgan(epochs: int = 15, batch_size: int = 64, latent_dim: int = 100) -> Tuple[AttentionCGANGenerator, List[float], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Attention GAN] Starting training on {device} ...")
    
    generator = AttentionCGANGenerator(latent_dim).to(device)
    discriminator = AttentionCGANDiscriminator().to(device)
    
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    dataset = ShapeDataset(num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    g_losses, d_losses = [], []
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            curr_batch_size = imgs.size(0)
            
            real_labels = torch.ones(curr_batch_size, 1, device=device)
            fake_labels = torch.zeros(curr_batch_size, 1, device=device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            outputs_real, _ = discriminator(imgs, labels)
            d_loss_real = adversarial_loss(outputs_real, real_labels)
            
            noise = torch.randn(curr_batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, 3, (curr_batch_size,), device=device)
            fake_imgs, _ = generator(noise, gen_labels)
            
            outputs_fake, _ = discriminator(fake_imgs.detach(), gen_labels)
            d_loss_fake = adversarial_loss(outputs_fake, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            outputs_tricked, _ = discriminator(fake_imgs, gen_labels)
            g_loss = adversarial_loss(outputs_tricked, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
        avg_g = epoch_g_loss / len(dataloader)
        avg_d = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss G: {avg_g:.4f} | Loss D: {avg_d:.4f}")
            
    print("[Attention GAN] Training complete!")
    return generator, g_losses, d_losses

def save_attention_predictions_and_maps(generator: AttentionCGANGenerator, output_path: str = "outputs/attention_gan_predictions.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval()
    
    labels = torch.tensor([0, 1, 2], device=device)
    noise = torch.randn(3, 100, device=device)
    
    with torch.no_grad():
        gen_imgs, attn_maps = generator(noise, labels)
        
    # Convert images to standard PIL/numpy format [0, 255]
    gen_imgs = (gen_imgs.cpu().numpy().transpose(0, 2, 3, 1) + 1.0) / 2.0
    gen_imgs = np.clip(gen_imgs * 255.0, 0, 255).astype(np.uint8)
    
    # Process attention maps: shape is [3, 1024, 1024] representing correlations between 32x32 pixels
    # We take the mean attention weights across query pixels to see which regions get most focus
    # Reshape back to [32, 32] spatial dimension and resize
    attn_maps_np = attn_maps.cpu().numpy() # [3, 1024, 1024]
    spatial_attn = np.mean(attn_maps_np, axis=1) # [3, 1024]
    spatial_attn = spatial_attn.reshape(3, 32, 32)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    names = ["Circle (Label: 0)", "Square (Label: 1)", "Triangle (Label: 2)"]
    
    for i in range(3):
        # Row 1: Generated Images
        axes[0, i].imshow(gen_imgs[i])
        axes[0, i].set_title(f"Gen: {names[i]}", fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: Attention Heatmaps
        im = axes[1, i].imshow(spatial_attn[i], cmap='jet', interpolation='bicubic')
        axes[1, i].set_title(f"Attention Heatmap", fontsize=10, fontweight='semibold')
        axes[1, i].axis('off')
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Attention GAN] Visual shape and attention map grid saved at: {output_path}")

if __name__ == "__main__":
    # Train Attention CGAN
    generator, g_losses, d_losses = train_attention_cgan(epochs=10)
    
    # Visualize generated predictions and attention maps
    save_attention_predictions_and_maps(generator)
