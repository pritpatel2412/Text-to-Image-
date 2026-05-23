import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Tuple, List

# Define the synthetic shapes dataset
class ShapeDataset(Dataset):
    """
    Generates synthetic geometric shapes (circle, square, triangle)
    on the fly to train the Conditional GAN.
    """
    def __init__(self, num_samples: int = 3000, img_size: int = 64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.labels = np.random.randint(0, 3, size=num_samples) # 0: circle, 1: square, 2: triangle

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx]
        
        # Create a black background image (RGB)
        img = Image.new("RGB", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(img)
        
        # Random size and position offsets
        size = np.random.randint(18, 26)
        cx, cy = self.img_size // 2, self.img_size // 2
        offset_x = np.random.randint(-6, 7)
        offset_y = np.random.randint(-6, 7)
        cx, cy = cx + offset_x, cy + offset_y
        
        # Random primary color
        colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "white"]
        color = np.random.choice(colors)
        
        if label == 0: # Circle
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif label == 1: # Square
            draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif label == 2: # Triangle
            points = [
                (cx, cy - size), 
                (cx - size, cy + size), 
                (cx + size, cy + size)
            ]
            draw.polygon(points, fill=color)
            
        # Convert image to float tensor normalized between [-1, 1]
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        # Transpose to PyTorch shape channel-first [C, H, W]
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        return img_tensor, int(label)

# Define the Generator
class CGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 3, embed_dim: int = 10, img_channels: int = 3):
        super(CGANGenerator, self).__init__()
        
        # Label embedding layer
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        
        # Combined input layer projection
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.ReLU(True)
        )
        
        # Upsampling convolutional pipeline: starts at [128, 16, 16] -> [64, 32, 32] -> [3, 64, 64]
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Upsample by 2x
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1), # Upsample by 2x
            nn.Tanh() # Normalizes spatial output between [-1, 1]
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Map labels to embed vector
        label_embed = self.label_embed(labels)
        # Concatenate latent noise and class conditioning embeddings
        x = torch.cat([noise, label_embed], dim=-1)
        x = self.fc(x)
        # Reshape to 4D tensor: [batch_size, 128, 16, 16]
        x = x.view(-1, 128, 16, 16)
        x = self.conv_blocks(x)
        return x

# Define the Discriminator
class CGANDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 3, embed_dim: int = 64, img_channels: int = 3):
        super(CGANDiscriminator, self).__init__()
        
        # Label embedding layer maps to feature map scale
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        
        # Convolutional pipeline: starts at [3 + embed_dim, 64, 64] -> [64, 32, 32] -> [128, 16, 16]
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels + embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid() # Probability output of being real vs fake
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = img.size()
        embed_dim = self.label_embed.embedding_dim
        # Embed label and project to [batch_size, embed_dim, 1, 1] spatial dimension
        label_embed = self.label_embed(labels).view(batch_size, embed_dim, 1, 1)
        # Expand to [batch_size, embed_dim, h, w]
        label_map = label_embed.expand(batch_size, embed_dim, h, w)
        
        # Concatenate spatial image with class label map channels
        x = torch.cat([img, label_map], dim=1)
        return self.conv_blocks(x)

# Setup complete training routine
def train_cgan(epochs: int = 15, batch_size: int = 64, latent_dim: int = 100) -> Tuple[CGANGenerator, List[float], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CGAN] Training starting on {device} ...")
    
    # Initialize networks
    generator = CGANGenerator(latent_dim).to(device)
    discriminator = CGANDiscriminator().to(device)
    
    # Losses & Optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Load dataset
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
            
            # Ground truths
            real_labels = torch.ones(curr_batch_size, 1, device=device)
            fake_labels = torch.zeros(curr_batch_size, 1, device=device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss on real images
            outputs_real = discriminator(imgs, labels)
            d_loss_real = adversarial_loss(outputs_real, real_labels)
            
            # Loss on generated images
            noise = torch.randn(curr_batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, 3, (curr_batch_size,), device=device)
            fake_imgs = generator(noise, gen_labels)
            
            outputs_fake = discriminator(fake_imgs.detach(), gen_labels)
            d_loss_fake = adversarial_loss(outputs_fake, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Tricking the discriminator
            outputs_tricked = discriminator(fake_imgs, gen_labels)
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
            
    print("[CGAN] Training complete!")
    return generator, g_losses, d_losses

def save_cgan_predictions(generator: CGANGenerator, output_path: str = "outputs/cgan_shapes.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval()
    
    # Generate shapes: circle (0), square (1), triangle (2)
    labels = torch.tensor([0, 1, 2], device=device)
    noise = torch.randn(3, 100, device=device)
    
    with torch.no_grad():
        gen_imgs = generator(noise, labels)
        
    # Convert images to standard PIL/numpy format [0, 255]
    gen_imgs = (gen_imgs.cpu().numpy().transpose(0, 2, 3, 1) + 1.0) / 2.0
    gen_imgs = np.clip(gen_imgs * 255.0, 0, 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ["Circle (Label: 0)", "Square (Label: 1)", "Triangle (Label: 2)"]
    
    for i in range(3):
        axes[i].imshow(gen_imgs[i])
        axes[i].set_title(names[i], fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[CGAN] Visual shape grid saved successfully at: {output_path}")

if __name__ == "__main__":
    # Test dataset
    ds = ShapeDataset(num_samples=10)
    print(f"ShapeDataset size: {len(ds)} | Sample image tensor shape: {ds[0][0].shape}")
    
    # Train CGAN
    generator, g_losses, d_losses = train_cgan(epochs=10)
    
    # Visualize generated predictions
    save_cgan_predictions(generator)
