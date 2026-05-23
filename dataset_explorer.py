import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datasets import load_dataset
from PIL import Image

class OxfordFlowersExplorer:
    """
    A professional-grade dataset explorer and profiler designed to load,
    analyze, and visualize the Oxford-102 Flowers dataset.
    Generates semantic descriptions, maps flower classes to English labels,
    and profiles dataset distributions and metadata.
    """
    # Standard mapping of the 102 Oxford Flower category IDs to English names
    FLOWER_CLASSES = {
        1: 'pink primrose', 2: 'hard-leaved pocket orchid', 3: 'canterbury bells', 4: 'sweet pea', 5: 'wild pansy',
        6: 'tiger lily', 7: 'moon orchid', 8: 'bird of paradise', 9: 'monkshood', 10: 'globe thistle',
        11: 'snapdragon', 12: "colt's foot", 13: 'kingcup', 14: 'spear thistle', 15: 'yellow iris',
        16: 'globe-flower', 17: 'purple coneflower', 18: 'peruvian lily', 19: 'balloon flower', 20: 'giant white arum lily',
        21: 'fire lily', 22: 'pincushion flower', 23: 'fritillary', 24: 'red ginger', 25: 'grape hyacinth',
        26: 'corn poppy', 27: 'toadflax', 28: 'yellow garden mum', 29: 'siam tulip', 30: 'lenten rose',
        31: 'barbeton daisy', 32: 'daffodil', 33: 'sword lily', 34: 'poinsettia', 35: 'bolero deep blue',
        36: 'wallflower', 37: 'marigold', 38: 'buttercup', 39: 'oxeye daisy', 40: 'common dandelion',
        41: 'petunia', 42: 'wild pansy', 43: 'primula', 44: 'sunflower', 45: 'pelargonium',
        46: 'bishop of llandaff', 47: 'gaura', 48: 'geranium', 49: 'orange dahlia', 50: 'pink-yellow dahlia',
        51: 'cautleya spicata', 52: 'japanese anemone', 53: 'black-eyed susan', 54: 'silverbush', 55: 'californian poppy',
        56: 'osteospermum', 57: 'spring crocus', 58: 'bearded iris', 59: 'windflower', 60: 'tree poppy',
        61: 'gazania', 62: 'azalea', 63: 'water lily', 64: 'rose', 65: 'thorn apple',
        66: 'morning glory', 67: 'passion flower', 68: 'lotus', 69: 'toad lily', 70: 'anthurium',
        71: 'frangipani', 72: 'clematis', 73: 'hibiscus', 74: 'columbine', 75: 'desert-rose',
        76: 'tree mallow', 77: 'magnolia', 78: 'cyclamen', 79: 'watercress', 80: 'canna lily',
        81: 'hippeastrum', 82: 'bee balm', 83: 'ball moss', 84: 'foxglove', 85: 'bougainvillea',
        86: 'yew', 87: 'yucca', 88: 'great masterwort', 89: 'siam tulip', 90: 'blackberry',
        91: 'cantua', 92: 'common tulip', 93: 'wild geranium', 94: 'colibri', 95: 'borage',
        96: 'love in a mist', 97: 'mallow', 98: 'chinese wild peach', 99: 'bromelia', 100: 'blanket flower',
        101: 'trumpet creeper', 102: 'blackberry lily'
    }

    def __init__(self, dataset_name: str = "nkirschi/oxford-flowers"):
        print(f"[OxfordFlowersExplorer] Loading dataset '{dataset_name}' ...")
        self.dataset = load_dataset(dataset_name)
        print("[OxfordFlowersExplorer] Dataset loaded successfully!")
        
    def get_flower_name(self, label_idx: int) -> str:
        """Helper to safely map standard integer folders (1-102) to standard names."""
        # Hugging Face class names are strings representing the folder numbers
        # E.g. class index 0 maps to '1' (folder 1)
        class_names = self.dataset['train'].features['label'].names
        folder_str = class_names[label_idx]
        folder_int = int(folder_str)
        return self.FLOWER_CLASSES.get(folder_int, f"unknown flower id {folder_int}")

    def generate_caption(self, flower_name: str, index: int) -> str:
        """Programmatically generates rich, descriptive sentences for learning text embeddings."""
        templates = [
            f"a beautiful close-up photograph of a vibrant {flower_name} with delicate petals.",
            f"a high-resolution macro shot of a blooming {flower_name} in a summer garden.",
            f"a pristine {flower_name} flower captured with soft lighting and natural bokeh.",
            f"an exquisite botanical study showing the detailed features of a {flower_name}.",
            f"a sharp, detailed capture of a colorful {flower_name} flower in full bloom."
        ]
        # Use simple indexing to assign a template deterministically
        return templates[index % len(templates)]

    def profile_dataset(self) -> Dict:
        """Profiles the dataset to extract rich stats (Task 3)."""
        train_ds = self.dataset['train']
        test_ds = self.dataset['test']
        
        # Calculate general size statistics
        num_train = len(train_ds)
        num_test = len(test_ds)
        total_images = num_train + num_test
        num_classes = len(train_ds.features['label'].names)
        
        # Process subset to extract image resolutions and build captions
        subset_size = min(num_train, 200) # Analyze first 200 samples for swift execution
        resolutions = []
        caption_lengths = []
        words = []
        
        for idx in range(subset_size):
            item = train_ds[idx]
            img = item['image']
            resolutions.append(img.size) # (width, height)
            
            # Map label to class name & generate prompt
            flower_name = self.get_flower_name(item['label'])
            caption = self.generate_caption(flower_name, idx)
            caption_lengths.append(len(caption))
            words.extend(caption.split())

        # Compile NLP vocabulary statistics
        unique_vocab = set(words)
        avg_caption_len = np.mean(caption_lengths)
        max_caption_len = np.max(caption_lengths)
        
        # Compile spatial resolution stats
        widths, heights = zip(*resolutions)
        avg_width, avg_height = np.mean(widths), np.mean(heights)
        
        stats = {
            "num_classes": num_classes,
            "total_images": total_images,
            "train_images": num_train,
            "test_images": num_test,
            "avg_resolution": f"{int(avg_width)}x{int(avg_height)}",
            "avg_description_length_chars": round(avg_caption_len, 2),
            "max_description_length_chars": int(max_caption_len),
            "vocabulary_size": len(unique_vocab)
        }
        return stats

    def create_visualization_grid(self, output_path: str = "outputs/dataset_samples.png") -> str:
        """Generates a grid displaying flower photos matched with their semantic descriptions."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "outputs", exist_ok=True)
        train_ds = self.dataset['train']
        
        # Select 5 diverse samples from the training split
        indices = [0, 15, 30, 45, 60]
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        fig.suptitle("Oxford-102 Flowers: Public Dataset Visualizer", fontsize=18, fontweight='bold', y=0.98)
        
        for i, idx in enumerate(indices):
            item = train_ds[idx]
            img = item['image']
            flower_name = self.get_flower_name(item['label'])
            caption = self.generate_caption(flower_name, idx)
            
            # Draw image
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Wrap text manually for clean labels
            words = caption.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(" ".join(current_line)) > 24:
                    lines.append(" ".join(current_line[:-1]))
                    current_line = [word]
            lines.append(" ".join(current_line))
            wrapped_caption = "\n".join(lines)
            
            axes[i].set_title(f"Class: {flower_name.upper()}\n{wrapped_caption}", fontsize=10, pad=10, fontweight='semibold')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OxfordFlowersExplorer] Visual grid successfully saved at: {output_path}")
        return output_path

if __name__ == "__main__":
    explorer = OxfordFlowersExplorer()
    
    print("\n--- Profiling Dataset Stats ---")
    stats = explorer.profile_dataset()
    for k, v in stats.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
        
    print("\n--- Generating Visualization Grid ---")
    explorer.create_visualization_grid()
