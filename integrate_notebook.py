import json
import os

def integrate_internship_tasks():
    notebook_path = "Text_to_Image.ipynb"
    print(f"[Integrator] Reading notebook '{notebook_path}' ...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    cells = notebook.get("cells", [])
    print(f"[Integrator] Current notebook has {len(cells)} cells.")
    
    # Check if we have already integrated the tasks to avoid duplicate injection
    for cell in cells:
        if "### INTERNSHIP TASK INTEGRATION" in "".join(cell.get("source", [])):
            print("[Integrator] Internship tasks are already integrated in this notebook!")
            return
            
    # We will read the code from our modular files to inject them
    def read_code_file(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines

    print("[Integrator] Reading modular files for injection...")
    preprocessor_code = read_code_file("preprocess.py")
    dataset_code = read_code_file("dataset_explorer.py")
    cgan_code = read_code_file("cgan.py")
    attention_code = read_code_file("attention_gan.py")
    pipeline_code = read_code_file("pipeline.py")

    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ### INTERNSHIP TASK INTEGRATION ###\n",
                "\n",
                "The following sections contain the complete integration of all **six internship tasks** directly within the training application codebase.\n",
                "\n",
                "1. **Task 1: Pre-trained Model Refinement** (using LoRA layers for illustration styling)\n",
                "2. **Task 2: Conditional GAN (CGAN) Shape Generator** (Circle, Square, Triangle from text labels)\n",
                "3. **Task 3: Public Dataset Exploration & Profiling** (Oxford-102 Flowers profiling & visualization)\n",
                "4. **Task 4: Text Preprocessing & Embedding Engine** (Hugging Face CLIP text tokenization and PCA spatial mapping)\n",
                "5. **Task 5: Self-Attention & Cross-Attention blocks** inside GAN convolutional blocks with attention map overlays\n",
                "6. **Task 6: Unified Pipeline** binding NLP, CGAN, Attention-Enhanced GAN, and Diffusion model routing logic"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 3: Public Dataset Exploration & Statistics Profiling (Oxford-102 Flowers)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": dataset_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 4: Text Preprocessing & CLIP Tokenization/Embedding Engine"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": preprocessor_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 2: Conditional GAN Shape Generator (Synthesizing Circles, Squares, Triangles from Text Labels)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cgan_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 5: Attention-Enhanced GAN with Spatial Attention Map Overlays"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": attention_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 1: Pre-trained Model Refinement & LoRA Adapter Tuning"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": read_code_file("refine_model.py")
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 6: Comprehensive Text-to-Image Generating Pipeline (NLP Preprocessor + Shape CGAN + Stable Diffusion Routing)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": pipeline_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Unified End-to-End Pipeline Verification Run"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Instantiate the unified pipeline\n",
                "unified_pipeline = UnifiedTextToImagePipeline()\n",
                "\n",
                "# 1. Test shape generation (Attention CGAN route)\n",
                "print(\"--- Verifying CGAN Route ---\")\n",
                "shape_img, shape_meta = unified_pipeline.generate(\"draw a crisp blue circle\", mode=\"Attention GAN\")\n",
                "print(\"Generated shape successfully. Metadata:\", shape_meta)\n",
                "display(shape_img)\n",
                "\n",
                "# 2. Test text preprocessing coordinates (Task 4 PCA)\n",
                "print(\"\\n--- Verifying NLP & Embedding PCA Projections ---\")\n",
                "prompts = [\n",
                "    \"a beautiful pink rose with dew drops\",\n",
                "    \"a vibrant yellow sunflower in a sunny meadow\",\n",
                "    \"a minimalistic pencil sketch of a rose\"\n",
                "]\n",
                "coords, labels = unified_pipeline.preprocessor.visualize_embeddings_pca(prompts)\n",
                "for coord, label in zip(coords, labels):\n",
                "    print(f\"Prompt: '{label}' -> 2D PCA Embeddings Coord: {coord.tolist()}\")\n"
            ]
        }
    ]
    
    # We append the new cells to the cells list
    notebook["cells"] = cells + new_cells
    print(f"[Integrator] Appended {len(new_cells)} new cells. Total cells: {len(notebook['cells'])}.")
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
        
    print(f"[Integrator] Updated notebook successfully saved at: {notebook_path}")

if __name__ == "__main__":
    integrate_internship_tasks()
