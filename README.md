# image-to-dense-caption

Generate comprehensive, dense portrait descriptions using a vision‑language model.

This repository provides a command-line tool (`infer.py`) that:

- Loads the `Qwen2.5-VL-7B-Instruct-abliterated` model
- Processes images (supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`)
- Produces rich descriptive paragraphs covering emotional expression, posture, clothing or nudity, body type, hair, and environmental context
- Outputs one `.txt` file per image

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/anto18671/image-to-dense-caption.git
cd image-to-dense-caption
```

### 2. Set up your Python environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate tqdm
```

### 4. Download the model

```bash
git lfs install
git clone https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated
```

Ensure the folder `Qwen2.5-VL-7B-Instruct-abliterated` sits in the same directory as `infer.py`.

---

## 🔧 Usage

1. Place your images in a subfolder (default: `images/`)
2. Run the script:

```bash
python infer.py
```

This will:

- Scan the folder for valid image files
- Generate a `.txt` with dense descriptions for each image

---

## 📁 Output Example

```
images/photo1.jpg      → images/photo1.txt
images/portrait.webp   → images/portrait.txt
```

Each `.txt` includes a paragraph describing emotional expression, posture, clothing/nudity status, body type, hair, and environment.

---

## 💡 Tips & Configurations

- **GPU**: Preferably use a GPU with ≥ 16 GB VRAM.
- **Memory options**:

  - Use 8‑bit quantization (via `bitsandbytes`) for lower VRAM.
  - Switch to `torch_dtype=torch.float16` if supported by your setup.

- **Custom folder**: Change the `image_folder` path in `infer.py` if needed.

---

## ✅ Troubleshooting

- **OSError / Model not found**: Confirm the model folder is correctly named and in place.
- **CUDA out-of-memory**:

  - Reduce VRAM usage by quantizing the model.
  - Run on CPU by removing `.to("cuda")`—will be slower.

- **Non‑image files**: Unsupported extensions are automatically skipped.

---

## ⚙️ Advanced / Future Features

- **JSON output**: Export descriptions in structured JSON for further processing.
- **Aggregate mode**: Process all images and output a combined JSON summary.
- **Web GUI**: Add a basic Flask or Gradio interface for interactive prompting.

---

## 📄 License

[MIT License](https://github.com/anto18671/image-to-dense-caption/blob/main/LICENSE) — this script

Model usage under Hugging Face terms (see [huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated](https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated) for details)
