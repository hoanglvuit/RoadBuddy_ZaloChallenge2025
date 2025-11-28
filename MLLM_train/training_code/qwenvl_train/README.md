# Qwen3VL-8B Fine-tuning Guide

## 📁 Project Structure
```
├── dataset.py          # Data format definition
├── model.py            # Model loading & training config
├── prompt.py           # Instruction prompts
├── train.json          # Training data
├── public_test.json    # Test data
├── seed.py             # Reproducibility seed
├── train.py            # SFT Trainer fine-tuning
├── run_inference.py    # Inference & CSV generation
└── dataset/
    ├── train/          # Top 4 frames from clip_train
    └── public_test/    # Top 4 frames from clip_test
```

## 🚀 Quick Start

### 1. Setup Server (Vast.ai)
- Rent server with **CUDA ≥ 12.8**
- Use **PyTorch template**
- Recommended: GPU with **VRAM ≥ 24GB**

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
```bash
mkdir -p dataset/train dataset/public_test
# Upload top 4 frames to respective folders
```

### 4. Train Model
```bash
python train.py
```

### 5. Run Inference
```bash
python run_inference.py
```

## 📄 Output
- Trained model checkpoint
- `predictions.csv` with results

## ⚡ Tips
- Monitor GPU memory usage
- Adjust batch size if OOM
- Use seed.py for reproducibility