# Efficient-Financial-NLP-Fine-Tuning-with-QLoRA
Fine-tuning the LLaMA large language model with QLoRA on the FLARE financial NLP benchmark. Supports multi-task learning across Sentiment Analysis, QA, NER, Headline Classification, and Stock Movement Prediction with efficient quantization and adapters.

# QLoRA Fine-tuning for Financial NLP on FLARE Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Parameter-Efficient Fine-Tuning of Meta-Llama-3.1-8B-Instruct with QLoRA for Financial NLP Tasks

This repository contains the official implementation of our paper: **"Parameter-Efficient Fine-Tuning of Meta-Llama-3-8B-Instruct with QLoRA for Financial NLP on the FLARE Benchmark: A Comparative Analysis"**

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project implements a two-stage curriculum learning approach for fine-tuning large language models on financial NLP tasks using QLoRA (Quantized Low-Rank Adaptation). We achieve competitive performance on the FLARE benchmark while reducing GPU memory requirements by 75%.

### Key Highlights

- **Model**: Meta-Llama-3.1-8B-Instruct
- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Benchmark**: FLARE (Financial Language understanding And Reasoning Evaluation)
- **Tasks**: Financial QA, Stock Movement Prediction, Sentiment Analysis, NER
- **Efficiency**: 75% GPU memory reduction, 0.17% trainable parameters

## ✨ Features

- 🚀 Two-stage curriculum learning (complex → simple tasks)
- 📊 Comprehensive metrics tracking and visualization
- 💾 Checkpoint management and model versioning
- 📈 Real-time GPU monitoring
- 🎨 Publication-ready plots and LaTeX tables
- 🔄 Reproducible training pipeline
- 🧪 Extensive evaluation suite

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 12.4+
- 40GB+ GPU (A100 recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/qlora-financial-nlp.git
cd qlora-financial-nlp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Google Colab

```python
# Run this in a Colab cell
!git clone https://github.com/yourusername/qlora-financial-nlp.git
%cd qlora-financial-nlp
!pip install -r requirements.txt
```

## 📁 Project Structure

```
qlora-financial-nlp/
├── configs/                    # Configuration files
│   ├── model_config.yaml      # QLoRA hyperparameters
│   ├── stage1_config.yaml     # Stage 1 training config
│   └── stage2_config.yaml     # Stage 2 training config
├── src/
│   ├── data/                  # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loaders.py         # Dataset loaders
│   │   └── processors.py      # Task-specific processors
│   ├── models/                # Model architecture
│   │   ├── __init__.py
│   │   ├── qlora_model.py     # QLoRA setup
│   │   └── adapters.py        # Multi-adapter management
│   ├── training/              # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py         # Main trainer
│   │   └── curriculum.py      # Curriculum learning
│   ├── evaluation/            # Evaluation suite
│   │   ├── __init__.py
│   │   ├── metrics.py         # Metric computation
│   │   └── evaluators.py      # Task evaluators
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── logging_utils.py   # Logging and tracking
│       ├── visualization.py   # Plotting functions
│       └── checkpoint.py      # Checkpoint management
├── scripts/                   # Executable scripts
│   ├── train_stage1.py        # Train Stage 1
│   ├── train_stage2.py        # Train Stage 2
│   ├── evaluate.py            # Run evaluation
│   └── generate_plots.py      # Generate paper plots
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_results_analysis.ipynb
├── tests/                     # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
├── outputs/                   # Training outputs (gitignored)
├── figures/                   # Generated figures
├── docs/                      # Documentation
│   ├── TRAINING.md           # Training guide
│   ├── EVALUATION.md         # Evaluation guide
│   └── API.md                # API documentation
├── .gitignore
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### 1. Prepare Data

```bash
python scripts/prepare_data.py --output_dir ./data
```

### 2. Train Stage 1 (Complex Tasks)

```bash
python scripts/train_stage1.py \
    --config configs/stage1_config.yaml \
    --output_dir ./outputs/stage1
```

### 3. Train Stage 2 (Simple Tasks)

```bash
python scripts/train_stage2.py \
    --config configs/stage2_config.yaml \
    --checkpoint ./outputs/stage1/final_model \
    --output_dir ./outputs/stage2
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --model_path ./outputs/stage2/final_model \
    --tasks all \
    --output_dir ./results
```

## 📚 Training

### Stage 1: Complex Tasks (QA + SMP)

Trains on Financial Question Answering (FinQA, ConvFinQA) and Stock Movement Prediction.

```python
from src.training import CurriculumTrainer

trainer = CurriculumTrainer(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_dir="./outputs/stage1"
)

trainer.train_stage1(
    qa_datasets=["FinQA", "ConvFinQA"],
    smp_datasets=["CIKM", "ACL", "BigData"],
    epochs=2,
    learning_rate=2e-4
)
```

**Training Configuration:**
- Batch size: 2 (effective: 8 with grad accumulation)
- Learning rate: 2e-4
- Max sequence length: 3500
- LoRA rank: 16, alpha: 32
- Training time: ~7 hours (A100 40GB)

### Stage 2: Simple Tasks (Sentiment + NER)

Loads Stage 1 adapters and continues with sentiment analysis and NER.

```python
trainer.train_stage2(
    sentiment_datasets=["FPB", "FiQA-SA"],
    ner_datasets=["FLARE-NER"],
    epochs=3,
    learning_rate=2e-4
)
```

**Training Configuration:**
- Batch size: 4 (effective: 8)
- Learning rate: 2e-4
- Max sequence length: 512
- LoRA rank: 32, alpha: 64
- Training time: ~1 hour (A100 40GB)

### Monitoring

Training metrics are automatically logged:
- TensorBoard: `tensorboard --logdir outputs/logs`
- Weights & Biases: Configure in `configs/logging.yaml`

## 📊 Evaluation

### Run Full Benchmark

```bash
python scripts/evaluate.py \
    --model_path ./outputs/stage2/final_model \
    --benchmark flare \
    --output_dir ./results
```

### Task-Specific Evaluation

```python
from src.evaluation import FLAREEvaluator

evaluator = FLAREEvaluator(model_path="./outputs/stage2/final_model")

# Evaluate QA
qa_results = evaluator.evaluate_qa(test_data="FinQA")

# Evaluate Sentiment
sentiment_results = evaluator.evaluate_sentiment(test_data="FPB")
```

## 📈 Results

### Performance on FLARE Benchmark

| Task | FinQA (EM) | ConvFinQA (EM) | FPB (F1) | FiQA-SA (F1) | SMP (Acc) | NER (F1) |
|------|------------|----------------|----------|--------------|-----------|----------|
| **Ours (QLoRA)** | **14.65** | **40.40** | 86.0 | 76.0 | **57.55** | 59.3 |
| FinMA-7B-Full | 4.0 | 20.0 | 87.0 | 79.0 | 54.5 | 69.0 |
| FinMA-30B | 11.0 | 40.0 | 88.0 | 87.0 | 46.0 | 62.0 |
| GPT-4 | 76.0 | 76.0 | 86.0 | 88.0 | 54.5 | 83.0 |

### Efficiency Metrics

| Metric | Full Fine-tuning | LoRA | **QLoRA (Ours)** |
|--------|------------------|------|------------------|
| GPU Memory | >40 GB | ~30 GB | **~7 GB** |
| Trainable Params | 8.03B (100%) | 13.6M (0.17%) | **13.6M (0.17%)** |
| Storage | 16.1 GB | 16.1 GB + 27 MB | **7.02 GB + 27 MB** |
| Training Speed | 1x | 1.2x | **2.5x** |

### Generated Figures

All figures are automatically generated in high resolution (300 DPI):

- `figures/training_curves.png` - Training/validation loss
- `figures/gpu_utilization.png` - GPU memory and utilization
- `figures/performance_comparison.png` - Task performance comparison
- `figures/efficiency_comparison.png` - Resource efficiency

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_use_double_quant: true

lora:
  stage1:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  
  stage2:
    r: 32
    alpha: 64
    dropout: 0.1
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", 
                     "gate_proj", "up_proj", "down_proj"]
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- [Training Guide](docs/TRAINING.md) - Comprehensive training instructions
- [Evaluation Guide](docs/EVALUATION.md) - Evaluation protocols
- [API Documentation](docs/API.md) - API reference

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{djagba2024qlora,
  title={Parameter-Efficient Fine-Tuning of Meta-Llama-3-8B-Instruct with QLoRA for Financial NLP on the FLARE Benchmark},
  author={Djagba, P. and Younoussi Saley, A. and Zeleke, A.},
  journal={arXiv preprint},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FLARE Benchmark team
- Hugging Face for transformers and PEFT libraries
- Meta AI for Llama models
- Michigan State University

## 📧 Contact

- **P. Djagba** - djagbapr@msu.edu
- **A. Younoussi Saley** - saley.younoussi@aims.ac.rw
- **A. Zeleke** - zeleke@msu.edu

---

**Note**: This is research code. For production use, additional testing and optimization may be required.
