# Parameter-Efficient Fine-Tuning of Meta-Llama-3-8B-Instruct with
QLoRA for Financial NLP on the FLARE Benchmark: A Comparative
Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Multi-Adapter QLoRA Fine-tuning of Meta-Llama-3.1-8B-Instruct for 6 Financial NLP Tasks on FLARE Benchmark

## 🎯 Overview

This project implements **task-specific multi-adapter QLoRA** fine-tuning for financial NLP. Each of the 6 tasks has its own independent LoRA adapter, trained on 8 datasets total from the FLARE benchmark.

### Tasks & Datasets (6 Tasks, 8 Datasets)

| Task | # Datasets | Datasets | Adapter Config |
|------|-----------|----------|----------------|
| **SA** - Sentiment Analysis | 2 | Financial Phrase Bank, FiQA-SA | r=8, α=16 |
| **HC** - Headline Classification | 1 | Gold News Headlines | r=8, α=16 |
| **NER** - Named Entity Recognition | 1 | Financial Agreements | r=16, α=32 |
| **QA** - Question Answering | 2 | FinQA, ConvFinQA | r=16, α=32 |
| **SMP** - Stock Movement Prediction | 3 | BigData22, ACL18, CIKM18 | r=16, α=32 |

### Key Features

- **Multi-Adapter Architecture**: Independent LoRA adapter per task
- **4-bit Quantization**: NF4 quantization with double quantization
- **Comprehensive Tracking**: GPU metrics, training curves, performance plots
- **Modular Design**: Easy to add/remove tasks and datasets

## 📁 Project Structure

```
qlora-financial-nlp/
├── configs/
│   ├── model_config.yaml           # Base model & quantization config
│   └── tasks/
│       ├── sa_config.yaml          # Sentiment Analysis
│       ├── hc_config.yaml          # Headline Classification
│       ├── ner_config.yaml         # Named Entity Recognition
│       ├── qa_config.yaml          # Question Answering
│       └── smp_config.yaml         # Stock Movement Prediction
│
├── data/                           # Raw datasets by task
│   ├── SA/                         # Sentiment Analysis
│   │   ├── financial_phrase_bank.csv
│   │   └── fiqa_sa.csv
│   ├── HC/                         # Headline Classification
│   │   └── gold_news_headlines.csv
│   ├── NER/                        # Named Entity Recognition
│   │   └── fin_agreements.csv
│   ├── QA/                         # Question Answering
│   │   ├── finqa.csv
│   │   └── convfinqa.csv
│   └── SMP/                        # Stock Movement Prediction
│       ├── bigdata22.csv
│       ├── acl18.csv
│       └── cikm18.csv
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py              # Dataset loaders per task
│   │   └── processors.py           # Task-specific data processors
│   │       ├── SAProcessor
│   │       ├── HCProcessor
│   │       ├── NERProcessor
│   │       ├── QAProcessor (FinQA + ConvFinQA)
│   │       └── SMPProcessor
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── qlora_model.py          # QLoRA base model setup
│   │   └── multi_adapter.py        # Multi-adapter management
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── task_trainer.py         # Single task trainer
│   │   └── multi_task_trainer.py   # Orchestrates all tasks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Task-specific metrics
│   │   └── evaluators.py           # Evaluation suite
│   │       ├── SAEvaluator (Macro F1)
│   │       ├── HCEvaluator (Accuracy)
│   │       ├── NEREvaluator (Entity F1)
│   │       ├── QAEvaluator (EM, F1)
│   │       └── SMPEvaluator (Accuracy, MCC)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics_tracker.py      # GPU & training metrics
│       ├── visualization.py        # Plot generation
│       └── checkpoint_manager.py   # Model checkpointing
│
├── scripts/
│   ├── download_data.py            # Download FLARE datasets
│   ├── train_task.py               # Train single task
│   ├── train_all.py                # Train all tasks
│   ├── evaluate_task.py            # Evaluate single task
│   ├── evaluate_all.py             # Full FLARE evaluation
│   └── generate_paper_outputs.py   # Generate plots & tables
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Explore datasets
│   ├── 02_single_task_demo.ipynb   # Train single task demo
│   ├── 03_multi_task_training.ipynb # Multi-task training
│   └── 04_results_analysis.ipynb   # Analyze results
│
├── outputs/                        # Training outputs (gitignored)
│   ├── adapters/                   # Saved LoRA adapters
│   │   ├── sa_adapter/
│   │   ├── hc_adapter/
│   │   ├── ner_adapter/
│   │   ├── qa_adapter/
│   │   └── smp_adapter/
│   ├── checkpoints/                # Training checkpoints
│   ├── logs/                       # Training logs
│   └── metrics/                    # Metrics JSON files
│
├── results/                        # Evaluation results
│   ├── figures/                    # Generated plots
│   │   ├── training_curves/
│   │   ├── gpu_utilization/
│   │   └── performance_comparison/
│   └── tables/                     # LaTeX tables
│
├── tests/
│   ├── test_data_loaders.py
│   ├── test_processors.py
│   ├── test_models.py
│   └── test_trainers.py
│
├── docs/
│   ├── SETUP.md                    # Setup guide
│   ├── TRAINING.md                 # Training guide per task
│   ├── EVALUATION.md               # Evaluation protocols
│   └── API.md                      # API documentation
│
├── .gitignore
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/qlora-financial-nlp.git
cd qlora-financial-nlp

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/download_data.py --output_dir ./data
```

### 3. Train Single Task

```bash
# Train Sentiment Analysis
python scripts/train_task.py \
    --task sa \
    --config configs/tasks/sa_config.yaml \
    --output_dir ./outputs/adapters/sa_adapter

# Train Question Answering
python scripts/train_task.py \
    --task qa \
    --config configs/tasks/qa_config.yaml \
    --output_dir ./outputs/adapters/qa_adapter
```

### 4. Train All Tasks

```bash
python scripts/train_all.py \
    --base_config configs/model_config.yaml \
    --output_dir ./outputs/adapters
```

### 5. Evaluate

```bash
# Evaluate single task
python scripts/evaluate_task.py \
    --task sa \
    --adapter_path ./outputs/adapters/sa_adapter

# Evaluate all tasks
python scripts/evaluate_all.py \
    --adapter_dir ./outputs/adapters \
    --output_dir ./results
```

## 📊 Task-Specific Configurations

### Simple Tasks (Classification-based)

**Sentiment Analysis (SA)**
- Datasets: FPB, FiQA-SA
- LoRA: r=8, α=16, dropout=0.1
- Modules: q_proj, v_proj
- Max length: 512
- Metric: Macro F1

**Headline Classification (HC)**
- Dataset: Gold News Headlines
- LoRA: r=8, α=16, dropout=0.1
- Modules: q_proj, v_proj
- Max length: 512
- Metric: Accuracy

### Complex Tasks (Reasoning-based)

**Question Answering (QA)**
- Datasets: FinQA, ConvFinQA
- LoRA: r=16, α=32, dropout=0.05
- Modules: q_proj, k_proj, v_proj, o_proj
- Max length: 3500
- Metrics: EM, F1

**Stock Movement Prediction (SMP)**
- Datasets: BigData22, ACL18, CIKM18
- LoRA: r=16, α=32, dropout=0.05
- Modules: q_proj, k_proj, v_proj, o_proj
- Max length: 2048
- Metrics: Accuracy, MCC

### Structured Tasks

**Named Entity Recognition (NER)**
- Dataset: Financial Agreements
- LoRA: r=16, α=32, dropout=0.1
- Modules: q_proj, k_proj, v_proj, o_proj
- Max length: 1024
- Metric: Entity F1

## 💻 Usage Examples

### Python API

```python
from src.training import MultiTaskTrainer
from src.models import QLoRAModelManager

# Initialize model
model_manager = QLoRAModelManager(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
)
model, tokenizer = model_manager.setup_base_model()

# Train specific task
trainer = MultiTaskTrainer(model, tokenizer)
trainer.train_task(
    task="sa",
    datasets=["fpb", "fiqa_sa"],
    epochs=3,
    batch_size=4
)

# Save adapter
model_manager.save_task_adapter("sa", "./outputs/adapters/sa_adapter")

# Load and use adapter
model_manager.load_task_adapter("sa", "./outputs/adapters/sa_adapter")
model_manager.set_active_adapter("sa")
```

### Command Line

```bash
# Train with custom hyperparameters
python scripts/train_task.py \
    --task qa \
    --lr 2e-4 \
    --epochs 2 \
    --batch_size 2 \
    --lora_r 16 \
    --lora_alpha 32

# Evaluate with metrics
python scripts/evaluate_task.py \
    --task qa \
    --adapter_path ./outputs/adapters/qa_adapter \
    --test_file ./data/QA/finqa_test.csv \
    --output_file ./results/qa_results.json
```

## 📈 Expected Results

### Performance Summary

| Task | Dataset(s) | Metric | Our QLoRA | Baseline |
|------|-----------|--------|-----------|----------|
| SA | FPB | F1 | 86.0 | 87.0 |
| SA | FiQA-SA | F1 | 76.0 | 79.0 |
| HC | Gold Headlines | Acc | - | - |
| QA | FinQA | EM | 14.65 | 4.0 |
| QA | ConvFinQA | EM | 40.40 | 20.0 |
| SMP | Combined | Acc | 57.55 | 54.5 |
| NER | Fin Agreements | F1 | 69 | 69.0 |

### Efficiency Metrics

- **GPU Memory**: ~7 GB (vs 40 GB full fine-tuning)
- **Training Speed**: 2.5x faster than full fine-tuning
- **Trainable Parameters**: 0.17% of total model
- **Storage per Adapter**: ~27 MB

## 🔧 Configuration Files

### model_config.yaml
```yaml
model:
  name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_use_double_quant: true

training:
  optim: "paged_adamw_32bit"
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.01
  fp16: false
  bf16: true
  gradient_checkpointing: true
```

### Task Config Example (sa_config.yaml)
```yaml
task: "sa"
description: "Sentiment Analysis"
datasets:
  - "financial_phrase_bank"
  - "fiqa_sa"

lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  max_length: 512

evaluation:
  metric: "macro_f1"
  batch_size: 8
```

## 📝 Citation

```bibtex
@article{djagba2025qlora,
  title={Parameter-Efficient Fine-Tuning of Meta-Llama-3-8B-Instruct with
QLoRA for Financial NLP on the FLARE Benchmark: A Comparative
Analysis},
  author={Djagba, P. and Younoussi Saley, A. and Zeleke, A.},
  year={2025}
}
```

## 📧 Contact

- P. Djagba
- A. Younoussi Saley - saley.younoussi@aims.ac.rw
- A. Zeleke

---

**Note**: Training times are approximate and depend on GPU hardware (tested on A100 40GB or plus is preferable).
