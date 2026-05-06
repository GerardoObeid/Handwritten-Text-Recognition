# Handwritten Text Recognition (HTR) Pipeline

## 📖 Project Overview

This repository contains a complete, end-to-end Deep Learning pipeline for Handwritten Text Recognition (HTR). Built with PyTorch, the system is designed to read and transcribe unconstrained handwritten text ranging from single lines to full paragraphs (up to 13 lines).

The custom architecture solves the "vanishing gradient" and sequence-length problems typical in paragraph-level HTR by using a specialized attention mechanism. It combines a CNN for feature extraction, a `RowColLSTM` for spatial context, an `IterativeWeightedCollapse` module to dynamically isolate individual lines of text, and a Bidirectional LSTM decoder optimized with Connectionist Temporal Classification (CTC) Loss.

## 🗂️ Repository Structure

The project is modularized for easy maintenance and experimentation:

```text
Handwritten-Text-Recognition/
├── data/
│   ├── make_splits.py      # Generates image crops and splits data (Train/Val/Test)
│   └── dataset.py          # PyTorch Dataset (IAMParagraphDataset) and custom collate_fn
├── models/
│   ├── attention.py        # RowColLSTM and IterativeWeightedCollapse modules
│   └── htr_network.py      # The main FullParagraphHTR architecture
├── results/                # Stores inference outputs and visualizations
├── testing/
│   └── model_testing.ipynb # Jupyter notebook for evaluating and visualizing attention
├── utils/
│   └── metrics.py          # Vocabulary, JIWER metrics (CER/WER), and decoding logic
├── weights/                # Directory for saved .pth (training) and .pt (production) models
├── train.py                # Main training loop with Differential Learning Rates
├── .gitignore              # Ignores large datasets, weights, and caches
└── README.md               # Project documentation
```
