# Protein Fitness Prediction Framework

A comprehensive framework for predicting protein fitness scores using combined sequence and structural data. This repository contains code to generate protein sequence embeddings using ESM-2, extract structural features from PDB files, and train various neural network models to predict protein fitness.

## Overview

ProteinFitPred integrates both sequence and structural information to predict protein fitness scores, leveraging deep learning techniques:

1. **Sequence Embeddings**: Generate high-quality protein sequence embeddings using ESM-2 large language model with LoRA fine-tuning.
2. **Structural Features**: Extract structural information from protein PDB files using a CNN-based approach.
3. **Combined Models**: Train neural networks that integrate both sequence and structural features for optimal prediction.

The framework follows the architecture described in the paper [insert paper reference if applicable].

## Repository Structure
├── make_embeddings.py    # Generate embeddings from protein sequences using ESM-2 <br>
├── cnn.py                # Extract structural features from PDB files <br>
├── baseline_nn.py        # Simple neural network using only sequence embeddings <br>
├── combined_nn.py        # Combined model using both sequence and structural embeddings <br>
└── README.md             # This file <br>
