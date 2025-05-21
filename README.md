# Diffusion Model Behavior in Low-Noise Regimes

This repository contains code and experiments for studying the behavior of diffusion models under low-noise conditions, with a focus on how models generalize vs memorize under different training conditions. We explore how score-based models behave when trained on varying dataset sizes, and evaluate their consistency, stability, and sensitivity to noise scale.

## Overview

We train and evaluate diffusion models using three core objective functions:
- **UNet denoisers** with reconstruction loss
- **Noise-conditional score networks (NCSN)**
- **Sliced score matching (SSM)**

The models are tested on a variety of datasets including:
- Subsets of **CelebA** with 1 to 100k samples
- **2D Gaussian mixtures** with interpretable structure
- **100D Gaussian mixtures** with isotropic and anisotropic covariances
- A **2D thin manifold** embedded in 100D space

## Repository Structure
- data/ Scripts for CelebA download and synthetic dataset generation
- train/ Training logic for each model type
- eval/ Scripts for trajectory divergence, cosine similarity, attractor stability
