# One-Shot Facial Recognition via Siamese Neural Networks

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV%20%7C%20Face%20Verification-green.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Siamese%20Twin%20Network-purple.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-LFW%20Benchmark-orange.svg)](http://vis-www.cs.umass.edu/lfw/)
[![Author](https://img.shields.io/badge/Author-Dheeraj%20Kumar%20Maradana-blue.svg)](https://github.com/dheerajkumar2005)

---

## 📌 Executive Summary

This repository implements a **One-Shot Facial Verification Pipeline** utilizing deep **Siamese Neural Networks (SNNs)**.

Rather than training a standard multiclass classifier that requires hundreds of images per identity, the Siamese architecture learns a continuous similarity metric space. Given two input face images, twin subnetworks with shared weights extract feature embedding vectors, and a specialized distance layer computes pairwise similarity to verify whether both images depict the same individual.

---



### Key Technical Aspects:
1. **L1 Distance Embedding**:
   $$d_1 = |f(x_1) - f(x_2)|$$
2. **Binary Cross-Entropy Loss**:
   Optimizes twin network weights to pull embeddings of matching identities together while repelling distinct identities beyond a margin boundary.
3. **Dataset Ingestion**:
   * Negative samples sourced from the **Labeled Faces in the Wild (LFW)** dataset (13,000+ faces).
   * Anchor and positive sample capture via real-time **OpenCV webcam video streams**.

---

## 📁 Repository Structure

```
├── Facial_Recognition.ipynb    # Complete pipeline: data loading, twin network training & real-time webcam verification
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/dheerajkumar2005/Siamese-Face-Recognition-Pipeline.git
cd Siamese-Face-Recognition-Pipeline

pip install opencv-python tensorflow numpy matplotlib

# Launch interactive verification notebook
jupyter notebook Facial_Recognition.ipynb
```
