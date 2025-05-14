# 📘 Document Classification with PyTorch and BERT

> 🗓️ **Project originally created on: January 14, 2024**

# 📘 Document Classification with PyTorch and BERT

This project explores two approaches to classifying humanitarian text documents into categories such as health, education, protection, and more. The dataset used is a real-world collection released by the DEEP project, focusing on international humanitarian response efforts.

I implement and compare two models:
- A **custom PyTorch classifier** using pretrained **Word2Vec embeddings**.
- A **fine-tuned DistilBERT transformer model** using Huggingface Transformers.

The goal is to build a clean, reproducible NLP pipeline that demonstrates how both traditional and transformer-based models can handle document classification tasks in real-world settings.

---

## ✨ Key Features

- Clean preprocessing pipeline with stemming, stopword removal, tokenization.
- Vocabulary building and out-of-vocabulary handling.
- Integration of pretrained **Google News Word2Vec embeddings**.
- Implementation of a **mean-pooling-based document classifier** in PyTorch.
- Fine-tuning of a **DistilBERT** model using Huggingface.
- Evaluation using **accuracy metrics**, early stopping, and logging with **TensorBoard** and **Weights & Biases (W&B)**.

---

## 🧠 Models Compared

### 1. PyTorch + Word2Vec
- Each document is converted into a fixed-size vector by averaging the embeddings of its words.
- Pretrained embeddings are used as initialization and fine-tuned during training.
- A fully connected classification layer maps embeddings to class probabilities.

### 2. DistilBERT Fine-Tuning
- Uses tokenization, attention masks, and CLS-based embedding extraction.
- Lightweight yet powerful transformer architecture.
- Only a few epochs needed for strong performance.

---

## 📁 Dataset

- A cleaned subset of the **HumSet** dataset, used for multilingual humanitarian text classification.
- More info: [https://blog.thedeep.io/humset/](https://blog.thedeep.io/humset/)

---

## 📌 Tools & Libraries

- PyTorch
- Huggingface Transformers
- NLTK
- Gensim (Word2Vec)
- Scikit-learn
- TensorBoard
- Weights & Biases

---

## 👤 Author and Creation Date 

**Adam Parkanyi**  
Machine Learning & NLP Enthusiast  

