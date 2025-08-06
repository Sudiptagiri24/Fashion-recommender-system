# 👗 Fashion Recommender System

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen?style=for-the-badge&logo=streamlit)](https://mainpy-kwxlg4capjfb7kaqknlwlm.streamlit.app/)

A deep learning-powered fashion recommendation system that helps users find visually similar fashion items using image embeddings and a ResNet50 model. Built using Streamlit and TensorFlow, and deployed live on the cloud.

---

## 🚀 Live Demo

👉 **[Click here to try the app](https://mainpy-kwxlg4capjfb7kaqknlwlm.streamlit.app/)**

---

## 📸 How It Works

1. Upload an image of a fashion item (e.g., t-shirt, dress, etc.)
2. The app extracts features using **ResNet50** (pretrained on ImageNet)
3. Embeddings are compared using **k-Nearest Neighbors**
4. Top-5 most visually similar fashion items are shown

---

## 🧠 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **Backend:** TensorFlow (ResNet50)
- **Similarity:** scikit-learn (k-NN)
- **Hosting:** Streamlit Community Cloud
- **Storage:** Embeddings from Google Drive

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sudiptagiri24/Fashion-recommender-system.git
cd Fashion-recommender-system
