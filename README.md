---


# Aura AI | Hybrid Skincare Recommendation Engine 🧠✨

Aura AI is a production-grade, Machine Learning-powered recommendation system built with Python and Streamlit. It transitions beyond simple rule-based heuristics to utilize a true **Hybrid Recommender Architecture**, blending Natural Language Processing (NLP) for ingredient analysis with Matrix Factorization for community-driven collaborative filtering.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## 🚀 Key Features

* **Content-Based Filtering (NLP):** Utilizes `TF-IDF` vectorization to map product ingredients into a high-dimensional feature space, matching them against user preference embeddings using Cosine Similarity.
* **Collaborative Filtering (SVD):** Implements Matrix Factorization via `TruncatedSVD` on the User-Item interaction matrix to uncover latent features and predict user ratings based on community behavior.
* **Dynamic Hybrid Scoring:** A tunable `alpha` parameter allows real-time adjustment between Content-Based (Biometric) and Collaborative (Community) weighting.
* **Maximal Marginal Relevance (MMR):** Enforces recommendation diversity by penalizing mathematically similar products in the final ranking sequence.
* **Offline Evaluation Suite:** Built-in calculation of `Precision@K`, `Recall@K`, and `Mean Average Precision (MAP)` to evaluate hyperparameter efficacy.
* **Explainable AI (XAI):** Visual score decomposition (Plotly) showing exactly *why* a product was recommended, breaking down the hybrid score components.

## 🏗️ Architecture

The system is built using Object-Oriented Programming (OOP) principles, ensuring modularity and scalability:

1.  **`DataLoader`**: Ingests and caches the raw CSV data.
2.  **`FeatureEngineer`**: Handles TF-IDF vectorization and data normalization.
3.  **`HybridRecommender`**: The core ML engine housing the SVD model, dot-product calculations, and MMR sorting logic.
4.  **`Evaluator`**: Computes ground-truth validation metrics against historical interactions.
5.  **`ExplainabilityModule`**: Generates Plotly-based visual representations of the AI's decision-making process.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/aura-ai-recommender.git](https://github.com/yourusername/aura-ai-recommender.git)
   cd aura-ai-recommender

```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


*(Ensure your `requirements.txt` includes: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`)*
4. **Ensure Data Files are present:**
Place `users.csv`, `products.csv`, and `interactions.csv` in the root directory.
5. **Run the application:**
```bash
streamlit run app.py

```



## 🎛️ Usage Guide

* **Live Inference Tab:** Select a user profile and tweak the ML Hyperparameters (Alpha, Diversity Penalty, Exploration Noise) in the sidebar to see real-time shifts in the product rankings.
* **Offline Evaluation Tab:** Run the evaluation suite to see how your current hyperparameter tuning affects MAP and Recall metrics against historical ground-truth data.
* **Vector Space Analysis:** Visualize the TF-IDF feature importance matrix that represents the selected user's ingredient preferences.



```

```