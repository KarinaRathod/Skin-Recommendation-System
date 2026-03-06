import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple

# =============================================================================
# 1. UI & CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Aura AI | ML Recommender",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .recommender-card {
            background: linear-gradient(145deg, #ffffff, #f8fafc);
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .tag {
            background: #eff6ff;
            color: #1d4ed8;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin: 0 4px 4px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 2. DATA LAYER
# =============================================================================
class DataLoader:
    """Handles raw data ingestion and basic integrity checks."""
    @staticmethod
    @st.cache_data
    def load_data():
        try:
            users = pd.read_csv('users.csv')
            products = pd.read_csv('products.csv')
            interactions = pd.read_csv('interactions.csv')
            return users, products, interactions
        except Exception as e:
            st.error(f"Data Loading Error: Ensure users.csv, products.csv, interactions.csv exist. Details: {e}")
            st.stop()

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    """Transforms raw data into ML-ready vectorized features."""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        self.scaler = MinMaxScaler()

    def process_products(self, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        df = products_df.copy()
        # Convert pip-separated ingredients to space-separated for TF-IDF
        df['Ingredients_Clean'] = df['Ingredients'].fillna('').apply(lambda x: str(x).replace('|', ' '))
        
        # 1. Content Embeddings (TF-IDF)
        tfidf_matrix = self.tfidf.fit_transform(df['Ingredients_Clean']).toarray()
        
        # 2. Normalize Prices
        df['Price_Norm'] = self.scaler.fit_transform(df[['Price']])
        
        return df, tfidf_matrix

    def get_cold_start_vector(self, user_row: pd.Series) -> np.ndarray:
        """Heuristic mapping to generate a synthetic TF-IDF vector for new users."""
        target_terms = []
        if user_row.get('Acne_Severity', 0) > 5: target_terms.extend(['Salicylic_Acid', 'Niacinamide'])
        if user_row.get('Dryness_Severity', 0) > 5: target_terms.extend(['Hyaluronic_Acid', 'Ceramides'])
        if user_row.get('Aging_Severity', 0) > 5: target_terms.extend(['Retinol', 'Peptides'])
        
        synthetic_text = " ".join(target_terms)
        return self.tfidf.transform([synthetic_text]).toarray()[0]

# =============================================================================
# 4. ML RECOMMENDER MODEL
# =============================================================================
class HybridRecommender:
    """Core ML Engine utilizing Matrix Factorization and Content Vectors."""
    
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.is_fitted = False

    def fit(self, users_df, products_df, interactions_df, feature_engineer):
        self.users = users_df.set_index('User_ID')
        self.products, self.tfidf_matrix = feature_engineer.process_products(products_df)
        self.interactions = interactions_df
        self.fe = feature_engineer
        
        # Build User-Item Interaction Matrix
        self.ui_matrix = interactions_df.pivot(index='User_ID', columns='Product_ID', values='User_Rating').fillna(0)
        self.user_ids = self.ui_matrix.index.tolist()
        self.product_ids = self.ui_matrix.columns.tolist()
        
        # Matrix Factorization (Collaborative Filtering)
        # Adjust n_components if matrix is smaller than requested components
        actual_components = min(self.n_components, self.ui_matrix.shape[1] - 1)
        if actual_components != self.n_components:
            self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
            
        self.user_factors = self.svd.fit_transform(self.ui_matrix)
        self.item_factors = self.svd.components_
        
        self.is_fitted = True

    def _get_user_content_vector(self, user_id: int) -> np.ndarray:
        """Builds a user profile vector based on historically liked items."""
        user_history = self.interactions[(self.interactions['User_ID'] == user_id) & (self.interactions['User_Rating'] >= 3.5)]
        
        if user_history.empty:
            # Cold Start Handle
            user_row = self.users.loc[user_id]
            return self.fe.get_cold_start_vector(user_row)
            
        # Weighted average of liked product TF-IDF vectors
        liked_product_ids = user_history['Product_ID'].tolist()
        ratings = user_history['User_Rating'].values
        
        idx_mask = self.products['Product_ID'].isin(liked_product_ids)
        liked_vectors = self.tfidf_matrix[idx_mask]
        
        if len(liked_vectors) == 0:
            return self.fe.get_cold_start_vector(self.users.loc[user_id])
            
        weights = ratings.reshape(-1, 1)[:len(liked_vectors)]
        user_vector = np.average(liked_vectors, axis=0, weights=weights.flatten())
        return user_vector

    def maximal_marginal_relevance(self, item_scores: dict, top_n: int, diversity_penalty: float) -> List[int]:
        """Applies MMR to ensure diverse recommendations."""
        selected = []
        candidates = list(item_scores.keys())
        
        while len(selected) < top_n and candidates:
            if not selected:
                # First item is just the highest scoring one
                best_item = max(candidates, key=lambda x: item_scores[x])
            else:
                # Calculate penalty based on similarity to already selected items
                best_item = None
                best_mmr_score = -float('inf')
                
                selected_indices = [self.products.index[self.products['Product_ID'] == idx].tolist()[0] for idx in selected]
                selected_vectors = self.tfidf_matrix[selected_indices]
                
                for cand in candidates:
                    cand_idx = self.products.index[self.products['Product_ID'] == cand].tolist()[0]
                    cand_vector = self.tfidf_matrix[cand_idx].reshape(1, -1)
                    
                    # Max cosine similarity to any already selected item
                    sims = cosine_similarity(cand_vector, selected_vectors)[0]
                    max_sim = max(sims) if len(sims) > 0 else 0
                    
                    # MMR Equation: alpha * Score - (1 - alpha) * Max_Similarity
                    mmr_score = (1 - diversity_penalty) * item_scores[cand] - (diversity_penalty * max_sim)
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_item = cand
                        
            selected.append(best_item)
            candidates.remove(best_item)
            
        return selected

    def recommend(self, user_id: int, top_n=5, alpha=0.5, diversity=0.2, explore_exploit=0.0):
        if not self.is_fitted: raise ValueError("Model not fitted.")
        
        # 1. Content-Based Scores (Vector Dot Product)
        user_vector = self._get_user_content_vector(user_id).reshape(1, -1)
        cb_scores = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        
        # Normalize CB scores to 0-1
        cb_scaler = MinMaxScaler()
        cb_scores_norm = cb_scaler.fit_transform(cb_scores.reshape(-1, 1)).flatten()
        
        # 2. Collaborative Filtering Scores (Matrix Reconstruction)
        if user_id in self.user_ids:
            u_idx = self.user_ids.index(user_id)
            cf_scores = np.dot(self.user_factors[u_idx, :], self.item_factors)
        else:
            cf_scores = np.zeros(len(self.product_ids)) # Cold start CF
            
        cf_scaler = MinMaxScaler()
        cf_scores_norm = cf_scaler.fit_transform(cf_scores.reshape(-1, 1)).flatten()
        
        # 3. Hybrid Aggregation
        hybrid_scores = (alpha * cb_scores_norm) + ((1 - alpha) * cf_scores_norm)
        
        # 4. Exploration noise injection
        if explore_exploit > 0:
            noise = np.random.normal(0, explore_exploit, len(hybrid_scores))
            hybrid_scores += noise
            
        # Map back to product IDs
        product_ids = self.products['Product_ID'].values
        score_dict = {pid: score for pid, score in zip(product_ids, hybrid_scores)}
        
        # 5. Apply MMR for Diversity
        final_ids = self.maximal_marginal_relevance(score_dict, top_n, diversity)
        
        # Prepare output dataframe
        results = []
        for pid in final_ids:
            idx = np.where(product_ids == pid)[0][0]
            prod_info = self.products.iloc[idx].to_dict()
            prod_info['Hybrid_Score'] = hybrid_scores[idx]
            prod_info['CB_Score'] = cb_scores_norm[idx]
            prod_info['CF_Score'] = cf_scores_norm[idx]
            # Confidence metric based on number of CF ratings + CB max similarity
            prod_info['Confidence'] = min(1.0, (cb_scores_norm[idx] * 0.6) + (0.4 if user_id in self.user_ids else 0.1))
            results.append(prod_info)
            
        return pd.DataFrame(results)

# =============================================================================
# 5. EVALUATOR MODULE
# =============================================================================
class Evaluator:
    """Computes ranking metrics for the recommender system."""
    
    @staticmethod
    def evaluate_model(recommender: HybridRecommender, test_interactions: pd.DataFrame, k=5):
        """Calculates Mean Precision@K and Mean Recall@K"""
        precisions = []
        recalls = []
        
        # Sample users to speed up evaluation in UI
        eval_users = test_interactions['User_ID'].unique()[:50] 
        
        for uid in eval_users:
            # Ground truth: items user rated >= 4.0
            user_data = test_interactions[(test_interactions['User_ID'] == uid) & (test_interactions['User_Rating'] >= 4.0)]
            relevant_items = set(user_data['Product_ID'].values)
            
            if not relevant_items: continue
            
            # Predict top K
            recs = recommender.recommend(uid, top_n=k, alpha=0.5, diversity=0.0)
            pred_items = set(recs['Product_ID'].values)
            
            hits = len(relevant_items.intersection(pred_items))
            precisions.append(hits / k)
            recalls.append(hits / len(relevant_items))
            
        return {
            f"MAP@{k}": np.mean(precisions),
            f"Mean Recall@{k}": np.mean(recalls)
        }



# =============================================================================
# 6. EXPLAINABILITY MODULE
# =============================================================================
class ExplainabilityModule:
    """Generates visual and textual explanations for AI predictions."""
    
    @staticmethod
    def plot_score_decomposition(row):
        fig = go.Figure(go.Bar(
            x=[row['CB_Score'], row['CF_Score']],
            y=['Content (Ingredients)', 'Collaborative (Community)'],
            orientation='h',
            marker_color=['#3b82f6', '#10b981']
        ))
        fig.update_layout(
            title="AI Score Decomposition",
            xaxis_title="Normalized Vector Score",
            height=200, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

# =============================================================================
# 7. MAIN STREAMLIT APP
# =============================================================================
@st.cache_resource
def init_system():
    users_df, products_df, interactions_df = DataLoader.load_data()
    fe = FeatureEngineer()
    model = HybridRecommender(n_components=20)
    model.fit(users_df, products_df, interactions_df, fe)
    return users_df, products_df, interactions_df, model

def main():
    inject_custom_css()
    users_df, products_df, interactions_df, recommender = init_system()

    # --- SIDEBAR: ML TUNING ---
    with st.sidebar:
        st.markdown("### 🧠 ML Hyperparameters")
        alpha = st.slider("Hybrid Alpha (CB vs CF)", 0.0, 1.0, 0.6, 
                          help="1.0 = Pure Content-Based. 0.0 = Pure Matrix Factorization.")
        diversity = st.slider("MMR Diversity Penalty", 0.0, 1.0, 0.2,
                              help="Penalizes items that are mathematically too similar to already recommended items.")
        exploration = st.slider("Exploration Noise", 0.0, 0.5, 0.05,
                                help="Injects Gaussian noise into final scores to surface novel products.")
        top_n = st.number_input("Top N Recommendations", min_value=1, max_value=20, value=4)
        
        st.divider()
        st.markdown("### 👤 User Context")
        user_list = users_df['User_ID'].head(100).tolist()
        selected_user = st.selectbox("Select Target User ID", user_list)
        user_data = users_df[users_df['User_ID'] == selected_user].iloc[0]

    # --- MAIN UI ---
    st.markdown("<h1>Aura ML | Hybrid Recommendation Architecture</h1>", unsafe_allow_html=True)
    
    tabs = st.tabs(["🚀 Live Inference", "📊 Offline Evaluation Metrics", "🧬 Vector Space Analysis"])
    
    # TAB 1: INFERENCE
    with tabs[0]:
        st.markdown(f"**Target Inference Profile:** User `{selected_user}` (Skin: {user_data['Skin_Type']}, Climate: {user_data['Climate']})")
        
        if st.button("Generate Matrix Factorization Outputs", type="primary"):
            with st.spinner("Computing dot products & applying MMR..."):
                predictions = recommender.recommend(
                    user_id=selected_user, 
                    top_n=top_n, 
                    alpha=alpha, 
                    diversity=diversity, 
                    explore_exploit=exploration
                )
            
            st.markdown("### 🎯 Top-K Ranked Candidates")
            for _, row in predictions.iterrows():
                # Extract Top TF-IDF Features (Ingredients)
                ings = [i.replace('_', ' ') for i in row['Ingredients_Clean'].split()]
                ing_html = "".join([f"<span class='tag'>{i}</span>" for i in ings])
                
                # Render UI Card
                st.markdown(f"""
                <div class="recommender-card">
                    <div style="display:flex; justify-content:space-between;">
                        <h3 style="margin:0; color:#0f172a;">{row['Brand']} • {row['Product_ID']}</h3>
                        <div style="text-align:right;">
                            <span style="font-size:1.5rem; font-weight:800; color:#3b82f6;">{(row['Hybrid_Score']*100):.1f}</span><br>
                            <span style="font-size:0.8rem; color:#64748b;">HYBRID SCORE</span>
                        </div>
                    </div>
                    <div style="color:#10b981; font-weight:700; margin-bottom:12px;">Est. Confidence: {row['Confidence']*100:.0f}%</div>
                    <div>{ing_html}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable XAI Module
                with st.expander(f"🔍 Explainability (Product {row['Product_ID']})"):
                    st.plotly_chart(ExplainabilityModule.plot_score_decomposition(row), use_container_width=True)

    # TAB 2: EVALUATION
    with tabs[1]:
        st.markdown("### 📈 Offline Ranking Metrics")
        st.write("Evaluating current model hyperparameters against historical interaction data...")
        
        if st.button("Run Evaluation Suite"):
            with st.spinner("Calculating Precision@K and Recall@K..."):
                metrics = Evaluator.evaluate_model(recommender, interactions_df, k=top_n)
                
                c1, c2 = st.columns(2)
                c1.metric(f"Mean Average Precision (MAP@{top_n})", f"{metrics[f'MAP@{top_n}']:.4f}")
                c2.metric(f"Mean Recall@{top_n}", f"{metrics[f'Mean Recall@{top_n}']:.4f}")
                st.caption("Note: Metrics are calculated dynamically based on the current hybrid weight (`alpha`). Tune the sidebar to optimize MAP.")

    # TAB 3: VECTOR SPACE
    with tabs[2]:
        st.markdown("### 🌌 User Profile Vectorization (TF-IDF)")
        st.write("Visualizing the mathematical representation of the user's preferences based on their interaction history.")
        
        user_vec = recommender._get_user_content_vector(selected_user)
        # Get top 10 features
        top_indices = user_vec.argsort()[-10:][::-1]
        feature_names = recommender.fe.tfidf.get_feature_names_out()
        
        top_features = [feature_names[i] for i in top_indices if user_vec[i] > 0]
        top_weights = [user_vec[i] for i in top_indices if user_vec[i] > 0]
        
        if top_features:
            fig = px.bar(x=top_weights, y=top_features, orientation='h', title="User Preference Embeddings (Top Features)")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cold Start: Not enough feature data to plot vector. Falling back to demographic heuristics.")

if __name__ == "__main__":
    main()