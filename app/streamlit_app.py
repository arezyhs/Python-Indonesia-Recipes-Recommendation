"""
Streamlit web application for Indonesian Recipe Recommendation System
"""
import streamlit as st
import pandas as pd
import sys
import os
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_processor import DataProcessor
from src.models.recommendation_system import RecipeRecommendationSystem


class RecipeApp:
    """Streamlit app for recipe recommendations."""
    
    def __init__(self):
        self.rec_system = None
        self.processor = None
        self.setup_page_config()
        
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="Indonesian Recipe Recommendations",
            page_icon="🍽️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @st.cache_resource
    def load_system(_self):
        """Load the recommendation system with caching."""
        try:
            # Initialize systems
            processor = DataProcessor()
            rec_system = RecipeRecommendationSystem()
            
            # Try to load pre-trained models
            if rec_system.load_models():
                st.success("✅ Pre-trained models loaded successfully!")
                return rec_system, processor
            else:
                # Train new models if not available
                st.info("🔄 Training new models...")
                with st.spinner("Processing data and training models..."):
                    df = processor.process_all()
                    rec_system.fit(df)
                    rec_system.save_models()
                
                st.success("✅ Models trained and saved successfully!")
                return rec_system, processor
                
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            return None, None
    
    def create_sidebar(self):
        """Create sidebar with navigation and controls."""
        with st.sidebar:
            # App Logo/Header
            st.markdown("""
            <div style="background: linear-gradient(90deg, #4CAF50, #45B7D1); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">🍽️</h2>
                <h3 style="color: white; margin: 5px 0; font-size: 18px;">Indonesian Recipe</h3>
                <p style="color: white; margin: 0; font-size: 14px; opacity: 0.9;">AI-Powered Finder</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation menu with improved styling
            selected = option_menu(
                menu_title="🧭 Navigation",
                options=["🔍 Recipe Search", "📊 Analytics", "ℹ️ About"],
                icons=["search", "bar-chart-line", "info-circle"],
                menu_icon="compass",
                default_index=0,
                styles={
                    "container": {"padding": "5px", "background-color": "#f8f9fa", "border-radius": "10px"},
                    "icon": {"color": "#4CAF50", "font-size": "18px"}, 
                    "nav-link": {
                        "font-size": "16px", 
                        "text-align": "left", 
                        "margin": "2px", 
                        "padding": "10px 15px",
                        "border-radius": "8px",
                        "--hover-color": "#e8f5e8"
                    },
                    "nav-link-selected": {
                        "background-color": "#4CAF50",
                        "color": "white",
                        "font-weight": "600"
                    },
                }
            )
            
            # Additional sidebar info
            st.markdown("---")
            st.markdown("""
            <div style="background: #f0f2f6; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0; color: #333; font-size: 14px;">💡 Tips Pencarian:</h4>
                <ul style="margin: 0; font-size: 12px; color: #666; line-height: 1.5;">
                    <li>Gunakan nama masakan: "rendang", "gado-gado"</li>
                    <li>Atau bahan utama: "ayam", "tahu", "udang"</li>
                    <li>Kombinasi: "ayam goreng kremes"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Model info in sidebar
            if hasattr(self.rec_system, 'evaluation_results'):
                eval_results = self.rec_system.evaluation_results
                st.markdown("""
                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4 style="margin: 0 0 10px 0; color: #2E7D32; font-size: 14px;">🎯 Model Performance:</h4>
                    <div style="font-size: 12px; color: #555;">
                        <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                            <span>SVM:</span><span><strong>{:.1%}</strong></span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                            <span>Random Forest:</span><span><strong>{:.1%}</strong></span>
                        </div>
                    </div>
                </div>
                """.format(
                    eval_results.get('svm_accuracy', 0),
                    eval_results.get('rf_accuracy', 0)
                ), unsafe_allow_html=True)
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 10px 0; color: #666; font-size: 12px;">
                <p style="margin: 5px 0;">🇮🇩 <strong>Indonesian Recipes</strong></p>
                <p style="margin: 5px 0;">Made with ❤️ using ML & NLP</p>
                <p style="margin: 5px 0; font-size: 10px; opacity: 0.7;">Portfolio Project 2025</p>
            </div>
            """, unsafe_allow_html=True)
            
            return selected
    
    def display_recipe_card(self, recipe: Dict[str, Any], show_full: bool = False):
        """Display a recipe card with ingredients and steps."""
        with st.container():
            # Recipe header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(recipe['title'])
            with col2:
                st.metric("Similarity", f"{recipe['similarity_score']:.3f}")
            
            # Category badge
            category_colors = {
                'Ayam': '#FF6B6B', 'Ikan': '#4ECDC4', 'Kambing': '#45B7D1',
                'Sapi': '#96CEB4', 'Tahu': '#FFEAA7', 'Tempe': '#DDA0DD',
                'Udang': '#98D8C8', 'Telur': '#F7DC6F', 'Lainnya': '#BDC3C7'
            }
            
            color = category_colors.get(recipe['category'], '#BDC3C7')
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 5px 10px; 
                        border-radius: 15px; display: inline-block; margin-bottom: 10px;">
                {recipe['category']}
            </div>
            """, unsafe_allow_html=True)
            
            if show_full or st.button(f"Show Recipe Details", key=f"btn_{recipe['recipe_index']}"):
                # Ingredients
                if recipe['ingredients']:
                    st.markdown("**🥘 Bahan-bahan:**")
                    for ingredient in recipe['ingredients']:
                        st.markdown(f"• {ingredient}")
                
                # Steps
                if recipe['steps']:
                    st.markdown("**👨‍🍳 Langkah-langkah:**")
                    for i, step in enumerate(recipe['steps'], 1):
                        st.markdown(f"{i}. {step}")
            
            st.markdown("---")
    
    def search_page(self):
        """Recipe search page."""
        st.title("🔍 Indonesian Recipe Finder")
        st.markdown("Temukan resep masakan Indonesia yang sesuai dengan selera Anda!")
        
        # Search section
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query = st.text_input(
                    "🔎 Masukkan kata kunci resep:",
                    placeholder="Contoh: ayam goreng, rendang, gado-gado...",
                    help="Ketik bahan makanan atau nama masakan yang ingin Anda cari"
                )
            
            with col2:
                num_recommendations = st.selectbox(
                    "Jumlah rekomendasi:",
                    [3, 5, 10, 15],
                    index=1
                )
        
        # Advanced options
        with st.expander("🔧 Opsi Lanjutan"):
            col1, col2 = st.columns(2)
            
            with col1:
                category_filter = st.selectbox(
                    "Filter berdasarkan kategori:",
                    ["Semua"] + list(self.rec_system.df['Category'].unique()) if self.rec_system and self.rec_system.df is not None else ["Semua"]
                )
            
            with col2:
                prediction_model = st.selectbox(
                    "Model prediksi kategori:",
                    ["SVM", "Random Forest"],
                    help="Pilih model untuk prediksi kategori resep"
                )
        
        if query:
            with st.spinner("🔄 Mencari resep yang cocok..."):
                try:
                    # Get recommendations
                    recommendations = self.rec_system.recommend_recipes(query, num_recommendations)
                    
                    # Filter by category if specified
                    if category_filter != "Semua":
                        recommendations = [r for r in recommendations if r['category'] == category_filter]
                    
                    if recommendations:
                        st.success(f"✅ Ditemukan {len(recommendations)} resep yang cocok!")
                        
                        # Show prediction for the query
                        try:
                            model_type = 'svm' if prediction_model == 'SVM' else 'rf'
                            predicted_category, confidence = self.rec_system.predict_category(query, model_type)
                            
                            st.info(f"🎯 Model memprediksi query Anda termasuk kategori: **{predicted_category}** "
                                   f"(Confidence: {confidence:.3f})")
                        except:
                            pass
                        
                        # Display recommendations
                        for i, recipe in enumerate(recommendations, 1):
                            st.markdown(f"### #{i}")
                            self.display_recipe_card(recipe)
                    else:
                        st.warning("❌ Tidak ada resep yang ditemukan untuk kata kunci tersebut.")
                        st.info("💡 Coba gunakan kata kunci yang berbeda atau lebih umum.")
                        
                except Exception as e:
                    st.error(f"❌ Error saat mencari resep: {e}")
    
    def analytics_page(self):
        """Analytics and statistics page."""
        st.title("📊 Recipe Analytics")
        st.markdown("Analisis dan statistik dari database resep Indonesia")
        
        if self.rec_system is None or self.rec_system.df is None:
            st.error("❌ Data tidak tersedia. Silakan kembali ke halaman pencarian untuk memuat data.")
            return
        
        df = self.rec_system.df
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resep", len(df))
        
        with col2:
            st.metric("Kategori", df['Category'].nunique())
        
        with col3:
            avg_ingredients = df['Ingredients'].apply(lambda x: len(str(x).split('--')) if pd.notna(x) else 0).mean()
            st.metric("Rata-rata Bahan", f"{avg_ingredients:.1f}")
        
        with col4:
            avg_steps = df['Steps'].apply(lambda x: len(str(x).split('--')) if pd.notna(x) else 0).mean()
            st.metric("Rata-rata Langkah", f"{avg_steps:.1f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            st.subheader("📈 Distribusi Kategori Resep")
            category_counts = df['Category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribusi Kategori Resep",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Text length distribution
            st.subheader("📏 Distribusi Panjang Teks")
            text_lengths = df['text_cleaned'].str.len()
            
            fig = px.histogram(
                x=text_lengths,
                nbins=30,
                title="Distribusi Panjang Teks Resep",
                color_discrete_sequence=['#4CAF50']
            )
            fig.update_layout(
                xaxis_title="Panjang Teks", 
                yaxis_title="Frekuensi",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance
        if hasattr(self.rec_system, 'evaluation_results'):
            st.subheader("🎯 Performa Model")
            
            eval_results = self.rec_system.evaluation_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("SVM Accuracy", f"{eval_results.get('svm_accuracy', 0):.3f}")
            
            with col2:
                st.metric("Random Forest Accuracy", f"{eval_results.get('rf_accuracy', 0):.3f}")
        
        # Top recipes by category
        st.subheader("🏆 Sample Resep per Kategori")
        
        for category in df['Category'].unique():
            if category != 'Lainnya':  # Skip 'Lainnya' category
                category_df = df[df['Category'] == category]
                if len(category_df) > 0:
                    sample_recipe = category_df.iloc[0]
                    
                    with st.expander(f"{category} - {sample_recipe['Title'].title()}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Bahan-bahan:**")
                            ingredients = str(sample_recipe['Ingredients']).split('--')[:5]  # First 5 ingredients
                            for ingredient in ingredients:
                                if ingredient.strip():
                                    st.markdown(f"• {ingredient.strip()}")
                            if len(str(sample_recipe['Ingredients']).split('--')) > 5:
                                st.markdown("• ...")
                        
                        with col2:
                            st.markdown("**Langkah-langkah:**")
                            steps = str(sample_recipe['Steps']).split('--')[:3]  # First 3 steps
                            for i, step in enumerate(steps, 1):
                                if step.strip():
                                    st.markdown(f"{i}. {step.strip()[:100]}...")
    
    def about_page(self):
        """About page with system information."""
        st.title("ℹ️ About Indonesian Recipe Recommendation System")
        
        # Project description
        st.markdown("""
        ## 🇮🇩 Sistem Rekomendasi Resep Masakan Indonesia
        
        Sistem rekomendasi ini menggunakan teknik **Machine Learning** dan **Natural Language Processing** 
        untuk memberikan rekomendasi resep masakan Indonesia berdasarkan input pengguna.
        
        ### 🎯 Fitur Utama:
        - **Pencarian Cerdas**: Menggunakan TF-IDF dan Cosine Similarity
        - **Klasifikasi Kategori**: SVM dan Random Forest untuk kategorisasi resep
        - **Interface Modern**: UI yang user-friendly dengan Streamlit
        - **Analytics**: Statistik dan visualisasi data resep
        
        ### 🧠 Teknologi yang Digunakan:
        - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Untuk ekstraksi fitur teks
        - **Cosine Similarity**: Untuk mengukur kemiripan antar resep
        - **Support Vector Machine (SVM)**: Untuk klasifikasi kategori resep
        - **Random Forest**: Model ensemble untuk klasifikasi
        - **Streamlit**: Framework web untuk interface
        - **NLTK**: Natural Language Processing
        
        ### 📊 Dataset:
        Dataset terdiri dari ribuan resep masakan Indonesia yang dikategorikan berdasarkan:
        """)
        
        # Categories
        if self.rec_system and self.rec_system.df is not None:
            categories = self.rec_system.df['Category'].value_counts()
            
            for category, count in categories.items():
                st.markdown(f"- **{category}**: {count} resep")
        
        # Model information
        if self.rec_system:
            st.markdown("### 🔧 Informasi Model:")
            model_info = self.rec_system.get_model_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"- **Total Resep**: {model_info.get('total_recipes', 0)}")
                st.markdown(f"- **Fitur TF-IDF**: {model_info.get('tfidf_features', 0)}")
                st.markdown(f"- **Model SVM**: {'✅ Tersedia' if model_info.get('svm_fitted', False) else '❌ Tidak tersedia'}")
            
            with col2:
                st.markdown(f"- **Model Random Forest**: {'✅ Tersedia' if model_info.get('rf_fitted', False) else '❌ Tidak tersedia'}")
                st.markdown(f"- **TF-IDF Vectorizer**: {'✅ Tersedia' if model_info.get('tfidf_fitted', False) else '❌ Tidak tersedia'}")
                st.markdown(f"- **Data**: {'✅ Dimuat' if model_info.get('data_loaded', False) else '❌ Tidak dimuat'}")
        
        # Contact/Credits
        st.markdown("""
        ---
        ### 👨‍💻 Credits
        
        Sistem ini dikembangkan sebagai project portfolio untuk demonstrasi kemampuan dalam:
        - Machine Learning & Data Science
        - Natural Language Processing
        - Web Application Development
        - Data Visualization
        
        **Teknologi Stack**: Python, Scikit-learn, NLTK, Streamlit, Plotly, Pandas, NumPy
        """)
    
    def run(self):
        """Run the Streamlit application."""
        # Custom CSS
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric > label {
            font-size: 16px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Load system
        with st.spinner("🔄 Loading recommendation system..."):
            self.rec_system, self.processor = self.load_system()
        
        if self.rec_system is None:
            st.error("❌ Failed to load recommendation system. Please check your setup.")
            return
        
        # Create navigation
        selected = self.create_sidebar()
        
        # Route to appropriate page
        if selected == "🔍 Recipe Search":
            self.search_page()
        elif selected == "📊 Analytics":
            self.analytics_page()
        elif selected == "ℹ️ About":
            self.about_page()


def main():
    """Main function to run the Streamlit app."""
    app = RecipeApp()
    app.run()


if __name__ == "__main__":
    main()