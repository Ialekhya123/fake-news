# Simplified version for deployment - minimal NLTK dependency
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import plotly.graph_objects as go
import plotly.express as px
import time

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .fake-news {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .real-news {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .analysis-section {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 1.2rem;
        color: #8e44ad;
        margin: 1rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .recommendation-section {
        font-family: 'Arial', 'Helvetica', sans-serif;
        font-size: 1.1rem;
        color: #e67e22;
        margin: 1rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .insight-item {
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #17a2b8;
        color: #2c3e50;
    }
    .insight-warning {
        color: #e74c3c;
        font-weight: bold;
    }
    .insight-info {
        color: #3498db;
        font-weight: bold;
    }
    .insight-success {
        color: #27ae60;
        font-weight: bold;
    }
    .recommendation-high {
        color: #27ae60;
        background-color: #d5f4e6;
        border-left-color: #27ae60;
    }
    .recommendation-moderate {
        color: #f39c12;
        background-color: #fef9e7;
        border-left-color: #f39c12;
    }
    .recommendation-low {
        color: #3498db;
        background-color: #ebf3fd;
        border-left-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

class SimpleFakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        # Simple stop words list (no NLTK dependency)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
            "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
            'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }
        
    def simple_preprocess_text(self, text):
        """Simple text preprocessing without NLTK"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)
        
        # Simple tokenization and stop word removal
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def extract_text_features(self, text):
        """Extract additional text-based features"""
        features = {}
        
        # Text length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Suspicious words/phrases
        suspicious_words = ['miracle', 'secret', 'government admits', 'scientists discover', 
                          'breakthrough', 'revolutionary', 'incredible', 'amazing', 'shocking',
                          'unbelievable', 'impossible', 'never before seen', 'world first']
        
        features['suspicious_word_count'] = sum(1 for word in suspicious_words if word.lower() in text.lower())
        
        return features
    
    def load_model(self, model_path='fake_news_model.pkl'):
        """Load the trained model and vectorizer"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, text):
        """Predict if a given text is fake or real news"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess the text
        processed_text = self.simple_preprocess_text(text)
        
        # Extract features
        features = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Extract text features for analysis
        text_features = self.extract_text_features(text)
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probability),
            'probabilities': {
                'Fake': probability[0],
                'Real': probability[1]
            },
            'text_features': text_features
        }

@st.cache_resource
def load_detector():
    """Load the fake news detector model"""
    try:
        detector = SimpleFakeNewsDetector()
        if detector.load_model():
            return detector
        else:
            return None
    except Exception as e:
        st.error(f"Error loading detector: {str(e)}")
        return None

def create_confidence_gauge(confidence, prediction):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {confidence:.1%}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_probability_bar(probabilities):
    """Create a bar chart for probabilities"""
    fig = px.bar(
        x=list(probabilities.keys()),
        y=list(probabilities.values()),
        color=list(probabilities.keys()),
        color_discrete_map={'Fake': '#ef5350', 'Real': '#66bb6a'},
        title="Prediction Probabilities"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    
    return fig

def analyze_text_features(text_features):
    """Analyze text features and provide insights"""
    insights = []
    
    if text_features['suspicious_word_count'] > 0:
        insights.append(f"<span class='insight-warning'>⚠️ Contains {text_features['suspicious_word_count']} suspicious words/phrases</span>")
    
    if text_features['exclamation_count'] > 2:
        insights.append("<span class='insight-warning'>⚠️ High number of exclamation marks (common in fake news)</span>")
    
    if text_features['capital_ratio'] > 0.3:
        insights.append("<span class='insight-warning'>⚠️ High use of capital letters (common in sensationalist content)</span>")
    
    if text_features['word_count'] < 5:
        insights.append("<span class='insight-info'>ℹ️ Very short text - consider providing more context</span>")
    
    if text_features['avg_word_length'] > 8:
        insights.append("<span class='insight-info'>ℹ️ Contains many long words (may indicate technical content)</span>")
    
    if not insights:
        insights.append("<span class='insight-success'>✅ Text features appear normal</span>")
    
    return insights

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze news articles and headlines to detect potential misinformation</p>', unsafe_allow_html=True)
    
    # Load model
    detector = load_detector()
    
    if detector is None:
        st.error("❌ Model not found! Please ensure the model file is uploaded to Streamlit Cloud.")
        st.info("Upload the 'fake_news_model.pkl' file to your Streamlit Cloud deployment.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Model Information")
    st.sidebar.markdown("""
    **Model Details:**
    - Algorithm: Naive Bayes (Best performing)
    - Features: TF-IDF Vectorization + Text Features
    - Accuracy: ~71% on test data
    - Dataset: 120 balanced examples
    
    **How it works:**
    1. Text preprocessing and cleaning
    2. Feature extraction using TF-IDF
    3. Additional text feature analysis
    4. Classification using trained model
    5. Confidence scoring
    """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter News Text")
        
        # Text input
        text_input = st.text_area(
            "Paste your news headline or article text here:",
            height=150,
            placeholder="Example: Scientists discover new breakthrough in renewable energy technology..."
        )
        
        # Predict button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # Add a small delay for better UX
                    time.sleep(0.5)
                    
                    try:
                        result = detector.predict(text_input)
                        
                        # Display result
                        st.subheader("📊 Analysis Results")
                        
                        # Result box
                        result_class = "real-news" if result['prediction'] == 'Real' else "fake-news"
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            Prediction: {result['prediction']} News
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence gauge
                        st.plotly_chart(create_confidence_gauge(result['confidence'], result['prediction']), use_container_width=True)
                        
                        # Probability breakdown
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric("Fake News Probability", f"{result['probabilities']['Fake']:.1%}")
                        with col_prob2:
                            st.metric("Real News Probability", f"{result['probabilities']['Real']:.1%}")
                        
                        # Probability bar chart
                        st.plotly_chart(create_probability_bar(result['probabilities']), use_container_width=True)
                        
                        # Text feature analysis
                        st.markdown('<h3 class="analysis-section">🔍 Text Analysis</h3>', unsafe_allow_html=True)
                        insights = analyze_text_features(result['text_features'])
                        
                        for insight in insights:
                            st.markdown(f"<div class='insight-item'>{insight}</div>", unsafe_allow_html=True)
                        
                        # Additional insights
                        st.markdown('<h3 class="recommendation-section">💡 Recommendations</h3>', unsafe_allow_html=True)
                        if result['confidence'] > 0.8:
                            st.markdown('<div class="info-box recommendation-high">✅ <strong>High confidence prediction</strong> - the model is very certain about this classification.</div>', unsafe_allow_html=True)
                        elif result['confidence'] > 0.6:
                            st.markdown('<div class="info-box recommendation-moderate">⚠️ <strong>Moderate confidence prediction</strong> - consider additional fact-checking.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box recommendation-low">ℹ️ <strong>Low confidence prediction</strong> - the model is uncertain. Manual verification recommended.</div>', unsafe_allow_html=True)
                        
                        # Warning about model limitations
                        st.markdown("""
                        <div class='info-box'>
                            <strong>⚠️ Important:</strong> This tool is for educational purposes only. 
                            Always verify information from multiple reliable sources and use critical thinking.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.subheader("📈 Quick Stats")
        
        # Sample predictions for demonstration
        sample_texts = [
            "Scientists discover new species in Amazon rainforest",
            "Aliens spotted in downtown New York, government denies everything",
            "Global climate summit reaches historic agreement",
            "Time travel machine invented by high school student",
            "New vaccine shows promising results in clinical trials",
            "Miracle cure discovered: one pill eliminates all diseases"
        ]
        
        st.markdown("**Sample Predictions:**")
        for text in sample_texts:
            try:
                result = detector.predict(text)
                color = "🟢" if result['prediction'] == 'Real' else "🔴"
                confidence_emoji = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.5 else "🔴"
                st.markdown(f"{color} **{result['prediction']}** {confidence_emoji} - {text[:25]}...")
            except:
                pass
        
        st.markdown("---")
        st.markdown("**Tips for better analysis:**")
        st.markdown("""
        - Include complete headlines or articles
        - Avoid very short text fragments
        - Consider the source and context
        - Use this as a tool, not the final verdict
        - Check multiple reliable sources
        """)
        
        # Model performance info
        st.markdown("---")
        st.markdown("**Model Performance:**")
        st.markdown("""
        - **Accuracy**: ~71%
        - **Dataset**: 120 examples
        - **Algorithm**: Naive Bayes
        - **Features**: TF-IDF + Text Analysis
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ This tool is for educational purposes. Always verify information from multiple reliable sources.</p>
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 