import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
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
    
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        print("Loading data...")
        data = pd.read_csv(file_path)
        
        print("Preprocessing text...")
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        # Extract additional features
        print("Extracting text features...")
        text_features = data['text'].apply(self.extract_text_features)
        feature_df = pd.DataFrame(text_features.tolist())
        data = pd.concat([data, feature_df], axis=1)
        
        # Remove empty texts after preprocessing
        data = data[data['processed_text'].str.len() > 0]
        
        print(f"Dataset shape: {data.shape}")
        print(f"Real news: {len(data[data['label'] == 1])}")
        print(f"Fake news: {len(data[data['label'] == 0])}")
        
        return data
    
    def extract_features(self, texts, fit=True):
        """Extract TF-IDF features from text with improved parameters"""
        if fit:
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 3),  # Include trigrams
                min_df=1,
                max_df=0.9,
                sublinear_tf=True,  # Apply sublinear tf scaling
                use_idf=True
            )
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)
        
        return features
    
    def train_model(self, X_train, y_train, model_type='logistic'):
        """Train the classification model with hyperparameter tuning"""
        if model_type == 'logistic':
            # Grid search for best parameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            
        elif model_type == 'naive_bayes':
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
            base_model = MultinomialNB()
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            raise ValueError("Model type must be 'logistic', 'naive_bayes', or 'random_forest'")
        
        print(f"Training {model_type} model...")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Create confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], 
                   yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy, report
    
    def save_model(self, model_path='fake_news_model.pkl'):
        """Save the trained model and vectorizer"""
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='fake_news_model.pkl'):
        """Load the trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, text):
        """Predict if a given text is fake or real news"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Extract features
        features = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probability),
            'probabilities': {
                'Fake': probability[0],
                'Real': probability[1]
            }
        }

def main():
    """Main training function"""
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load and preprocess data
    data = detector.load_data('sample_news_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'], data['label'], 
        test_size=0.2, random_state=42, stratify=data['label']
    )
    
    # Extract features
    print("Extracting features...")
    X_train_features = detector.extract_features(X_train, fit=True)
    X_test_features = detector.extract_features(X_test, fit=False)
    
    # Train model with different algorithms and compare
    models = ['logistic', 'naive_bayes', 'random_forest']
    best_accuracy = 0
    best_model_type = None
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        detector.train_model(X_train_features, y_train, model_type=model_type)
        accuracy, report = detector.evaluate_model(X_test_features, y_test)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_type = model_type
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model_type.upper()} with accuracy: {best_accuracy:.4f}")
    print(f"{'='*50}")
    
    # Retrain with best model
    detector.train_model(X_train_features, y_train, model_type=best_model_type)
    
    # Save model
    detector.save_model()
    
    # Test predictions
    test_texts = [
        "Scientists discover new breakthrough in renewable energy technology",
        "Aliens spotted in downtown New York, government denies everything",
        "Global climate summit reaches historic agreement",
        "Time travel machine invented by high school student",
        "New vaccine shows promising results in clinical trials",
        "Miracle cure discovered: one pill eliminates all diseases"
    ]
    
    print("\nTest Predictions:")
    print("-" * 80)
    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        print(f"Probabilities: Fake={result['probabilities']['Fake']:.3f}, Real={result['probabilities']['Real']:.3f}")
        print("-" * 80)

if __name__ == "__main__":
    main() 