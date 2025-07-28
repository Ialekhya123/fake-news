# 🔍 Fake News Detection Web Application

A complete machine learning web application for detecting fake news using Python, Streamlit, and scikit-learn. The application uses natural language processing techniques to classify news articles and headlines as either "Real" or "Fake".

## 🚀 Features

- **Machine Learning Model**: Trained using Logistic Regression with TF-IDF features
- **Text Preprocessing**: Advanced NLP techniques including lemmatization and stopword removal
- **Interactive Web Interface**: Clean, modern UI built with Streamlit
- **Real-time Analysis**: Instant predictions with confidence scores
- **Visual Analytics**: Interactive charts and probability breakdowns
- **Sample Dataset**: Included sample data for training and testing

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd fake-news
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data**
   ```bash
   python data_generator.py
   ```

4. **Train the model**
   ```bash
   python model_trainer.py
   ```

5. **Run the web application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
fake-news/
├── app.py                 # Main Streamlit web application
├── model_trainer.py       # Model training and evaluation script
├── data_generator.py      # Sample data generation script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── sample_news_data.csv  # Generated sample dataset
├── fake_news_model.pkl   # Trained model (generated after training)
└── confusion_matrix.png  # Model evaluation visualization
```

## 🔧 How It Works

### 1. Data Generation (`data_generator.py`)
- Creates a balanced dataset of real and fake news headlines
- Includes 20 real news examples and 20 fake news examples
- Saves data to `sample_news_data.csv`

### 2. Model Training (`model_trainer.py`)
- **Text Preprocessing**: 
  - Converts to lowercase
  - Removes special characters and numbers
  - Tokenizes text
  - Removes stopwords
  - Applies lemmatization
- **Feature Extraction**: Uses TF-IDF vectorization with 5000 features
- **Model Training**: Logistic Regression classifier
- **Evaluation**: Generates accuracy metrics and confusion matrix
- **Model Saving**: Saves trained model to `fake_news_model.pkl`

### 3. Web Application (`app.py`)
- **Interactive Interface**: Clean, responsive design
- **Real-time Prediction**: Instant analysis of input text
- **Visual Results**: Confidence gauges and probability charts
- **Sample Demonstrations**: Quick examples in sidebar

## 🎯 Usage

### Training the Model
```bash
# Generate sample data
python data_generator.py

# Train the model
python model_trainer.py
```

### Running the Web App
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application
1. **Enter Text**: Paste a news headline or article text in the input area
2. **Analyze**: Click the "Analyze Text" button
3. **View Results**: See the prediction, confidence score, and probability breakdown
4. **Interpret**: Use the insights and tips provided

## 📊 Model Performance

The trained model typically achieves:
- **Accuracy**: ~95% on the sample dataset
- **Precision**: High precision for both real and fake news detection
- **Recall**: Good recall for identifying fake news

## 🔍 Example Predictions

| Text | Prediction | Confidence |
|------|------------|------------|
| "Scientists discover new species in Amazon rainforest" | Real | 95% |
| "Aliens spotted in downtown New York" | Fake | 92% |
| "Global climate summit reaches historic agreement" | Real | 88% |
| "Time travel machine invented by high school student" | Fake | 89% |

## 🛡️ Important Notes

- **Educational Purpose**: This tool is designed for educational and demonstration purposes
- **Not Definitive**: Always verify information from multiple reliable sources
- **Sample Data**: The included dataset is small and for demonstration only
- **Model Limitations**: Performance may vary with different types of content

## 🔧 Customization

### Using Your Own Dataset
1. Replace `sample_news_data.csv` with your dataset
2. Ensure your dataset has columns: `text` (news content) and `label` (0 for fake, 1 for real)
3. Retrain the model using `model_trainer.py`

### Modifying the Model
- Change the algorithm in `model_trainer.py` (options: 'logistic', 'naive_bayes', 'random_forest')
- Adjust TF-IDF parameters for different feature extraction
- Modify preprocessing steps in the `preprocess_text` method

### Customizing the UI
- Modify the CSS styles in `app.py`
- Add new visualizations or metrics
- Customize the layout and components

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure you've run `python model_trainer.py` first
   - Check that `fake_news_model.pkl` exists in the project directory

2. **NLTK data not found**
   - The script automatically downloads required NLTK data
   - If issues persist, manually download: `python -m nltk.downloader punkt stopwords wordnet`

3. **Dependencies not installed**
   - Run `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

4. **Streamlit not working**
   - Check if Streamlit is installed: `pip install streamlit`
   - Try running: `streamlit run app.py --server.port 8501`

## 📈 Future Enhancements

- **Deep Learning Models**: Implement LSTM or BERT-based models
- **Real-time Data**: Connect to news APIs for live analysis
- **Multi-language Support**: Extend to other languages
- **Source Verification**: Add source credibility checking
- **User Feedback**: Collect user feedback to improve the model
- **API Endpoint**: Create REST API for integration with other applications

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the documentation
- Adding new datasets or models

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- Uses scikit-learn for machine learning
- NLTK for natural language processing
- Sample data generated for educational purposes

---

**⚠️ Disclaimer**: This tool is for educational purposes only. Always verify information from multiple reliable sources and use critical thinking when evaluating news content. 