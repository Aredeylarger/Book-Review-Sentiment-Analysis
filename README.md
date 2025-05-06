# Sentiment Analysis Project


## Overview
This repository contains a sentiment analysis project that classifies e-commerce reviews as positive or negative using machine learning and natural language processing techniques. The project evaluates multiple models, including Logistic Regression, SGDClassifier, Multinomial Naive Bayes, and VADER sentiment analysis. The best-performing model, Logistic Regression with TF-IDF vectorization, achieves an accuracy of 90.51% on the test dataset. A Streamlit-based online web interface is provided for real-time sentiment prediction, allowing users to input reviews and receive instant sentiment analysis.

## Data Location
The File data file is a large and here is the link to the data on dropbox for your download.
https://www.dropbox.com/scl/fo/kjzl0yi4s1ejuqgbrbjtq/AMY-13ewK99DQ7aqRGeP3yc?rlkey=ol3r7w307bfmuvsoe26oz0nqo&st=jmzb2nb9&dl=0

## Features
- Data Preprocessing: Cleans text data by removing stopwords and preparing it for model training.
- Feature Extraction: Uses Bag of Words (BoW) and TF-IDF vectorization for text representation.
- Machine Learning Models:
    - Logistic Regression
    - SGDClassifier
    - Multinomial Naive Bayes
    - VADER Sentiment Analyzer
- Model Evaluation: Provides accuracy, precision, recall, and F1-score metrics for comprehensive evaluation.
- Online Web Interface: A user-friendly Streamlit web application enables real-time sentiment prediction through a browser-based interface.
- Model Persistence: Saves the trained Logistic Regression model and TF-IDF vectorizer using joblib for deployment.

## Dataset
The dataset consists of e-commerce reviews split into training and test sets:
   - Training Data: e commerce reviews train.csv
   - Test Data: e commerce reviews test.csv

Each dataset contains two columns: labels (__label__1 for negative, __label__2 for positive) and text (the review text).

## Installation
1: Clone the repository:
      git clone https://github.com/your-username/sentiment-analysis-project.git
      cd sentiment-analysis-project

2: Install the required dependencies:
      pip install -r requirements.txt
Ensure you have Python 3.7+ installed.

3: Download the NLTK data required for preprocessing:
      import nltk
      nltk.download('stopwords')
      nltk.download('punkt_tab')
      nltk.download('vader_lexicon')

## Usage
#### 1: Run the Jupyter Notebook:
      Open Sentiment Analysis - II.ipynb in Jupyter Notebook or Google Colab.
      Execute the cells to preprocess data, train models, evaluate performance, and save the best model.
      The notebook includes code to generate classification reports and accuracy scores for all models.

#### 2: Run the Streamlit Web Interface:
      Ensure the trained model (logistic_regression_model.pkl) and vectorizer (vectorizer.pkl) are in the project directory.
      Launch the Streamlit app to access the online web interface:

      streamlit run app.py
      
  Open your browser and navigate to http://localhost:8501.
  Enter a review in the text area and click "Analyze Sentiment" to view the predicted sentiment (POSITIVE or NEGATIVE).

#### 3: Deploy with ngrok (Optional):
To make the Streamlit web interface publicly accessible online, install pyngrok:
       
          pip install pyngrok

#### Set up an ngrok tunnel:

          from pyngrok import ngrok
          public_url = ngrok.connect(port='8501')
          print("Streamlit app is running at:", public_url)

Access the public URL to use the web interface from anywhere.

## Project Structure

    sentiment-analysis-project/
    ├── e commerce reviews train.csv      # Training dataset
    ├── e commerce reviews test.csv       # Test dataset
    ├── Sentiment Analysis - II.ipynb      # Main Jupyter Notebook
    ├── app.py                            # Streamlit web interface
    ├── logistic_regression_model.pkl      # Trained Logistic Regression model
    ├── vectorizer.pkl                    # TF-IDF vectorizer
    ├── requirements.txt                  # Python dependencies
    └── README.md                         # This file

Model Performance

The following table summarizes the accuracy of each model on the test dataset:
| Model                     | Vectorization | Accuracy (%) |
|---------------------------|---------------|--------------|
| Logistic Regression       | TF-IDF        | 90.51        |
| Logistic Regression       | BoW           | 89.98        |
| SGDClassifier             | TF-IDF        | 86.00        |
| SGDClassifier             | BoW           | 89.98        |
| Multinomial Naive Bayes   | TF-IDF        | 84.69        |
| Multinomial Naive Bayes   | BoW           | 85.34        |
| VADER                     | N/A           | 69.27        |


Logistic Regression with TF-IDF vectorization was selected as the final model for the Streamlit web interface due to its superior performance.

## Requirements

The requirements.txt file includes all necessary Python packages. Key dependencies include:

    pandas
    numpy
    scikit-learn
    nltk
    streamlit
    joblib
    pyngrok (optional, for public deployment)

Install them using:

      pip install -r requirements.txt


## Notes
The dataset file paths in the notebook are set for Google Drive. Update them to local paths if running outside Google Colab.
The app.py script assumes the remove_stopwords function is defined. If not, modify the script to skip stopword removal or include the function from the notebook.
For ngrok deployment, a verified ngrok account and authtoken are required. Sign up at ngrok's website and follow the instructions to set up your authtoken.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact opeyemiadelaja"gmail.com.
