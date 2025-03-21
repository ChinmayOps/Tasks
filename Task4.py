import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import os
import urllib.request
import zipfile

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set random seed for reproducibility
np.random.seed(42)

def download_data():
    """Download and extract the SMS Spam Collection dataset"""
    print("Downloading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Download zip file
    zip_path = "data/smsspamcollection.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")
        
        print("Dataset successfully downloaded and extracted.")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_data():
    """Load the SMS Spam Collection dataset"""
    try:
        # Try to load the dataset
        data = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
        print("Dataset successfully loaded!")
        return data
    except FileNotFoundError:
        print("File not found! Trying to download...")
        if download_data():
            try:
                data = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
                print("Dataset successfully loaded after download!")
                return data
            except FileNotFoundError:
                pass
                
        # If still not found, create a small sample dataset for demonstration
        print("Creating sample dataset for demonstration...")
        data = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam', 'ham'],
            'message': [
                'Hey, how are you doing?',
                'CONGRATULATIONS! You won $5000! Claim now!',
                'I\'ll be home late tonight',
                'FREE ENTRY! WIN $1000 this week!',
                'Don\'t forget to buy milk'
            ]
        })
        return data

def explore_data(data):
    """Explore and visualize the dataset"""
    print(f"\nDataset shape: {data.shape}")
    
    # Check class distribution
    class_distribution = data['label'].value_counts()
    print("\nClass distribution:")
    print(class_distribution)
    
    # Calculate percentages
    total = len(data)
    ham_percent = (class_distribution['ham'] / total) * 100
    spam_percent = (class_distribution['spam'] / total) * 100
    
    print(f"Ham: {ham_percent:.2f}%")
    print(f"Spam: {spam_percent:.2f}%")
    
    # Basic statistics on message length
    data['message_length'] = data['message'].apply(len)
    print("\nMessage length statistics:")
    print(data['message_length'].describe())
    
    # Visualize class distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=data)
    plt.title('Class Distribution: Spam vs Ham')
    plt.ylabel('Count')
    
    # Compare message length between spam and ham
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x='message_length', hue='label', bins=30, kde=True)
    plt.title('Message Length Distribution')
    plt.xlabel('Message Length (characters)')
    plt.xlim(0, 300)  # Focus on the main distribution
    
    plt.tight_layout()
    plt.savefig('data_exploration.png')
    print("Visualization saved as 'data_exploration.png'")
    
    return data

def preprocess_text(text):
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing non-alphabetic characters
    3. Removing stopwords
    4. Stemming words
    """
    # Initialize Porter Stemmer and stopwords
    stemmer = PorterStemmer()
    
    # Download NLTK resources if not already downloaded
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and stem
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    # Join back into a string
    return ' '.join(cleaned_words)

def prepare_data(data):
    """Prepare data for machine learning"""
    # Apply preprocessing to the 'message' column
    print("\nPreprocessing text data...")
    data['cleaned_message'] = data['message'].apply(preprocess_text)
    
    # Convert labels to binary (0 for ham, 1 for spam)
    data['label_binary'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_message'], 
        data['label_binary'], 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate different classification models"""
    print("\nTraining and evaluating models...")
    
    # Define different classifiers to try
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    # Compare different classifiers
    results = {}
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Create a pipeline with the current classifier
        pipeline = Pipeline([
            ('vect', CountVectorizer(max_features=5000)),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier)
        ])
        
        # Train and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_model_name = name
        
        print(f"{name} accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    
    # Visualize the comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values())
    
    # Add accuracy values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0.9, 1.0)  # Adjust y-axis for better visualization
    plt.title('Classifier Comparison')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nModel comparison visualization saved as 'model_comparison.png'")
    
    print(f"\nThe best classifier is {best_model_name} with an accuracy of {best_accuracy:.4f}")
    
    return best_model, best_model_name

def test_with_new_messages(model):
    """Test the model with new, unseen messages"""
    print("\nTesting model with new messages...")
    
    # Sample new messages
    new_messages = [
        "Hey, how are you doing? Want to meet up for coffee?",
        "CONGRATULATIONS! You've been selected to win a free iPhone! Click here to claim your prize now!",
        "Your Amazon order #12345 has been shipped and will arrive tomorrow.",
        "URGENT: Your bank account has been compromised. Please update your details at www.security-bank.com",
        "Reminder: Your doctor's appointment is scheduled for tomorrow at 2 PM."
    ]
    
    # Preprocess the new messages
    new_messages_cleaned = [preprocess_text(msg) for msg in new_messages]
    
    # Make predictions
    predictions = model.predict(new_messages_cleaned)
    probabilities = model.predict_proba(new_messages_cleaned)[:, 1]  # Probability of being spam
    
    # Display results
    print("\nPrediction results:")
    for i, msg in enumerate(new_messages):
        prediction = "Spam" if predictions[i] == 1 else "Ham"
        probability = probabilities[i]
        print(f"\nMessage: {msg}")
        print(f"Prediction: {prediction}")
        print(f"Spam Probability: {probability:.4f}")
    
    return

def save_model(model, filename="spam_classifier_model.pkl"):
    """Save the trained model to a file"""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"\nModel saved as '{filename}'")

def load_model(filename="spam_classifier_model.pkl"):
    """Load a trained model from a file"""
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"\nModel loaded from '{filename}'")
    return loaded_model

def main():
    """Main function to run the spam detection pipeline"""
    print("======== SPAM EMAIL DETECTION WITH MACHINE LEARNING ========")
    
    # Load and explore data
    data = load_data()
    data = explore_data(data)
    
    # Prepare data for machine learning
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train and evaluate models
    best_model, best_model_name = train_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Test the model with new messages
    test_with_new_messages(best_model)
    
    # Save the model
    save_model(best_model)
    
    print("\n======== SPAM DETECTION MODEL TRAINING COMPLETE ========")
    print(f"Best model: {best_model_name}")
    print("Model saved and ready for use!")
    print("\nTo use this model in other applications, load it with:")
    print("  loaded_model = load_model('spam_classifier_model.pkl')")
    print("  prediction = loaded_model.predict([preprocess_text('your message here')])")

if __name__ == "__main__":
    main()