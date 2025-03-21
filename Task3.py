import nltk
import random
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK data (uncomment these lines if running for the first time)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Sample knowledge base - you can expand this or load from a file
knowledge_base = {
    "greetings": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Greetings! How may I assist you?"
    ],
    "farewell": [
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back if you have more questions.",
        "Bye! It was nice chatting with you."
    ],
    "name": [
        "I'm NLPBot, your friendly AI assistant.",
        "My name is NLPBot. I'm here to help answer your questions.",
        "I go by NLPBot. How can I assist you today?"
    ],
    "capabilities": [
        "I can answer simple questions based on my knowledge base. I'm always learning!",
        "I can chat with you and try to answer your questions to the best of my abilities.",
        "I'm a simple NLP-based chatbot that can respond to various queries."
    ],
    "weather": [
        "I don't have access to real-time weather data, but I'd be happy to chat about other topics!",
        "Unfortunately, I can't check the weather for you. Is there something else I can help with?",
        "I'm not connected to weather services. Maybe try a weather app or website?"
    ],
    "thanks": [
        "You're welcome! Is there anything else I can help with?",
        "Glad I could help! Feel free to ask more questions.",
        "No problem at all! Let me know if you need anything else."
    ],
    "default": [
        "I'm not sure I understand. Could you rephrase that?",
        "I don't have information about that yet. I'm still learning!",
        "Interesting question! Unfortunately, I don't have a specific answer for that."
    ]
}

# Additional information for pattern matching
patterns = {
    "greetings": ["hello", "hi", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "farewell", "cya", "good night"],
    "name": ["who are you", "what's your name", "what are you called", "your name"],
    "capabilities": ["what can you do", "help", "features", "abilities", "what are you able to do"],
    "weather": ["weather", "temperature", "forecast", "raining", "sunny"],
    "thanks": ["thanks", "thank you", "appreciate it", "grateful"]
}

class NLPChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        
        # Create a corpus of all pattern words for similarity matching
        self.corpus = []
        self.corpus_labels = []
        for intent, pattern_list in patterns.items():
            for pattern in pattern_list:
                self.corpus.append(pattern)
                self.corpus_labels.append(intent)
        
        # Fit vectorizer on the corpus
        if self.corpus:
            self.X = self.vectorizer.fit_transform(self.corpus)
    
    def preprocess(self, text):
        """Preprocess the text by tokenizing, removing punctuation and stopwords, and lemmatizing"""
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in string.punctuation and token not in self.stop_words]
        
        return " ".join(tokens)
    
    def get_intent(self, user_input):
        """Determine the intent of the user's input using TF-IDF and cosine similarity"""
        # If the corpus is empty, return default
        if not self.corpus:
            return "default"
            
        processed_input = self.preprocess(user_input)
        
        # If after preprocessing we have no tokens, return default
        if not processed_input.strip():
            return "default"
        
        # Transform user input with our fitted vectorizer
        user_vector = self.vectorizer.transform([processed_input])
        
        # Calculate similarities between user input and our corpus
        similarities = cosine_similarity(user_vector, self.X).flatten()
        
        # Get the index of the most similar text
        max_similarity_idx = similarities.argmax()
        
        # If the similarity is too low, return default
        if similarities[max_similarity_idx] < 0.3:
            return "default"
        
        # Return the intent corresponding to the most similar pattern
        return self.corpus_labels[max_similarity_idx]
    
    def get_response(self, user_input):
        """Generate a response based on the user's input"""
        intent = self.get_intent(user_input)
        
        # Get responses for the identified intent
        responses = knowledge_base.get(intent, knowledge_base["default"])
        
        # Return a random response from the selected category
        return random.choice(responses)
    
    def chat(self):
        """Run the chatbot interactively"""
        print("NLPBot: Hello! Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("NLPBot: Goodbye! Have a great day!")
                break
                
            response = self.get_response(user_input)
            print(f"NLPBot: {response}")


# Create an instance of our chatbot
def main():
    chatbot = NLPChatbot()
    chatbot.chat()

if __name__ == "__main__":
    main()