import sys
import os

# Get the absolute path to the directory containing 'main.py'
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (containing 'data_preprocessing.py') to sys.path
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)


from data_preprocessing import load_and_preprocess_data
from text_classification_model import TextClassifier
from train_model import train_and_evaluate

if __name__ == "__main__":
    data_path = "/Users/CieMarte/Documents/ML Projects/YouTubeStats/_Top50_viewed_video_from_each_channels.csv"  
    df = load_and_preprocess_data(data_path)

    model = TextClassifier(num_classes=2, embedding_dim=100, vocab_size=10000, hidden_dim=128)

    train_and_evaluate(model, df)
