import torch
import gensim 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

def tokenize_text(text):
    # Implement the tokenization logic here using nltk or any other library
    pass

def prepare_text_data(df):
    # Tokenize the text in the DataFrame
    df['tokens'] = df['title'].apply(lambda x: word_tokenize(x.lower()))

    # Train a Word2Vec model on the tokens to get word embeddings
    model = Word2Vec(df['tokens'], min_count=1)

    # Convert tokens to embeddings
    df['embeddings'] = df['tokens'].apply(lambda x: [model.wv[word] for word in x])

    return df

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1
