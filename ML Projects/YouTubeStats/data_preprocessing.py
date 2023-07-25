import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download necessary resources for tokenization (do this only once)
#nltk.download('punkt')

def load_and_preprocess_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Drop irrelevant columns
    df.drop(columns=['channelTitle', 'publishedAt', 'viewCount', 'likeCount', 'commentCount', 'duration'], inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Tokenize the text
    df['tokens'] = df['title'].apply(lambda x: word_tokenize(x.lower()))

    return df
