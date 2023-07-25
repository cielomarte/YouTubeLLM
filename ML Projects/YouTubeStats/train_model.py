import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train_and_evaluate(model, df):
    # Define hyperparameters
    num_classes = 2
    embedding_dim = 100
    hidden_dim = 128
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10

    # Tokenize text and convert to numerical representations
    # Assuming 'tokens' column is already present in df from data_preprocessing.py
    # You can use 'embeddings' column if you have already converted text to embeddings.
    # If not, you can implement token_to_embeddings function similar to the one shown in the previous response.
    text_data = df['tokens'].tolist()
    vocab_size = len(set(word for sentence in text_data for word in sentence))

    # Prepare data loaders
    # Assuming 'labels' column contains the class labels for classification.
    # Replace 'labels' with the actual column name from your dataset.
    labels = df['title'].astype(int).tolist()
    dataset = [(torch.tensor(tokens), torch.tensor(label)) for tokens, label in zip(text_data, labels)]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = TextClassifier(num_classes, embedding_dim, vocab_size, hidden_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "text_classification_model.pth")
