import os
import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
EMBEDDINGS_DIR = "/scratch/akj/DMS_ProteinGym_embeddings"  
SAMPLES_PER_FILE = 10  # Number of sequences to use from each file
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Simple MLP model for regression
class FitnessMLP(nn.Module):
    def __init__(self, input_size):
        super(FitnessMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE // 2, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

def load_embedding_data():
    """Load embedding data from all files and take the first SAMPLES_PER_FILE samples from each."""
    all_embedding_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*_with_embeddings.csv"))
    
    # Check if we found any files
    if not all_embedding_files:
        raise ValueError(f"No embedding files found in {EMBEDDINGS_DIR}")
    
    print(f"Found {len(all_embedding_files)} embedding files")
    
    all_data_samples = []
    
    for file_path in tqdm(all_embedding_files, desc="Loading data"):
        try:
            # Load the file
            df = pd.read_csv(file_path)
            
            # Check if the file has the required columns
            embedding_cols = [col for col in df.columns if 'embedding_' in col]
            if not embedding_cols or 'DMS_score' not in df.columns:
                print(f"Skipping {file_path}: Missing required columns")
                continue
            
            # Take first N samples
            samples = df.head(SAMPLES_PER_FILE)
            if len(samples) > 0:
                all_data_samples.append(samples)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Combine all samples
    if not all_data_samples:
        raise ValueError("No valid data found in any of the embedding files")
    
    combined_data = pd.concat(all_data_samples, ignore_index=True)
    print(f"Total samples collected: {len(combined_data)}")
    
    return combined_data

def prepare_data(data):
    """Extract features (embeddings) and target (fitness scores) from the data."""
    # Extract embedding columns
    embedding_cols = [col for col in data.columns if 'embedding_' in col]
    X = data[embedding_cols].values
    y = data['DMS_score'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader, X_test_tensor, y_test_tensor

def train_and_evaluate():
    """Train the model and evaluate it using Spearman correlation."""
    # Load data
    data = load_embedding_data()
    
    # Prepare data
    train_loader, test_loader, X_test, y_test = prepare_data(data)
    
    # Get input size from the data
    input_size = X_test.shape[1]
    print(f"Input size (number of embedding dimensions): {input_size}")
    
    # Initialize model
    model = FitnessMLP(input_size).to(DEVICE)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training loop
    best_spearman = -np.inf
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            # Test on all test data at once to calculate Spearman correlation
            X_test = X_test.to(DEVICE)
            predictions = model(X_test).cpu().numpy()
            
            # Calculate Spearman correlation
            spearman_corr, p_value = spearmanr(predictions, y_test.numpy())
            
            # Save best model
            if spearman_corr > best_spearman:
                best_spearman = spearman_corr
                torch.save(model.state_dict(), "best_fitness_model.pt")
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}, "
                  f"Spearman Correlation: {spearman_corr:.4f}, p-value: {p_value:.4f}")
    
    # Load best model and make final evaluation
    model.load_state_dict(torch.load("best_fitness_model.pt"))
    model.eval()
    
    with torch.no_grad():
        X_test = X_test.to(DEVICE)
        final_predictions = model(X_test).cpu().numpy()
        final_spearman, final_p = spearmanr(final_predictions, y_test.numpy())
    
    print(f"\nFinal Evaluation:")
    print(f"Best Spearman Correlation: {final_spearman:.4f}")
    print(f"p-value: {final_p:.6f}")
    
    return model, final_spearman

if __name__ == "__main__":
    model, correlation = train_and_evaluate()
    print(f"Training complete. Model saved with Spearman correlation: {correlation:.4f}")