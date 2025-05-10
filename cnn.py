import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Paths
PDB_DIR = "/scratch/akj/pdb_files"
OUTPUT_DIR = "/scratch/akj/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
MAX_DISTANCE = 20.0  # Clip distances at 20 Å
EMBEDDING_DIM = 256  # Structural embedding dimension
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_ca_coordinates(pdb_file):
    """Extract alpha carbon coordinates from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Extract CA atoms
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"].get_coord())
    
    return np.array(ca_atoms)

def compute_distance_map(coords):
    """Compute pairwise distance matrix between all CA atoms."""
    num_residues = coords.shape[0]
    distance_map = np.zeros((num_residues, num_residues))
    
    for i in range(num_residues):
        for j in range(num_residues):
            distance_map[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    return distance_map

def normalize_distance_map(distance_map):
    """Clip distances at MAX_DISTANCE and normalize to [0, 1]."""
    clipped_map = np.minimum(distance_map, MAX_DISTANCE)
    normalized_map = clipped_map / MAX_DISTANCE
    return normalized_map

def process_pdb_file(pdb_file):
    """Process a single PDB file to create a normalized distance map."""
    try:
        # Extract CA coordinates
        ca_coords = extract_ca_coordinates(pdb_file)
        
        # Compute distance map
        distance_map = compute_distance_map(ca_coords)
        
        # Normalize distance map
        norm_distance_map = normalize_distance_map(distance_map)
        
        # Convert to PyTorch tensor with batch dimension
        tensor_map = torch.tensor(norm_distance_map, dtype=torch.float32).unsqueeze(0)
        
        return tensor_map
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None

class StructuralCNN(nn.Module):
    """CNN to extract structural embeddings from distance maps."""
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super(StructuralCNN, self).__init__()
        
        # Three convolutional blocks (3×3 kernels, stride 1, padding 1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling will give us a 128-dimensional vector
        # Final linear layer to project to embedding dimension
        self.fc = nn.Linear(128, embedding_dim)
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Final projection
        x = self.fc(x)
        
        return x

def visualize_distance_map(distance_map, output_path):
    """Visualize and save a distance map."""
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_map.squeeze(), cmap='viridis')
    plt.colorbar(label='Normalized Distance')
    plt.title('CA-CA Distance Map')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Get all PDB files
    pdb_files = glob.glob(os.path.join(PDB_DIR, "*.pdb"))
    
    if not pdb_files:
        print(f"No PDB files found in {PDB_DIR}")
        return
    
    print(f"Found {len(pdb_files)} PDB files")
    
    # Initialize CNN model
    model = StructuralCNN().to(DEVICE)
    
    # Process PDB files to create distance maps
    distance_maps = []
    protein_names = []
    
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        tensor_map = process_pdb_file(pdb_file)
        if tensor_map is not None:
            distance_maps.append(tensor_map)
            protein_names.append(os.path.basename(pdb_file))
            
            # Visualize a sample of distance maps (e.g., the first 5)
            if len(distance_maps) <= 5:
                vis_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(pdb_file)}_distance_map.png")
                visualize_distance_map(tensor_map, vis_path)
    
    if not distance_maps:
        print("No valid distance maps were generated")
        return
    
    print(f"Successfully processed {len(distance_maps)} PDB files")
    
    # Create a dummy training loop (for demonstration)
    # In a real scenario, you would train this model on actual fitness data
    
    # Convert list of tensors to a single tensor
    distance_maps_tensor = torch.cat(distance_maps, dim=0).to(DEVICE)
    
    # Define optimizer and loss function (using MSE loss as a placeholder)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Dummy target embeddings (for demonstration)
    dummy_targets = torch.randn(len(distance_maps), EMBEDDING_DIM).to(DEVICE)
    
    # Training loop (only for demonstration)
    num_epochs = 5
    print("\nTraining CNN (demonstration):")
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(distance_maps_tensor)
        
        # Compute loss
        loss = criterion(outputs, dummy_targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Extract structural embeddings from the trained model
    model.eval()
    with torch.no_grad():
        structural_embeddings = model(distance_maps_tensor).cpu().numpy()
    
    # Save structural embeddings
    np.save(os.path.join(OUTPUT_DIR, "structural_embeddings.npy"), structural_embeddings)
    
    # Save protein names for reference
    with open(os.path.join(OUTPUT_DIR, "protein_names.txt"), "w") as f:
        for name in protein_names:
            f.write(f"{name}\n")
    
    print(f"\nStructural embeddings extracted and saved to {OUTPUT_DIR}")
    print(f"Embeddings shape: {structural_embeddings.shape}")

if __name__ == "__main__":
    main()