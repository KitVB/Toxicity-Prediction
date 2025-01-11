import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_features=2048):  # ECFP size is 1024 bits by default
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.fc1 = Linear(input_features, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.fc3 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = global_mean_pool(x, batch)  # Pooling for graph classification
        x = self.fc3(x)
        return torch.sigmoid(x)  # Binary classification

# 1. Define a function to preprocess a single SMILES string and generate input data
def preprocess_smiles(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Generate ECFP fingerprints
    ecfp_gen = AllChem.GetMorganGenerator(radius=radius)
    ecfp = ecfp_gen.GetFingerprint(mol)

    # Convert fingerprint to numpy array and then to tensor
    ecfp_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(ecfp, ecfp_array)
    x = torch.tensor(ecfp_array, dtype=torch.float).view(1, -1)  # Shape: (1, nBits)

    return Data(x=x, edge_index=torch.tensor([[0], [0]]))  # Dummy edge_index

# 2. Define a function to load the trained model
def load_trained_model(model_path, hidden_channels, input_features=2048):
    model = GCN(hidden_channels, input_features)
    model.load_state_dict(torch.load(model_path, weights_only = True, map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. Predict toxicity for a single molecule
def predict_toxicity(smiles, model):
    try:
        # Preprocess the SMILES string
        data = preprocess_smiles(smiles)
        
        # Perform the prediction
        with torch.no_grad():
            out = model(data.x.float(), data.edge_index, torch.tensor([0]))  # Single graph (batch = 0)
            probability = out.item()  # Extract the single value

        # Interpret the result
        toxicity = "Toxic" if probability > 0.5 else "Non-Toxic"
        return toxicity, probability

    except Exception as e:
        return str(e), None

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None  # Return None for invalid SMILES
    except:
        return None  # Return None for invalid SMILES

# Streamlit App
st.title("Molecule Toxicity Predictor")
st.write("Enter a SMILES string to predict the toxicity of a molecule.")

# Input field for SMILES
smiles_input = st.text_input("Enter SMILES string:")

# Define the path to the saved model
model_path = "gcn_ecfp_model_64.pth"  # Replace with your model path
hidden_channels = 64  # Adjust based on your model architecture

# Load the trained model
model = load_trained_model(model_path, hidden_channels)

if st.button("Predict"):
    if smiles_input:
        # Canonicalize the SMILES input
        canonical_smiles = canonicalize_smiles(smiles_input)

        if canonical_smiles is None:
            st.error("Invalid SMILES string. Please enter a valid SMILES.")
        else:
            st.write(f"Canonical SMILES: {canonical_smiles}")

            # Predict toxicity
            result, probability = predict_toxicity(canonical_smiles, model)

            if probability is not None:
                st.success(f"Predicted Toxicity: {result}")
                st.write(f"Probability: {probability:.4f}")
            else:
                st.error(f"Error during prediction: {result}")
    else:
        st.error("Please enter a SMILES string.")
