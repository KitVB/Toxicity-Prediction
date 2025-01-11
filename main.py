import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from torch_geometric.data import Data

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
    model.load_state_dict(torch.load(model_path, weights_only=True))
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
