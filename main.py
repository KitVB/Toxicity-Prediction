import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import joblib

# GCN Model for ECFP fingerprints
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

# Function to preprocess a single SMILES string and generate input data for GCN
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

# Function to load the trained GCN model
def load_trained_gcn_model(model_path, hidden_channels, input_features=2048):
    model = GCN(hidden_channels, input_features)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the trained stacking classifier model
def load_stacking_classifier(model_path):
    return joblib.load(model_path)

# Calculate RDKit descriptors for the stacking classifier
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Get all descriptor names and create calculator
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    # Calculate descriptors
    descriptors = calculator.CalcDescriptors(mol)
    
    # Convert to DataFrame with proper column names
    df = pd.DataFrame([descriptors], columns=descriptor_names)
    return df

# Predict toxicity using GCN model
def predict_toxicity_gcn(smiles, model):
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

# Predict toxicity using stacking classifier
def predict_toxicity_stacking(smiles, model):
    try:
        # Calculate descriptors for the molecule
        descriptors_df = calculate_descriptors(smiles)
        
        # Predict using the stacking classifier
        probability = model.predict_proba(descriptors_df)[0, 1]
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

# Model selection dropdown
model_option = st.selectbox(
    "Select Model",
    ["GCN with ECFP Fingerprints", "Stacking Classifier with RDKit Descriptors"]
)

# Input field for SMILES
smiles_input = st.text_input("Enter SMILES string:")

# Define the paths to the saved models
gcn_model_path = "gcn_ecfp_model_64.pth"
stacking_model_path = "stacking_classifier_model.pkl"
hidden_channels = 64  # For GCN model

# Load the appropriate model based on selection
if model_option == "GCN with ECFP Fingerprints":
    model = load_trained_gcn_model(gcn_model_path, hidden_channels)
    st.info("Using Graph Convolutional Network model with ECFP fingerprints")
else:
    model = load_stacking_classifier(stacking_model_path)
    st.info("Using Stacking Classifier with RDKit descriptors (Random Forest, LightGBM, SVM)")

if st.button("Predict"):
    if smiles_input:
        # Canonicalize the SMILES input
        canonical_smiles = canonicalize_smiles(smiles_input)

        if canonical_smiles is None:
            st.error("Invalid SMILES string. Please enter a valid SMILES.")
        else:
            st.write(f"Canonical SMILES: {canonical_smiles}")

            # Predict toxicity based on selected model
            if model_option == "GCN with ECFP Fingerprints":
                result, probability = predict_toxicity_gcn(canonical_smiles, model)
            else:
                result, probability = predict_toxicity_stacking(canonical_smiles, model)

            if probability is not None:
                st.success(f"Predicted Toxicity: {result}")
                st.write(f"Probability: {probability:.4f}")
                
                # Display additional information about the prediction
                st.write("---")
                st.write("### Model Details")
                if model_option == "GCN with ECFP Fingerprints":
                    st.write("**Model Type:** Graph Convolutional Network")
                    st.write("**Input Features:** Extended Connectivity Fingerprints (ECFP)")
                else:
                    st.write("**Model Type:** Stacking Classifier")
                    st.write("**Base Models:** Random Forest, LightGBM, SVM")
                    st.write("**Meta Learner:** Logistic Regression")
                    st.write("**Input Features:** RDKit Molecular Descriptors")
            else:
                st.error(f"Error during prediction: {result}")
    else:
        st.error("Please enter a SMILES string.")
