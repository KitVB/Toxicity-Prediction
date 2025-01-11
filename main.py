import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, Descriptors
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from PIL import Image
from io import BytesIO
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# Substructure definitions
substructures = {
    'Aromatic_Amines': 'c1ccc(cc1)N',
    'Nitro_Groups': '[N+](=O)[O-]',
    'Halogenated_Compounds_Cl': 'Cl',
    'Halogenated_Compounds_Br': 'Br',
    'Halogenated_Compounds_F': 'F',
    'Aldehydes_Ketones': 'C=O',
    'Alcohols_Phenols': '[OX2H]',
    'Carboxylic_Acids': 'C(=O)O',
    'Esters': 'C(=O)O',
    'Amides': 'C(=O)N',
    'Quinones': 'C1=CC(=O)C=CC1=O',
    'Sulfonamides': 'S(=O)(=O)N',
    'Epoxides': 'O1CO1'
}

# Function to check substructures
def check_substructures(mol, substructures):
    results = {}
    for name, smarts in substructures.items():
        substructure_mol = Chem.MolFromSmarts(smarts)
        results[name] = mol.HasSubstructMatch(substructure_mol)
    return results

# Function to calculate molecular descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol_wt = Descriptors.MolWt(mol)
        exact_mol_wt = Descriptors.ExactMolWt(mol)
        heavy_atom_mol_wt = Descriptors.HeavyAtomMolWt(mol)
        smiles_length = len(smiles)
        return pd.Series([mol_wt, exact_mol_wt, heavy_atom_mol_wt, smiles_length])
    else:
        return pd.Series([None, None, None, None])

# Generate 2D image of the molecule
def generate_molecule_image(mol):
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    else:
        return None


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
st.write("Enter a SMILES string to predict the toxicity of a molecule, visualize its structure, and explore molecular properties.")

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

            # Generate molecule and image
            mol = Chem.MolFromSmiles(canonical_smiles)
            img = generate_molecule_image(mol)
            if img:
                st.image(img, caption="2D Structure", use_column_width=False)

            # Check substructures
            substructure_results = check_substructures(mol, substructures)
            st.write("### Substructure Presence")
            st.write(pd.DataFrame.from_dict(substructure_results, orient='index', columns=['Present']))

            # Calculate and display descriptors
            descriptors = calculate_descriptors(canonical_smiles)
            descriptors_df = pd.DataFrame([descriptors], columns=['MolWt', 'ExactMolWt', 'HeavyAtomMolWt', 'SMILES_Length'])
            st.write("### Molecular Descriptors")
            st.write(descriptors_df)

            # Predict toxicity
            result, probability = predict_toxicity(canonical_smiles, model)

            if probability is not None:
                st.success(f"Predicted Toxicity: {result}")
                st.write(f"Probability: {probability:.4f}")
            else:
                st.error(f"Error during prediction: {result}")
    else:
        st.error("Please enter a SMILES string.")
