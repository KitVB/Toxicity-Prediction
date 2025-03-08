import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import joblib
import os

# New GCN Model with dropout
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layer
        return torch.sigmoid(self.fc(x))

# Function to convert SMILES to molecular graph
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features (using atomic number as a simple feature)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # Edge indices (bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i))  # Undirected graph
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=atom_features, edge_index=edge_indices)

# Function to load the trained GCN model
def load_trained_gcn_model(model_path, num_features, hidden_dim, num_classes, dropout_rate=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, hidden_dim, num_classes, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Load the trained stacking classifier model (keeping the original function)
def load_stacking_classifier(model_path):
    try:
        return joblib.load(model_path)
    except (ImportError, ModuleNotFoundError) as e:
        st.error(f"Error loading stacking classifier model: The required dependencies are not installed. {str(e)}")
        return None
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading stacking classifier model: {str(e)}")
        return None

# Calculate RDKit descriptors for the stacking classifier (keeping the original function)
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Use the exact 61 descriptors that were used during training
    trained_descriptor_names = [
        'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 
        'BCUT2D_MRLOW', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BalabanJ',
        'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
        'Chi3n', 'Chi4n', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3',
        'EState_VSA4', 'EState_VSA5', 'EState_VSA8', 'ExactMolWt',
        'FpDensityMorgan3', 'HeavyAtomMolWt', 'Kappa1', 'Kappa2',
        'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge',
        'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge',
        'MolLogP', 'MolMR', 'MolWt', 'NumRotatableBonds', 'PEOE_VSA1',
        'PEOE_VSA10', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA1',
        'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA5',
        'TPSA', 'VSA_EState3', 'VSA_EState7', 'fr_Al_OH', 'fr_Al_OH_noTert',
        'fr_NH0', 'fr_allylic_oxid', 'fr_ester', 'fr_ether', 'fr_unbrch_alkane',
        'qed'
    ]
    
    # Verify we have the correct number of descriptors
    if len(trained_descriptor_names) != 61:
        st.warning(f"Warning: Expected 61 descriptors but found {len(trained_descriptor_names)}.")
    
    try:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(trained_descriptor_names)
        
        # Calculate descriptors
        descriptors = calculator.CalcDescriptors(mol)
        
        # Convert to DataFrame with proper column names
        df = pd.DataFrame([descriptors], columns=trained_descriptor_names)
        return df
    except Exception as e:
        st.error(f"Error calculating descriptors: {str(e)}")
        st.info("This may be because one or more of the descriptor names is not available in your RDKit version.")
        
        # Fall back to available descriptors
        available_desc_names = [desc[0] for desc in Descriptors._descList]
        st.warning("Falling back to available descriptors. Model prediction may not be accurate.")
        
        # Filter to only use descriptors that are available in both lists
        valid_descriptors = [desc for desc in trained_descriptor_names if desc in available_desc_names]
        st.info(f"Using {len(valid_descriptors)} out of 61 required descriptors.")
        
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(valid_descriptors)
        descriptors = calculator.CalcDescriptors(mol)
        df = pd.DataFrame([descriptors], columns=valid_descriptors)
        return df

# Predict toxicity using the new GCN model
def predict_toxicity_gcn(smiles, model, device):
    try:
        # Convert SMILES to graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            return "Invalid molecule structure", None
        
        # Move graph to device
        graph = graph.to(device)
        
        # Create a batch with just this one graph
        batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
        
        # Perform the prediction
        with torch.no_grad():
            output = model(graph.x, graph.edge_index, batch)
            probability = output.item()  # Extract the single value

        # Interpret the result
        toxicity = "Toxic" if probability > 0.5 else "Non-Toxic"
        return toxicity, probability

    except Exception as e:
        return str(e), None

# Predict toxicity using stacking classifier (keeping the original function)
def predict_toxicity_stacking(smiles, model):
    try:
        # Calculate descriptors for the molecule
        descriptors_df = calculate_descriptors(smiles)
        
        # Get the number of features
        num_features = descriptors_df.shape[1]
        if num_features != 61:
            st.warning(f"Warning: Model expects 61 features but got {num_features}. " +
                      "Prediction may not be accurate.")
        
        try:
            # Try prediction with shape check disabled first
            if hasattr(model, 'predict_proba') and hasattr(model, '_predict_proba_lr'):
                # This is likely a StackingClassifier with a final_estimator that has predict_proba
                probability = model.predict_proba(descriptors_df, predict_disable_shape_check=True)[0, 1]
            else:
                # Fall back to regular predict_proba
                probability = model.predict_proba(descriptors_df)[0, 1]
        except Exception as inner_e:
            if "predict_disable_shape_check" in str(inner_e):
                # If the parameter wasn't accepted, try without it
                probability = model.predict_proba(descriptors_df)[0, 1]
            else:
                # Some other error occurred
                raise inner_e
                
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

# Check if models exist
gcn_model_path = "toxicity_gcn_model.pth"  # Updated path for the new GCN model
stacking_model_path = "stacking_classifier_model.pkl"

# Model parameters for the new GCN
num_features = 1
hidden_dim = 64
num_classes = 1
dropout_rate = 0.5

# Determine which models are available
gcn_available = os.path.exists(gcn_model_path)
stacking_available = os.path.exists(stacking_model_path)

available_models = []
if gcn_available:
    available_models.append("GCN with Molecular Graph Structure")
if stacking_available:
    available_models.append("Stacking Classifier with RDKit Descriptors")

if not available_models:
    st.error("No models found. Please make sure at least one model file is available.")
    st.stop()

# Model selection dropdown
model_option = st.selectbox(
    "Select Model",
    available_models
)

# Input field for SMILES
smiles_input = st.text_input("Enter SMILES string:")

# Load the appropriate model based on selection
model = None
device = None
if model_option == "GCN with Molecular Graph Structure":
    try:
        model, device = load_trained_gcn_model(
            gcn_model_path, 
            num_features, 
            hidden_dim, 
            num_classes, 
            dropout_rate
        )
        st.info("Using Graph Convolutional Network model with molecular graph structure")
    except Exception as e:
        st.error(f"Error loading GCN model: {str(e)}")
        st.stop()
elif model_option == "Stacking Classifier with RDKit Descriptors":
    model = load_stacking_classifier(stacking_model_path)
    if model is None:
        st.warning("""
        Unable to load the Stacking Classifier model. This could be due to missing dependencies.
        
        The Stacking Classifier requires these libraries:
        - lightgbm
        - scikit-learn with specific versions
        
        Please ensure these libraries are installed in your environment.
        """)
        st.stop()
    else:
        st.info("Using Stacking Classifier with RDKit descriptors (Random Forest, LightGBM, SVM)")

if st.button("Predict"):
    if not model:
        st.error("No model is loaded. Please check the model configuration.")
    elif smiles_input:
        # Canonicalize the SMILES input
        canonical_smiles = canonicalize_smiles(smiles_input)

        if canonical_smiles is None:
            st.error("Invalid SMILES string. Please enter a valid SMILES.")
        else:
            st.write(f"Canonical SMILES: {canonical_smiles}")

            try:
                # Predict toxicity based on selected model
                if model_option == "GCN with Molecular Graph Structure":
                    result, probability = predict_toxicity_gcn(canonical_smiles, model, device)
                else:
                    result, probability = predict_toxicity_stacking(canonical_smiles, model)

                if probability is not None:
                    st.success(f"Predicted Toxicity: {result}")
                    st.write(f"Probability: {probability:.4f}")
                    
                    # Display additional information about the prediction
                    st.write("---")
                    st.write("### Model Details")
                    if model_option == "GCN with Molecular Graph Structure":
                        st.write("**Model Type:** Graph Convolutional Network")
                        st.write("**Input Features:** Molecular Graph Structure (Atom Features)")
                        st.write("**Architecture:** 2-layer GCN with dropout")
                    else:
                        st.write("**Model Type:** Stacking Classifier")
                        st.write("**Base Models:** Random Forest, LightGBM, SVM")
                        st.write("**Meta Learner:** Logistic Regression")
                        st.write("**Input Features:** RDKit Molecular Descriptors")
                else:
                    st.error(f"Error during prediction: {result}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Please enter a SMILES string.")
