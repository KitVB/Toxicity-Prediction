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
import os

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

# Calculate RDKit descriptors for the stacking classifier
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
gcn_model_path = "gcn_ecfp_model_64.pth"
stacking_model_path = "stacking_classifier_model.pkl"
hidden_channels = 64  # For GCN model

# Determine which models are available
gcn_available = os.path.exists(gcn_model_path)
stacking_available = os.path.exists(stacking_model_path)

available_models = []
if gcn_available:
    available_models.append("GCN with ECFP Fingerprints")
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
if model_option == "GCN with ECFP Fingerprints":
    try:
        model = load_trained_gcn_model(gcn_model_path, hidden_channels)
        st.info("Using Graph Convolutional Network model with ECFP fingerprints")
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
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Please enter a SMILES string.")
