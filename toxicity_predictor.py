#!/usr/bin/env python3
import os
import sys
import torch
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
import joblib

# Color codes for terminal output
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# GCN Model for ECFP fingerprints
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_features=2048):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.fc1 = Linear(input_features, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.fc3 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(x)

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
        print(f"{TermColors.ERROR}Error loading stacking classifier model: The required dependencies are not installed. {str(e)}{TermColors.ENDC}")
        return None
    except FileNotFoundError:
        print(f"{TermColors.ERROR}Error: Model file '{model_path}' not found.{TermColors.ENDC}")
        return None
    except Exception as e:
        print(f"{TermColors.ERROR}Unexpected error loading stacking classifier model: {str(e)}{TermColors.ENDC}")
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
        print(f"{TermColors.WARNING}Warning: Expected 61 descriptors but found {len(trained_descriptor_names)}.{TermColors.ENDC}")
    
    try:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(trained_descriptor_names)
        
        # Calculate descriptors
        descriptors = calculator.CalcDescriptors(mol)
        
        # Convert to DataFrame with proper column names
        df = pd.DataFrame([descriptors], columns=trained_descriptor_names)
        return df
    except Exception as e:
        print(f"{TermColors.ERROR}Error calculating descriptors: {str(e)}{TermColors.ENDC}")
        print(f"{TermColors.BLUE}This may be because one or more of the descriptor names is not available in your RDKit version.{TermColors.ENDC}")
        
        # Fall back to available descriptors
        available_desc_names = [desc[0] for desc in Descriptors._descList]
        print(f"{TermColors.WARNING}Falling back to available descriptors. Model prediction may not be accurate.{TermColors.ENDC}")
        
        # Filter to only use descriptors that are available in both lists
        valid_descriptors = [desc for desc in trained_descriptor_names if desc in available_desc_names]
        print(f"{TermColors.BLUE}Using {len(valid_descriptors)} out of 61 required descriptors.{TermColors.ENDC}")
        
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
            print(f"{TermColors.WARNING}Warning: Model expects 61 features but got {num_features}. Prediction may not be accurate.{TermColors.ENDC}")
        
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

def main():
    parser = argparse.ArgumentParser(description='Predict molecular toxicity from SMILES strings.')
    parser.add_argument('--smiles', type=str, help='SMILES string of the molecule')
    parser.add_argument('--model', type=str, choices=['gcn', 'stacking'], default='gcn',
                        help='Model to use for prediction: gcn (Graph Convolutional Network) or stacking (Stacking Classifier)')
    parser.add_argument('--gcn-model-path', type=str, default='gcn_ecfp_model_64.pth',
                        help='Path to the GCN model file')
    parser.add_argument('--stacking-model-path', type=str, default='stacking_classifier_model.pkl',
                        help='Path to the stacking classifier model file')
    parser.add_argument('--batch', type=str, help='Path to a CSV file with SMILES strings (must have a column named "SMILES")')
    parser.add_argument('--output', type=str, help='Path to save the output CSV file for batch processing')
    
    args = parser.parse_args()
    
    # Check model availability
    available_models = []
    gcn_available = os.path.exists(args.gcn_model_path)
    stacking_available = os.path.exists(args.stacking_model_path)
    
    if gcn_available:
        available_models.append('gcn')
    if stacking_available:
        available_models.append('stacking')
    
    if not available_models:
        print(f"{TermColors.ERROR}Error: No models found. Please ensure at least one model file is available.{TermColors.ENDC}")
        sys.exit(1)
    
    if args.model not in available_models:
        print(f"{TermColors.ERROR}Error: Selected model '{args.model}' is not available. Available models: {', '.join(available_models)}{TermColors.ENDC}")
        sys.exit(1)
    
    # Load the selected model
    model = None
    if args.model == 'gcn':
        try:
            hidden_channels = 64  # For GCN model
            model = load_trained_gcn_model(args.gcn_model_path, hidden_channels)
            print(f"{TermColors.BLUE}Using Graph Convolutional Network model with ECFP fingerprints{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.ERROR}Error loading GCN model: {str(e)}{TermColors.ENDC}")
            sys.exit(1)
    elif args.model == 'stacking':
        model = load_stacking_classifier(args.stacking_model_path)
        if model is None:
            print(f"{TermColors.ERROR}Unable to load the Stacking Classifier model. Please ensure the required dependencies are installed.{TermColors.ENDC}")
            sys.exit(1)
        print(f"{TermColors.BLUE}Using Stacking Classifier with RDKit descriptors (Random Forest, LightGBM, SVM){TermColors.ENDC}")
    
    # Batch processing
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"{TermColors.ERROR}Error: Batch file '{args.batch}' not found.{TermColors.ENDC}")
            sys.exit(1)
        
        try:
            # Read the batch file
            batch_df = pd.read_csv(args.batch)
            if 'SMILES' not in batch_df.columns:
                print(f"{TermColors.ERROR}Error: Batch file must contain a 'SMILES' column.{TermColors.ENDC}")
                sys.exit(1)
            
            results = []
            
            print(f"{TermColors.HEADER}Processing {len(batch_df)} molecules...{TermColors.ENDC}")
            
            # Process each SMILES
            for i, row in batch_df.iterrows():
                smiles = row['SMILES']
                print(f"Processing molecule {i+1}/{len(batch_df)}: {smiles}")
                
                # Canonicalize SMILES
                canonical_smiles = canonicalize_smiles(smiles)
                if canonical_smiles is None:
                    print(f"{TermColors.WARNING}  Invalid SMILES: {smiles}{TermColors.ENDC}")
                    results.append({
                        'SMILES': smiles,
                        'Canonical_SMILES': None,
                        'Prediction': 'Error',
                        'Probability': None,
                        'Error': 'Invalid SMILES'
                    })
                    continue
                
                # Predict toxicity
                if args.model == 'gcn':
                    prediction, probability = predict_toxicity_gcn(canonical_smiles, model)
                else:
                    prediction, probability = predict_toxicity_stacking(canonical_smiles, model)
                
                if probability is not None:
                    print(f"  {TermColors.GREEN}Prediction: {prediction}, Probability: {probability:.4f}{TermColors.ENDC}")
                    results.append({
                        'SMILES': smiles,
                        'Canonical_SMILES': canonical_smiles,
                        'Prediction': prediction,
                        'Probability': probability,
                        'Error': None
                    })
                else:
                    print(f"{TermColors.ERROR}  Error: {prediction}{TermColors.ENDC}")
                    results.append({
                        'SMILES': smiles,
                        'Canonical_SMILES': canonical_smiles,
                        'Prediction': 'Error',
                        'Probability': None,
                        'Error': prediction
                    })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"{TermColors.GREEN}Results saved to {args.output}{TermColors.ENDC}")
            else:
                print("\nResults:")
                print(results_df)
            
        except Exception as e:
            print(f"{TermColors.ERROR}Error during batch processing: {str(e)}{TermColors.ENDC}")
            sys.exit(1)
    
    # Single SMILES processing
    elif args.smiles:
        canonical_smiles = canonicalize_smiles(args.smiles)
        
        if canonical_smiles is None:
            print(f"{TermColors.ERROR}Invalid SMILES string: {args.smiles}{TermColors.ENDC}")
            sys.exit(1)
        
        print(f"{TermColors.BLUE}Canonical SMILES: {canonical_smiles}{TermColors.ENDC}")
        
        try:
            # Predict toxicity
            if args.model == 'gcn':
                prediction, probability = predict_toxicity_gcn(canonical_smiles, model)
            else:
                prediction, probability = predict_toxicity_stacking(canonical_smiles, model)
            
            if probability is not None:
                print(f"{TermColors.GREEN}Predicted Toxicity: {prediction}{TermColors.ENDC}")
                print(f"Probability: {probability:.4f}")
                
                # Display model details
                print("\nModel Details:")
                if args.model == 'gcn':
                    print("Model Type: Graph Convolutional Network")
                    print("Input Features: Extended Connectivity Fingerprints (ECFP)")
                else:
                    print("Model Type: Stacking Classifier")
                    print("Base Models: Random Forest, LightGBM, SVM")
                    print("Meta Learner: Logistic Regression")
                    print("Input Features: RDKit Molecular Descriptors")
            else:
                print(f"{TermColors.ERROR}Error during prediction: {prediction}{TermColors.ENDC}")
                sys.exit(1)
                
        except Exception as e:
            print(f"{TermColors.ERROR}Error during prediction: {str(e)}{TermColors.ENDC}")
            sys.exit(1)
    
    else:
        # Interactive mode if no SMILES or batch file is provided
        print(f"{TermColors.HEADER}Molecule Toxicity Predictor{TermColors.ENDC}")
        print(f"Using model: {args.model}")
        
        while True:
            smiles_input = input("\nEnter SMILES string (or 'q' to quit): ")
            
            if smiles_input.lower() == 'q':
                break
            
            if not smiles_input:
                print(f"{TermColors.ERROR}Please enter a SMILES string.{TermColors.ENDC}")
                continue
            
            canonical_smiles = canonicalize_smiles(smiles_input)
            
            if canonical_smiles is None:
                print(f"{TermColors.ERROR}Invalid SMILES string. Please enter a valid SMILES.{TermColors.ENDC}")
                continue
            
            print(f"{TermColors.BLUE}Canonical SMILES: {canonical_smiles}{TermColors.ENDC}")
            
            try:
                # Predict toxicity
                if args.model == 'gcn':
                    prediction, probability = predict_toxicity_gcn(canonical_smiles, model)
                else:
                    prediction, probability = predict_toxicity_stacking(canonical_smiles, model)
                
                if probability is not None:
                    print(f"{TermColors.GREEN}Predicted Toxicity: {prediction}{TermColors.ENDC}")
                    print(f"Probability: {probability:.4f}")
                else:
                    print(f"{TermColors.ERROR}Error during prediction: {prediction}{TermColors.ENDC}")
            
            except Exception as e:
                print(f"{TermColors.ERROR}Error during prediction: {str(e)}{TermColors.ENDC}")

if __name__ == "__main__":
    main()