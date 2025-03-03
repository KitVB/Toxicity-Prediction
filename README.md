# Toxicity-Prediction

A website and a command-line tool for predicting molecular toxicity using machine learning models.

Link to the website : https://toxicity-prediction.streamlit.app/

## Features

- Support for two prediction models:
  - Graph Convolutional Network (GCN) with ECFP fingerprints
  - Stacking Classifier with RDKit descriptors (Random Forest, LightGBM, SVM)
- Color-coded terminal output for better readability
- Three operating modes:
  1. Single molecule prediction via command line argument
  2. Batch processing from a CSV file
  3. Interactive mode for multiple predictions

## Installation

First, make sure to install the required dependencies:

```bash
pip install torch torchvision torch-geometric rdkit numpy pandas joblib lightgbm scikit-learn
```

## Usage

Make the `toxicity_predictor.py` executable:

```bash
chmod +x toxicity_predictor.py
```

### Single Molecule Prediction

```bash
# Using GCN model (default)
python toxicity_predictor.py --smiles "CC(=O)Oc1ccccc1C(=O)O" 

# Using stacking classifier model
python toxicity_predictor.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --model stacking
```

### Batch Processing

Process multiple molecules from a CSV file:

```bash
python toxicity_predictor.py --batch molecules.csv --output results.csv --model stacking
```

The input CSV file must contain a column named "SMILES".

### Interactive Mode

Run the script without the `--smiles` or `--batch` arguments:

```bash
python toxicity_predictor.py --model gcn
```

This will start an interactive session where you can enter multiple SMILES strings.

### Custom Model Paths

You can specify custom paths to your model files:

```bash
python toxicity_predictor.py --gcn-model-path "/path/to/gcn_model.pth" --stacking-model-path "/path/to/stacking_model.pkl"
```

## Help

For more information on available options:

```bash
python toxicity_predictor.py --help
```

## Model Details

### GCN Model
- Uses Extended Connectivity Fingerprints (ECFP) as input features
- Neural network architecture with hidden layers
- Trained on binary toxicity data

### Stacking Classifier
- Ensemble of Random Forest, LightGBM, and SVM models
- Uses RDKit molecular descriptors as input features
- Meta-learner: Logistic Regression
- Required descriptors: 61 RDKit descriptors including MolWt, MolLogP, TPSA, etc.

## Error Handling

The tool includes robust error handling for:
- Invalid SMILES strings
- Missing model files
- Missing dependencies
- Feature mismatch between training and inference
- File I/O errors

## Requirements

- Python 3.6+
- PyTorch
- RDKit
- pandas
- numpy
- joblib
- LightGBM
- scikit-learn
