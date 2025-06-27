
Ventilator Pressure Prediction: A Deep Learning Approach
This project provides a complete, research-level pipeline for the Kaggle "Ventilator Pressure Prediction" competition. It includes advanced feature engineering, training of both Transformer and LSTM models, hyperparameter tuning, and a final weighted ensemble with post-processing to generate a submission.

Project Structure
The project code is organized in the src/ directory. The data and saved models are expected to be in a separate, untracked directory as shown below.

.
├── ventilator-pressure-prediction/  <-- **Not tracked by Git**. Contains data and models.
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── saved_models_transformer/
│   └── saved_models_lstm_final/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── requirements.txt
└── README.md

Setup and Installation
1. Prerequisites
Python 3.8+

An NVIDIA GPU with CUDA is highly recommended for training.

2. Environment Setup
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt

3. Kaggle API Setup & Data Download
This project requires the Kaggle API to download the dataset.

Get your API credentials from your Kaggle account page and place the kaggle.json file in ~/.kaggle/.

Create the data directory:

mkdir -p ventilator-pressure-prediction/data

Download and unzip the data:

# From the root of the project directory
kaggle competitions download -c ventilator-pressure-prediction -p ventilator-pressure-prediction/data/
unzip ventilator-pressure-prediction/data/ventilator-pressure-prediction.zip -d ventilator-pressure-prediction/data/

This will place train.csv, test.csv, and sample_submission.csv inside the correct data/ folder.

Running the Pipeline
The entire project can be run from the main.py script. The script uses absolute paths defined in src/config.py, so ensure they are correct for your system.

# From the root of the project directory
python main.py
