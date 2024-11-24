# Student Loan Risk Prediction Model

This project implements a deep learning model to predict student loan repayment success using TensorFlow/Keras. The model analyzes various student and loan-related features to assess credit risk.

## Requirements

- Python 3.10+
- TensorFlow 2.x
- pandas
- scikit-learn
- numpy

Install dependencies using:
```bash
pip install tensorflow pandas scikit-learn numpy
```

## Dataset Structure

The model expects input data in CSV format with the following columns:

- payment_history (float)
- location_parameter (float)
- stem_degree_score (float)
- gpa_ranking (float)
- alumni_success (float)
- study_major_code (float)
- time_to_completion (float)
- finance_workshop_score (float)
- cohort_ranking (float)
- total_loan_score (float)
- financial_aid_score (float)
- credit_ranking (int, target variable: 0 or 1)

## Using the Jupyter Notebook

1. Clone this repository
2. Navigate to the project directory
3. Launch Jupyter Notebook:
```bash
jupyter notebook
```
4. Open `student_loans_with_deep_learning.ipynb`
5. Run all cells in sequence

## Using the Standalone Model

### Loading the Model

```python
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = tf.keras.models.load_model('student_loans.keras')
```

### Preparing Input Data

```python
# Load your data
data = pd.read_csv('your_data.csv')

# Create the feature set (exclude credit_ranking if present)
X = data.drop('credit_ranking', axis=1) if 'credit_ranking' in data.columns else data

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Make predictions
predictions = model.predict(X_scaled)
# Convert to binary predictions (0 or 1)
binary_predictions = (predictions >= 0.5).astype(int)
```

## Model Performance

The current model achieves:
- Accuracy: 74%
- Precision: 74%
- Recall: 74%
- F1-Score: 74%

Performance metrics for individual classes:
- Class 0 (Not Successful): 71% precision, 76% recall
- Class 1 (Successful): 77% precision, 72% recall

## Model Architecture

The neural network consists of:
- Input layer: 11 features
- First hidden layer: 6 neurons (ReLU activation)
- Second hidden layer: 3 neurons (ReLU activation)
- Output layer: 1 neuron (Sigmoid activation)

## Notes

- All input features should be numeric and properly scaled
- The model returns probabilities between 0 and 1, where values ≥ 0.5 indicate likely successful loan repayment
- Missing values should be handled before feeding data to the model
- Consider retraining the model periodically with updated data to maintain accuracy

## Contributing

Not open to contributions at this time.