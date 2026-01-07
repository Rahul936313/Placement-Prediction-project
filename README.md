# Student Placement Prediction System

A machine learning project that predicts student placement probability based on academic and skill-based criteria.https://github.com/Rahul936313/Placement-Prediction-project/edit/main/README.md
# The app is live on  https://placement-prediction-project-1.onrender.com/
## Features

- **Neural Network Model**: Deep learning model for placement prediction
- **Web Interface**: Streamlit-based frontend for easy predictions
- **Batch Prediction**: Support for predicting multiple students at once

## Project Structure

```
placement prediction project/
├── DS-placement.ipynb              # Main notebook with model training
├── college_student_placement_dataset.csv  # Training dataset
├── predict_placement.py            # Standalone prediction script
├── app.py                          # Streamlit web application
├── requirement.txt                 # Python dependencies
└── README.md                       # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirement.txt
```

## Usage

### Step 1: Train the Model

1. Open `DS-placement.ipynb` in Jupyter Notebook
2. Run all cells to train the model
3. The model will be saved as `placement_prediction_model.pth`

### Step 2: Run the Web Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown (usually `http://localhost:8501`)

3. Enter student details:
   - IQ (Intelligence Quotient)
   - Previous Semester Result
   - CGPA (Cumulative Grade Point Average)
   - Communication Skills (0-10)
   - Projects Completed

4. Click "Predict Placement" to get the prediction percentage

### Alternative: Command Line Prediction

You can also use the standalone script:

```bash
python predict_placement.py
```

Or import it in your code:

```python
from predict_placement import PlacementPredictor

predictor = PlacementPredictor()
percentage = predictor.predict(
    IQ=110,
    Prev_Sem_Result=8.5,
    CGPA=8.7,
    Communication_Skills=8,
    Projects_Completed=3
)
print(f"Placement Probability: {percentage:.2f}%")
```

## Model Details

- **Architecture**: Neural Network (5 → 64 → 32 → 16 → 1)
- **Input Features**: 
  - IQ
  - Previous Semester Result
  - CGPA
  - Communication Skills
  - Projects Completed
- **Output**: Placement probability (0-100%)
- **Preprocessing**: PCA transformation + StandardScaler normalization

## Notes

- The model uses 5 features after PCA transformation
- Input features are normalized before prediction
- Predictions are probabilities, with ≥50% indicating "Placed"

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- Streamlit
- pandas, numpy
- matplotlib, seaborn

## Troubleshooting

1. **Model file not found**: Make sure you've run the notebook and saved the model
2. **Import errors**: Install all requirements: `pip install -r requirement.txt`
3. **Prediction errors**: Check that input values are within valid ranges

## License

This project is for educational purposes.

