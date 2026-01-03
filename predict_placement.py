"""
Standalone Placement Prediction Script
This script loads the trained model and makes predictions
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

class PlacementPredictor:
    def __init__(self, model_path='placement_prediction_model.pth'):
        """
        Initialize the predictor by loading the saved model
        
        Parameters:
        - model_path: Path to the saved model file
        """
        # Load the saved model
        # Note: weights_only=False is needed for PyTorch 2.6+ because the checkpoint
        # contains scikit-learn objects (StandardScaler) which are not pure PyTorch weights
        # This is safe since we're loading our own trained model
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # Define the model architecture (must match training)
        class PlacementPredictionNN(torch.nn.Module):
            def __init__(self, input_size=5):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, 64)
                self.fc2 = torch.nn.Linear(64, 32)
                self.fc3 = torch.nn.Linear(32, 16)
                self.fc4 = torch.nn.Linear(16, 1)
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(0.3)
                self.sigmoid = torch.nn.Sigmoid()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.sigmoid(self.fc4(x))
                return x
        
        # Initialize and load model
        self.model = PlacementPredictionNN(input_size=5)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler - try from checkpoint first, then from separate pickle file
        try:
            self.scaler = checkpoint['scaler']
        except (KeyError, AttributeError):
            # Fallback: try loading from separate pickle file
            scaler_path = 'scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                raise FileNotFoundError("Scaler not found in checkpoint or scaler.pkl file")
        
        # Load PCA components and feature names
        self.pca_components = checkpoint['pca_components']
        self.feature_names = checkpoint.get('feature_names', 
            ['IQ', 'Prev_Sem_Result', 'CGPA', 'Communication_Skills', 'Projects_Completed'])
        
        print("Model loaded successfully!")
        print(f"Features: {self.feature_names}")
    
    def predict(self, IQ, Prev_Sem_Result, CGPA, Communication_Skills, Projects_Completed):
        """
        Predict placement percentage
        
        Parameters:
        - IQ: Intelligence Quotient
        - Prev_Sem_Result: Previous Semester Result
        - CGPA: Cumulative Grade Point Average
        - Communication_Skills: Communication Skills Score
        - Projects_Completed: Number of Projects Completed
        
        Returns:
        - placement_percentage: Probability of placement (0-100%)
        """
        try:
            # Create feature array in the correct order
            # Based on the training: IQ, Prev_Sem_Result, CGPA, Communication_Skills, Projects_Completed
            features = np.array([[IQ, Prev_Sem_Result, CGPA, Communication_Skills, Projects_Completed]])
            
            # Apply PCA transformation
            features_pca = np.matmul(features, self.pca_components)
            
            # Normalize using the same scaler
            features_scaled = self.scaler.transform(features_pca)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(features_tensor).item()
            
            # Convert to percentage
            placement_percentage = prediction * 100
            
            return placement_percentage
        
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
    
    def predict_batch(self, students_data):
        """
        Predict placement for multiple students
        
        Parameters:
        - students_data: List of dictionaries with student information
        
        Returns:
        - List of dictionaries with predictions
        """
        results = []
        for i, student in enumerate(students_data):
            try:
                percentage = self.predict(
                    student['IQ'],
                    student['Prev_Sem_Result'],
                    student['CGPA'],
                    student['Communication_Skills'],
                    student['Projects_Completed']
                )
                results.append({
                    'Student_ID': i + 1,
                    'Placement_Percentage': percentage,
                    'Status': 'Placed' if percentage >= 50 else 'Not Placed',
                    **student
                })
            except Exception as e:
                results.append({
                    'Student_ID': i + 1,
                    'Error': str(e),
                    **student
                })
        return results


if __name__ == "__main__":
    # Example usage
    try:
        predictor = PlacementPredictor()
        
        # Test prediction
        print("\n" + "="*60)
        print("Testing Prediction Function")
        print("="*60)
        
        test_student = {
            'IQ': 110,
            'Prev_Sem_Result': 8.5,
            'CGPA': 8.7,
            'Communication_Skills': 8,
            'Projects_Completed': 3
        }
        
        percentage = predictor.predict(**test_student)
        print(f"\nTest Student:")
        print(f"  IQ: {test_student['IQ']}")
        print(f"  CGPA: {test_student['CGPA']}")
        print(f"  Projects: {test_student['Projects_Completed']}")
        print(f"  Predicted Placement Percentage: {percentage:.2f}%")
        print(f"  Placement Status: {'Placed' if percentage >= 50 else 'Not Placed'}")
        
    except FileNotFoundError:
        print("Error: Model file 'placement_prediction_model.pth' not found!")
        print("Please run the notebook first to train and save the model.")
    except Exception as e:
        print(f"Error: {str(e)}")

