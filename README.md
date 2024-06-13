# Student Performance Prediction and Educational Interventions

This project aims to predict student performance and provide educational interventions based on influential factors. We use the Student Performance Data Set, combining math and Portuguese language datasets to build a predictive model and offer tailored recommendations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Overview](#project-overview)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Educational Interventions](#educational-interventions)
- [Example Usage](#example-usage)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd student-performance-prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the dataset files (`student-mat.csv` and `student-por.csv`) in the project directory.
2. Run the script:
   ```bash
   python main.py
   ```

## Project Overview

This project involves the following steps:
1. Loading and combining the student performance datasets.
2. Cleaning and preprocessing the data.
3. Splitting the data into training and testing sets.
4. Training a RandomForestRegressor model to predict student performance.
5. Evaluating the model's performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
6. Analyzing feature importance to understand influential factors.
7. Providing educational interventions based on model predictions.

## Model Training

The model training involves data loading, preprocessing, feature and label separation, train-test split, and training the RandomForestRegressor model.

## Evaluation

Evaluate the model using MAE and RMSE to measure its performance.

## Feature Importance

Analyze feature importance to understand the influential factors that affect student performance.

## Educational Interventions

Provide tailored recommendations based on influential features identified during the model training process.

## Example Usage

Example of recommending interventions for a student, showcasing how the model's predictions can be used to provide personalized educational support.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
