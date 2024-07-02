# Laptop Price Prediction for SmartTech Co.

## Project Overview
SmartTech Co. has partnered with our data science team to develop a robust machine learning model that predicts laptop prices accurately. As the market for laptops continues to expand with a myriad of brands and specifications, having a precise pricing model becomes crucial for both consumers and manufacturers.

### Data Sources
- Original Dataset: laptop.csv
- External Resources: None

## Methodologies and Tools
- **Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Streamlit
- **Methodologies:** Data Exploration, Data Preprocessing, Feature Engineering, Model Development, Hyperparameter Tuning, Real-time Predictions, Interpretability and Insights

## Key Findings
The project achieved:
- Development of a reliable machine learning model (XGBoost) for accurate laptop price prediction.
- Insights into influential features affecting laptop prices, aiding market positioning strategies.

## Visuals
![Visualization Example](link-to-image.png)
[Link to interactive dashboard, if applicable]

## Documentation

### `ML_Project_(Part_1).ipynb`: 
This notebook covers the initial stages of the project, including data exploration, data cleaning, and preprocessing. The original dataset `laptop.csv` is used for this purpose. After preprocessing, the cleaned dataset is saved as `new_laptop.xlsx`.

### `ML_Project(Part_2).ipynb`: 
In this notebook, feature selection and encoding techniques are applied to the preprocessed dataset (`new_laptop.xlsx`). The encoded data is saved as `encoded_laptop.xlsx`. Additionally, machine learning models are developed, trained, and evaluated. The best-performing model, XGBoost, is selected and saved as `best_xgb_model.pkl`.

### Files and Descriptions
- `laptop.csv`: Original dataset containing information about laptops.
- `new_laptop.xlsx`: Preprocessed dataset obtained after data cleaning and preprocessing.
- `encoded_laptop.xlsx`: Dataset with encoded features generated from `new_laptop.xlsx`.
- `best_xgb_model.pkl`: Trained XGBoost model, selected as the best-performing model for laptop price prediction.
- `feature_df_laptop.xlsx`: Dataset used for the Streamlit UI. It contains simplified features without the target variable (price) and some complex or unnecessary features to facilitate a streamlined user interface.
- `LaptopPricePrediction.py`: Python file containing the Streamlit application for the laptop price prediction model. It provides a user-friendly interface for predicting laptop prices based on the trained model. Users can run the application locally by executing the provided command in their command prompt.
- `README.md`: This file, providing an overview of the project and descriptions of the files present in the repository.

### Dependencies:
1. Pandas
2. NumPy
3. Matplotlib
4. Seaborn
5. Scikit-learn
6. Joblib
7. Streamlit
8. XGBoost
9. os
10. re
11. openxyl

### To reproduce the analysis:
1. Clone the repository.
2. Install dependencies.
3. Run the Streamlit application for laptop price prediction, execute the following command in the terminal:
```bash
streamlit run LaptopPricePrediction.py
```

## Role and Contributions
I played a pivotal role in:
- Conducting data exploration, preprocessing, and feature engineering.
- Developing and evaluating machine learning models.
- Creating the Streamlit application for real-time predictions.

## Tags
Machine Learning, Data Visualization, Statistical Analysis, Streamlit, XGBoost, Prediction, Laptop Sales
