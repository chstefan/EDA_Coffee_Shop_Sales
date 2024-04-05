# Coffee Shop Sales EDA & Forecasting

This project consists of a comprehensive exploratory data analysis (EDA) and sales forecasting model for a coffee shop's sales data. The aim is to understand sales trends, patterns, and factors influencing sales, ultimately aiding in making informed business decisions.

## Project Structure

The analysis and model development is structured as follows:

1. **Data Loading and Initial Exploration**: Load the sales data and perform initial exploration to understand the dataset structure, including viewing the first few rows and getting dataset information.

2. **Data Preprocessing**:
   - Handling missing values.
   - Combining date and time into a single datetime column and dropping the original columns.
   - Removing duplicates.
   - Standardizing numerical values.

3. **Feature Engineering**:
   - Generating dummy variables for categorical features.
   - Displaying summary statistics for numerical columns.

4. **Exploratory Data Analysis (EDA)**:
   - Visualizing the distribution of numerical attributes using histograms.
   - Analyzing the correlation between numerical variables through a heatmap.
   - Categorical analysis by visualizing the distribution of product categories.
   - Time series analysis to observe monthly sales trends.

5. **Modeling for Sales Forecasting**:
   - Building a linear regression model to forecast sales.
   - Utilizing a combination of time-related, categorical, and numerical features.
   - Employing a preprocessing pipeline for encoding categorical features.
   - Evaluating model performance using RMSE (Root Mean Square Error).

## Requirements

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Usage

To replicate this analysis and model:

1. **Data Preparation**: Ensure your dataset is in the correct format and path as mentioned in the script.

2. **Execution**: Run each code cell sequentially, from data loading to model evaluation.

3. **Customization**: Modify paths, features selection, and preprocessing steps as necessary to suit your dataset.

## Note

- The dataset path used in this project (`/Users/chantalstefan/Documents/GitProjects/Coffee_Shop_Sales_EDA/Coffee_Sales_Data.csv`) is specific to the author's local environment. Please adjust the `data_path` variable accordingly.
- The script assumes certain structure and column names for the dataset. Ensure your data matches or adjust the script accordingly.

## Contributions

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](#) if you want to contribute.


