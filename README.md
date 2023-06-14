 Laptop Price Prediction project in Machine Learning:

# Laptop Price Prediction

This project aims to predict the prices of laptops using machine learning techniques. By analyzing various features of laptops such as processor type, RAM size, storage capacity, and more, we can build a predictive model that estimates the price of a laptop.

## Dataset

The dataset used for this project is sourced from kaggle = https://tinyurl.com/yxyjs6yh  sheet=https://tinyurl.com/4vypweah and consists of labeled data points of laptops with their corresponding prices and features. The dataset contains the following columns:

- company: The company of the laptop.
- Processor: The type of processor used in the laptop.
- RAM: The size of the RAM in gigabytes.
- Storage: The storage capacity of the laptop in gigabytes.
- ScreenSize: The size of the laptop's screen in inches.
- GPU: The graphics processing unit (GPU) used in the laptop.
- OpSys: The operating system installed on the laptop.
- Weight: The weight of the laptop in kilograms.
- Price: The price of the laptop in the target currency.

## Installation

To run this project locally, 

Open the `https://drive.google.com/file/d/1rgG-PkhuOMbaObfmQiLpqSrUUuS_H0lG/view?usp=drive_link` file in the Jupyter Notebook and execute the cells to run the project.

## Usage

In the Jupyter Notebook, you will find the following sections:

1. Data Exploration: Analyzing the dataset, checking for missing values, and performing exploratory data analysis (EDA).

2. Data Preprocessing: Handling missing values, encoding categorical variables, and performing feature scaling.

3. Model Building: Splitting the data into training and testing sets, training various machine learning models such as linear regression, Linear Regression, Lasso Regression,KNN k-nearest neighbors algorithm,decision tree, and XGBoost.

4. Model Evaluation: Evaluating the trained models using evaluation metrics such as mean absolute error (MAE), root mean squared error (RMSE), and R-squared score.

5. Price Prediction: Using the best-performing model to predict the price of a new laptop based on its features.

## Results

After evaluating various models, we found that the decision tree model achieved the highest accuracy with an R2 score: 0.750533767883519. This indicates that our model can explain 75% of the variance in laptop prices.

## Conclusion

The Laptop Price Prediction project demonstrates the application of machine learning techniques to predict laptop prices based on their features. By utilizing the provided dataset and training different models, we were able to achieve accurate price predictions. The trained model can be used by retailers, customers, or manufacturers to estimate laptop prices and make informed decisions.

## Contributing

Contributions to this project are welcome. If you have any suggestions or improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

-https://tinyurl.com/yxyjs6yh  for providing the dataset used in this project.
-https://tinyurl.com/4vypweah = xl sheets link
- [Scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/) libraries for machine learning algorithms and tools.
- [Pandas](https://pandas.pydata.org/) 
- [NumPy](https://numpy.org/)

 libraries for data manipulation and analysis.
- [Matplotlib](https://matplotlib.org/) 
- [Seaborn](https://seaborn.pydata.org/) libraries for data visualization.
