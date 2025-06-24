import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer


def load_data(filepath):
    """
    Reads in the data

    Returns:
        The data in a pandas dataframe format
    """
    return pd.read_csv(filepath)

def wrangle(df):
    """
    Cleans and preprocess the data for analysis

    Returns:
        A clean data for predicting house prices
    """
    #Round up bathrooms and floors to whole number and also change the datatype to int
    df[["bathrooms", "floors"]] = df[["bathrooms", "floors"]].round().astype(int)

    #Rename the price column
    df.rename(columns={"price": "price_($)"}, inplace=True)

    #Rmove the top and bottom 3% of the data to mitigate outliers issue
    low, high = df["sqft_living"].quantile([0.3, 0.97])
    mask_ft = df["sqft_living"].between(low, high)
    df = df[mask_ft]
    
    return df

def explore_data(df):
    """
    Creates visualizations like boxplot, histogram and scatterpplot to better understand the data 

    Returns:
        Visualizations like boxplot, histogram and scatterplot
    """
    #Let's create a boxplot of the sqft_living to show any outliers in the column
    plt.boxplot(df["sqft_living"], vert=False)
    plt.xlabel("Sqft of house")
    plt.title("Distribution of house sizes");

    #A scatter plot showing the relationship between sqft living and price
    plt.scatter(x=df["sqft_living"], y=df["price_($)"])
    plt.xlabel("Sqft of house")
    plt.ylabel("Price_($)");


def cal_corr(df):
    """
    calculates pearson's correlation coefficient between price and sqft_living

    Returns:
        The pearson correlation 
    """
    p_corr = df["sqft_living"].corr(df["price_($)"])
    
    return p_corr

def split_data(df, features, target):
    """
    splits the data into train and test set
    
    Returns:
        the split data
    """
    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_baseline_model(y_train):
    """
    calculates the baseline model and mae of the training data
    
    Returns:
        the mae of the training data and the baseline model
    """
    #Calculates the baseline mean
    y_train_mean = y_train.mean()
    y_pred_baseline = [y_train_mean] * len(y_train)
    mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
    
    return mae_baseline, y_train_mean

def train_model(X_train, y_train):
    """
    Trains and fits the model
    
    Returns:
        model
    """
    model = make_pipeline(
        SimpleImputer(),
        LinearRegression()
    )

    #Fits the model
    model.fit(X_train, y_train)

    return model

def evaluate_model(model,X_test, y_test):
    """
    Predicts the price of house Using the test set and calculates the mae of the test set
    
    Returns:
        the mae of test set
    """
    #Predict house prices using the test set
    y_pred_test = model.predict(X_test)

    #Calculates the mae of the X_test
    mae_test = mean_absolute_error(y_test, y_pred_test)

    return mae_test, y_pred_test

def display_pred_values(y_test, y_pred_test):
    """
    Displays the predicted values of your model with the test data side by side with the real test data.
    Just to visually know how your model performs
    
    Returns:
        y_test and y_pred_test
    """
    data = pd.DataFrame({'Real Values': y_test , 'Predicted Values': y_pred_test})

    return data

#Main script to call the functions
if __name__ == "__main__":
    # Load and wrangle data
    filepath = "kc_house_data.csv"
    features = ["sqft_living"]
    target = "price_($)"

    df = load_data(filepath)
    print(f"Data shape before wrangling is {df.shape}")

    df = wrangle(df)
    print(f"Data shape after cleaning: {df.shape}")

    p_corr = cal_corr(df)
    print(f"The correlation between price and sqft_living is {p_corr}")

    X_train, X_test, y_train, y_test = split_data(df, features, target)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    mae_baseline, y_train_mean = build_baseline_model(y_train)
    print(f"Actual mean is {round(y_train_mean, 2)}")
    print(f"MAE of baseline model is {round(mae_baseline, 2)}")

    model = train_model(X_train, y_train)
    print(f"Model has been trained successfully...")

    mae_test, y_pred_test = evaluate_model(model, X_test, y_test)
    if mae_test < mae_baseline:
        print(f"Congratulations!!! You beat the baseline model.\nYou beat the baseline at about {mae_baseline - mae_test}.")
    else:
        print("Ugh! you didn't beat the baseline model.")

    data = display_pred_values(y_test, y_pred_test)
    print(f"\nReal vs Predicted values (first 10 rows):\n{data.head(10)}")
