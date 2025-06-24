import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from category_encoders import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline



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
    df[['bathrooms', 'floors']] = df[['bathrooms', 'floors']].round().astype(int)
    df.rename(columns={'price':'price_($)'}, inplace=True)
    low, high = df['sqft_living'].quantile([0.02, 0.98])
    mask_quantile = df['sqft_living'].between(low, high)
    df = df[mask_quantile]

    #Check the low and the high cardinality columns
    df.select_dtypes("object").nunique()

    #The date is the only object coumn here which is unnecessary to building our model. So we should drop it
    df.drop(columns="date", inplace=True)

    #Leaky columns are columns that we don't have access to when we're predicting after building our model in real life
    # Lets drop leaky column
    # Columns like yr_built, yr_renovated, zipcode, id  are leaky
    df.drop(columns=["yr_built","yr_renovated","id","zipcode"], inplace=True)

    #Check columns with multicollinearity
    corr = df.select_dtypes("number").drop(columns="price_($)").corr()
    #sns.heatmap(corr)

    #Columns like sqft_lot15,sqft_living15, sqft_above, are highly correlated with each other
    df.drop(columns=["sqft_lot15", "sqft_living15", "sqft_above"], inplace=True)

    return df

    

def data_exploration(df):
    """
    Creates visualizations like Histogram and scatter plot to understand the distribution of House prices 

    Returns:
        Visualizations like Histogram and 3D scatterplot
    """
    plt.hist(df["price_($)"] / 1e3)
    plt.xlabel("House_prices(1e6)")
    plt.ylabel("Count")
    plt.title("Distribution of house prices")

    #Let's create a scatterpplot which shows apartment price as a distribution of sqft_living
    plt.scatter(x=df["sqft_living"], y=df["price_($)"]/1e3)
    plt.xlabel("sqft_Living")
    plt.ylabel("Price($)")
    plt.title("Price vs House sqft_living")

def split_data(df, features, target):
    """
    splits the data into train and test set
    
    Returns:
        the split data
    """
    y = df.iloc[:,0].values
    X = df.iloc[:, 1:]

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

def display_values(y_test, y_pred_test):
    """
    Displays the True values of the test set and the predicted values

    Returns:
        the values in a DataFrame
    """
    data = pd.DataFrame({'Real Values' : y_test, 'Predicted values' : y_pred_test})

    return data

if __name__ == '__main__':

    #Loads the data
    filepath = "kc_house_data.csv"

    #Pipeline
    df = load_data(filepath)
    print(f"Data shape before wrangling is {df.shape}")
    
    df =wrangle(df)
    print(f"Data shape after wrangling is {df.shape}")

    #features and target columns
    features = df.iloc[:, 1:]
    target = df.iloc[:,0].values

    X_train, X_test, y_train, y_test = split_data(df, features, target)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    mae_baseline, y_train_mean = build_baseline_model(y_train)
    print(f"Actual mean house price is ${round(y_train_mean, 2)}")
    print(f"MAE of baseline model is ${round(mae_baseline, 2)}")

    model = train_model(X_train, y_train)

    mae_test, y_pred_test = evaluate_model(model, X_test, y_test)
    if mae_test < mae_baseline:
        print(f"Congratulations!!! You beat the baseline model.\n You beat the baseline at about ${round(mae_baseline - mae_test, 2)}.")
    else:
        print("Ugh! you didn't beat the baseline model.")

    data = display_values(y_test, y_pred_test)
    print(f"Below are the predicted house prices using the model. \n{data[:10]}")