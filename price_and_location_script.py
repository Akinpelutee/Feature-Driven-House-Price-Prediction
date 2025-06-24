import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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
    low, high = df["sqft_living"].quantile([0.2, 0.98])
    mask_ft = df["sqft_living"].between(low, high)
    df = df[mask_ft]

    return df

def data_exploration(df):
    """
    Creates visualizations like Scatter mapbox, 3D Scatter plot to understand the relationship between price and locations 

    Returns:
        Visualizations like Mapbox and 3D scatterplot
    """
    #Let's create a Sactter mapbox to better understand house_prices with thier locations
    fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat="lat",
    lon="long",
    width=600,  # Width of map
    height=600,  # Height of map
    color="price_($)",
    hover_data=["price_($)"],  # Display price when hovering mouse over house
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()

    #The below shows a 3D Scatter plot
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="long",
        y="lat",
        z="price_($)",
        labels={"lon": "longitude", "lat": "latitude", "price_($)": "price"},
        width=600,
        height=500,
    )
    
    # Refine formatting
    fig.update_traces(
        marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
        selector={"mode": "markers"},
    )
    
    # Display figure
    fig.show()

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

def display_model_equation(model):
    """
    Displays the model's equation 

    Returns:
        the model's equation with the intercept and coefficients
    """
    intercept = model.named_steps["linearregression"].intercept_.round(2)
    coefficient = model.named_steps["linearregression"].coef_.round(2)
    model_equ = print(f"price = {intercept} + {coefficient[0]} * longitude + {coefficient[1]} * latitude")

    return model_equ, intercept, coefficient

def display_pred_values(y_test, y_pred_test):
    """
    Displays the predicted values of your model with the test data side by side with the real test data.
    Just to visually know how your model performs
    
    Returns:
        y_test and y_pred_test
    """
    data = pd.DataFrame({'Real Values': y_test , 'Predicted Values': y_pred_test})

    return data


if __name__ == '__main__':

    #Loads the data
    filepath = "kc_house_data.csv"

    #features and target columns
    features = ["lat", "long"]
    target = "price_($)"

    #Pipeline
    df = load_data(filepath)
    print(f"Data shape before wrangling is {df.shape}")
    
    df = wrangle(df)
    print(f"Data shape after cleaning: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df, features, target)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    mae_baseline, y_train_mean = build_baseline_model(y_train)
    print(f"Actual mean is {round(y_train_mean, 2)}")
    print(f"MAE of baseline model is {round(mae_baseline, 2)}")

    model = train_model(X_train, y_train)

    mae_test, y_pred_test = evaluate_model(model, X_test, y_test)
    if mae_test < mae_baseline:
        print(f"Congratulations!!! You beat the baseline model.\nYou beat the baseline at about {mae_baseline - mae_test}.")
    else:
        print(f"Ugh! you didn't beat the baseline model.")

    model_equ, intercept, coefficient = display_model_equation(model)
    print(f"price = {intercept} + {coefficient[0]} * longitude + {coefficient[1]} * latitude")

    data = display_pred_values(y_test, y_pred_test)
    print(f"\nReal vs Predicted values (first 10 rows):\n{data.head(10)}")