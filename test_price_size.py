import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

from price_and_size_script import load_data, wrangle, cal_corr, split_data, build_baseline_model, train_model, evaluate_model, display_pred_values

@pytest.fixture
def test_data():
    """Creates variables and respective data"""
    data = {
        'price': [
            100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 2000, 1900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000],
        'sqft_living': [
            1000, 2000, 3000, 1500, 2500, 3500, 1200, 2200, 3200, 1800, 2800, 3800, 1300, 2300, 3300, 1900, 2900, 3900, 1400, 2400],
        'sqft_lot': [
            10, 20, 35, 25, 15, 46, 58, 37, 58, 48, 12, 22, 38, 28, 18, 50, 60, 40, 55, 65],
        'date': [
            '2027-01-02', '2020-02-03', '2020-03-03', '2020-02-05', '2020-02-04', '2020-02-04', '2020-02-04', '2020-02-04', '2020-02-04', '2020-02-04',
            '2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06', '2021-07-07', '2021-08-08', '2021-09-09', '2021-10-10'],
        'bathrooms': [
            2.5, 3.7, 1.2, 2.1, 3.2, 1.5, 2.8, 3.1, 1.9, 2.4, 2.6, 3.8, 1.1, 2.3, 3.4, 1.6, 2.9, 3.2, 1.8, 2.5],
        'floors': [
            1.1, 2.2, 3.3, 1.4, 2.5, 3.6, 1.7, 2.8, 3.1, 1.2, 2.3, 3.4, 1.5, 2.6, 3.7, 1.8, 2.9, 3.2, 1.3, 2.4],
    }
    return pd.DataFrame(data)
    

    
def test_load_data(test_data):
    """tests the load_data function"""
    test_data.to_csv('test_data.csv', index=False)
    loaded_data = load_data('test_data.csv')
    pd.testing.assert_frame_equal(test_data, loaded_data)

def test_wrangle_dtypes(test_data):
    wrangled_data = wrangle(test_data)
    assert wrangled_data['bathrooms'].dtype == int
    assert wrangled_data['floors'].dtype == int

def test_wrangle_column_renamed(test_data):
    wrangled_data = wrangle(test_data)
    assert 'price_($)' in wrangled_data.columns
    assert 'price' not in wrangled_data.columns

def test_wrangle_outlier(test_data):
    test_data.loc[len(test_data)] = {
        'price': 200, 'sqft_living': 10000, 'sqft_lot': 15, 'sqft_lot': 23, 'date':'2020-02-07', 'bathrooms': 3, 'floors': 5
    }
    wrangled_data = wrangle(test_data)
    assert wrangled_data['sqft_living'].min() >= test_data['sqft_living'].quantile(0.03)
    assert wrangled_data['sqft_living'].max() <= test_data['sqft_living'].quantile(0.97)

def test_wrangle_return_type(test_data):
    wrangled_data = wrangle(test_data)
    assert isinstance(wrangled_data, pd.DataFrame)

def test_cal_corr(test_data):
    wrangled_data = wrangle(test_data)
    exp_corr = wrangled_data['sqft_living'].corr(wrangled_data['price_($)'])
    act_corr = cal_corr(wrangled_data)
    assert np.isclose(act_corr, exp_corr)

def test_split_data_type(test_data):
    #Asserts whether it returns dataframe and series for the features respectively
    wrangled_data = wrangle(test_data)
    features = ['sqft_living']
    target = 'price_($)'
    X_train, X_test, y_train, y_test = split_data(wrangled_data, features, target)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_split_data_shape(test_data):
    #Checks if we have correct length of splits
    wrangled_data = wrangle(test_data)
    X_train, X_test, y_train, y_test = split_data(wrangled_data, ['sqft_living'], 'price_($)')
    assert X_train.shape[0] + X_train.shape[0] == len(wrangled_data)
    assert X_test.shape[0] + X_test.shape[0] == len(wrangled_data)
    assert_frame_equal(X_train.shape[1], len(features))
    assert_frame_equal(X_test.shape[1], len(target))

    

    
    
    
    