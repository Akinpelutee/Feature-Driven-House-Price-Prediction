# Feature-Driven-House-Price-Prediction
This Test driven development project used full feature engineering process to carefully predict house prices using linear regression model.

Predicting house prices seem to be one of the most popular machine learning project in the world of data, but doing some of the things i incorporated in this project can really set your project apart. In this machine learning project, house prices was predicted using house size, house location (lat and long) and finally, all the variables were taken into account using the King County House Price Sales dataset.

There are three models built for this project which takes into consideration size of the house, it's location and all variables. All the python scripts for this project are all available HERE on my github repo. There are two main concepts that intruiged me in this project, which are basically writing efficient pytest for the price_size.py script (TDD i.e Test Driven Development) and a form of feature engineering which involves the issue of multicollinearity, handling leaky columns and removing low and high cardinallity column for object data types. All terms will be discussed in later part of this write up.


For this project, I used the mean absolute error metric to access my model's performance. MAE gives you an idea of how far your predictions are far from your actual values on average. A lower MAE indicates better predictive performance.
An initial baseline model was created using the mean value of the house price and the the house price itself. The mean absolute error(MAE baseline model) was then computed by getting the sum of differences between each price and the mean price then divide by the number of observations. The initial MAE of the baseline model resulted at about $207,826.46 which is quite much. Now, the solution to this is to beat the baseline mae, we need to keep the mean absolute error as low as possible.


This represents a decrease of approximately 55.8% in MAE indicating that my model is able to make predictions that are on average, $115,890.77 closer to the actual value compared to the baseline.


Before reaching for advanced algorithms, invest in understanding your features how they relate, overlap, or leak information.

You can read more on the project here.
