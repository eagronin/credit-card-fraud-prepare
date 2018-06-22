# Data Preparation

This section examines summary statistics for the fraud_data.csv dataset.  It then splits the dataset into training and test sets to train several models and evaluate their effectiveness in detecting fraud in credit card transactions. This project focuses on selecting the appropriate model evaluation metrics when classes are imbalanced.

A description of the dataset and code for importing the data are provided in the [previous section](https://eagronin.github.io/credit-card-fraud-acquire/).

Construction of the model and analysis are presented in the [next section](https://eagronin.github.io/credit-card-fraud-analyze/).

The following code outputs summary statistcis for each of the features:

```python
# Read data and output summary statistics 
df = read_transactions_data()

print(round(df.describe().transpose(), 3))
print('\nThe number of missing values across all attributes and samples: ', df.isnull().sum().sum())
```

The summary statistics below show that there are 21,693 transactions in the data, of which 1.6% are fraudulant.  The average transaction amount being substantially higher than the median suggests that there is a relatively small number of very large transactions that drive the mean upward.  The dataset has no missing values.

```
          count    mean      std     min    25%     50%     75%       max
V1      21693.0  -0.032    2.107 -41.929 -0.929   0.008   1.316     2.452
V2      21693.0   0.048    1.691 -40.804 -0.593   0.075   0.820    21.467
V3      21693.0  -0.092    1.870 -31.104 -0.963   0.177   1.021     4.070
V4      21693.0   0.058    1.540  -4.849 -0.850  -0.013   0.772    12.115
V5      21693.0  -0.034    1.531 -32.092 -0.698  -0.064   0.615    29.162
V6      21693.0  -0.023    1.341 -20.368 -0.779  -0.282   0.384    21.393
V7      21693.0  -0.074    1.597 -41.507 -0.565   0.031   0.564    34.303
V8      21693.0   0.002    1.413 -38.987 -0.206   0.023   0.328    20.007
V9      21693.0  -0.044    1.159 -13.434 -0.670  -0.074   0.590     9.126
V10     21693.0  -0.091    1.355 -24.403 -0.555  -0.099   0.445    12.702
V11     21693.0   0.067    1.154  -3.996 -0.739   0.006   0.786    12.019
V12     21693.0  -0.094    1.365 -18.600 -0.439   0.127   0.614     3.970
V13     21693.0  -0.001    0.990  -3.845 -0.634  -0.019   0.652     4.099
V14     21693.0  -0.091    1.356 -19.214 -0.438   0.045   0.490     6.441
V15     21693.0  -0.004    0.917  -4.499 -0.582   0.049   0.642     5.720
V16     21693.0  -0.055    1.096 -14.130 -0.493   0.060   0.525     6.443
V17     21693.0  -0.098    1.425 -24.019 -0.499  -0.076   0.390     6.609
V18     21693.0  -0.033    0.937  -9.499 -0.513  -0.019   0.495     3.790
V19     21693.0   0.022    0.844  -4.400 -0.444   0.022   0.485     4.850
V20     21693.0  -0.002    0.728 -21.025 -0.210  -0.057   0.139    13.120
V21     21693.0   0.012    0.850 -21.454 -0.225  -0.024   0.193    27.203
V22     21693.0   0.004    0.741  -8.887 -0.538   0.007   0.530     8.362
V23     21693.0  -0.002    0.630 -21.304 -0.162  -0.012   0.147    15.626
V24     21693.0  -0.002    0.600  -2.767 -0.356   0.037   0.432     4.014
V25     21693.0  -0.000    0.521  -4.542 -0.317   0.012   0.354     5.542
V26     21693.0   0.002    0.478  -1.855 -0.326  -0.045   0.239     3.463
V27     21693.0   0.002    0.425  -7.764 -0.070   0.002   0.096     9.880
V28     21693.0   0.003    0.302  -6.520 -0.053   0.012   0.082     9.876
Amount  21693.0  86.776  235.644   0.000  5.370  21.950  76.480  7712.430
Class   21693.0   0.016    0.127   0.000  0.000   0.000   0.000     1.000

The number of missing values across all attributes and samples:  0
```

The following code splits the sample into training and test sets:

```python
# Split the data into X_train, X_test, y_train, y_test
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
```

Features need to be scaled before we train the models described in the [next section](https://eagronin.github.io/credit-card-fraud-analyze/).  The code below fits a scaler to the training data and transforms both the training and test data using the fitted scaler.  

It is important to note that the scaler should be fitted to the training data only (rather than to the entire dataset) in order to prevent leakage of information from the test data.

```python
# Scale the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Next step: [Analysis](https://eagronin.github.io/credit-card-fraud-analyze/)
