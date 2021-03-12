from numpy.lib.ufunclike import isneginf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression as LR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class SaleDate_to_Year:
    '''Convert saledate column to year
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        X['saleyear'] = pd.to_datetime(X['saledate']).dt.year
        X.drop('saledate', axis = 1, inplace = True)
        return X, y 

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)


    def __repr__(self):
            return f'Convert Saledate to Year'

class NaNFiller:
    '''Fill any NaN values with column means
    '''
    def __init__(self):
        self.column_means = None

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X,y = X

        self.column_means = X.mean(axis = 0)

        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X

        X.fillna(self.column_means, inplace = True)

        return X, y 

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)


    def __repr__(self):
            return f'Fill NaN values with the mean of the columns'

class TooManyCategoriesDropper:
    '''Drop any categorical column that has too many values to use.
    '''
    def __init__(self, max_values =10):
        self.max_values = 20
        self.columns = None

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X,y = X

        self.columns = X.columns[[(X[i].nunique()>=self.max_values) & (X[i].dtype == 'object') for i in X.columns]]

        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X

        X.drop(self.columns, axis=1, inplace = True)

        return X, y 

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)


    def __repr__(self):
            return f'ColumnDropper for columns with more than {self.max_values} values'
    
class ColumnOneHotter:
    '''Preform OneHotEncoding on any object column with less than max_value unique values
    '''
    def __init__(self, max_values =10):
        self.max_values = 20
        self.columns = None

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X,y = X

        self.columns = X.columns[[(X[i].nunique()<self.max_values) & (X[i].dtype == 'object') for i in X.columns]]

        self.encoders = {column: OHE(handle_unknown = 'ignore').fit(X[column].values.reshape(-1,1)) for column in self.columns}

        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X

        for column in self.columns:
            data = self.encoders[column].transform(X[column].values.reshape(-1,1))

            feature_names = self.encoders[column].get_feature_names()
            feature_names = [f'{column}_{f}' for f in feature_names]

            data = pd.DataFrame(data.todense())

            data.columns = feature_names

            X.drop(column, axis = 1, inplace = True)
            X = pd.concat([X, data], axis = 1)

        return X, y 

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)


    def __repr__(self):
        if self.columns is not None:
            return f'OneHotEncoder for {len(self.columns)} columns with less than {self.max_values} values'
        return f'Unfitted OneHotEncoder for columns with less than {self.max_values} values'

class ColumnDropper:
    '''Drop Specified Columns
    '''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X

        for column in self.columns:
            if column in X.columns:
                X.drop(column, axis = 1, inplace = True)

        return X,y

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)

    def __repr__(self):
        return f'Drop {self.columns}'

class ColumnClipper:
    '''Clip specified columns to range of values
    '''
    def __init__(self, column, lower, upper) :
        self.column = column
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        X[self.column] = X[self.column].clip(self.lower, self.upper)

        return X, y

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)

max_features = 20

pipeline = Pipeline([ 
    ('sale_date_to_year', SaleDate_to_Year()),
    ('drop_IDs', ColumnDropper(['SalesID','MachineID'])),
    ('trim_year', ColumnClipper('YearMade', 1900, 2020)),
    ('OneHotEncoder', ColumnOneHotter(max_features)),
    ('ColumnDropper', TooManyCategoriesDropper(max_features)),
    ('NaNFiller', NaNFiller())
                    ])

tf_model = Sequential()
tf_model.add(Dense(256, activation = 'relu'))
tf_model.add(Dense(128, activation = 'relu'))
tf_model.add(Dense(64, activation = 'relu'))
tf_model.add(Dense(32, activation = 'relu'))
tf_model.add(Dense(1, activation = 'linear'))
tf_model.compile(optimizer = 'adam', loss = 'mse')

if __name__ == '__main__':

    try:
        X = pd.read_csv('prepared_data.csv')
        y = X.pop('y')
        print('Found preprared data, so reading that in')
    
    except:
        print('Could not find prepared data, so generating it anew')
        frac = 1
        df = pd.read_csv('data/Train.zip', low_memory = False).sample(frac = frac).reset_index(drop = True)

        y = np.log(df.pop('SalePrice'))

        X,y = pipeline.fit_transform(df,y)

        prepared_data = X.copy()
        prepared_data['y'] = y
        prepared_data.to_csv('prepared_data.csv', index = False)

    X_train, X_test, y_train, y_test = tts(X,y)

    # model = KNeighborsRegressor(40).fit(X_train,y_train)
    # model = RFR(n_estimators = 400, max_depth = 10, max_features = 'sqrt')
    # model = LR()
    
    history = tf_model.fit(X_train, y_train, epochs = 20, validation_data = (X_test, y_test)).history

    print(mse(y_test, tf_model.predict(X_test))**.5)

    fig, ax = plt.subplots()
    ax.plot(history['loss'], label = 'Train')
    ax.plot(history['val_loss'], label = 'Test')
    ax.set_yscale('log')
    plt.legend()
    plt.show()