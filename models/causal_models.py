from models.model import *

class Law:
    def __init__(self, data):
        self.data = data
        self.model = {'RS->G': LinearRegression(),
                      'RS->L': LinearRegression(), 
                      'RS->F': LinearRegression()}

    def fit(self):
        for name in self.model.keys():
            X_name, y_name = name.split('->')
            X_name = list(X_name)
            X = self.data[X_name].values
            y = self.data[y_name].values
            self.model[name].fit(X, y)
    
    def __call__(self, X):
        # The input is RS, and the output is F.
        return self.model['RS->G'](X), self.model['RS->L'](X), self.model['RS->F'](X)


class COMPAS:
    def __init__(self, data):
        self.data = data
        self.model = {'RS->G': LinearRegression(),
                      'RS->L': LinearRegression(), 
                      'RS->F': LinearRegression()}

    def fit(self):
        for name in self.model.keys():
            X_name, y_name = name.split('->')
            X_name = list(X_name)
            X = self.data[X_name].values
            y = self.data[y_name].values
            self.model[name].fit(X, y)
    
    def __call__(self, X):
        # The input is RS, and the output is F.
        return self.model['RS->G'](X), self.model['RS->L'](X), self.model['RS->F'](X)

