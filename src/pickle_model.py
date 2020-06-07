import numpy as np
import pickle

class MyModel():

    def fit():
        pass

    def predict_proba():
        return np.random.uniform()


def get_data(datafile):
    ....
    return X, y

if __name__ == '__main__':
    X, y = get_data('data/data.json')
    model = MyModel()
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)