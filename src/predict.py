import joblib
import pandas as pd


def load_model():

    model = joblib.load("models/best_model.pkl")

    return model


def predict(data):

    model = load_model()

    prediction = model.predict(data)

    probability = model.predict_proba(data)[:,1]

    return prediction, probability


if __name__ == "__main__":

    sample = pd.read_csv("data/processed/X_test.csv").iloc[:5]

    pred, prob = predict(sample)

    print("Predictions:", pred)
    print("Probabilities:", prob)