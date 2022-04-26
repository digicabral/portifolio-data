import pickle
import pandas as pd

# Load data
def load_data():
    new_data = pd.DataFrame([[1,1.5,5,6]])
    return new_data

# Load model
def load_model():
    with open("trained_classifier.pkl","rb") as file:
        model = pickle.load(file)
    return model

# Make predictions
def make_predictions(data, model):
    return model.predict(data)

# Write results
def write_results(predictions):
    print(predictions)


def run():
    new_data = load_data()
    model = load_model()
    predictions = make_predictions(data=new_data, model=model)
    write_results(predictions=predictions)

if __name__ == "__main__":
    run()