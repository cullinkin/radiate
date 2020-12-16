# Standard:
import json

# 3rdparty:
import pandas
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Local:
from common import CSV, MOLECULES, MOLECULE_MAP, dataset_what
from plots import plot_results


def create_model():
    print(f"\n***** Initializing Model *****\n")
    model = Sequential()
    model.add(Dense(10, input_dim=5, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    print(f"\n***** Keras Model Created *****\n")

    print(f"\n***** Compiling Keras Model *****\n")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def load():
    print(f"\n***** Loading Molecule Dataset *****\n")
    final = pandas.read_csv(f"{CSV}/all_molecules.csv")
    dataset_what(final)

    # Parsing Labels and Deleting Columns:
    labels = final["name"].map(MOLECULE_MAP).to_numpy()
    features = final.drop(columns=["name", "temperature", "pressure"], axis=1)

    #  Gaussian Noising the Spectrum Columns:
    for column in list(features.columns):
        if "spectrum" in column:
            noise = np.random.normal(0, 0.1, len(features))
            features[column] = features[column] + noise

    print(f"\n***** Splitting Dataset into Train and Test *****\n")
    x_train, x_test, y_train, y_test = train_test_split(
        features.to_numpy(), labels, test_size=0.2, random_state=0
    )
    
    return x_train, x_test, y_train, y_test


def train():
    print(f"\n***** Initializing *****\n")
    x_train, x_test, y_train, y_test = load()
    y_test_original = y_test.copy()

    print(f"\n***** One-Hot Encoding Dataset Labels *****\n")
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = to_categorical(encoder.transform(y_train))
    y_test = to_categorical(encoder.transform(y_test))

    model = create_model()
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=2, batch_size=5)
    model.save("gas_predictor")
    y_pred_original = encoder.transform([yp.argmax() for yp in model.predict(x_test)])
    y_pred = to_categorical(y_pred_original)

    results = classification_report(y_test, y_pred, target_names=list(MOLECULES.keys()))
    print(f"\nClassification Report:\n{results}\n")
    plot_results(history)

def load_and_predict():
    print(f"\n***** Initializing *****\n")
    x_train, x_test, y_train, y_test = load()
    y_test_original = y_test.copy()

    print(f"\n***** One-Hot Encoding Dataset Labels *****\n")
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = to_categorical(encoder.transform(y_train))
    y_test = to_categorical(encoder.transform(y_test))

    model = load_model("gas_predictor")
    print("\n***** Predicting the Gas Concentrations *****\n")
    y_pred_original = encoder.transform([yp.argmax() for yp in model.predict(x_test)])
    y_pred = to_categorical(y_pred_original)

    print(y_test_original[0])
    print(y_pred[0])
    results = confusion_matrix(y_test_original, y_pred_original)
    tpr = (results[0][0] / (results[0][0] + results[0][1])) * 100
    fpr = (results[1][0] / (results[1][0] + results[1][1])) * 100
    print(f"Confusion Matrix:\n{results}\n")
    print(f"True Positive Rate (TPR): {tpr}")
    print(f"False Positive Rate (FPR): {fpr}")


if __name__ == "__main__":
    #train()
    load_and_predict()
