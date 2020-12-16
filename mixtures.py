# External:
import keras
import pandas
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Local:
from common import CSV, MOLECULES, dataset_what


def load():
    print(f"\n***** Loading Mixtures Dataset *****\n")
    final = pandas.read_csv(f"{CSV}/mixtures.csv")
    dataset_what(final)

    # Parsing Labels and Deleting Columns:
    labels = np.array(list(zip(
        final["x_CO2"].to_numpy(),
        final["x_CH4"].to_numpy(),
        final["x_NO"].to_numpy(),
    )))
    features = final.drop(
        columns=["x_CO2", "x_CH4", "x_NO"],
        axis=1,
    )

    print(f"\n***** Splitting Dataset into Train and Test *****\n")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=0
    )
    print(f"Training Datasets Length: {len(x_train)}")
    print(f"Testing Datasets Length: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def baseline():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_dim=4, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(3, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    model.summary()
    return model


def plot_differences(y_test, y_pred):
    calculations = [(np.abs((yt + 1) - (yp + 1)) / (yt + 1)) * 100 for yt, yp in zip(y_test, y_pred)]

    for i, molecule in enumerate(list(MOLECULES.keys())):
        actuals = [yt[i] for yt in y_test]
        predictions = [yp[i] for yp in y_pred]
        differences = [df[i] for df in calculations]

        mean = np.mean(differences)
        std = np.std(differences)
        print(f"\n{molecule} Difference Mean: {mean}")
        print(f"{molecule} Difference StdDev: {std}\n")

        plt.figure()
        plt.scatter(actuals, predictions, label="True", color="m")
        plt.plot(actuals, actuals, label="Predicted", color="k")
        plt.title(f"{molecule} Predictions")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.legend()
        plt.savefig(f"plots/{molecule}_predictions")
        plt.clf


def load_and_predict():
    print(f"\n***** Initializing *****\n")
    x_train, x_test, _, y_test = load()
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = keras.models.load_model("mixture_predictor")
    print("\n***** Predicting the Gas Concentrations *****\n")
    y_pred = model.predict(x_test_scaled)
    plot_differences(y_test, y_pred)


def train():
    print(f"\n***** Initializing *****\n")
    x_train, _, y_train, _ = load()
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    print("\n***** Creating and Fitting the Model *****\n")
    model = baseline()
    model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=1, batch_size=1)
    model.save("mixture_predictor")


if __name__ == "__main__":
    train()
    load_and_predict()
