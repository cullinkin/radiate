# Standard:
import os
import json
import shutil
import random
from itertools import combinations, product

# 3rdparty:
import pandas
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from hapi import (
    db_begin,
    fetch,
    absorptionCoefficient_Voigt,
    molecularMass,
    absorptionSpectrum,
    transmittanceSpectrum,
    radianceSpectrum,
)

# Local:
from common import (
    HAPI,
    CSV,
    MOLECULES,
    COLUMNS,
    MIXTURES,
    NU_MIN,
    NU_MAX,
    NU_MIN_MIX,
    NU_MAX_MIX,
    dataset_what,
)
random.seed(42)


# Restart the HAPI Database from Scratch:
def restart():
    try:
        shutil.rmtree(HAPI)
    except OSError:
        pass

    os.makedirs(HAPI)
    db_begin(HAPI)


# Populate the HAPI Database with Data:
def create(molecules, nu_min, nu_max):
    for molecule, spec in molecules.items():
        print(f"\n***** Loading Molecule: {molecule} *****\n")
        print(f"*** Minimum Wave Number (nu): {nu_min} ***")
        print(f"*** Maximum Wave Number (nu): {nu_max} ***")
        fetch(molecule, spec[0], spec[1], nu_min, nu_max)


# Transform the HAPI Database Data into a Usable CSV Format:
def transform_data(molecules, nu_min, nu_max):
    for molecule, spec in molecules.items():
        print(f"\n***** Converting Molecule {molecule} Data to CSV *****\n")
        dataset = pandas.DataFrame(columns=COLUMNS)
        environment = {"T": 296, "p": 1, "l": 1}
        wave_number, absorption_coefficient = absorptionCoefficient_Voigt(
            Components=(spec,),
            SourceTables=molecule,
            OmegaRange=(nu_min, nu_max),
            OmegaStep=0.1,
            HITRAN_units=False,
            Environment=environment,
        )
        _, absorption_spectrum = absorptionSpectrum(
            wave_number, absorption_coefficient, Environment=environment
        )
        _, transmittance_spectrum = transmittanceSpectrum(
            wave_number, absorption_coefficient, Environment=environment
        )
        dataset["wave_number"] = wave_number
        dataset["mass"] = molecularMass(spec[0], spec[1])
        dataset["name"] = molecule
        dataset["absorption_coefficient"] = absorption_coefficient
        dataset["absorption_spectrum"] = absorption_spectrum
        dataset["transmittance_spectrum"] = transmittance_spectrum
        dataset.to_csv(f"{CSV}/{molecule}.csv")


# Combine Molecule Datasets:
def create_dataset(molecules):
    datasets = []
    for molecule, _ in molecules.items():
        print(f"\n***** Loading Molecule {molecule} to Dataframe *****\n")
        dataset = pandas.read_csv(f"{CSV}/{molecule}.csv", index_col=0)
        datasets.append(dataset)

    # Joining Datasets, Random Shuffling, and Dropping Columns:
    final = pandas.concat(datasets, ignore_index=True, sort=False)
    final.to_csv(f"{CSV}/all_molecules.csv", index=False)
    dataset_what(final)


# Calculate Mixture Property Based on Gas Concentrations:
def calculate_property(fractions, datasets, column):
    portions = {}
    for name, fraction in fractions.items():
        portions[name] = fraction * datasets[name][column].to_numpy()

    results = [sum([x, y, z]) for x, y, z in zip(portions["CO2"], portions["CH4"], portions["NO"])]
    return results


# Simylate All Possible Mixtures of the 3 Gases:
def simulate_mixtures(molecules):
    concentrations = list(range(1, 101))
    unique = [pair for pair in combinations(concentrations, 3) if sum(pair) == 100]
    expanded = [list(product(*[mix]*3)) for mix in unique]
    all_mixtures = [np.array(mix) / 100 for mixture in expanded for mix in mixture if sum(mix) == 100]
    print(f"Total Number of Mixtures: {len(all_mixtures)}")
    
    datasets = {}
    for molecule, _ in molecules.items():
        print(f"\n***** Loading Molecule {molecule} to Dataframe *****\n")
        datasets[molecule] = pandas.read_csv(f"{CSV}/{molecule}.csv", index_col=0)
        dataset_what(datasets[molecule])
    wave_numbers = datasets["CO2"]["wave_number"].values

    mixtures = []
    names = list(molecules.keys())
    chosen_mixtures = random.sample(all_mixtures, 100)
    print(f"Chosen Mixtures: {chosen_mixtures}")
    for i in trange(len(chosen_mixtures), desc="Simulating Mixtures"):
        mixture = pandas.DataFrame(columns=MIXTURES)

        fractions = dict(zip(names, chosen_mixtures[i]))
        mixture["absorption_coefficient"] = calculate_property(fractions, datasets, "absorption_coefficient")
        mixture["absorption_spectrum"] = calculate_property(fractions, datasets, "absorption_spectrum")
        mixture["transmittance_spectrum"] = calculate_property(fractions, datasets, "transmittance_spectrum")
        mixture["wave_number"] = wave_numbers
        mixture["x_CO2"] = fractions["CO2"]
        mixture["x_CH4"] = fractions["CH4"]
        mixture["x_NO"] = fractions["NO"]
        mixtures.append(mixture)

    final = pandas.concat(mixtures, ignore_index=True, sort=False)
    final.to_csv(f"{CSV}/mixtures.csv", index=False)
    dataset_what(final)


if __name__ == "__main__":
    #data = {"components": (NU_MIN, NU_MAX), "mixtures": (NU_MIN_MIX, NU_MAX_MIX)}
    data = {"mixtures": (NU_MIN_MIX, NU_MAX_MIX)}
    for item, waves in data.items():
        restart()
        create(MOLECULES, waves[0], waves[1])
        transform_data(MOLECULES, waves[0], waves[1])
        if item == "components":
            create_dataset(MOLECULES)
        elif item == "mixtures":
            simulate_mixtures(MOLECULES)

