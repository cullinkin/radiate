import os
import numpy
import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hapi import (
    db_begin,
    getStickXY,
    PROFILE_VOIGT,
    PROFILE_LORENTZ,
    absorptionCoefficient_Voigt,
    absorptionSpectrum,
    transmittanceSpectrum,
    radianceSpectrum,
    convolveSpectrum,
    SLIT_MICHELSON,
    abundance,
    molecularMass,
    moleculeName,
    isotopologueName,
)
from common import HEAT, DATA, HAPI, MOLECULES


def sticks(molecule, path):
    # Plot Stick Spectrum:
    print(f"\n***** Plotting Stick Spectrums *****\n")
    x, y = getStickXY(molecule)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.title("Stick Spectrum")
    plt.xlabel("Wave Number (nu)")

    # Zoom Stick Spectrum:
    plt.subplot(2, 1, 2)
    plt.plot(x, y, ".")
    plt.xlim([4020, 4035])
    plt.title("Zoomed Stick Spectrum")
    plt.xlabel("Wave Number (nu)")
    plt.tight_layout()
    plt.savefig(f"{path}/stick_spectrums.png")
    plt.clf()


def lineshapes(path):
    print(f"\n***** Plotting Lineshapes *****\n")
    wn = numpy.arange(3900, 4100, 1)  # get wavenumber range of interest
    voi = PROFILE_VOIGT(4000, 0.1, 0.3, wn)[0]  # calc Voigt
    lor = PROFILE_LORENTZ(4000, 0.3, wn)  # calc Lorentz
    diff = voi - lor  # calc difference
    plt.figure()
    plt.subplot(2, 1, 1)  # upper panel
    plt.plot(wn, voi, "red", wn, lor, "blue")  # plot both profiles
    plt.legend(["Voigt", "Lorentz"])  # show legend
    plt.title("Voigt and Lorentz Profiles")  # show title
    plt.subplot(2, 1, 2)  # lower panel
    plt.plot(wn, diff)  # plot difference
    plt.title("Voigt-Lorentz Residual")  # show title
    plt.tight_layout()
    plt.savefig(f"{path}/lineshapes.png")
    plt.clf()


def absorptions(molecule, ids, path):
    print(f"\n***** Plotting Absorption Coefficients *****\n")
    plt.figure()
    colors = ["r", "g", "b", "y"]
    for i, temperature in enumerate([296]):
        nu, coef = absorptionCoefficient_Voigt(
            (ids,),
            molecule,
            OmegaStep=0.01,
            HITRAN_units=False,
            GammaL="gamma_self",
            Environment={"p": 1, "T": temperature},
        )
        plt.subplot(2, 2, i + 1)
        plt.plot(nu, coef, colors[i])
        plt.title(f"{molecule} Absorption (p=1atm, T={temperature}K)")
        plt.xlabel("Wavenumber")
        plt.ylabel("Absorption Coefficient")

    plt.tight_layout()
    plt.savefig(f"{path}/absorption_coefficients.png")
    plt.clf()


def spectrums(molecule, ids, path):
    print(f"\n***** Plotting Spectrums *****\n")
    environment = {"p": 1, "T": 296, "l": 1000}
    nu1, coef = absorptionCoefficient_Voigt(
        (ids,),
        molecule,
        OmegaStep=0.01,
        HITRAN_units=False,
        GammaL="gamma_self",
        Environment=environment,
    )
    nu2, absorp = absorptionSpectrum(nu1, coef, Environment=environment)
    nu3, transm = transmittanceSpectrum(nu1, coef, Environment=environment)
    nu4, radian = radianceSpectrum(nu1, coef, Environment=environment)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(nu1, coef, "r")
    plt.title(f"{molecule} Absorption Coefficient")
    plt.subplot(2, 2, 2)
    plt.plot(nu2, absorp, "g")
    plt.title(f"{molecule} Absorption Spectrum")
    plt.subplot(2, 2, 3)
    plt.plot(nu3, transm, "b")
    plt.title(f"{molecule} Transmittance Spectrum")
    plt.subplot(2, 2, 4)
    plt.plot(nu4, radian, "y")
    plt.title(f"{molecule} Radiance Spectrum")
    plt.tight_layout()
    plt.savefig(f"{path}/spectrums.png")
    plt.clf()


def hires(molecule, ids, path):
    print(f"\n***** Plotting HiRes *****\n")
    environment = {"p": 1, "T": 296, "l": 1000}
    nu1, coef = absorptionCoefficient_Voigt(
        (ids,),
        molecule,
        OmegaStep=0.01,
        HITRAN_units=False,
        GammaL="gamma_self",
        Environment=environment,
    )
    nu2, transm = transmittanceSpectrum(nu1, coef, Environment=environment)
    nu3, conv, _, _, _ = convolveSpectrum(
        nu2,
        transm,
        SlitFunction=SLIT_MICHELSON,
        Resolution=1.0,
        AF_wing=20.0,
    )
    plt.figure()
    plt.plot(nu2, transm, "red", nu3, conv, "blue")
    plt.legend(["HI-RES", "Michelson"])
    plt.tight_layout()
    plt.savefig(f"{path}/hires.png")
    plt.clf()


def properties(molecule, ids):
    print(f"\n***** Getting Properties *****\n")
    ab = abundance(ids[0], ids[1])
    mass = molecularMass(ids[0], ids[1])
    molname = moleculeName(ids[0])
    isoname = isotopologueName(ids[0], ids[1])
    print(f"{molecule} Abundance: {ab}")
    print(f"{molecule} Molecular Mass: {mass}")
    print(f"{molecule} Molecule Name: {molname}")
    print(f"{molecule} Iso Name: {isoname}")


def plot_results(results):
    results = dict(results.history)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(results["loss"], label="train")
    plt.plot(results["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(results["accuracy"], label="train")
    plt.plot(results["val_accuracy"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/results.png")
    plt.clf()


def plot_mixture(mix, datasets, mixture):
    mix = mix.tolist()
    plt.subplot(2, 2, 1)
    plt.plot(wave_numbers, datasets["CO2"]["absorption_coefficient"], color="m")
    plt.title(f"CO2 Absorption")
    plt.xlabel("Wavenumber")
    plt.ylabel("Absorption Coefficient")
    plt.subplot(2, 2, 2)
    plt.plot(wave_numbers, datasets["CH4"]["absorption_coefficient"], color="b")
    plt.title(f"CH4 Absorption")
    plt.xlabel("Wavenumber")
    plt.ylabel("Absorption Coefficient")
    plt.subplot(2, 2, 3)
    plt.plot(wave_numbers, datasets["NO"]["absorption_coefficient"], color="g")
    plt.title(f"NO Absorption")
    plt.xlabel("Wavenumber")
    plt.ylabel("Absorption Coefficient")
    plt.subplot(2, 2, 4)
    plt.plot(wave_numbers, mixture["absorption_coefficient"], color="k")
    plt.title(f"{mix[0]}%CO2-{mix[1]}%CH4-{mix[2]}%NO, Absorption")
    plt.xlabel("Wavenumber")
    plt.ylabel("Absorption Coefficient")
    plt.tight_layout()
    plt.savefig(f"plots/mixture_absorption.png")
    plt.clf()


if __name__ == "__main__":
    db_begin(HAPI)
    for molecule, spec in MOLECULES.items():
        properties(molecule, spec)
        folder = f"{HEAT}/plots/{molecule}"
        try:
            os.makedirs(folder)
        except OSError:
            pass
        sticks(molecule, folder)
        lineshapes(folder)
        absorptions(molecule, spec, folder)
        spectrums(molecule, spec, folder)
        hires(molecule, spec, folder)
