# External:
import pandas

# Set Up Relevant Paths:
CODE = "/Users/ck/Documents/Code"
HEAT = f"{CODE}/heat"
DATA = f"{CODE}/data/heat"
HAPI = f"{DATA}/hapi"
CSV = f"{DATA}/csv"

# Dataset Characteristics:
MOLECULES = {
    "CO2": (2, 1),
    "CH4": (6, 1),
    "NO": (8, 1),
}
MOLECULE_MAP = {
    "CO2": 0,
    "CH4": 1,
    "NO": 2,
}
NU_MIN = 2000
NU_MAX = 5000
NU_MIN_MIX = 3900
NU_MAX_MIX = 4100
COLUMNS = [
    "name",
    "mass",
    "wave_number",
    "absorption_coefficient",
    "absorption_spectrum",
    "transmittance_spectrum",
]
MIXTURES = [
    "x_CO2",
    "x_CH4",
    "x_NO",
    "wave_number",
    "absorption_coefficient",
    "absorption_spectrum",
    "transmittance_spectrum",
]

# Print Dataset Characteristics:
def dataset_what(data):
    print(f"Dataset Length: {len(data)}\n")
    if isinstance(data, pandas.DataFrame):
        print(f"Dataset Columns:\n{list(data.columns)}\n")
        print(f"Dataset Describe:\n{data.describe()}\n")
        print(f"Dataset Head:\n{data.head()}\n")
        print(f"Dataset Tail:\n{data.tail()}\n")
    print(f"Dataset Valid: {(data.isnull().sum().sum() == 0)}\n")
