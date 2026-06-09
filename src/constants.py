from pathlib import Path

# File Paths
DIRECTORY_PATH = Path(__file__).parent.parent
CONFIG_PATH = DIRECTORY_PATH / "configs"
DATA_PATH_DICT = {
    0: DIRECTORY_PATH / "boreal_data" / "full" / "COPERNICUS" / "S2_SERIES_SEAS_FORMAT_JPEG",
    1: DIRECTORY_PATH / "boreal_data" / "full" / "COPERNICUS" / "S2_SERIES_SEAS_FORMAT_JPEG_NEG",
}
DATA_PATH_DICT_UPDATE = {
    0: DIRECTORY_PATH / "boreal_data" / "full" / "COPERNICUS" / "UPDATE" / "S2_SERIES_SEAS_FORMAT",
    1: DIRECTORY_PATH / "boreal_data" / "full" / "COPERNICUS" / "UPDATE" / "S2_SERIES_SEAS_FORMAT_NEG_HARD",
}

# Data Constants
CANADA_REF_COLS = ["tile_id", "valid_names", "flag_tile_select", "region", "file_id", "year", "fwinx_mean"]
BANDS_ALL = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]  # Keep in order between all and subsplits
BANDS_10 = ["B4", "B3", "B2", "B8"]
BANDS_20 = ["B5", "B6", "B7", "B8A", "B11", "B12"]
BANDS_60 = ["B1", "B9", "B10"]
LABEL_IMG_RES = 264
LABEL_IMG_RES_LOW = 260
LABEL_RES = 26
ORDER_INDEX = [10, 2, 1, 0, 4, 5, 6, 3, 7, 11, 12, 8, 9]
MSCLIP_ORDER_10 = [2, 1, 0, 4, 5, 6, 3, 7, 8, 9]
S2_UINT_TO_REFLECTANCE = 10000.0 / 255.0

# Environment Canada Constants
TAB_SOURCE_COLS = {
    "era5": [
        "total_precipitation_sum_mean",
        "skin_temperature_mean",
        "temperature_2m_mean",
        "volumetric_soil_water_layer_1_mean",
        "wind_speed_10m_mean",
        "relative_humidity_2m_mean",
        "vapor_pressure_2m_mean",
    ],
    "modis": ["LST_Day_1km_mean", "NDVI", "EVI", "Fpar_500m", "Lai_500m"],  # "LST_Day_1km_max","LST_Day_1km_min"
    "cds": ["drtcode_mean", "fwinx_mean"],
}
MID_SOURCE = ["modis11", "modis13_15"]
LOW_SOURCE = ["era5", "cds"]
ENV_SOURCE_COLS = {
    "era5": [
        "total_precipitation_sum_mean",
        "skin_temperature_mean",
        "temperature_2m_mean",
        "volumetric_soil_water_layer_1_mean",
        "wind_speed_10m_mean",
        "relative_humidity_2m_mean",
        "vapor_pressure_2m_mean",
    ],
    "modis11": ["LST_Day_1km_mean"],
    "modis13_15": ["NDVI", "EVI", "Fpar_500m", "Lai_500m"],
    "cds": ["drtcode_mean", "fwinx_mean"],
}

# Validation Splits
TRAIN_FILTER = {"year": [2015, 2016, 2017, 2018, 2019, 2020, 2021], "region": []}
VAL_FILTER = {"year": [2022], "region": []}
TEST_FILTER = {"year": [2023], "region": []}
PREPROCESSING_SUFFIX = "one"

# Land Cover Validation
LAND_COVER_FOLDERS = ["./boreal_data/LC", "./boreal_data/NEG_LC"]

LAND_COVER_DICT_IDS = {
    0: "Unknown",
    1: "Temperate or sub-polar needleleaf forest",
    2: "Sub-polar taiga needleleaf forest",
    3: "Tropical or subtropical broadleaf evergreen forest",
    4: "Tropical or subtropical broadleaf deciduous forest",
    5: "Temperate or sub-polar broadleaf deciduous forest",
    6: "Mixed forest",
    7: "Tropical or sub-tropical shrubland",
    8: "Temperate or sub-polar shrubland",
    9: "Tropical or sub-tropical grassland",
    10: "Temperate or sub-polar grassland",
    11: "Sub-polar or polar shrubland-lichen-moss",
    12: "Sub-polar or polar grassland-lichen-moss",
    13: "Sub-polar or polar barren-lichen-moss",
    14: "Wetland",
    15: "Cropland",
    16: "Barren land",
    17: "Urban",
    18: "Water",
    19: "Snow and ice",
}

HF_IMG_COLUMNS = ["doy", "10x", "20x", "60x", "loc", "labels", "region", "tile_id", "file_id"]
HF_MIX_TAB_COLUMNS = [
    "doy",
    "10x",
    "20x",
    "60x",
    "loc",
    "labels",
    "region",
    "tile_id",
    "file_id",
    "tab_cds",
    "tab_modis",
    "tab_era5",
    "env_doy",
]
HF_MIX_SPA_COLUMNS = [
    "doy",
    "10x",
    "20x",
    "60x",
    "loc",
    "labels",
    "region",
    "tile_id",
    "file_id",
    "env_cds",
    "env_cds_loc",
    "env_modis11",
    "env_modis11_loc",
    "env_modis13_15",
    "env_modis13_15_loc",
    "env_era5",
    "env_era5_loc",
    "env_doy",
]
HF_ENV_TAB_COLUMNS = ["labels", "region", "tile_id", "file_id", "tab_cds", "tab_modis", "tab_era5", "env_doy"]
HF_ENV_SPA_COLUMNS = [
    "labels",
    "region",
    "tile_id",
    "file_id",
    "env_cds",
    "env_cds_loc",
    "env_modis11",
    "env_modis11_loc",
    "env_modis13_15",
    "env_modis13_15_loc",
    "env_era5",
    "env_era5_loc",
    "env_doy",
]
