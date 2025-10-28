from src.data_preprocessing import DataPreprocessor
from src.data_preprocessing.load_config import load_config
from pathlib import Path
import pytest


@pytest.fixture
def preprocessor(config_path):
    return DataPreprocessor(config=load_config(config_path))
