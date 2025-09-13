"""
Pytest configuration and fixtures for Plitka Pro Project TDD tests
Version: v4.6.01
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_prompt_data():
    """Sample prompt data for testing"""
    return {
        "simple": "100% red",
        "multi_color": "60% red, 40% blue",
        "complex": "50% red, 30% blue, 20% green",
        "coded": "70% RED, 30% BLUE"
    }

@pytest.fixture
def sample_colormap_data():
    """Sample colormap data for testing"""
    return {
        "red": (255, 0, 0, 255),
        "blue": (0, 0, 255, 255),
        "green": (0, 255, 0, 255),
        "white": (255, 255, 255, 255)
    }

@pytest.fixture
def mock_predictor():
    """Mock predictor for testing without full model initialization"""
    # This will be implemented as we refactor the Predictor class
    pass


