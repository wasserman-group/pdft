"""
Unit and regression test for the pdft package.
"""

# Import package, test suite, and other packages as needed
import pdft
import pytest
import sys

def test_pdft_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pdft" in sys.modules
