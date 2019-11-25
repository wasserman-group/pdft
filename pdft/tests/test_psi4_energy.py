""" 

Tests calculations that can be done with Psi4

"""

import psi4
import numpy as np
import pdft
import pytest

@pytest.fixture()
def he_dimer():

    he2 = psi4.geometry("""
    0 1

    He 0.0 0.0 0.0
    He 0.0 0.0 6.0

    units bohr
    symmetry c1
    """) 
    return he2

@pytest.fixture()
def hydrogen():

    h = psi4.geometry("""
    0 2

    H 0.0 0.0 0.0

    units bohr
    symmetry c1
    """)

    return h


def test_LDA_energy_restricted(he_dimer):

    psi4.core.clean()
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=he_dimer)
    helium = pdft.Molecule(he_dimer, "cc-pVDZ", "SVWN")
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_LDA_unrestricted_two_shells(he_dimer):

    psi4.core.clean()

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=he_dimer)
    helium = pdft.U_Molecule(he_dimer, "cc-pVDZ", "SVWN")
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_LDA_unrestricted_one_shell(hydrogen):

    psi4.core.clean()

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=hydrogen)
    helium = pdft.U_Molecule(hydrogen, "cc-pVDZ", "SVWN")
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)





