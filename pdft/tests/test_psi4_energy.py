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

@pytest.fixture()
def ethylene():

    c2h4 = psi4.geometry("""
    0 1
    H      1.1610      0.0661      1.0238
    C      0.6579     -0.0045      0.0639
    H      1.3352     -0.0830     -0.7815
    C     -0.6579      0.0045     -0.0639
    H     -1.3355      0.0830      0.7812
    H     -1.1608     -0.0661     -1.0239
    units angstrom
    symmetry c1
    """)

    return c2h4


def test_LDA_energy_restricted(he_dimer):

    psi4.core.clean()
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=he_dimer)
    helium = pdft.Molecule(he_dimer, "cc-pVDZ", "SVWN")
    helium.scf()
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_LDA_unrestricted_two_shells(he_dimer):

    psi4.core.clean()

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=he_dimer)
    helium = pdft.U_Molecule(he_dimer, "cc-pVDZ", "SVWN")
    helium.scf()
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_LDA_unrestricted_one_shell(hydrogen):

    psi4.core.clean()

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=hydrogen)
    helium = pdft.U_Molecule(hydrogen, "cc-pVDZ", "SVWN")
    helium.scf()
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_pbe_restricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "rks"})
    psi4_energy = psi4.energy("PBE/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.Molecule(ethylene, "aug-cc-pvdz", "pbe")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_pbe_urestricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("PBE/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.U_Molecule(ethylene, "aug-cc-pvdz", "pbe")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-3)


def test_tpss_restricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "rks"})
    psi4_energy = psi4.energy("TPSS/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.Molecule(ethylene, "aug-cc-pvdz", "tpss")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_tpss_urestricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("TPSS/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.U_Molecule(ethylene, "aug-cc-pvdz", "tpss")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-3)

def test_pbe0_restricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "rks"})
    psi4_energy = psi4.energy("pbe0/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.Molecule(ethylene, "aug-cc-pvdz", "pbe0")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_pbe0_urestricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("pbe0/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.U_Molecule(ethylene, "aug-cc-pvdz", "pbe0")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-3)

def test_b3lyp_restricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "rks"})
    psi4_energy = psi4.energy("b3lyp/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.Molecule(ethylene, "aug-cc-pvdz", "b3lyp")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_b3lyp_urestricted(ethylene):

    psi4.core.clean()
    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("b3lyp/aug-cc-pvdz", molecule=ethylene)

    ethy = pdft.U_Molecule(ethylene, "aug-cc-pvdz", "b3lyp")
    ethy.scf()
    pdft_energy = ethy.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)








