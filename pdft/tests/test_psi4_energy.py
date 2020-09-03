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
    psi4.core.be_quiet()

    psi4.set_options({
                  "DFT_SPHERICAL_POINTS" : 6,
                  "DFT_RADIAL_POINTS":     12,})

    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=he_dimer)
    helium = pdft.RMolecule(he_dimer, "cc-pVDZ", "SVWN")
    helium.scf()
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_LDA_unrestricted_two_shells(he_dimer):

    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({
                "DFT_SPHERICAL_POINTS" : 6,
                "DFT_RADIAL_POINTS":     12,})

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=he_dimer)
    helium = pdft.UMolecule(he_dimer, "cc-pVDZ", "SVWN")
    helium.scf()
    pdft_energy = helium.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)

def test_LDA_unrestricted_one_shell(hydrogen):

    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({
            "DFT_SPHERICAL_POINTS" : 6,
            "DFT_RADIAL_POINTS":     12,})

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("SVWN/cc-pVDZ", molecule=hydrogen)
    h = pdft.UMolecule(hydrogen, "cc-pVDZ", "SVWN")
    h.scf()
    pdft_energy = h.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-4)


def test_GGA_unrestricted_one_shell(hydrogen):

    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({
            "DFT_SPHERICAL_POINTS" : 6,
            "DFT_RADIAL_POINTS":     12,})

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("pbe/cc-pVDZ", molecule=hydrogen)
    h = pdft.UMolecule(hydrogen, "cc-pvdz", "pbe")
    h.scf()

    pdft_energy = h.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-2)    

def test_GGA_unrestricted_two_shelss(he_dimer):

    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({
            "DFT_SPHERICAL_POINTS" : 6,
            "DFT_RADIAL_POINTS":     12,})

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("pbe/cc-pVDZ", molecule=he_dimer)
    h = pdft.UMolecule(he_dimer, "cc-pvdz", "pbe")
    h.scf()

    pdft_energy = h.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-2)  
  
def test_GGA_unrestricted_two_shells(he_dimer):

    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({
            "DFT_SPHERICAL_POINTS" : 6,
            "DFT_RADIAL_POINTS":     12,})

    psi4.set_options({"reference" : "rks"})
    psi4_energy = psi4.energy("pbe/cc-pVDZ", molecule=he_dimer)
    h = pdft.RMolecule(he_dimer, "cc-pvdz", "pbe")
    h.scf()

    pdft_energy = h.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-2)  

def test_META_unrestricted_two_shells(he_dimer):

    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({
            "DFT_SPHERICAL_POINTS" : 6,
            "DFT_RADIAL_POINTS":     12,})

    psi4.set_options({"reference" : "rks"})
    psi4_energy = psi4.energy("M05/cc-pVDZ", molecule=he_dimer)
    h = pdft.RMolecule(he_dimer, "cc-pvdz", "M05")
    h.scf()

    pdft_energy = h.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-2)  


def test_META_unrestricted_one_shell(he_dimer):

    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({
            "DFT_SPHERICAL_POINTS" : 6,
            "DFT_RADIAL_POINTS":    12 ,})

    psi4.set_options({"reference" : "uks"})
    psi4_energy = psi4.energy("tpss/cc-pvtz", molecule=he_dimer)
    dimer = pdft.UMolecule(he_dimer, "cc-pvtz", "tpss")
    dimer.scf()

    pdft_energy = h.energy

    assert np.isclose(psi4_energy, pdft_energy, rtol=1e-2)    