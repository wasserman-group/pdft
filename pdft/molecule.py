"""
Molecule.py

Defitions for molecule class.
Molecule object will be used to describe the fragment systems.

This will be replaced by psi4.wfn hopefully in the future when
1) Psi4 has a way to add external potential.
2) The "bug" of sharing mints for different psi4 systems is fixed.
The reason I want this to be replaced is that I don't want to reinvent the wheel.

"""
import numpy as np
# from opt_einsum import contract

import psi4

from pdft.xc import functional_factory
from pdft.xc import xc
from pdft.xc import u_xc

class Molecule():
    # From scf
    Ca = None
    Cb = None
    Cocca = None
    Coccb = None
    Da = None
    Db = None
    Da_0 = None
    Db_0 = None
    Fa = None
    Fb = None
    energy = None
    frag_energy = None
    energetics = None
    eigs_a = None
    eigs_b = None

    # Potentials
    vha_a = None
    vha_b = None
    vxc_a = None
    vxc_b = None
    vp_a = None
    vp_b = None

    # Grid
    grid = None
    omegas = None
    phi = None
    # self.Da_r        = None
    # self.Db_r        = None
    ingredients = None
    orbitals = None

    def __init__(self, geometry, basis, method,
                 mints=None, jk=None, vpot=None,
                 ):

        # basics
        self.geometry = geometry
        self.basis_label = basis
        self.method = method
        self.Enuc = geometry.nuclear_repulsion_energy()

        #Psi4 objects
        self.wfn         = psi4.core.Wavefunction.build(self.geometry, self.basis_label)
        self.mints       = mints if mints is not None else psi4.core.MintsHelper(self.wfn.basisset())

        #From psi4 objects
        self.basis       = self.wfn.basisset()
        self.nbf         = self.basis.nbf()
        self.nalpha      = self.wfn.nalpha()
        self.nbeta       = self.wfn.nbeta()
        self.ndocc       = self.nalpha + self.nbeta

        #From methods
        self.jk          = jk if jk is not None else self.form_JK()
        self.S           = self.mints.ao_overlap()
        self.A           = self.form_A()
        self.H           = self.form_H()
        #Options

    def form_JK(self, K=True, memory=1.25e8):
        """
        Constructs a psi4 JK object from input basis
        """
        jk = psi4.core.JK.build(self.basis)
        jk.set_memory(int(memory)) #1GB
        jk.set_do_K(K)
        jk.initialize()
        jk.print_header()
        return jk

    def form_A(self):
        """
        Constructs matrix A = S^(1/2) required to orthonormalize the Fock Matrix
        """
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        return A

    def form_H(self):
        """
        Forms core matrix
        H =  T + V
        """
        self.V = self.mints.ao_potential()
        self.T = self.mints.ao_kinetic()
        H = self.T.clone()
        H.add(self.V)
        return H

    def build_orbitals(self, diag, ndocc):
        """
        Diagonalizes matrix

        Parameters
        ----------
        diag: psi4.core.Matrix
            Fock matrix

        Returns
        -------
        C: psi4.core.Matrix
            Molecular orbitals coefficient matrix

        Cocc: psi4.core.Matrix
            Occupied molecular orbitals coefficient matrix

        D: psi4.core.Matrix
            One-particle density matrix

        eigs: psi4.core.Vector
            Eigenvectors of Fock matrix
        """
        Fp = psi4.core.triplet(self.A, diag, self.A, True, False, True)
        nbf = self.A.shape[0]
        Cp = psi4.core.Matrix(nbf, nbf)
        eigvecs = psi4.core.Vector(nbf)
        Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)
        C = psi4.core.doublet(self.A, Cp, False, False)

        Cocc = psi4.core.Matrix(nbf, ndocc)
        Cocc.np[:] = C.np[:, :ndocc]

        D = psi4.core.doublet(Cocc, Cocc, False, True)
        return C, Cocc, D, eigvecs

    def scf(self, maxiter=100, hamiltonian=["kinetic", "external", "hartree", "xc"],
            vp_Fock_updown=None, xxxtra_Fock_updown=None,
            get_matrices=False,
            diis=True, energetic=False):
        """
        Performs scf cycle
        Parameters
        ----------
        vp: psi4.core.Matrix
            Vp_matrix to be added to KS matrix
        """
        # At the beginning, check if the given vp is the same as vp of last SCF.
        if vp_Fock_updown is not None and \
                (self.vp_a is not None and self.vp_b is not None):
            if vp_Fock_updown is None:
                raise NameError("You want to add an xxxtra component to Hamiltonian but didn't specify it")
            # If YES, no need to run SCF one more time.
            if np.allclose(vp_Fock_updown[0].np, self.vp_a.np) and np.allclose(vp_Fock_updown[1].np, self.vp_b.np):
                return
            # Otherwise, update vp_a and vp_b
            else:
                self.vp_a = vp_Fock_updown[0]
                self.vp_b = vp_Fock_updown[1]

        # Restricted/Unrestricted
        # Initial Guess.
        if self.Da is None and self.Db is None:
            Ca, Cocca, Da, eigs_a = self.build_orbitals(self.H, self.nalpha)
            # Set to X_b = X_a if restricted == True without depending on child class.
            Cb, Coccb, Db, eigs_b = self.build_orbitals(self.H, self.nbeta)

        if self.Da is not None and self.Da is not None:
            Ca, Cocca, Da, eigs_a = self.Ca, self.Cocca, self.Da, self.eigs_a
            Cb, Coccb, Db, eigs_b = self.Cb, self.Coccb, self.Db, self.eigs_b



        if diis is True:
            diisa_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")
            diisb_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")

        if self.energy is not None:
            Eold = self.energy
        else:
            Eold = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")

        for SCF_ITER in range(maxiter + 1):
            self.jk.C_left_add(Cocca)
            self.jk.C_left_add(Coccb)
            self.jk.compute()
            self.jk.C_clear()

            if get_matrices:
                self.vha_a = psi4.core.Matrix(self.nbf, self.nbf)
                self.vha_b = psi4.core.Matrix(self.nbf, self.nbf)
                self.vxc_a = psi4.core.Matrix(self.nbf, self.nbf)
                self.vxc_b = psi4.core.Matrix(self.nbf, self.nbf)
                self.vee_a = psi4.core.Matrix(self.nbf, self.nbf)
                self.vee_b = psi4.core.Matrix(self.nbf, self.nbf)
                self.vks_a = psi4.core.Matrix(self.nbf, self.nbf)
                self.vks_b = psi4.core.Matrix(self.nbf, self.nbf)

            Fa = psi4.core.Matrix(self.nbf, self.nbf)
            Fb = psi4.core.Matrix(self.nbf, self.nbf)

            if "kinetic" in hamiltonian:
                Fa.axpy(1.0, self.T.clone())
                Fb.axpy(1.0, self.T.clone())

            if "external" in hamiltonian:
                Fa.axpy(1.0, self.V.clone())
                Fb.axpy(1.0, self.V.clone())

            if "hartree" in hamiltonian:
                Fa.axpy(1.0, self.jk.J()[0])
                Fa.axpy(1.0, self.jk.J()[1])
                Fb.axpy(1.0, self.jk.J()[0])
                Fb.axpy(1.0, self.jk.J()[1])

                if get_matrices is True:
                    # Hartree
                    self.vha_a.axpy(1.0, self.jk.J()[0])
                    self.vha_a.axpy(1.0, self.jk.J()[1])
                    self.vha_b.axpy(1.0, self.jk.J()[0])
                    self.vha_b.axpy(1.0, self.jk.J()[1])
                    # Kohn_Sham Potential
                    self.vks_a.axpy(1.0, self.jk.J()[0])
                    self.vks_a.axpy(1.0, self.jk.J()[1])
                    self.vks_b.axpy(1.0, self.jk.J()[0])
                    self.vks_b.axpy(1.0, self.jk.J()[1])

            if "xc" in hamiltonian:
                if self.functional.is_x_hybrid() is True:
                    alpha = self.functional.x_alpha()
                    Fa.axpy(-alpha, self.jk.K()[0])
                    Fb.axpy(-alpha, self.jk.K()[1])

                    if get_matrices is True:
                        self.vee_a.axpy(-alpha, self.jk.K()[0])
                        self.vee_b.axpy(-alpha, self.jk.K()[1])
                        self.vxc_a.axpy(-alpha, self.jk.K()[0])
                        self.vxc_b.axpy(-alpha, self.jk.K()[1])
                else:
                    alpha = 0.0

                # Correlation Hybrid?
                if self.functional.is_c_hybrid() is True:
                    raise NameError("Correlation hybrids are not avaliable")

                # Exchange Correlation
                ks_e, Vxc_a, Vxc_b, self.ingredients, self.orbitals, self.grid, self.potential = self.get_xc(Da, Db,
                                                                                                             Ca.np,
                                                                                                             Cb.np)
                # XC already scaled by alpha
                Vxc_a = psi4.core.Matrix.from_array(Vxc_a)
                Vxc_b = psi4.core.Matrix.from_array(Vxc_b)
                Fa.axpy(1.0, Vxc_a)
                Fb.axpy(1.0, Vxc_b)
                
                if get_matrices:
                    self.vxc_a.axpy(1.0, Vxc_a)
                    self.vxc_b.axpy(1.0, Vxc_b)

            if vp_Fock_updown is not None:
                Fa.axpy(1.0, vp_Fock_updown[0])
                Fb.axpy(1.0, vp_Fock_updown[1])
                    
            if "xxxtra" in hamiltonian:
                if xxxtra_Fock_updown is None:
                    raise NameError("You want to add an xxxtra component to Hamiltonian but didn't specify it")
                Fa.axpy(1.0, xxxtra_Fock_updown[0])
                Fb.axpy(1.0, xxxtra_Fock_updown[1])

            # DIIS
            if diis:
                diisa_e = psi4.core.triplet(Fa, Da, self.S, False, False, False)
                diisa_e.subtract(psi4.core.triplet(self.S, Da, Fa, False, False, False))
                diisa_e = psi4.core.triplet(self.A, diisa_e, self.A, False, False, False)
                diisa_obj.add(Fa, diisa_e)

                diisb_e = psi4.core.triplet(Fb, Db, self.S, False, False, False)
                diisb_e.subtract(psi4.core.triplet(self.S, Db, Fb, False, False, False))
                diisb_e = psi4.core.triplet(self.A, diisb_e, self.A, False, False, False)
                diisb_obj.add(Fb, diisb_e)

                # dRMSa = diisa_e.rms()
                # dRMSb = diisb_e.rms()

                dRMS = 0.5 * (np.mean(diisa_e.np ** 2) ** 0.5 + np.mean(diisb_e.np ** 2) ** 0.5)

                Fa = diisa_obj.extrapolate()
                Fb = diisb_obj.extrapolate()

            # Define Energetics
            if "kinetic" in hamiltonian:
                energy_kinetic = 1.0 * self.T.vector_dot(Da) + 1.0 * self.T.vector_dot(Db)
            else:
                energy_kinetic = 0.0

            if "external" in hamiltonian:
                energy_external = 1.0 * self.V.vector_dot(Da) + 1.0 * self.V.vector_dot(Db)
            else:
                energy_external = 0.0

            if "hartree" in hamiltonian:
                energy_hartree_a = 0.5 * (self.jk.J()[0].vector_dot(Da) + self.jk.J()[1].vector_dot(Da))
                energy_hartree_b = 0.5 * (self.jk.J()[0].vector_dot(Db) + self.jk.J()[1].vector_dot(Db))
            else:
                energy_hartree_a = 0.0
                energy_hartree_b = 0.0

            if "xc" in hamiltonian:
                energy_exchange_a = -0.5 * alpha * (self.jk.K()[0].vector_dot(Da))
                energy_exchange_b = -0.5 * alpha * (self.jk.K()[1].vector_dot(Db))
                energy_ks = 1.0 * ks_e
            else:
                energy_exchange_a = 0.0
                energy_exchange_b = 0.0
                energy_ks = 0.0

            if "xxxtra" in hamiltonian:
                # Warning, xxxtra energy should be able to be computed as a contraction.
                energy_xtra_a = xxxtra_Fock_updown[0].vector_dot(Da)
                energy_xtra_b = xxxtra_Fock_updown[1].vector_dot(Db)

            energy_nuclear = 1.0 * self.Enuc
            energy_partition = 0.0
            SCF_E = energy_kinetic + energy_external + energy_hartree_a + energy_hartree_b \
                    + energy_partition + energy_ks + energy_exchange_a + energy_exchange_b + \
                    energy_nuclear

            # print("Iter", SCF_ITER, "Energy", SCF_E, "dE", abs(SCF_E - Eold), "dRMS", dRMS)

            if diis:
                if (abs(SCF_E - Eold) < E_conv and dRMS < D_conv):
                    break
            else:
                if abs(SCF_E - Eold) < E_conv:
                    break

            Eold = SCF_E
            # Diagonalize Fock matrix
            Ca, Cocca, Da, eigs_a = self.build_orbitals(Fa, self.nalpha)
            Cb, Coccb, Db, eigs_b = self.build_orbitals(Fb, self.nbeta)

        # ks_e, _, _, self.ingredients, self.orbitals, self.grid, self.potential = self.get_xc(Da, Db, Ca.np, Cb.np,
        #                                                                                      get_ingredients=get_ingredients,
        #                                                                                      get_orbitals=get_orbitals,
        #                                                                                      vxc=None)

        if energetic:
            self.energetics = {"Kinetic": energy_kinetic,
                               "External": energy_external,
                               "Hartree": energy_hartree_a + energy_hartree_b,
                               "Exact Exchange": energy_exchange_a + energy_exchange_b,
                               "Exchange-Correlation": energy_ks,
                               "Nuclear": energy_nuclear,
                               "Partition": energy_partition,
                               "Total": SCF_E}

        self.energy = SCF_E
        self.frag_energy = SCF_E - energy_partition
        self.Da, self.Db = Da, Db
        self.Fa, self.Fb = Fa, Fb
        self.Ca, self.Cb = Ca, Cb
        # self.vks_a, self.vks_b    =
        self.Cocca, self.Coccb = Cocca, Coccb
        self.eigs_a, self.eigs_b = eigs_a, eigs_b


class UMolecule(Molecule):
    """
    UMolecule defines an unrestricted KS-DFT system. Each fragment is defined a one UMolecule object.
    The reason UMolecule is created instead of using Psi4's wfn calculation is that:
    1) There is currently no ideal way to add a partition/embedding potential to some specific psi4.wfn.
    2) There used to be a bug.
    """
    def __init__(self, geometry, basis, method,
                 mints=None, jk=None, vpot=None,
                 ):
        super().__init__(geometry, basis, method,
                         mints, jk, vpot,
                         )

        self.restricted = False
        self.functional = functional_factory(self.method, False, deriv=1)

        self.Vpot = vpot if vpot is not None else psi4.core.VBase.build(self.wfn.basisset(), self.functional, "UV")
        self.Vpot.initialize()
        self.nblocks = self.Vpot.nblocks()

        D = psi4.core.Matrix(self.nbf,self.nbf)
        self.Vpot.set_D([D, D])
        self.Vpot.properties()[0].set_pointers(D, D)

        point_function = self.Vpot.properties()[0]
        point_function.set_ansatz(0)
        point_function.set_deriv(0)
        if self.functional.is_gga():
            point_function.set_ansatz(1)
            point_function.set_deriv(1)
        elif self.functional.is_meta():
            point_function.set_ansatz(2)
            point_function.set_deriv(2)

    def get_xc(self, Da, Db, Ca, Cb,
               get_ingredients=False, get_orbitals=False, vxc=None):
        self.Vpot.set_D([Da, Db])
        self.Vpot.properties()[0].set_pointers(Da, Db)
        ks_e, Vxc_a, Vxc_b, ingredients, orbitals, grid, potential = u_xc(Da, Db, Ca, Cb,
                                                                          self.wfn, self.Vpot,
                                                                          get_ingredients, get_orbitals, vxc)

        return ks_e, Vxc_a, Vxc_b, ingredients, orbitals, grid, potential
