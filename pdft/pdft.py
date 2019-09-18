"""
pdft.py
"""


import psi4
import numpy as np


def build_orbitals(diag, A, ndocc):
    """
    Diagonalizes matrix

    Parameters
    ----------
    diag: psi4.core.Matrix
        Fock matrix

    A: psi4.core.Matrix
        A = S^(1/2), Produces orthonormalized Fock matrix

    ndocc: integer
        Number of occupied orbitals 

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
    Fp = psi4.core.triplet(A, diag, A, True, False, True)

    nbf = A.shape[0]
    Cp = psi4.core.Matrix(nbf, nbf)
    eigvecs = psi4.core.Vector(nbf)
    Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.doublet(A, Cp, False, False)

    Cocc = psi4.core.Matrix(nbf, ndocc)
    Cocc.np[:] = C.np[:, :ndocc]

    D = psi4.core.doublet(Cocc, Cocc, False, True)
    return C, Cocc, D, eigvecs

def fouroverlap(wfn,geometry,basis, mints):
        """
        Calculates four overlap integral with Density Fitting method.

        Parameters
        ----------
        wfn: psi4.core.Wavefunction
            Wavefunction object of molecule

        geometry: psi4.core.Molecule
            Geometry of molecule

        basis: str
            Basis set used to build auxiliary basis set

        Return
        ------
        S_densityfitting: numpy array
            Four overlap tensor
        """
        aux_basis = psi4.core.BasisSet.build(geometry, "DF_BASIS_SCF", "",
                                             "JKFIT", basis)
        S_Pmn = np.squeeze(mints.ao_3coverlap(aux_basis, wfn.basisset(),
                                              wfn.basisset()))
        S_PQ = np.array(mints.ao_overlap(aux_basis, aux_basis))
        S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-12)
        d_mnQ = np.einsum('Pmn,PQ->mnQ',S_Pmn,S_PQinv)
        S_densityfitting = np.einsum('Pmn,PQ,Qrs->mnrs', S_Pmn, S_PQinv, S_Pmn, optimize=True)
        return S_densityfitting, d_mnQ, S_Pmn, S_PQ


def xc(D, Vpot, functional='lda'):
    """
    Calculates the exchange correlation energy and exchange correlation
    potential to be added to the KS matrix

    Parameters
    ----------
    D: psi4.core.Matrix
        One-particle density matrix
    
    Vpot: psi4.core.VBase
        V potential 

    functional: str
        Exchange correlation functional. Currently only supports RKS LSDA 

    Returns
    -------

    e_xc: float
        Exchange correlation energy
    
    Varr: numpy array
        Vxc to be added to KS matrix
    """
    nbf = D.shape[0]
    Varr = np.zeros((nbf, nbf))
    
    total_e = 0.0
    
    points_func = Vpot.properties()[0]
    superfunc = Vpot.functional()

    e_xc = 0.0
    
    # First loop over the outer set of blocks
    for l_block in range(Vpot.nblocks()):
        
        # Obtain general grid information
        l_grid = Vpot.get_block(l_block)
        l_w = np.array(l_grid.w())
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())
        l_npoints = l_w.shape[0]

        points_func.compute_points(l_grid)

        # Compute the functional itself
        ret = superfunc.compute_functional(points_func.point_values(), -1)
        
        e_xc += np.vdot(l_w, np.array(ret["V"])[:l_npoints])
        v_rho = np.array(ret["V_RHO_A"])[:l_npoints]
    
        # Recompute to l_grid
        lpos = np.array(l_grid.functions_local_to_global())
        points_func.compute_points(l_grid)
        nfunctions = lpos.shape[0]
        
        # Integrate the LDA
        phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]

        # LDA
        Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho, l_w, phi, optimize=True)
        
        # Sum back to the correct place
        Varr[(lpos[:, None], lpos)] += 0.5*(Vtmp + Vtmp.T)

    return e_xc, Varr

class Molecule():
    def __init__(self, geometry, basis, method, mints=None, restricted=True):
        #basics
        self.geometry   = geometry
        self.basis      = basis
        self.method     = method
        self.restricted = restricted
        self.Enuc = self.geometry.nuclear_repulsion_energy()

        #Psi4 objects
        self.wfn        = psi4.core.Wavefunction.build(self.geometry, self.basis)
        self.functional = psi4.driver.dft.build_superfunctional(method, restricted=self.restricted)[0]
        #self.diis_obj   = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 

        if mints == None:
            self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        else:
            self.mints = mints
        
        if restricted == True:
            resctricted_label = "RV"
        elif restricted == False:
            restricted_label  = "UV"
        self.Vpot       = psi4.core.VBase.build(self.wfn.basisset(), self.functional, resctricted_label)

        #From psi4 objects
        self.nbf        = self.wfn.nso()
        self.ndocc      = self.wfn.nalpha()

        #From methods
        self.jk             = self.form_JK()
        self.S              = self.mints.ao_overlap()
        self.A              = self.form_A()
        self.H              = self.form_H()
        self.D, self.energy, self.grid, self.phi = self.scf()

    def initialize(self):
        """
        Initializes functional and V potential objects
        """
        #Functional
        self.functional.set_deriv(2)
        self.functional.allocate()

        #External Potential
        self.Vpot.initialize()


    def form_H(self):
        """
        Forms core matrix 
        H =  T + V
        """
        V = self.mints.ao_potential()
        T = self.mints.ao_kinetic()
        H = T.clone()
        H.add(V)

        return H

    def form_JK(self):
        """
        Constructs a psi4 JK object from input basis
        """
        jk = psi4.core.JK.build(self.wfn.basisset())
        jk.set_memory(int(1.25e8)) #1GB
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

    def scf(self, maxiter=30, vp_add=False, vp_matrix=None):
        """
        Performs scf calculation to find energy and density

        Parameters
        ----------
        vp: Bool
            Introduces a non-zero vp matrix

        vp_matrix: psi4.core.Matrix
            Vp_matrix to be added to KS matrix

        Returns
        -------

        """
        if vp_add == False:
            vp = psi4.core.Matrix(self.nbf,self.nbf)

        if vp_add == True:
            vp = vp_matrix

        self.initialize()

        C, Cocc, D, eigs = build_orbitals(self.H, self.A, self.ndocc)

        diis_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 

        Eold = 0.0
        E = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")

        for SCF_ITER in range(maxiter+1):

            self.jk.C_left_add(Cocc)
            self.jk.compute()
            self.jk.C_clear()

            #Bring core matrix
            F = self.H.clone()

            #Exchange correlation energy/matrix
            self.Vpot.set_D([D])
            self.Vpot.properties()[0].set_pointers(D)
            ks_e ,Vxc, grid, phi = xc(D, self.Vpot)
            Vxc = psi4.core.Matrix.from_array(Vxc)

            #add components to matrix
            F.axpy(2.0, self.jk.J()[0])
            F.axpy(1.0, Vxc)
            F.axpy(1.0, vp)

            #DIIS
            diis_e = psi4.core.triplet(F, D, self.S, False, False, False)
            diis_e.subtract(psi4.core.triplet(self.S, D, F, False, False, False))
            diis_e = psi4.core.triplet(self.A, diis_e, self.A, False, False, False)
            diis_obj.add(F, diis_e)
            dRMS = diis_e.rms()

            SCF_E = 2.0 * self.H.vector_dot(D)
            SCF_E += 2.0 * self.jk.J()[0].vector_dot(D)
            SCF_E += ks_e
            SCF_E += self.Enuc

            #print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'
            #       % (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))

            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                break

            Eold = SCF_E

            #DIIS extrapolate
            F = diis_obj.extrapolate()

            #Diagonalize Fock matrix
            C, Cocc, D, eigs = build_orbitals(F, self.A, self.ndocc)

            if SCF_ITER == maxiter:
                raise Exception("Maximum number of SCF cycles exceeded.")

        #print('\nFinal SCF energy: %.8f hartree' % SCF_E)
        #print(F'Core                 : {2.0 * self.H.vector_dot(D)}')
        #print(F'Hartree              : {2.0 * self.jk.J()[0].vector_dot(D)}')
        #print(F'Exchange Correlation : {ks_e}')
        #print(F'Nuclear Repulsion    : {self.Enuc}')

        return D.np, SCF_E, grid, phi


class Embedding:
    def __init__(self, fragments, molecule):
        #basics
        self.fragments = fragments
        self.nfragments = len(fragments)
        self.molecule = molecule

        #from mehtods
        self.fragment_densities = self.get_density_sum()

    def get_density_sum(self):
        sum = self.fragments[0].D.copy()
        for i in range(1,len(self.fragments)):
            sum +=  self.fragments[i].D
        return sum

    def find_vp(self, beta, guess=None, maxiter=10):
        """
        Given a target function, finds vp_matrix to be added to each fragment
        ks matrix to match full molecule energy/density

        Parameters
        ----------
        beta: positive float
            Coefficient for delta_n = beta * (molecule_density  - sum_fragment_densities)

        Returns
        -------
        vp: psi4.core.Matrix
            Vp to be added to fragment ks matrix

        """
        if guess==None:
            vp =  psi4.core.Matrix.from_array(np.zeros_like(self.molecule.D))
        #else:
        #    vp_guess

        for scf_step in range(maxiter+1):

            total_densities = np.zeros_like(self.molecule.D)
            total_energies = 0.0
            density_convergence = 0.0

            for i in range(self.nfragments):

                density, energy = self.fragments[i].scf(vp_add=True, vp_matrix=vp)
                
                total_densities += density
                total_energies  += energy

            if np.isclose( total_densities.sum(),self.molecule.D.sum(), atol=1e-16) :
                break

            #if scf_step == maxiter:
            #    raise Exception("Maximum number of SCF cycles exceeded for vp.")

            print(F'Iteration: {scf_step} Delta_E = {total_energies - self.molecule.energy} Delta_D = {total_densities.sum() - self.molecule.D.sum()}')

            delta_vp =  beta * (total_densities - self.molecule.D)       
            S, D_mnQ, S_pmn, Spq = fouroverlap(self.fragments[0].wfn, self.fragments[0].geometry, "STO-3G", self.mints)
            #S_2, d_2, S_pmn_2, Spq_2 = fouroverlap(self.fragments[1].wfn, self.fragments[1].geometry, "STO-3G")

            delta_vp =  psi4.core.Matrix.from_array( np.einsum('ijmn,mn->ij', S, delta_vp))
            #delta_vp = psi4.core.Matrix.from_array(delta_vp)

            vp.axpy(1.0, delta_vp)

        return vp
