"""
pdft.py
"""




import psi4
import qcelemental as qc
import numpy as np
import os
import pdft

from xc import functional_factory
from xc import xc
from xc import u_xc

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def basis_to_grid(mol, mat, blocks=True):
    """
    Turns a matrix expressed in the basis of mol to its grid representation

    Parameters
    ----------
    mol: Pdft.Molecule

    mat: Numpy array
        Matrix representation in the basis of mol


    Returns
    -------
    frag_phi = List
        Phi separated in blocks
    
    frag_pos = List
        Positions separated in blocks

    full_mat = Numpy array
        Grid representation of mat on the grid. Order not explicit. 
    """
    nbf = mol.nbf
    Vpot = mol.Vpot
     
    points_func = Vpot.properties()[0]
    superfunc = Vpot.functional()

    full_phi, fullx, fully, fullz, fullw, full_mat = [], [], [], [], [], []
    frag_phi, frag_w, frag_mat, frag_pos = [],[],[],[]

    # Loop Over Blocks
    for l_block in range(Vpot.nblocks()):

        # Obtain general grid information
        l_grid = Vpot.get_block(l_block)

        l_w = np.array(l_grid.w())
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())
        frag_w.append(l_w)

        for i in range(len(l_x)):
            fullx.append(l_x[i])
            fully.append(l_y[i])
            fullz.append(l_z[i])
            fullw.append(l_w[i])

        l_npoints = l_w.shape[0]
        points_func.compute_points(l_grid)

        # Recompute to l_grid
        lpos = np.array(l_grid.functions_local_to_global())
        frag_pos.append(lpos)
        points_func.compute_points(l_grid)
        nfunctions = lpos.shape[0]

        phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]
        
        l_mat = mat[(lpos[:, None], lpos)]
        mat_r = np.einsum('pm,mn,pn->p', phi, l_mat, phi, optimize=True)
        frag_mat.append(mat_r[:l_npoints])
        #frag_mat.append(mat_r)

        for i in range(len(mat_r)):
            full_mat.append(mat_r[i])
            
        for i in range(len(phi)):
            full_phi.append(phi[i])
            
    x, y, z= np.array(fullx), np.array(fully), np.array(fullz)
    full_mat = np.array(full_mat)
    full_w = np.array(fullw)
        
    if blocks is True:
        return frag_mat, frag_w
    if blocks is False: 
        return full_mat, [x,y,z,full_w] 

def grid_to_basis(mol, frag_phi, frag_pos, f):
    """
    Turns quantity expressed on the grid to basis set. 
    Missing parameter to restrict number of points of the grid involved

    Parameters
    ----------
    
    mol: Pdft.Molecule

    frag_phi = List
        Phi separated in blocks

    frag_pos = List
        Positions separated in blocks

    f = Numpy array
        Grid representation of mat on the grid. Order not explicit.


    """

    nbf = mol.nbf
    basis_grid_matrix = np.empty((0, nbf ** 2))
    
    for block in range(len(frag_phi)):
        appended = np.zeros((len(frag_phi[block]), nbf**2))
        for points in range(len(frag_phi[block])):
            appendelements = np.zeros((1, nbf))
            appendelements[0, frag_pos[block]] = frag_phi[block][points,:]
            appended[points, :] = np.squeeze((appendelements.T.dot(appendelements)).reshape(nbf ** 2, 1))
        appended = appended.reshape(len(frag_phi[block]), nbf ** 2)
        basis_grid_matrix = np.append(basis_grid_matrix, appended, axis=0)
            
    mat = np.linalg.lstsq(np.array(basis_grid_matrix), f, rcond=-1e-16)
    mat = mat[0].reshape(nbf, nbf)
    mat = 0.5 * (mat + mat.T)
    
    return mat

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

    C_ort = Cp

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


class Molecule():
    def __init__(self, geometry, basis, method, hybrid=None, mints=None, jk=None, restricted=True):
        #basics
        self.geometry   = geometry
        self.basis      = basis
        self.method = method
        self.restricted = restricted
        self.Enuc       = self.geometry.nuclear_repulsion_energy()

        #Psi4 objects
        self.wfn        = psi4.core.Wavefunction.build(self.geometry, self.basis)
        self.functional = functional_factory(self.method, True)
        self.mints = mints if mints is not None else psi4.core.MintsHelper(self.wfn.basisset())
        self.Vpot       = psi4.core.VBase.build(self.wfn.basisset(), self.functional, "RV")
        self.Vpot.initialize()

        #From psi4 objects
        self.nbf        = self.wfn.nso()
        self.ndocc      = self.wfn.nalpha()

        #From methods
        self.jk             = jk if jk is not None else self.form_JK()
        self.S              = self.mints.ao_overlap()
        self.A              = self.form_A()
        self.H              = self.form_H()

        #From SCF
        self.C              = None
        self.Cocc           = None
        self.D              = None
        self.energy         = None
        self.frag_energy    = None
        self.energetics     = None
        self.eigs           = None
        self.vks            = None
        
        self.D_r            = None  #Populate
        self.F              = None  #Populate
        self.orbitals       = None  #Populate
        self.D_0            = None  #Populate

        #For basis/grid
        self.omegas         = None  #Populate
        self.phi            = None  #Populate
        self.pos            = None  #Populate


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

    def form_JK(self, K=True):
        """
        Constructs a psi4 JK object from input basis
        """
        jk = psi4.core.JK.build(self.wfn.basisset())
        jk.set_memory(int(1.25e8)) #1GB
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

    def get_orbitals_r(self, orta=False):

        """
        Turns the C matrix into a list of matrices with each orbital
        """
        orbitals = []
        omega    = []
        pos      = []
        nbf      = self.nbf
        
        for orb_i in range(nbf):
            if orta is False:
                orbital = np.einsum('p,q->pq', self.C.np[:,orb_i], self.C.np[:,orb_i])
            if orta is True:
                orbital = np.einsum('p,q->pq', self.C_ort.np[:,orb_i], self.C_ort.np[:,orb_i])
 
            mat, phi, w, pos = basis_to_grid(self, orbital)
            orbitals.append(mat)
            omega.append(w)
            pos.append(pos)
        return orbitals, omega[0], pos

    def get_plot(self):
        plot = qc.models.Molecule.from_data(self.geometry.save_string_xyz())
        return plot

    def scf(self, maxiter=50, vp_matrix=None, print_scf=False):
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


        if vp_matrix is None:
            vp = psi4.core.Matrix(self.nbf,self.nbf)
            #self.initialize()
            C, Cocc, D, eigs = build_orbitals(self.H, self.A, self.ndocc)

        if vp_matrix is not None:
            vp = vp_matrix
            C, Cocc, D, eigs = self.C, self.Cocc, self.D, self.eigs
        

        diis_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 

        Eold = 0.0
        E = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")

        for SCF_ITER in range(maxiter+1):

            self.jk.C_left_add(Cocc)
            self.jk.compute()
            self.jk.C_clear()

            #Set DFT pointers for grid calculations
            self.Vpot.set_D([D])
            self.Vpot.properties()[0].set_pointers(D)

            #Bring core matrix
            F = self.H.clone()
            F.axpy(2.0, self.jk.J()[0])

            #Hybrid in Exchange?
            if self.functional.is_x_hybrid() is True:
                alpha = self.functional.x_alpha()
                F.axpy(-alpha, self.jk.K()[0])
            elif self.functional.is_x_hybrid() is False:
                alpha = 0.0

            #Hybrid in Correlation
            if self.functional.is_c_hybrid() is True:
                raise NameError("correlation hybrids are not avaliable")

            # DFT Compoenents
            ks_e ,Vxc = xc(D, self.Vpot)
            Vxc = psi4.core.Matrix.from_array(Vxc)
            F.axpy(1.0, Vxc)

            #PDFT Components
            F.axpy(2.0, vp)

            #DIIS
            diis_e = psi4.core.triplet(F, D, self.S, False, False, False)
            diis_e.subtract(psi4.core.triplet(self.S, D, F, False, False, False))
            diis_e = psi4.core.triplet(self.A, diis_e, self.A, False, False, False)
            diis_obj.add(F, diis_e)
            dRMS = diis_e.rms()

            #Define Energetics
            energy_core      =  2.0 * self.H.vector_dot(D)
            energy_hartree   =  2.0 * self.jk.J()[0].vector_dot(D)
            energy_exchange  = -1.0 * alpha * self.jk.K()[0].vector_dot(D)
            energy_ks        =  1.0 * ks_e
            energy_partition =  2.0 * vp.vector_dot(D)
            energy_nuclear   =  1.0 * self.Enuc

            #Add to total energy
            SCF_E = energy_core + energy_hartree + energy_nuclear + energy_partition + energy_ks + energy_exchange

            #Print Convergence
            if print_scf is True:
                print('SCF Iter%3d: % 18.14f   % 1.5E   %1.5E'% (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))

            #Convergence?
            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                break

            Eold = SCF_E

            #DIIS extrapolate
            F = diis_obj.extrapolate()

            #Diagonalize Fock matrix
            C, Cocc, D, eigs = build_orbitals(F, self.A, self.ndocc)


            if SCF_ITER == maxiter:
                raise Exception("Maximum number of SCF cycles exceeded.")

            
        energetics = {"Core" : energy_core,
                      "Hartree" : energy_hartree, 
                      "Exact Exchange" : energy_exchange, 
                      "Exchange-Correlation" : energy_ks, 
                      "Nuclear" : energy_nuclear, 
                      "Partition" : energy_partition,
                      "Total" : SCF_E}

        self.C              = C
        self.Cocc           = Cocc
        self.D              = D
        self.energy         = SCF_E
        self.frag_energy    = SCF_E - 2.0 * vp.vector_dot(D) 
        self.energetics     = energetics
        self.eigs           = eigs
        self.F              = F

        self.orbitals, self.omegas, self.pos = self.get_orbitals_r(orta=False)
        self.D_r, _, _, _           = basis_to_grid(self, self.D.np)

        if vp_matrix == None:
            self.D_0          = D

        return

class Embedding:
    def __init__(self, fragments, molecule):
        #basics
        self.fragments  = fragments
        self.nfragments = len(fragments)
        self.molecule   = molecule
        self.nblocks    = len(self.molecule.D_r)
        self.vp_update  = None

        #from mehtods
        self.frag_densities   = self.get_density_sum()
        self.frag_densities_r = self.get_density_sum_r()

    def get_energies(self):
        total = []
        for i in range(len(self.fragments)):
            total.append(self.fragments[i].energies)
        total.append(self.molecule.energies)
        pandas = pd.concat(total,axis=1)
        return pandas

    def get_density_sum(self):
        sum = self.fragments[0].D.np.copy()
        for i in range(1,len(self.fragments)):
            sum +=  self.fragments[i].D.np
        return sum

    def get_density_sum_r(self):
        sum_density_r = []
        for block in range(self.nblocks):
            block_sum = np.zeros_like(self.molecule.D_r[block])
            for i_frag in range(self.nfragments):
                block_sum += self.fragments[i_frag].D_r[block]
            sum_density_r.append(block_sum)
        return sum_density_r

    def get_delta_d(self, option):
        if option is "grid":
            delta_d = []
            sum_frag_densities = self.get_density_sum_r()
            mean = 0.0
            integral = 0.0
            for block in range(self.nblocks):
                d_d = self.molecule.D_r[block] - sum_frag_densities[block]
                delta_d.append( d_d )
                mean     += d_d.mean()
                integral += np.abs(np.einsum( 'p,p->', d_d, self.molecule.omegas[block]))

            return delta_d, mean, integral

        elif option is "matrix":
            sum_frag_densities = self.get_density_sum()
            d_d = self.molecule.D - sum_frag_densities

            return d_d

    def vp_difference(self, beta, delta_d):

        d_vp = np.zeros_like(self.molecule.D.np)
        for block in range(self.nblocks):
            omega = self.molecule.omegas[block]
            phi   = self.molecule.phi[block]
            pos   = self.molecule.pos[block]
            delta_vp = beta * (delta_d[block])  
            Vtmp = np.einsum('pb,p,p,pa->ab', phi, delta_vp, omega, phi, optimize=True)
            d_vp[(pos[:, None], pos)] += 0.5 * (Vtmp + Vtmp.T)  

        return d_vp

    def get_hessian(self):
        """
        To get the Hessian operator on the basis set xi_p = phi_i*phi_j as a matrix.
        :return: Hessian matrix as np.array self.molecule.nbf**2 x self.molecule.nbf**2
        """
        eri = self.molecule.mints.ao_eri()
        three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())

        hess = np.zeros((self.molecule.nbf, self.molecule.nbf))
        # =  np.zeros((self.molecule.nbf, self.molecule.nbf))
        for frag in self.fragments:
            # GET dvp
            # matrices for epsilon_i - epsilon_j. M
            occ = frag.ndocc
            epsilon_occ_a   = frag.eigs.np[:occ, None]
            epsilon_unocc_a = frag.eigs.np[occ:]
            epsilon_a       = epsilon_occ_a - epsilon_unocc_a
            epsilon_a       = np.reciprocal(epsilon_a)

            phi_unocc = frag.C.np[:, :occ]
            phi_occ   = frag.C.np[:, occ:]
        
            hess += np.einsum('ai,bj,ci,dj,ij,amb,cnd -> mn', phi_unocc, phi_occ, phi_unocc, phi_occ, epsilon_a, three_overlap, three_overlap, optimize=True)

        hess = -0.5 * (hess + hess.T)
        return hess

    def get_jacobian(self, d_d):
        four_overlap  = np.squeeze(self.molecule.mints.ao_eri())
        three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())

        jacobian = np.einsum("uv,uiv->i", d_d, three_overlap, optimize=True)
        jac = np.einsum("u,ui->i", d_d.reshape(self.molecule.nbf**2), four_overlap.reshape(self.molecule.nbf**2, self.molecule.nbf**2), optimize=True)
        jacobian *= -1

        return jacobian

    def vp_wuyang(self, d_d, maxiter=21, svd_rcond=1e-3, regul_const=None, a_rho_var=1e-4, vp_norm_conv=1e-6, printflag=True):

            hessian  = self.get_hessian()
            jacobian = self.get_jacobian(d_d)
            three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())

            #SVD
            hessian_inv = np.linalg.pinv(hessian, rcond=svd_rcond)

            #dvp = -1.0 * hessian_inv.dot(1.0 * jacobian)
            #dvp = np.einsum('ijm,m->ij', three_overlap, dvp)
            #dvp = 0.5 * (dvp + dvp.T)

            dvp_2 = -1.0 * hessian_inv * jacobian
            dvp_2 = 0.5 * (dvp_2 + dvp_2.T)


            return dvp_2

    def vp_yang(self, d_d, svd):

        nbf = self.molecule.nbf
        Vpot = self.molecule.Vpot
        points_func = Vpot.properties()[0]
        superfunc = Vpot.functional()

        # Loop Over Blocks
        for block in range(Vpot.nblocks()):

            # Obtain general grid information
            grid = Vpot.get_block(block)
            w = np.array(grid.w())
            npoints = w.shape[0]
            points_func.compute_points(grid)

            # Recompute to grid
            pos = np.array(grid.functions_local_to_global())
            points_func.compute_points(grid)
            nfunctions = pos.shape[0]

            phi = np.array(points_func.basis_values()["PHI"])[:npoints, :nfunctions]
            
            #l_mat = mat[(lpos[:, None], lpos)]
            #mat_r = np.einsum('pm,mn,pn->p', phi, l_mat, phi, optimize=True)

            d_vp = np.zeros_like(self.molecule.D.np)
            d_vp_block = np.zeros_like(self.fragments[0].D_r[block])  


            for point in range(len(self.molecule.D_r[block])):
                for frag in range(self.nfragments):

                    sum_density = d_d[block]
                    nocc = self.fragments[frag].ndocc
                    eigs = self.fragments[frag].eigs.np
                    x = np.zeros_like(self.molecule.orbitals[0][block])
                    orbitals = self.fragments[frag].orbitals

                    for occ in range(0, nocc):
                        for unocc in range(nocc, nbf):
                            denominator = eigs[occ] - eigs[unocc]
                            x += ((orbitals[occ][block]   * 
                                   orbitals[unocc][block] * 
                                   orbitals[unocc][block][point] * 
                                   orbitals[occ][block][point]) / denominator) 

                numerador = (sum_density[point] - self.molecule.D_r[block][point]) 
                vp_sum = (numerador / (x)) * (w[point])

            Vtmp = np.einsum('pb,p,p,pa->ab', phi, vp_sum, w, phi, optimize=True)
            d_vp[(pos[:, None], pos)] += 0.5 * (Vtmp + Vtmp.T)
        
        return d_vp

        # for block in range(self.nblocks):
        #     vp = np.zeros_like(self.fragments[0].D.np)
        #     #Grid action
        #     vp_block = np.zeros_like(self.fragments[0].D_r[block])
        #     for point in range(len(self.fragments[0].D_r[block])):
        #         for frag in range(self.nfragments):
        #             sum_density = d_d[block]
        #             nocc = self.fragments[frag].ndocc
        #             nbf  = self.fragments[frag].nbf
        #             eigs = self.fragments[frag].eigs.np
        #             omega = self.fragments[0].omegas[block]
        #             phi   = self.molecule.phi[block]
        #             pos   = self.molecule.pos[block]
        #             x = np.zeros_like(self.molecule.orbitals[0][block]) 
        #             orbitals = self.fragments[frag].orbitals

        #             for occ in range(0, nocc):
        #                 for unocc in range(nocc, nbf):
        #                     denominator = eigs[occ] - eigs[unocc]
        #                     x += ((orbitals[occ][block]   * 
        #                            orbitals[unocc][block] * 
        #                            orbitals[unocc][block][point] * 
        #                            orbitals[occ][block][point]) / denominator) 


        #         numerador = (sum_density[point] - self.molecule.D_r[block][point]) * (omega[point])
        #         vp_sum = numerador / (x)


            # Vtmp = np.einsum('pb,p,p,pa->ab', phi, vp_block, omega, phi, optimize=True)
            # vp[(pos[:, None], pos)] += 0.5 * (Vtmp + Vtmp.T)

    def vp_zmp(self):


        delta_vp = (0.5) * np.einsum('imlj, ml->ij', self.molecule.mints.ao_eri().np, self.molecule.D.np - (self.fragments[0].D.np + self.fragments[1].D.np))

        # for i in range(delta_vp.shape[0]):
        #     for j in range(delta_vp.shape[0]):
        #         if np.abs(delta_vp[i,j]) < 1e-6:
        #             delta_vp[i,j] = 0

        #delta_vp = psi4.core.Matrix.from_array(delta_vp)
        return delta_vp

    def vp_parr(self, beta, d_d):
        d_vp       = np.zeros_like(self.molecule.D.np)

        for block in range(self.nblocks):
            vp_block = np.zeros_like(self.fragments[0].D_r[block])

            delta_d  = d_d[block]
            occ      = self.molecule.ndocc
            orbital  = self.molecule.orbitals
            eigs     = self.molecule.eigs.np   
            eig_homo = self.molecule.eigs.np[occ]
            omega    = self.molecule.omegas[block]
            phi      = self.fragments[0].phi[block]
            pos      = self.fragments[0].pos[block]
            #works for 2 fragments
            max_homo_frag = max( max(self.fragments[0].eigs.np[:self.fragments[0].ndocc]), max(self.fragments[1].eigs.np[:self.fragments[1].ndocc]) )
            max_homo      = self.molecule.eigs.np[self.molecule.ndocc - 1 ]

            fraction = np.zeros_like(self.fragments[0].D_r[block])
            for i in range(occ):
                fraction += np.abs(orbital[i][block] * orbital[i][block]) / eigs[i]

            G = fraction + max((-1/max_homo_frag + 1/max_homo) * (delta_d))
            delta_vp = beta * delta_d / (G)

  
            Vtmp = np.einsum('pb,p,p,pa->ab', phi, delta_vp, omega, phi, optimize=True)
            d_vp[(pos[:, None], pos)] += 0.5 * (Vtmp + Vtmp.T)  



        return d_vp

    def vp_handler(self, beta=1.0, method="difference", maxiter=100, atol=5e-4, svd=1e-3):

        vp = psi4.core.Matrix(self.molecule.nbf,self.molecule.nbf)

        log_error = -1
        l1_error   = []
        part_energy = []
        scf_number = []


        for scf_step in range(maxiter+1):

            
            for i in range(self.nfragments):
                #Update_Fragments
                self.fragments[i].scf(vp_matrix=vp)


            #check convergence
            delta_d, dd_mean, integral = self.get_delta_d(option="grid")
            delta_d_mn = self.get_delta_d(option="matrix")
            log_error_scf = np.log10(np.abs(integral))

            #if np.floor(log_error_scf) < np.floor(log_error):
            #    log_error = log_error_scf
            #    beta /= 1.2

            #print(F"Iteration: {scf_step} | beta: {beta} | Delta_D_Mean: {dd_mean} | Delta_D: {integral}")
            l1_error.append(integral)
            scf_number.append(scf_step)
            e_p = self.molecule.energy  - (self.fragments[0].energy + self.fragments[1].energy - self.fragments[1].Enuc - self.fragments[0].Enuc) - self.molecule.Enuc
            part_energy.append(e_p)

            if np.abs(integral) < atol:
                break
            #if scf_step == maxiter:
            #    raise Exception("Maximum number of SCF cycles exceeded for vp.")


            #Get delta vp
            if method == "wuyang":
                d_vp = self.vp_wuyang(delta_d_mn, svd_rcond=svd)
            elif method == "difference":
                d_vp = self.vp_difference(beta, delta_d)
            elif method == "parr":
                d_vp = self.vp_parr(beta, delta_d)
            elif method == "yang":
                d_vp = self.vp_yang(delta_d, svd)
            elif method == "zpm":
                d_vp = self.vp_zmp()
            else:
                raise KeyError("I don't know that method")
            
            # d_vp = vp
            d_vp = psi4.core.Matrix.from_array(d_vp)
            #Update vp
            vp.axpy(beta, d_vp)
            self.vp_update = vp

            #Plotting vp info

            if np.mod(scf_step,1) == 0: 

                x_1, y_1   = get_sep_axis(self.fragments[0], self.fragments[0].D.np)
                x_2, y_2   = get_sep_axis(self.fragments[0], self.fragments[1].D.np)
                x_m, y_m   = get_sep_axis(self.molecule, self.molecule.D.np)
                #x_ks, y_ks = get_sep_axis(self.fragment[0], self.fragment[0].vks.np)
                x_vp, y_vp   = get_sep_axis(self.molecule, vp.np)

                fig = make_subplots(2,2,
                                    x_title=f"Iteration: {scf_step} | Delta_D: {integral:1.8f} | Frag_Energy : {(self.fragments[0].energy + self.fragments[0].energy):1.8f} | Partition_Energy: {e_p:1.8f}",
                                    subplot_titles=("Density Difference L1 error", "vp(x)", "Ep", "Density Difference"))

                marker = {"color":"grey", "size":3}       

                #UL
                fig.add_trace(go.Scatter(x=np.array(scf_number), y=np.array(l1_error)), row=1, col=1)
                fig.update_xaxes(row=1, col=1)

                #UR
                fig.add_trace(go.Scatter(x=x_vp, y=y_vp, mode="lines",marker=marker), row=1, col=2)
                fig.update_yaxes(range=(-0.5,0.5), row=1, col=2)
                fig.update_xaxes(range=(-4,4), row=1, col=2)
                                
                #LL
                fig.add_trace(go.Scatter(x=np.array(scf_number), y=np.array(part_energy)), row=2, col=1)

                #LR
                fig.add_trace(go.Scatter(x=x_1, y=(y_1 + y_2 - y_m), mode="lines",marker={"color":"blue", "size":3} ), row=2, col=2)
                fig.update_xaxes(range=(-6,6), row=2, col=2)
                fig.update_yaxes(range=(-0.01,0.01), row=2, col=2)


                fig.update_layout(template="plotly_white")
                fig.show()

        pio.write_image(fig, f"/zmp_{self.molecule.method}_{self.molecule.basis}.png", format='png')

        return vp

class U_Molecule():
    def __init__(self, geometry, basis, method, mints=None, jk=None):
        #From Input
        self.geometry   = geometry
        self.basis      = basis
        self.method     = method
        self.Enuc = self.geometry.nuclear_repulsion_energy()

        #Psi4 objects
        self.wfn        = psi4.core.Wavefunction.build(self.geometry, self.basis)
        self.functional = functional_factory(self.method, False)
        #self.functional = psi4.driver.dft.build_superfunctional(method, restricted=False)[0]
        self.mints = mints if mints is not None else psi4.core.MintsHelper(self.wfn.basisset())
        self.Vpot       = psi4.core.VBase.build(self.wfn.basisset(), self.functional, "UV")
        self.Vpot.initialize()

        #self.S4 = fouroverlap(self.wfn, self.geometry, self.basis, self.mints)[0]

        #From psi4 objects
        self.nbf        = self.wfn.nso()
        self.ndocc      = self.wfn.nalpha()
        self.nalpha     = self.wfn.nalpha()
        self.nbeta      = self.wfn.nbeta()

        #From methods
        self.jk             = jk if jk is not None else self.form_JK()
        self.S              = self.mints.ao_overlap()
        self.A              = self.form_A()
        self.H              = self.form_H()

        #From SCF calculation
        self.D_a             = None
        self.Da_r           = None
        self.D_b             = None
        self.Db_r           = None
        self.energy         = None
        self.frag_energy    = None
        self.energetics     = None
        self.eigs_a          = None
        self.eigs_b          = None
        self.vha_a          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
        self.vha_b          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
        self.vxc_a          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
        self.vxc_b          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
        self.vee_a          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
        self.vee_b          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
        self.F_a             = None
        self.F_b             = None
        self.C_a             = None
        self.C_b             = None
        self.Cocc_a          = None
        self.Cooc_b          = None
        self.orbitals_a     = None
        self.orbitals_b     = None
        self.Da_0          = None
        self.Db_0          = None

        #For basis/grid
        self.omegas_a       = None
        self.omegas_b       = None

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

    def form_JK(self, K=True):
        """
        Constructs a psi4 JK object from input basis
        """
        jk = psi4.core.JK.build(self.wfn.basisset())
        jk.set_memory(int(1.25e8)) #1GB
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

    def get_plot(self):
        plot = qc.models.Molecule.from_data(self.geometry.save_string_xyz())
        return plot

    def axis_plot(self, axis, matrices, labels=None, xrange=[-8,8], yrange=[-1.0, 1.0], threshold=1e-8, 
                  return_array=False):

        y_arrays = []

        for i, matrix in enumerate(matrices):

            density_grid, grid = basis_to_grid(self, matrix, blocks=False)

            x = []
            y = []
    
            if axis is "z":
                for i in range(len(grid[0])):
                    if np.abs(grid[0][i]) < threshold:
                        if np.abs(grid[1][i]) < threshold:
                            x.append((grid[2][i]))
                            y.append(density_grid[i])

            elif axis is "y":
                for i in range(len(grid[0])):
                    if np.abs(grid[0][i]) < threshold:
                        if np.abs(grid[2][i]) < threshold:
                            x.append((grid[1][i]))
                            y.append(density_grid[i])

            elif axis is "x":
                for i in range(len(grid[0])):
                    if np.abs(grid[1][i]) < threshold:
                        if np.abs(grid[2][i]) < threshold:
                            x.append((grid[0][i]))
                            y.append(density_grid[i])


            x = np.array(x)
            y = np.array(y)

            indx = x.argsort()
            x = x[indx]
            y = y[indx]

            if labels is None:

                plt.plot(x,y)

            if labels is not None:
                
                plt.plot(x,y,label=labels[i])
                plt.legend()

            if return_array is True:
                y_arrays.append((x, y))

        plt.xlim(xrange)
        plt.ylim(yrange)

        plt.show()

        if return_array is True:
            return y_arrays



    def scf(self, maxiter=50, vp=None, print_energies=False):
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
        if vp is None:
            vp_a = psi4.core.Matrix(self.nbf,self.nbf)
            vp_b = psi4.core.Matrix(self.nbf,self.nbf)

            C_a, Cocc_a, D_a, eigs_a = build_orbitals(self.H, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = build_orbitals(self.H, self.A, self.nbeta)

        elif vp is not None:
            vp_a = vp[0]
            vp_b = vp[1]

            C_a, Cocc_a, D_a, eigs_a = self.C_a, self.Cocc_a, self.D_a, self.eigs_a
            C_b, Cocc_b, D_b, eigs_b = self.C_b, self.Cocc_b, self.D_b, self.eigs_b        


        diisa_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 
        diisb_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 

        Eold = 0.0
        E = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE") 

        for SCF_ITER in range(maxiter+1):

            self.jk.C_left_add(Cocc_a)
            self.jk.C_left_add(Cocc_b)
            self.jk.compute()
            self.jk.C_clear()

            #Bring core matrix
            F_a = self.H.clone()
            F_b = self.H.clone()

            self.vha_a          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
            self.vha_b          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
            self.vxc_a          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
            self.vxc_b          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
            self.vee_a          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))
            self.vee_b          = psi4.core.Matrix.from_array(np.zeros_like(self.H.np))


            #Exchange Hybrid?
            if self.functional.is_x_hybrid() is True:
                alpha = self.functional.x_alpha()
                F_a.axpy(-alpha, self.jk.K()[0])
                F_b.axpy(-alpha, self.jk.K()[1])

                self.vee_a.axpy(-alpha, self.jk.K()[0])
                self.vee_b.axpy(-alpha, self.jk.K()[1])

            elif self.functional.is_x_hybrid() is False:
                alpha = 0.0

            #Correlation Hybrid?
            if self.functional.is_c_hybrid() is True:
                raise NameError("correlation hybrids are not availiable")


            #Exchange correlation energy/matrix
            self.Vpot.set_D([D_a,D_b])
            self.Vpot.properties()[0].set_pointers(D_a, D_b)

            ks_e ,Vxc_a, Vxc_b = u_xc(D_a, D_b, self.Vpot)
            Vxc_a = psi4.core.Matrix.from_array(Vxc_a)
            Vxc_b = psi4.core.Matrix.from_array(Vxc_b)

            #Hartree
            F_a.axpy(1.0, self.jk.J()[0])
            F_a.axpy(1.0, self.jk.J()[1]) 
            F_b.axpy(1.0, self.jk.J()[0])
            F_b.axpy(1.0, self.jk.J()[1])    

            #Exchange Correlation. Already contains scaling by alpha. 
            F_a.axpy(1.0, Vxc_a)
            F_b.axpy(1.0, Vxc_b)
            

            self.vha_a.axpy(1.0, self.jk.J()[0])
            self.vha_a.axpy(1.0, self.jk.J()[1])
            self.vha_b.axpy(1.0, self.jk.J()[0])
            self.vha_b.axpy(1.0, self.jk.J()[1])
            self.vxc_a.axpy(1.0, Vxc_a)
            self.vxc_b.axpy(1.0, Vxc_b)

            F_a.axpy(1.0, vp_a)
            F_b.axpy(1.0, vp_b)
            
            #DIIS
            diisa_e = psi4.core.triplet(F_a, D_a, self.S, False, False, False)
            diisa_e.subtract(psi4.core.triplet(self.S, D_a, F_a, False, False, False))
            diisa_e = psi4.core.triplet(self.A, diisa_e, self.A, False, False, False)
            diisa_obj.add(F_a, diisa_e)

            diisb_e = psi4.core.triplet(F_b, D_b, self.S, False, False, False)
            diisb_e.subtract(psi4.core.triplet(self.S, D_b, F_b, False, False, False))
            diisb_e = psi4.core.triplet(self.A, diisb_e, self.A, False, False, False)
            diisb_obj.add(F_b, diisb_e)

            dRMSa = diisa_e.rms()
            dRMSb = diisb_e.rms()

            #Define Energetics
            energy_core          =  1.0 * self.H.vector_dot(D_a) + 1.0 * self.H.vector_dot(D_b)
            energy_hartree_a     =  0.5 * (self.jk.J()[0].vector_dot(D_a) + self.jk.J()[1].vector_dot(D_a))
            energy_hartree_b     =  0.5 * (self.jk.J()[0].vector_dot(D_b) + self.jk.J()[1].vector_dot(D_b))
            energy_exchange_a    = -0.5 * alpha * (self.jk.K()[0].vector_dot(D_a))
            energy_exchange_b    = -0.5 * alpha * (self.jk.K()[1].vector_dot(D_b))
            energy_ks            =  1.0 * ks_e
            energy_partition     =  1.0 * vp_a.vector_dot(D_a) + vp_b.vector_dot(D_b)
            energy_nuclear       =  1.0 * self.Enuc

            SCF_E = energy_core + energy_hartree_a + energy_hartree_b + energy_partition + energy_ks + energy_exchange_a + energy_exchange_b + energy_nuclear 

            dRMS = 0.5 * (np.mean(diisa_e.np**2)**0.5 + np.mean(diisb_e.np**2)**0.5)

            # print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'
            #        % (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))

            if (abs(SCF_E - Eold) < E_conv and abs(dRMS < D_conv)):
               break

            # if (abs(SCF_E - Eold) < E_conv):
            #     break

            Eold = SCF_E

            #DIIS extrapolate
            F_a = diisa_obj.extrapolate()
            F_b = diisb_obj.extrapolate()

            #Diagonalize Fock matrix
            C_a, Cocc_a, D_a, eigs_a = build_orbitals(F_a, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = build_orbitals(F_b, self.A, self.nbeta)

            #if SCF_ITER == maxiter:
            #    raise Exception("Maximum number of SCF cycles exceeded.")

        energetics = {"Core" : energy_core,
                "Hartree" : energy_hartree_a + energy_hartree_b, 
                "Exact Exchange" : energy_exchange_a + energy_exchange_b, 
                "Exchange-Correlation" : energy_ks, 
                "Nuclear" : energy_nuclear, 
                "Partition" : energy_partition,
                "Total" : SCF_E}
        self.D_a                       = D_a
        self.D_b                       = D_b
        self.energy                    = SCF_E
        self.frag_energy               = SCF_E - energy_partition
        self.energetics                = energetics
        self.eigs_a                    = eigs_a
        self.eigs_b                    = eigs_b
        self.F_a                       = F_a
        self.F_b                       = F_b
        self.C_a                       = C_a
        self.C_b                       = C_b
        self.Cocc_a                    = Cocc_a
        self.Cocc_b                    = Cocc_b
        self.Da_r, self.omegas_a       = basis_to_grid(self, self.D_a.np)
        self.Db_r, self.omegas_b       = basis_to_grid(self, self.D_b.np)

        orbitals_a = []
        orbitals_b = []

        for i in range(self.nbf):
            orba_i = np.zeros_like(self.C_a.np)
            orbb_i = np.zeros_like(self.C_b.np)
            orba_i[:,i] = self.C_a.np[:,i]
            orbb_i[:,i] = self.C_b.np[:,i]

            orb_a_r, _ = basis_to_grid(self, orba_i)
            orb_b_r, _ = basis_to_grid(self, orbb_i)

            orbitals_a.append(orb_a_r)
            orbitals_b.append(orb_b_r)

        self.orbitals_a = orbitals_a
        self.orbitals_b = orbitals_b

        if vp is None:
            self.Da_0                 = D_a
            self.Db_0                 = D_b

        return

class U_Embedding:
    def __init__(self, fragments, molecule):
        #basics
        self.fragments  = fragments
        self.nfragments = len(fragments)
        self.nblocks    = len(molecule.Da_r)
        self.molecule   = molecule
        self.orbitals_a = None
        self.orbitals_b = None
        self.omegas_b   = None
        self.omegas_a   = None

        #from mehtods
        self.fragment_densities = self.get_density_sum()

    def get_energies(self):
        total = []
        for i in range(len(self.fragments)):
            total.append(self.fragments[i].energies)
        total.append(self.molecule.energies)
        pandas = pd.concat(total,axis=1)
        return pandas

    def get_density_sum(self):
        suma_a = self.fragments[0].D_a.np.copy()
        suma_b = self.fragments[0].D_b.np.copy()
        for i in range(1, self.nfragments):
            suma_a += self.fragments[i].D_a.np
            suma_b += self.fragments[i].D_b.np

        return suma_a, suma_b

    def get_density_sum_r(self):
        sum_density_a_r = []
        sum_density_b_r = []

        for block in range(self.nblocks):
            block_sum_a = np.zeros_like(self.molecule.Da_r[block])
            block_sum_b = np.zeros_like(self.molecule.Db_r[block])

            for frag in self.fragments:
                block_sum_a += frag.Da_r[block]
                block_sum_b += frag.Db_r[block]
            
            sum_density_a_r.append(block_sum_a)
            sum_density_b_r.append(block_sum_b)

        return sum_density_a_r, sum_density_b_r

    def get_delta_d(self, option):

        if option is "grid":
            delta_d_a = []
            delta_d_b = []

            sum_frag_densities_a, sum_frag_densities_b = self.get_density_sum_r()
            l1_error = 0.0
            for block in range(self.nblocks):
                dd_a = self.molecule.Da_r[block] - sum_frag_densities_a[block]
                dd_b = self.molecule.Db_r[block] - sum_frag_densities_b[block]

                delta_d_a.append(dd_a)
                delta_d_b.append(dd_b)

                l1_error += np.abs(np.einsum('p,p->', dd_a, self.molecule.omegas_a[block]))
                l1_error += np.abs(np.einsum('p,p->', dd_b, self.molecule.omegas_b[block]))

            return delta_d_a, delta_d_b, l1_error

        elif option is "matrix":
            sum_frag_densities_a, sum_frag_densities_b = self.get_density_sum()
            dd_a = self.molecule.D_a - sum_frag_densities_a
            dd_b = self.molecule.D_b - sum_frag_densities_b

            return dd_a, dd_b

    def get_vp_zmp(self, dd_a_nm, dd_b_nm):
        
        delta_vp_a = (-0.5) * np.einsum('imlj, ml->ij', self.molecule.mints.ao_eri().np, dd_a_nm)
        delta_vp_b = (-0.5) * np.einsum('imlj, ml->ij', self.molecule.mints.ao_eri().np, dd_b_nm)

        return delta_vp_a, delta_vp_b


    def get_vp_dd(self, dd_a_nm, dd_b_nm):

        #delta_vp_a = (-0.5) * np.einsum('imlj, ml->ij', self.S4, dd_a_nm)
        #delta_vp_b = (-0.5) * np.einsum('imlj, ml->ij', self.S4, dd_b_nm)

        dd = dd_a_nm + dd_b_nm

        return dd, dd

    def get_vp_parr(self, q, sum_density_a, sum_density_b):
        vp_in_blocks = []
        d_vp_a       = np.zeros_like(self.molecule.D_a.np)
        d_vp_b       = np.zeros_like(self.molecule.D_b.np)
        for spin in ["alpha", "beta"]:
            for block in range(len(self.omegas_a[0])):
                vp_block = np.zeros_like(self.fragments[0].Da_r[block])

                if spin == "alpha":
                    delta_d  = self.molecule.Da_r[block] - sum_density_a[block]
                    occ      = self.molecule.nalpha
                    orbital  = self.molecule.orbitals_a   
                    eigs     = self.molecule.eig_a.np   
                    eig_homo = self.molecule.eig_a.np[occ]
                    omega    = self.molecule.omegas_a[block]
                    #works for 2 fragments
                    max_homo_frag = max( max(self.fragments[0].eig_a.np[:self.fragments[0].nalpha]), max(self.fragments[1].eig_a.np[:self.fragments[1].nalpha]) )
                    max_homo      = self.molecule.eig_a.np[self.molecule.nalpha - 1 ]

                if spin == "beta":
                    delta_d  = self.molecule.Db_r[block] - sum_density_b[block] 
                    occ      = self.molecule.nbeta
                    orbital  = self.molecule.orbitals_b
                    eigs     = self.molecule.eig_b.np
                    eig_homo = self.molecule.eig_b.np[occ]
                    omega    = self.molecule.omegas_b[block]
                    #works for 2 fragments
                    max_homo_frag = max( max(self.fragments[0].eig_b.np[:self.fragments[0].nbeta]), max(self.fragments[1].eig_b.np[:self.fragments[1].nbeta]) )
                    max_homo      = self.molecule.eig_b.np[self.molecule.nbeta - 1 ]

                fraction = np.zeros_like(self.fragments[0].Da_r[block])
                for i in range(occ):
                    fraction += np.abs(orbital[i][block] * orbital[i][block]) / eigs[i]

                #G = fraction + max(0, (1/eig_homo - 1/eigs[occ]) * (delta_d))
                G = fraction + max((-1/max_homo_frag + 1/max_homo) * (delta_d))
                delta_vp = q * delta_d / (G)

                if spin == "alpha":
                    Vtmp = np.einsum('pb,p,p,pa->ab', self.fragments[0].phi_a[block], delta_vp, omega, self.fragments[0].phi_a[block], optimize=True)
                    d_vp_a[(self.fragments[0].pos_a[block][:, None], self.fragments[0].pos_a[block])] += 0.5 * (Vtmp + Vtmp.T)  

                if spin == "beta":
                    Vtmp = np.einsum('pb,p,p,pa->ab', self.fragments[0].phi_a[block], delta_vp, omega, self.fragments[0].phi_a[block], optimize=True)
                    d_vp_b[(self.fragments[0].pos_a[block][:, None], self.fragments[0].pos_a[block])] += 0.5 * (Vtmp + Vtmp.T)  


        return d_vp_a,  d_vp_b

    def vp_wuyang_r_old(self, spin, sum_density):

        vp_in_blocks = []
        for block in range(len(self.omegas_a[0])):
            vp = np.zeros_like(self.fragments[0].Da.np)
            #Grid action
            vp_block = np.zeros_like(self.fragments[0].Da_r[block])
            for point in range(len(self.fragments[0].Da_r[block])):
                for frag in range(self.nfragments):
                    nocc = self.fragments[frag].ndocc
                    nbf  = self.fragments[frag].nbf
                    if spin == "alpha":
                        eigs = self.fragments[frag].eig_a.np
                        omega = self.fragments[0].omegas_a[block]
                        x = np.zeros_like(self.orbitals_a[frag][0][block]) 
                        orbitals = self.orbitals_a
                    if spin == "beta":
                        eigs = self.fragments[frag].eig_b.np
                        omega = self.fragments[0].omegas_b[block]
                        x = np.zeros_like(self.orbitals_b[frag][0][block]) 
                        orbitals = self.orbitals_b
                    
                    for occ in range(0, nocc):
                        for unocc in range(nocc, nbf):
                            denominator = eigs[occ] - eigs[unocc]
                            x += ((orbitals[frag][occ][block]   * 
                                orbitals[frag][unocc][block] * 
                                orbitals[frag][unocc][block][point] * 
                                orbitals[frag][occ][block][point]) / denominator) 

                numerador = (sum_density[block][point] - self.molecule.Da_r[block][point]) * (omega[point])
                vp_sum = numerador / (x)
                vp_block += vp_sum
            vp_in_blocks.append(vp_block)
            if spin == "alpha":
                #grid action
                #delta_d = sum_density[block] - self.molecule.Da_r[block]
                #omega = self.fragments[0].omegas_a[block]
                #vp_container = (delta_d) * (omega) * (1/x)
                #vp_in_blocks.append(vp_container)

                #basis_action
                Vtmp = np.einsum('pb,p,p,pa->ab', self.fragments[0].phi_a[block], vp_block, omega, self.fragments[0].phi_a[block], optimize=True)
                vp[(self.fragments[0].pos_a[block][:, None], self.fragments[0].pos_a[block])] += 0.5 * (Vtmp + Vtmp.T)  

            if spin == "beta":
                #grid action
                #delta_d = sum_density[block] - self.molecule.Db_r[block]
                #omega = self.fragments[0].omegas_b[block]
                #vp_container = (delta_d) * (omega) * (1/x)
                #vp_in_blocks.append(vp_container)

                #basis_action
                Vtmp = np.einsum('pb,p,p,pa->ab', self.fragments[0].phi_a[block], vp_block, omega, self.fragments[0].phi_a[block], optimize=True)
                vp[(self.fragments[0].pos_a[block][:, None], self.fragments[0].pos_b[block])] += 0.5 * (Vtmp + Vtmp.T)  

        return vp

    def vp_wuyang_r(self, dd_a, dd_b, rcond=1e-6):
        points_func = self.molecule.Vpot.properties()[0]
        superfunc = self.molecule.Vpot.functional()
        d_vp = np.zeros_like(self.molecule.D_a.np)

        # Loop Over Blocks
        for l_block in range(self.nblocks):

            # Obtain general grid information
            l_grid = self.molecule.Vpot.get_block(l_block)

            l_w = np.array(l_grid.w())
            l_npoints = l_w.shape[0]
            points_func.compute_points(l_grid)

            # Recompute to l_grid
            lpos = np.array(l_grid.functions_local_to_global())
            points_func.compute_points(l_grid)
            nfunctions = lpos.shape[0]
            phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]

            dvp = np.zeros_like(l_w)

            #HESSIAN
            density_difference = np.zeros_like(self.molecule.Da_r[l_block])
            x = np.zeros_like(self.fragments[0].orbitals_a[0][l_block])
            for i_point in range(len(x)):
                #xb = np.zeros_like(orbs_b[block])
                for frag in self.fragments:

                    nalpha = frag.nalpha
                    nbeta = frag.nbeta
                    nbf = frag.nbf

                    eigs_a = frag.eigs_a.np
                    eigs_b = frag.eigs_b.np

                    orbs_a = frag.orbitals_a
                    orbs_b = frag.orbitals_b

                    for occ in range(0, nalpha):
                        for vir in range(nalpha, nbf):
                            denominator = eigs_a[occ] - eigs_a[vir]
                            x += (orbs_a[occ][l_block] * orbs_a[vir][l_block] * orbs_a[vir][l_block][i_point] * orbs_a[occ][l_block][i_point]) / denominator

                    for occ in range(0, nbeta):
                        for vir in range(nbeta, nbf):
                            denominator = eigs_b[occ] - eigs_b[vir]
                            x += (orbs_b[occ][l_block] * orbs_b[vir][l_block] * orbs_b[vir][l_block][i_point] * orbs_b[occ][l_block][i_point]) / denominator

                density_difference[i_point] = dd_a[l_block][i_point] + dd_b[l_block][i_point]
            
                norm = np.linalg.norm(x)

                if np.abs(norm) >= 5e-11:
                    dvp += density_difference[i_point] * l_w[i_point] / (-1.0 * x)


            counter2 = 0
            # counter = 0
            # for i in range(len(x)):
            #     #Relatively good one 4.0e-09
            #     if np.abs(x[i]) <= 4.0e-12:
            #         dvp[i] = 0
            #         counter += 1
            #     else:
            #         dvp[i] = density_difference[i] * l_w[i] / (-1.0 * x[i])
                
            # if counter/len(x) == 1.0:
            #     counter2 += 1

            ### Without svd
            ##dvp = density_difference * l_w / (-1.0 * x)

            ### Without dividing by x -> It is perfectly symmetrical! Assymetry comes from the denominator
            ##dvp = density_difference * l_w * -1.0


            Vtmp = np.einsum('pb,p,p,pa->ab', phi, dvp, l_w, phi, optimize=True)
            d_vp[(lpos[:, None], lpos)] += 0.5 * (Vtmp + Vtmp.T)

        print("Number of blocks with all zero", counter2)


        return d_vp, d_vp

    def vp_wuyang(self, dd_a_mn, dd_b_mn, svd_cond=1e-3, beta=1.0):

        # beta = 1.0
        # svd_cond = 1e-6

        four_overlap = fouroverlap(self.molecule.wfn, self.molecule.geometry, 
                                    self.molecule.basis, self.molecule.mints)[0]

        #Hessian
        hess = np.zeros((self.molecule.nbf**2, self.molecule.nbf**2))

        for frag in self.fragments:
            e_occ_a = frag.eigs_a.np[:frag.nalpha, None]
            e_occ_b = frag.eigs_b.np[:frag.nbeta, None]
            e_vir_a = frag.eigs_a.np[frag.nalpha:]
            e_vir_b = frag.eigs_b.np[frag.nbeta:]

            epsilon_a = e_occ_a - e_vir_a
            epsilon_b = e_occ_b - e_vir_b

            hess += np.einsum('ai,bj,ci,dj,ij,amnb,cuvd -> mnuv', frag.C_a.np[:, :frag.nalpha], frag.C_a.np[:, frag.nalpha:],
                                frag.C_a.np[:, :frag.nalpha], frag.C_a.np[:, frag.nalpha:], np.reciprocal(epsilon_a),
                                four_overlap, four_overlap, optimize=True).reshape(self.molecule.nbf**2, self.molecule.nbf**2)

            hess += np.einsum('ai,bj,ci,dj,ij,amnb,cuvd -> mnuv', frag.C_b.np[:, :frag.nbeta], frag.C_b.np[:, frag.nbeta:],
                                frag.C_b.np[:, :frag.nbeta], frag.C_b.np[:, frag.nbeta:], np.reciprocal(epsilon_b),
                                four_overlap, four_overlap, optimize=True).reshape(self.molecule.nbf**2, self.molecule.nbf**2)

        #negative sign here?
        hess = -0.5 * (hess + hess.T)
        #Regularization?

        #Jacobian
        jac = np.einsum("u,ui->i", (dd_a_mn + dd_b_mn).reshape(self.molecule.nbf**2),
                        four_overlap.reshape(self.molecule.nbf**2, self.molecule.nbf**2), optimize=True)
        #Regularization??

        #SVD
        hess_inv = np.linalg.pinv(hess, rcond=svd_cond)
        d_vp = hess_inv.dot(beta * jac)

        print("vp norm", np.linalg.norm(d_vp, ord=1))

        d_vp = -1.0 * d_vp.reshape(self.molecule.nbf, self.molecule.nbf)
        d_vp = np.einsum('ijmn,mn->ij', four_overlap, d_vp)
        d_vp = 0.5 * (d_vp * d_vp.T)

        return d_vp, d_vp

    def vp_wuyang_nm(self, da_mn, db_mn, rcond=1e-6):


        nbf = self.molecule.nbf
        dd = da_mn + db_mn

        nalpha =  self.molecule.nalpha
        nbeta = self.molecule.nbeta
        
        for frag in self.fragments:

            if self.molecule.nalpha == self.molecule.nbeta:
                
                x = np.zeros((nbf, nbf, nbf, nbf))

                Ca = frag.C_a.np
                Cb = frag.C_b.np

                #eigs_a = frag.eigs_a.np[:nalpha,None] - frag.eigs_a.np[nalpha:]
                #eigs_b = frag.eigs_b.np[:nbeta, None] - frag.eigs_b.np[nbeta:]

                for i in range(0, self.molecule.nalpha):
                    for a in range(self.molecule.nalpha, nbf):
                        x += np.einsum('mi, na, li, sa -> mnls', Ca[None,i], Ca[None,a], Ca[None,i], Ca[None,a], optimize=True) / (frag.eigs_a.np[i] - frag.eigs_a.np[a])
                        x += np.einsum('mi, na, li, sa -> mnls', Cb[None,i], Cb[None,a], Cb[None,i], Cb[None,a], optimize=True) / (frag.eigs_b.np[i] - frag.eigs_b.np[a])
                        #x += np.einsum('m, n, l, s -> mnls', Ca[:,i], Ca[:,a], Ca[:,i], Ca[:,a], optimize=True) / (frag.eigs_a.np[i] - frag.eigs_a.np[a])
                        #x += np.einsum('m, n, l, s -> mnls', Cb[:,i], Cb[:,a], Cb[:,i], Cb[:,a], optimize=True) / (frag.eigs_b.np[i] - frag.eigs_b.np[a])


                #x_y += np.einsum('mi, na, li, sa, ia -> mnls', Ca[:,:nalpha], Ca[:,nalpha:], Ca[:,:nalpha], Ca[:,nalpha:], np.reciprocal(eigs_a),optimize=True)
                #x_y += np.einsum('mi, na, li, sa, ia -> mnls', Cb[:,:nbeta], Cb[:,nbeta:], Cb[:,:nbeta], Cb[:,nbeta:], np.reciprocal(eigs_b),optimize=True)

            # else:

            #     xa = np.zeros((nbf, nbf, nbf, nbf))
            #     xb = np.zeros((nbf, nbf, nbf, nbf))

            #     for i in range(0, self.molecule.nalpha):
            #         for a in range(self.molecule.nalpha, nbf):
            #             xa += np.einsum('m, n, l, s -> mnls', frag.C_a.np[None,i], frag.C_a.np[None,a], frag.C_a.np[None,i], frag.C_a.np[None,a], optimize=True) / (frag.eigs_a.np[i] - frag.eigs_a.np[a])

            #     for i in range(0, self.molecule.nbeta):
            #         for a in range(self.molecule.nbeta, nbf):
            #             xb += np.einsum('m, n, l, s -> mnls', frag.C_b.np[None,i], frag.C_b.np[None,a], frag.C_b.np[None,i], frag.C_b.np[None,a], optimize=True) / (frag.eigs_b.np[i] - frag.eigs_b.np[a])



        #x = 0.5 * (x + x.T)
        x_inv = np.linalg.pinv(x, rcond=rcond)
        #print("min value of x_inv", np.min(np.abs(x_inv)))
        dvp = np.einsum('mnls, ls -> mn', x_inv, dd)
        dvp = 0.5 * (dvp + dvp.T)

        return dvp, dvp
        
    def vp_handler(self,method, rcond ,q, guess=None, maxiter=40, atol=2e-6, plot=True, **kwargs):

        l1_error_list = []
        #VP SCF CYCLE
        vp_a = psi4.core.Matrix(self.molecule.nbf, self.molecule.nbf) 
        vp_b = psi4.core.Matrix(self.molecule.nbf, self.molecule.nbf) 

        self.S4 = four_overlap = fouroverlap(self.molecule.wfn, self.molecule.geometry, self.molecule.basis, self.molecule.mints)[0]

        if guess is "external":
            vp_a.axpy(0.5, self.molecule.V)
            vp_b.axpy(0.5, self.molecule.V)
        if guess is "h_ext":

            v_nad_ha = (self.molecule.vha_a.np * self.molecule.D_a.np) + (self.molecule.vha_b.np * self.molecule.D_b.np)
            v_nad_ext = self.molecule.V.np * (self.molecule.D_a.np + self.molecule.D_b.np)

            for frag in self.fragments:
                v_nad_ha  -= (frag.vha_a.np * frag.D_a.np + frag.vha_b.np * frag.D_b.np)
                v_nad_ext -= frag.V.np * (frag.D_a.np + frag.D_b.np)

            vnad = v_nad_ha + v_nad_ext
            self.vnad_hxc = vnad
            vnad = psi4.core.Matrix.from_array(vnad)

            vp_a.axpy(0.5, vnad)
            vp_b.axpy(0.5, vnad)


        for scf_step in range(maxiter+1):

            total_energies = 0.0

            for frag in self.fragments:
                frag.scf(vp=[vp_a, vp_b])
                total_energies += frag.energy

            ############   CHECK CONVERGENCE  ###############
            dd_a, dd_b, l1_error = self.get_delta_d(option="grid")
            dd_a_mn, dd_b_mn = self.get_delta_d(option="matrix") 
            li_error_matrix = (np.sum(np.abs(dd_a_mn + dd_b_mn)))

            ep = (self.molecule.energy - self.molecule.Enuc) - (self.fragments[0].energy - self.fragments[0].Enuc + self.fragments[1].energy - self.fragments[1].Enuc)

            print(F"Vp scf cycle: {scf_step} | Density Difference: {l1_error:.5f} | Density Difference matrix: {li_error_matrix:.5f}| Ep: {ep}") 
            l1_error_list.append(l1_error)
            if l1_error < atol:
                break

            if method == "wuyang_nm":
                delta_vp_a, delta_vp_b = self.vp_wuyang_nm(dd_a_mn, dd_b_mn, rcond=rcond)
            elif method == 'dd':
                delta_vp_a, delta_vp_b = self.get_vp_dd(dd_a_mn, dd_b_mn)
            elif method == "zpm":
                delta_vp_a, delta_vp_b = self.get_vp_zmp(dd_a_mn, dd_b_mn)
            elif method  == "wuyang_r":
                delta_vp_a, delta_vp_b = self.vp_wuyang_r(dd_a, dd_b, rcond=rcond)

            ############   UPDATE VP   ###############
            delta_vp_a = psi4.core.Matrix.from_array(delta_vp_a)
            delta_vp_b = psi4.core.Matrix.from_array(delta_vp_b)
            
            vp_a.axpy(q * l1_error, delta_vp_a)
            vp_b.axpy(q * l1_error, delta_vp_b)


            self.vp_update = [vp_a, vp_b]


            if plot is True:
                ############   PLOT  ###############

                da_sum, db_sum = self.get_density_sum()
                x, y = get_sep_axis(self.molecule, da_sum + db_sum)
                x_mol, y_mol = get_sep_axis(self.molecule, self.molecule.D_a.np + self.molecule.D_b.np)
                x_vp, y_vp = get_sep_axis(self.molecule, vp_a.np + vp_b.np)
                x_dvp, y_dvp = get_sep_axis(self.molecule, delta_vp_a.np + delta_vp_b.np)

                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,5))

                p1 = ax[0][0]
                p2 = ax[1][0]
                p3 = ax[0][1]
                p4 = ax[1][1]

                p1.plot(l1_error_list)
                p1.set_xlabel("SCF Cycle")
                p1.set_ylabel("L1 Density Difference Error")

                p2.plot(x, y)
                p2.plot(x_mol, y_mol, label="mol density")
                p2.set_xlabel("X")
                p2.set_ylabel("Fragment Density")
                p2.legend()
                p2.set_ylim(-0.01, 0.2)
                p2.set_xlim(-10,10)

                p3.plot(x_vp,y_vp, label="vp")
                #p3.plot(x_dvp, y_dvp, label="dvp")
                # p3.plot(vnad_xc_x, vnad_xc_y, label="vnad_xc", linestyle=":")
                # p3.plot(vnad_ha_x, vnad_ha_y, label="vnad_ha", linestyle=":")
                p3.set_xlabel("X")
                p3.set_ylabel("vp")
                p3.set_xlim(-7,7)
                #p3.set_ylim(-1.0,1.0)
                p3.legend()

                p4.plot(x_mol, np.abs(y - y_mol))
                p4.set_xlabel("X")
                p4.set_ylabel("n_mol - sum(n_i)")
                p4.set_xlim(-10,10)

                fig.tight_layout()
                plt.show()


            if scf_step == maxiter:
                raise Exception("Maximum number of SCF cycles exceeded for vp.")

        return
        


    def find_vp(self, beta, guess=None, maxiter=10, atol=2e-4):
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
            vp_a = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Da.np))
            vp_b = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp =  [ vp_a , vp_b ]
        #else:
        #    vp_guess

        for scf_step in range(maxiter+1):

            total_density_a = np.zeros_like(self.molecule.Da.np)
            total_density_b = np.zeros_like(self.molecule.Db.np)
            total_energies = 0.0
            density_convergence = 0.0

            for i in range(self.nfragments):

                self.fragments[i].scf(vp=vp)

                total_density_a += self.fragments[i].Da.np 
                total_density_b += self.fragments[i].Db.np 

                total_energies  += self.fragments[i].frag_energy

            #if np.isclose( total_densities.sum(),self.molecule.D.sum(), atol=1e-5) :
            if np.allclose(total_density_a + total_density_b, self.molecule.Da.np, atol):
                break

            #if scf_step == maxiter:
            #    raise Exception("Maximum number of SCF cycles exceeded for vp.")

            print(F'Iteration: {scf_step} Delta_E = {total_energies - self.molecule.energy} Delta_D = {total_density_a.sum() + total_density_b.sum() - (self.molecule.Da.np.sum() + self.molecule.Db.np.sum())}')

            sumfrag_a, sumphi_a, sumw_a, sumpos_a = basis_to_grid(self.fragments[0].nbf, self.fragments[0].Vpot, total_density_a)
            sumfrag_b, sumphi_b, sumw_b, sumpos_b = basis_to_grid(self.fragments[0].nbf, self.fragments[0].Vpot, total_density_b)

            #sumfrag_a, sumphi_a, sumw_a, sumpos_a = basis_to_grid(self.fragments[0] ,total_density_a)
            #sumfrag_b, sumphi_b, sumw_b, sumpos_b = basis_to_grid(self.fragments[0] ,total_density_b)

            mol_a, molphi_a, wphi_a, summol_a = basis_to_grid(self.molecule.nbf, self.molecule.Vpot, self.molecule.Da.np)
            mol_b, molphi_b, wphi_b, summol_b = basis_to_grid(self.molecule.nbf, self.molecule.Vpot, self.molecule.Da.np)

            #mol_a, molphi_a, wphi_a, summol_a = basis_to_grid(self.molecule, self.molecule.Da.np)
            #mol_b, molphi_b, wphi_b, summol_b = basis_to_grid(self.molecule, self.molecule.Db.np)

            delta_vp_a_g = []
            delta_vp_b_g = []

            for block in range(len(sumfrag_a)):
                delta_vp_a_g.append(beta * (sumfrag_a[block] - mol_a[block]) / mol_a[block])
                delta_vp_b_g.append(beta * (sumfrag_b[block] - mol_b[block]) / mol_b[block])
            
            delta_vp_a = np.zeros_like(self.fragments[0].H.np)
            delta_vp_b = np.zeros_like(self.fragments[0].H.np)
    
            for block in range(len(sumphi_a)):
                partial_a = np.einsum('pb,p,p,pa->ab', sumphi_a[block], delta_vp_a_g[block], sumw_a[block], sumphi_a[block], optimize=True)
                delta_vp_a[(sumpos_a[block][:, None], sumpos_a[block])] += 0.5*(partial_a + partial_a.T)

                partial_b = np.einsum('pb,p,p,pa->ab', sumphi_b[block], delta_vp_b_g[block], sumw_b[block], sumphi_b[block], optimize=True)
                delta_vp_b[(sumpos_b[block][:, None], sumpos_b[block])] += 0.5*(partial_b + partial_b.T)

            delta_vp_a = psi4.core.Matrix.from_array(delta_vp_a)
            delta_vp_b = psi4.core.Matrix.from_array(delta_vp_b)
            
            vp_a.axpy(1.0, delta_vp_a)
            vp_b.axpy(1.0, delta_vp_b)

            vp_total.axpy(1.0, vp_a)
            vp_total.axpy(1.0, vp_b)

        return vp_a, vp_b, vp_total
