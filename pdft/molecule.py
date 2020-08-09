"""

Molecule.py
Defitions for molecule class

"""
import numpy as np
from opt_einsum import contract

import matplotlib.pyplot as plt
import psi4

from .xc import functional_factory
from .xc import xc
from .xc import u_xc

class Molecule():

    def __init__(self, geometry, basis, method, 
                 mints = None, jk = None, vpot = None,
                 ):
        
        #basics
        self.geometry    = geometry
        self.basis_label = basis
        self.method      = method
        self.Enuc        = geometry.nuclear_repulsion_energy()

        #Psi4 objects
        #self.wfn         = psi4.core.Wavefunction.build(self.geometry, self.basis_label)
        self.set_wfn()
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
        #From scf
        self.Ca          = None
        self.Cb          = None
        self.Cocca       = None
        self.Coccb       = None
        self.Da          = None
        self.Db          = None
        self.Da_0        = None
        self.Db_0        = None
        self.Fa          = None
        self.Fb          = None
        self.energy      = None
        self.frag_energy = None
        self.energetics  = None
        self.eigs_a      = None
        self.eigs_b      = None
        
        #Potentials
        self.vha_a       = None
        self.vha_b       = None
        self.vxc_a       = None
        self.vxc_b       = None

        #Grid
        self.grid        = None
        self.omegas      = None
        self.phi         = None
        #self.Da_r        = None
        #self.Db_r        = None
        self.ingredients = None
        self.orbitals    = None

        #Options

    def form_JK(self, K=True):
        """
        Constructs a psi4 JK object from input basis
        """
        jk = psi4.core.JK.build(self.basis)
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

    def scf(self, maxiter=100, hamiltonian = ["kinetic", "external", "hartree", "xc"],
                              vp_mn=None, xfock_nm = None,
                              get_matrices=True, get_ingredients=False, get_orbitals=False,
                              diis=False):
        """
        Performs scf cycle

        Parameters
        ----------
        vp: psi4.core.Matrix
            Vp_matrix to be added to KS matrix
        """

        #Restricted/Unrestricted
        #Initial Guess. 
        if self.Da is None and self.Db is None:
            Ca, Cocca, Da, eigs_a = self.build_orbitals(self.H, self.nalpha)
            #Set to X_b = X_a if restricted == True without depending on child class. 
            Cb, Coccb, Db, eigs_b = self.build_orbitals(self.H, self.nbeta)

        if self.Da is not None and self.Da is not None:
            Ca, Cocca, Da, eigs_a = self.Ca, self.Cocca, self.Da, self.eigs_a
            Cb, Coccb, Db, eigs_b = self.Cb, self.Coccb, self.Db, self.eigs_b
                
        if diis is True:
            diisa_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 
            diisb_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")

        Eold = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE") 

        for SCF_ITER in range(maxiter+1):
            self.jk.C_left_add(Cocca)
            self.jk.C_left_add(Coccb)
            self.jk.compute()
            self.jk.C_clear()

            if get_matrices is True:
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
                    #Hartree
                    self.vha_a.axpy(1.0, self.jk.J()[0])
                    self.vha_a.axpy(1.0, self.jk.J()[1])             
                    self.vha_b.axpy(1.0, self.jk.J()[0])
                    self.vha_b.axpy(1.0, self.jk.J()[1])
                    #Kohn_Sham Potential 
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

                #Correlation Hybrid?
                if self.functional.is_c_hybrid() is True:
                    raise NameError("Correlation hybrids are not avaliable")

                #Exchange Correlation
                ks_e, Vxc_a, Vxc_b, self.ingredients, self.orbitals, self.grid, self.potential = self.get_xc(Da, Db, Ca.np, Cb.np, 
                                                                                                False, False, vxc=None)

                #XC already scaled by alpha  
                Vxc_a = psi4.core.Matrix.from_array(Vxc_a)
                Vxc_b = psi4.core.Matrix.from_array(Vxc_b)          
                self.vxc_a.axpy(1.0, Vxc_a)
                self.vxc_b.axpy(1.0, Vxc_b)
                Fa.axpy(1.0, Vxc_a)
                Fb.axpy(1.0, Vxc_b)
                # Fa.axpy(1.0, vp_a)
                # Fb.axpy(1.0, vp_b)

                if get_matrices is True:
                    self.vxc_a.axpy(1.0, Vxc_a)
                    self.vxc_b.axpy(1.0, Vxc_b)

            if "xxxtra" in hamiltonian:
                if xfock_nm is None:
                    raise NameError("You want to add an xxxtra component to Hamiltonian but didn't specify it")
                Fa.axpy(1.0, xfock_nm[0])
                Fb.axpy(1.0, xfock_nm[1])

            #DIIS
            if diis is True:
                diisa_e = psi4.core.triplet(Fa, Da, self.S, False, False, False)
                diisa_e.subtract(psi4.core.triplet(self.S, Da, Fa, False, False, False))
                diisa_e = psi4.core.triplet(self.A, diisa_e, self.A, False, False, False)
                diisa_obj.add(Fa, diisa_e)

                diisb_e = psi4.core.triplet(Fb, Db, self.S, False, False, False)
                diisb_e.subtract(psi4.core.triplet(self.S, Db, Fb, False, False, False))
                diisb_e = psi4.core.triplet(self.A, diisb_e, self.A, False, False, False)
                diisb_obj.add(Fb, diisb_e)

                dRMSa = diisa_e.rms()
                dRMSb = diisb_e.rms()

                dRMS = 0.5 * (np.mean(diisa_e.np**2)**0.5 + np.mean(diisb_e.np**2)**0.5)

                Fa = diisa_obj.extrapolate()
                Fb = diisb_obj.extrapolate()

            #Define Energetics
            if True:
                if "kinetic" in hamiltonian:
                    energy_kinetic = 1.0 * self.T.vector_dot(Da) + 1.0 * self.T.vector_dot(Db)
                else:
                    energy_kinetic = 0.0

                if "external" in hamiltonian:
                    energy_external = 1.0 * self.V.vector_dot(Da) + 1.0 * self.V.vector_dot(Db)
                else:
                    energy_external = 0.0

                if "hartree" in hamiltonian:
                    energy_hartree_a     =  0.5 * (self.jk.J()[0].vector_dot(Da) + self.jk.J()[1].vector_dot(Da))
                    energy_hartree_b     =  0.5 * (self.jk.J()[0].vector_dot(Db) + self.jk.J()[1].vector_dot(Db))
                else:
                    energy_hartree_a = 0.0
                    energy_hartree_b = 0.0

                if "xc" in hamiltonian:
                    energy_exchange_a    = -0.5 * alpha * (self.jk.K()[0].vector_dot(Da))
                    energy_exchange_b    = -0.5 * alpha * (self.jk.K()[1].vector_dot(Db))
                    energy_ks            =  1.0 * ks_e
                else:
                    energy_exchange_a    = 0.0
                    energy_exchange_b    = 0.0
                    energy_ks            = 0.0

                if "xxxtra" in hamiltonian:
                    #Warning, xxxtra energy should be able to be computed as a contraction. 
                    energy_xtra_a = xfock_nm[0].vector_dot(Da)
                    energy_xtra_b = xfock_nm[1].vector_dot(Db)

            energy_nuclear = 1.0 * self.Enuc
            energy_partition = 0.0
            SCF_E = energy_kinetic + energy_external + energy_hartree_a + energy_hartree_b + energy_partition + energy_ks + energy_exchange_a + energy_exchange_b + energy_nuclear 
           
            #if np.mod(SCF_ITER, 5.0) == 0:
            #    print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'% (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))

            if abs(SCF_E - Eold) < E_conv:
            #if (abs(SCF_E - Eold) < E_conv and abs(dRMS < 1e-5)):
               break

            Eold = SCF_E

            #Diagonalize Fock matrix
            Ca, Cocca, Da, eigs_a = self.build_orbitals(Fa, self.nalpha)
            Cb, Coccb, Db, eigs_b = self.build_orbitals(Fb, self.nbeta)
        
        ks_e, _, _, self.ingredients, self.orbitals, self.grid, self.potential = self.get_xc(Da, Db, Ca.np, Cb.np, 
                                                                                                    get_ingredients=get_ingredients, 
                                                                                                    get_orbitals=get_orbitals, 
                                                                                                    vxc=None)

        self.energetics = {"Kinetic" : energy_kinetic,
                           "External" : energy_external,
                           "Hartree" : energy_hartree_a + energy_hartree_b, 
                           "Exact Exchange" : energy_exchange_a + energy_exchange_b, 
                           "Exchange-Correlation" : energy_ks, 
                           "Nuclear" : energy_nuclear, 
                           "Partition" : energy_partition,
                           "Total" : SCF_E}

        self.energy               = SCF_E
        self.frag_energy          = SCF_E - energy_partition
        self.Da, self.Db          = Da, Db
        self.Fa, self.Fb          = Fa, Fb
        self.Ca, self.Cb          = Ca, Cb
        #self.vks_a, self.vks_b    = 
        self.Cocca, self.Coccb    = Cocca, Coccb
        self.eigs_a, self.eigs_b  = eigs_a, eigs_b
        
        #Stores everything in wfn object
        self.set_wfn()

    def basis_to_grid(self, mat, blocks=True):
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
        Vpot = self.Vpot
        
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

            if mat.ndim == 2:
                l_mat = mat[(lpos[:, None], lpos)]
                mat_r = contract('pm,mn,pn->p', phi, l_mat, phi)

            if mat.ndim == 1:
                l_mat = mat[lpos]
                mat_r =mat_r = contract('pm,m->p', phi, l_mat)

            frag_mat.append((mat_r[:l_npoints]))
            #frag_mat.append(mat_r)

            for i in range(len(mat_r)):
                full_mat.append(mat_r[i])
                
            for i in range(len(phi)):
                full_phi.append(phi[i])
                
        x, y, z= np.array(fullx), np.array(fully), np.array(fullz)
        full_mat = np.array(full_mat)
        full_w = np.array(fullw)
            
        if blocks is True:
            return np.array(frag_mat), frag_w
        if blocks is False: 
            return full_mat, [x,y,z,full_w] 

    def to_grid(self, Duv, Duv_b=None, vpot=None):
        """
        For any function on double ao basis: f(r) = Duv*phi_u(r)*phi_v(r), e.g. the density.
        If Duv_b is not None, it will take Duv + Duv_b.
        One should use the same wfn for all the fragments and the entire systems since different geometry will
        give different arrangement of xyzw.
        :return: The value of f(r) on grid points.
        """
        if vpot is None:
            vpot = self.Vpot
        points_func = vpot.properties()[0]
        f_grid = np.array([])
        # Loop over the blocks
        if Duv.ndim == 2:
            for b in range(vpot.nblocks()):
                # Obtain block information
                block = vpot.get_block(b)
                points_func.compute_points(block)
                npoints = block.npoints()
                lpos = np.array(block.functions_local_to_global())

                # Compute phi!
                phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

                # Build a local slice of D
                if Duv_b is None:
                    lD = Duv[(lpos[:, None], lpos)]
                else:
                    lD = Duv[(lpos[:, None], lpos)] + Duv_b[(lpos[:, None], lpos)]
                # Copmute rho
                f_grid = np.append(f_grid, np.einsum('pm,mn,pn->p', phi, lD, phi))
        elif Duv.ndim==1:
            for b in range(vpot.nblocks()):
                # Obtain block information
                block = vpot.get_block(b)
                points_func.compute_points(block)
                npoints = block.npoints()
                lpos = np.array(block.functions_local_to_global())

                # Compute phi!
                phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

                # Build a local slice of D
                if Duv_b is None:
                    lD = Duv[lpos]
                else:
                    lD = Duv[lpos] + Duv_b[lpos]

                # Copmute rho
                f_grid = np.append(f_grid, np.einsum('pm,m->p', phi, lD))
        return f_grid

    def axis_plot_r(self, functions, axis="z", labels=None, 
                                               xrange=None, 
                                               yrange=None, 
                                               logplot=False, return_array=False, 
                                               threshold=1e-11):

        y_out_arrays = []
        x_out_array = []

        for j, function in enumerate(functions):

            x_out = []
            y_out = []

            for block in range(len(function)):

                x = self.grid["x"][block]
                y = self.grid["y"][block]
                z = self.grid["z"][block]

                if axis == "z":
                    for i_point in range(len(x)):
                        if np.abs(x[i_point]) < threshold:
                            if np.abs(y[i_point]) < threshold:
                                x_out.append(z[i_point])
                                y_out.append(function[block][i_point])
                
                elif axis == "x":
                    for i_point in range(len(x)):
                        if np.abs(y[i_point]) < threshold:
                            if np.abs(z[i_point]) < threshold:
                                x_out.append(x[i_point])
                                y_out.append(function[block][i_point])

                elif axis == "y":
                    for i_point in range(len(x)):
                        if np.abs(x[i_point]) < threshold:
                            if np.abs(z[i_point]) < threshold:
                                x_out.append(y[i_point])
                                y_out.append(function[block][i_point])


            x_out = np.array(x_out)
            y_out = np.array(y_out)
            indx = x_out.argsort()
            x_out = x_out[indx]
            y_out = y_out[indx]

            if labels is None:
                plt.plot(x_out,y_out)
                if logplot is not False:
                    plt.xscale('log')
            elif labels is not None:
                plt.plot(x_out,y_out,label=labels[j])
                plt.legend(fontsize=18)
            # if return_array is True:
            #     y_arrays.append((x_out,y_out))
            if xrange is not None:
                plt.xlim(xrange)
            if yrange is not None:
                plt.ylim(yrange)
            
            y_out_arrays.append(y_out)
        x_out_array = x_out

        plt.show()

        if return_array is True:
            return x_out_array, y_out_arrays

    def axis_plot(self, axis, matrices, labels=None, xrange=None, yrange=None, threshold=1e-11, 
                  return_array=False):
        """

        For a given matrix in AO basis set, plots the value for that matrix along a given axis. 

        """    
        y_arrays = []

        for j, matrix in enumerate(matrices):

            #matrix = matrix.np
            density_grid, grid = self.basis_to_grid(matrix, blocks=False)

            x = []
            y = []
    
            if axis == "z":
                for i in range(len(grid[0])):
                    if np.abs(grid[0][i]) < threshold:
                        if np.abs(grid[1][i]) < threshold:
                            x.append((grid[2][i]))
                            y.append(density_grid[i])

            elif axis == "y":
                for i in range(len(grid[0])):
                    if np.abs(grid[0][i]) < threshold:
                        if np.abs(grid[2][i]) < threshold:
                            x.append((grid[1][i]))
                            y.append(density_grid[i])

            elif axis == "x":
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
            y_arrays.append(y)

            if labels is None:
                plt.plot(x,y)
            elif labels is not None:
                plt.plot(x,y,label=labels[j])
                plt.legend()
            if return_array is True:
                y_arrays.append((x, y))
            if xrange is not None:
                plt.xlim(xrange)
            if yrange is not None:
                plt.ylim(yrange)


        plt.show()

        if return_array is True:
            return x, y_arrays

    def set_wfn(self):
        """
        Sets scf output to wfn object
        """

        _, wfn = psi4.energy(self.method+"/"+self.basis_label, molecule=self.geometry, return_wfn=True)

        # matrices = {"Ca" : self.Ca, "Cb" : self.Cb, 
        #             "Da" : self.Da, "Db" : self.Db,
        #             "Fa" : self.Fa, "Fb" : self.Fb,
        #             "H"  : self.H,
        #             "S"  : self.S,
        #             #"Lagrangian" : None,
        #             "AO2SO" : wfn.aotoso(),
        #             #"gradient" : None, 
        #             #"Hessian" : None
        #             }

        # vectors = {"epsilon_a" : self.eigs_a, "epsilon_b" : self.eigs_b,
        #            #"frequencies" : None
        #            }

        # dimensions = {"doccpi": wfn.doccpi(), 
        #               "soccpi" : wfn.soccpi(), 
        #               "frzcpi" : wfn.frzcpi(), 
        #               "frzvpi" : wfn.frzvpi(),
        #               "nalphapi" : wfn.nalphapi(),
        #               "nbetapi" : wfn.nbetapi(), 
        #               "nmopi" : wfn.nmopi(),
        #               "nsopi" : wfn.nsopi()
        #               }

        # integers = {"nalpha" : self.nalpha, "nbeta" : self.nbeta, 
        #             #"nfrzc" : None, 
        #             #"nirrep": None,
        #             #"nmo" : None, 
        #             #"nso" : None, 
        #             #"print" : None
        #             }
                
        # name = {#"name" : None
        #         }

        # booleans = {#"PCM_enabled" : None, 
        #             "same_a_b_dens" : True, 
        #             "same_a_b_orbs" : True,
        #             #"density_fitted" : None,
        #             # 
        #             }

        # floats = {"energy" : self.energy, 
        #           #"efzc" : None, 
        #           #"dipole_field_x" : None,
        #           #"dipole_field_y" : None,
        #           #"dipole_field_z" : None
        #           }        

        
        # wfn = psi4.core.Wavefunction(self.geometry, self.basis, matrices,
        #                                                         vectors, 
        #                                                         dimensions, 
        #                                                         integers,
        #                                                         name,
        #                                                         booleans, 
        #                                                         floats)

        self.wfn = wfn

class RMolecule(Molecule):

    def __init__(self, geometry, basis, method, 
                 mints=None, jk=None, vpot = None,
                 ):
        super().__init__(geometry, basis, method, 
                         mints, jk, vpot,
                         )

        #Psi4 objects 
        self.functional = functional_factory(self.method, True, deriv=1)
        self.Vpot       = vpot if vpot is not None else psi4.core.VBase.build(self.wfn.basisset(), self.functional, "RV")
        self.Vpot.initialize()
        self.nblocks = self.Vpot.nblocks()

        if self.nalpha != self.nbeta:
            raise ValueError("RMolecule can't be used with that electronic configuration")

    def get_xc(self, Da, Db, Ca, Cb, 
               get_ingredients, get_orbitals, vxc):
        self.Vpot.set_D([Da])
        self.Vpot.properties()[0].set_pointers(Da)
        ks_e, Vxc, ingredients, orbitals, grid, potential = xc(Da, Ca, 
                                              self.wfn, self.Vpot,
                                              get_ingredients, get_orbitals, vxc)

        return ks_e, Vxc, Vxc, ingredients, orbitals, grid, potential

class UMolecule(Molecule):

    def __init__(self, geometry, basis, method, 
                 mints = None, jk = None, vpot = None,
                 ):
        super().__init__(geometry, basis, method, 
                        mints, jk, vpot, 
                        )

        self.restricted = False
        self.functional = functional_factory(self.method, False, deriv=1)

        self.Vpot       = vpot if vpot is not None else psi4.core.VBase.build(self.wfn.basisset(), self.functional, "UV")
        self.Vpot.initialize()
        self.nblocks = self.Vpot.nblocks()

    def get_xc(self, Da, Db, Ca, Cb, 
               get_ingredients, get_orbitals, vxc):
        self.Vpot.set_D([Da, Db])
        self.Vpot.properties()[0].set_pointers(Da, Db)  
        ks_e, Vxc_a, Vxc_b, ingredients, orbitals, grid, potential = u_xc(Da, Db, Ca, Cb, 
                                                        self.wfn, self.Vpot,
                                                        get_ingredients, get_orbitals, vxc)
        
        return ks_e, Vxc_a, Vxc_b, ingredients, orbitals, grid, potential
