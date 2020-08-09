import numpy as np 
from opt_einsum import contract
import matplotlib.pyplot as plt
import psi4

import scipy.optimize as optimizer

from .xc import u_xc
# from .inv.wuyang
# from .inv.oucarter

class Inversion():
    def __init__(self, fragments, molecule, target_wfn, **kwargs):
        
        #Basics
        self.frags    = fragments
        self.nfrag    = len(fragments)
        self.molecule = molecule
        self.target   = target_wfn
        self.nbf      = self.molecule.nbf
        self.nblocks  = self.assert_grid_elements()

        #From methods
        self.frag_e     = None 
        self.frag_da_nm = None
        self.frag_db_nm = None
        self.frag_da_r  = None
        self.frag_db_r  = None
        self.dd_a_mn    = None
        self.dd_b_mn    = None
        self.frag_orbitals_r = None
        #self.get_frag_energies()
        #self.get_frag_densities_nm()
        #self.get_frag_densities_r()
        
        #From inversion
        self.vp     = None
        self.ep     = None

    def assert_grid_elements(self):
        """
        Asserts if density on the grid is avaliable. 
        If so, asserts that grid points are same between molecules and fragments. 
        """

        #if len(self.molecule.ingredients["density"]["da"]) == 0:
        #    raise ValueError("Density on the grid not avaliable for molecule. Please run scf with get_ingredients as True")

        #if len(self.frags[0].ingredients["density"]["da"]) == 0:
        #    raise ValueError("Density on the grid not avaliable for molecule. Please run scf with get_ingredients as True")

        #Checks that the number of points in each block for framgent is same wrt molecule
        #mol_points = psi4.driver.p4util.python_helpers._core_vbase_get_np_xyzw(self.molecule.Vpot)
        #fra_points = psi4.driver.p4util.python_helpers._core_vbase_get_np_xyzw(self.frags[0].Vpot)

        #if len(mol_points[0]) != len(fra_points[0]):
        #    raise ValueError("Grid of fragments does not match Grid of molecule. Verify nuclei charges or use shared Vpot")

        #else:
        return self.frags[0].Vpot.nblocks()

    def get_vha_nad(self):
        """
        Calculates vha_nad on the grid
        """

        vha_nad  = self.molecule.potential["vha"].copy()
        for i_frag in self.frags:
            vha_nad -= i_frag.potential["vha"].copy()
        return vha_nad

    def get_vxc_nad(self):
        """
        Calculates vxc_nad on the grid
        """

        vxc_nad_a = self.molecule.potential["vxc_a"].copy()
        for i_frag in self.frags:
            vxc_nad_a -= i_frag.potential["vxc_a"].copy()

        return vxc_nad_a

    def get_frag_energies(self):
        """
        Adds fragment energies for fragments
        """
        frag_energy = 0.0
        for frag in self.frags:
            frag_energy += frag.energy
        self.frag_energy = frag_energy

    def get_frag_densities_nm(self):
        """
        Adds fragment densities in their AO orbital representation 
        """
        sum_a = psi4.core.Matrix.from_array(np.zeros((self.nbf, self.nbf)))
        sum_b = psi4.core.Matrix.from_array(np.zeros((self.nbf, self.nbf)))
        for frag in self.frags:
            sum_a.axpy(1.0, frag.Da)
            sum_b.axpy(1.0, frag.Db)
        self.frag_da_nm = sum_a
        self.frag_db_nm = sum_b

    def get_frag_densities_r(self):
        """
        Adds fragment densities on the grid in blocks
        """
        sum_a = []
        sum_b = []

        for block in range(self.nblocks):
            block_sum_a = np.zeros_like(self.molecule.ingredients["density"]["da"][block])
            block_sum_b = np.zeros_like(self.molecule.ingredients["density"]["db"][block])
            for frag in self.frags:
                block_sum_a += frag.ingredients["density"]["da"][block]
                block_sum_b += frag.ingredients["density"]["db"][block]
            sum_a.append(block_sum_a)
            sum_b.append(block_sum_b)
        self.frag_da_r = sum_a
        self.frag_db_r = sum_b

    def get_delta_density(self, option):
        """
        Calculates density difference on the grid/AObasis for current frag densities
        """ 
        if option == "grid":
            dd_a = []
            dd_b = []
            l1error = 0.0
            for block in range(self.nblocks): 
                block_dd_a =  self.frag_da_r[block] - self.molecule.ingredients["density"]["da"][block]
                block_dd_b =  self.frag_db_r[block] - self.molecule.ingredients["density"]["db"][block]
                dd_a.append(block_dd_a)
                dd_b.append(block_dd_b)
                l1error += np.abs(contract('p,p->', block_dd_a, self.molecule.grid["w"][block]))     
                l1error += np.abs(contract('p,p->', block_dd_b, self.molecule.grid["w"][block]))   
            return np.array(dd_a), np.array(dd_b), l1error

        if option == "matrix":
            dd_a = self.target.Da().np - self.frag_da_nm.np
            dd_b = self.target.Db().np - self.frag_db_nm.np
            return dd_a, dd_b

    def get_ep(self):
        """
        Finds partition energy for current fragment densities
        """
        ep = self.molecule.energy - self.molecule.Enuc
        for frag in self.frags:
            ep -= (frag.energy - frag.Enuc)
        self.ep = ep

    def reintegrate_ao(self, function):

        print("This is my function shape")
        print(function.shape)
        print("Len of first block", function[0])

        f_nm = np.zeros((self.nbf, self.nbf))
        points_func = self.frags[0].Vpot.properties()[0]

        for block in range(self.nblocks):
            grid_block = self.frags[0].Vpot.get_block(block)
            points_func.compute_points(grid_block)
            npoints = grid_block.npoints()
            lpos = np.array(grid_block.functions_local_to_global())
            w = np.array(grid_block.w())
            phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
            print(phi.shape)
            print(function[block])
            vtmp = contract('pb,p,p,pa->ab', phi, function[block], w, phi)
            f_nm[(lpos[:, None], lpos)] += 0.5 * (vtmp + vtmp.T)

        return f_nm

    # Inversion Procedures

    def vp_handler(self, method, beta, maxiter=40, atol=1e-5,
                  guess=None, print_scf=True, plot_scf=True):

        vp_a = psi4.core.Matrix(self.nbf, self.nbf)
        vp_b = psi4.core.Matrix(self.nbf, self.nbf)

        #GRID GUESSES
        if guess == "hartree":
            vp_a_r = self.get_vha_nad()
            vp_b_r = vp_a_r.copy()

        elif guess == "xc":
            vp_a_r = self.get_vxc_nad()
            vp_b_r = vp_a_r.copy()

        elif guess == "hxc":
            vp_a_r = self.get_vha_nad()
            vp_a_r += self.get_vxc_nad()
            vp_b_r = vp_a_r.copy()

        # elif guess == "fermiamaldi":
            



        else:
            print("You have chosen no initial guess")
            #vp_a_r = np.zeros_like(self.frags[0].ingredients["density"]["da"]) 
            #vp_b_r = np.zeros_like(self.frags[0].ingredients["density"]["db"]) 
            # vp_a_r = np.zeros_like(self.get_vha_nad())
            # vp_a_r = np.zeros_like(self.get_vha_nad())

        #vp_a = self.reintegrate_ao(vp_a_r)
        #vp_b = self.reintegrate_ao(vp_b_r)
        #vp_a = psi4.core.Matrix.from_array(vp_a)
        #vp_b = psi4.core.Matrix.from_array(vp_b)

        l1_list = []
        ep_list = []
        for step in range(maxiter+1):
            print(f"External scf cycle {step}")
            #Update fragment densities
            for frag in self.frags:
                frag.scf(vxc=[vp_a, vp_b], get_ingredients=True)

            #Check convergence
            self.get_frag_energies()
            #self.get_frag_densities_r()
            self.get_frag_densities_nm()
            self.get_ep()
           
            self.dd_a_mn, self.dd_b_mn  = self.get_delta_density(option="matrix")
            #Grid dependant quantities
            #dd_a, dd_b, error = self.get_delta_density(option="grid")
            #l1_list.append(error)
            #ep_list.append(self.ep)
            #if print_scf is True:
            #    print(F"vp scf cycle: {step} | Density Difference: {error:.5f} | Ep: {self.ep:.5f} | Ef: {self.frag_energy:.4f}")

            if print_scf is True:
                print(F"scf outter cycle: {step} | Density Difference: {np.linalg.norm(self.dd_a_mn + self.dd_b_mn)}")

            #Choose inversion method
            if method == "zpm":
                dvp_a, dvp_b = self.vp_zpm(dd_a_mn, dd_b_mn)
            elif method == "dd":
                dvp_a, dvp_b = self.vp_dd(dd_a_mn, dd_b_mn)
            elif method == "zc":
                dvp_a, dvp_b = self.vp_zc(dd_a_mn, dd_b_mn)
            elif method == "wy_r":
                dvp_a, dvp_b = self.vp_wy_r(dd_a, dd_b)
            elif method == "wy_nm":
                dvp_a, dvp_b = self.wuyangscipy()
 
            #Update vp
            dvp_a = psi4.core.Matrix.from_array(dvp_a)
            dvp_b = psi4.core.Matrix.from_array(dvp_b)
            vp_a.axpy(beta, dvp_a)
            vp_b.axpy(beta, dvp_b)
            self.vp = [vp_a, vp_b]

            #Plot
            if plot_scf is True:
                vp, grid = self.molecule.basis_to_grid(vp_a.np + vp_b.np, blocks=False)
                dvp, _ = self.molecule.basis_to_grid(dvp_a.np + dvp_b.np, blocks=False)
                nmol,  _ = self.molecule.basis_to_grid(self.molecule.Da.np + self.molecule.Db.np, blocks=False)
                nfrag, _ = self.molecule.basis_to_grid(self.frag_da_nm.np + self.frag_db_nm.np, blocks=False)
                x, ys = self.axis_plot("z", [vp, dvp, nmol, nfrag], grid)

                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,5))

                p1 = ax[0][0]
                p2 = ax[1][0]
                p3 = ax[0][1]
                p4 = ax[1][1]

                p1.plot(l1_list, label="L1 Error")
                p1.set_xlabel("SCF Cycle")
                p1.set_ylabel("L1 Density Difference Error")
                p1.legend()

                p2.plot(x, ys[2], label="mol_density")
                p2.plot(x, ys[3], label="frag_density")
                p2.set_xlabel("z")
                p2.set_ylabel("Fragment Density")
                p2.legend()
                p2.set_ylim(-0.01, 0.2)
                p2.set_xlim(-10,10)

                p3.plot(x,ys[0], label="vp")
                #   p3.plot(x,ys[1], label="dvp")
                #p3.plot(x_dvp, y_dvp, label="dvp")
                # p3.plot(vnad_xc_x, vnad_xc_y, label="vnad_xc", linestyle=":")
                # p3.plot(vnad_ha_x, vnad_ha_y, label="vnad_ha", linestyle=":")
                p3.set_xlabel("X")
                p3.set_ylabel("vp")
                p3.set_xlim(-7,7)
                #p3.set_ylim(-1.0,1.0)
                p3.legend()


                p4.plot(x, np.log10(np.abs(ys[2] - ys[3])), label="log10(DD)")
                p4.set_xlabel("X")
                p4.set_ylabel("n_mol - sum(n_i)")
                #p4.set_xlim(-10,10)
                p4.legend()

                fig.tight_layout()
                plt.show()

            if step == maxiter:
                raise Exception("Maximum number of SCF cycles exceeded for vp")


    #Methods for Wu Yang Inversion
    #Notes

    def lagr_WuYang(self, v):

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v[:self.nbf]) + self.initial_guess) 
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v[self.nbf:]) + self.initial_guess)

        self.frags[0].scf(  maxiter=0, 
                            hamiltonian=["kinetic", "external", "xxxtra"], xfock_nm=[Vks_a, Vks_b],
                            get_ingredients=True)

        L = - self.frags[0].T.vector_dot(self.frags[0].Da) - self.frags[0].T.vector_dot(self.frags[0].Db)
        L += - self.frags[0].V.vector_dot(self.frags[0].Da) - self.frags[0].V.vector_dot(self.frags[0].Db) 
        L += - Vks_a.vector_dot(self.frags[0].Da) - Vks_b.vector_dot(self.frags[0].Db) 
        L +=  self.frags[0].V.vector_dot(self.target.Da()) + self.frags[0].V.vector_dot(self.target.Db()) 
        L +=  Vks_a.vector_dot(self.target.Da()) + Vks_b.vector_dot(self.target.Db()) 
        
        return L

    def grad_WuYang(self, v):

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v[:self.nbf]) + self.initial_guess) 
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v[self.nbf:]) + self.initial_guess)

        self.frags[0].scf(  maxiter=0, 
                            hamiltonian=["kinetic", "external", "xxxtra"], xfock_nm=[Vks_a, Vks_b],
                            get_ingredients=True)

        dd_a_mn = self.target.Da().np - self.frags[0].Da.np
        dd_b_mn = self.target.Db().np - self.frags[0].Db.np

        grad_a = contract("uv,uvi->i", dd_a_mn, self.three_overlap)
        grad_b = contract("uv,uvi->i", dd_b_mn, self.three_overlap)
        grad = np.concatenate((grad_a, grad_b))

        return grad
    
    def hess_WuYang(self, v):

        Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v[:self.nbf]) + self.initial_guess)
        Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v[self.nbf:]) + self.initial_guess)

        self.frags[0].scf(  maxiter=0, #Solved non-self consistently
                            hamiltonian=["kinetic", "external", "xxxtra"], xfock_nm=[Vks_a, Vks_b],
                            get_ingredients=True)

        epsilon_occ_a = self.frags[0].eigs_a.np[:self.frags[0].nalpha, None]
        epsilon_occ_b = self.frags[0].eigs_b.np[:self.frags[0].nbeta, None]
        epsilon_unocc_a = self.frags[0].eigs_a.np[self.frags[0].nalpha:]
        epsilon_unocc_b = self.frags[0].eigs_b.np[self.frags[0].nbeta:]
        epsilon_a = epsilon_occ_a - epsilon_unocc_a
        epsilon_b = epsilon_occ_b - epsilon_unocc_b

        hess = np.zeros((self.nbf*2, self.nbf*2))
        # Alpha electrons
        hess[0:self.nbf, 0:self.nbf] = - 1.0 * contract('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                             self.frags[0].Ca.np[:, :self.frags[0].nalpha],
                                                                                             self.frags[0].Ca.np[:, self.frags[0].nalpha:],
                                                                                             self.frags[0].Ca.np[:, :self.frags[0].nalpha],
                                                                                             self.frags[0].Ca.np[:, self.frags[0].nalpha:],
                                                                                             np.reciprocal(epsilon_a), self.three_overlap,
                                                                                             self.three_overlap)
        # Beta electrons
        hess[self.nbf:, self.nbf:] = - 1.0 * contract('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                           self.frags[0].Cb.np[:, :self.frags[0].nbeta],
                                                                                           self.frags[0].Cb.np[:, self.frags[0].nbeta:],
                                                                                           self.frags[0].Cb.np[:, :self.frags[0].nbeta],
                                                                                           self.frags[0].Cb.np[:, self.frags[0].nbeta:],
                                                                                           np.reciprocal(epsilon_b),self.three_overlap,
                                                                                           self.three_overlap)
        hess = (hess + hess.T)

        return hess
    
    def wuyang(self, guess,
                          opt_method="trust-krylov",
                          #opt_method="BFGS" 
                          ):

        self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.frags[0].wfn.basisset(),
                                                                         self.frags[0].wfn.basisset(),
                                                                         self.frags[0].wfn.basisset()))        

        if guess == "fermiamaldi":
            print("Initial Guess: Fermi Amaldi")
            #Molecule is assumed to be the same as target one, i.e. occupied orbitals are the same
            wfn = self.target
            nocc =  wfn.nalpha() + wfn.nbeta()
            Ca = wfn.Ca().np.copy()
            Cb = wfn.Cb().np.copy()
            Cocca = psi4.core.Matrix.from_array(Ca[:,:wfn.nalpha()])
            Coccb = psi4.core.Matrix.from_array(Cb[:,:wfn.nbeta()])

            self.frags[0].scf(maxiter=0)
            self.frags[0].jk.C_left_add(Cocca)
            self.frags[0].jk.C_left_add(Coccb)
            self.frags[0].jk.compute()
            self.frags[0].jk.C_clear()
            self.initial_guess = (nocc-1)/nocc*(self.frags[0].jk.J()[0].np + self.frags[0].jk.J()[1].np)
            self.initial_guess = psi4.core.Matrix.from_array(self.initial_guess)
            self.frags[0].scf(  maxiter=0,
                                hamiltonian=["kinetic", "external", "xxxtra"], 
                                xfock_nm=[self.initial_guess, self.initial_guess],
                                get_ingredients=True)

        dda_grid, _ = self.frags[0].basis_to_grid(self.frags[0].Da - self.target.Da().np, blocks=True)
        ddb_grid, w = self.frags[0].basis_to_grid(self.frags[0].Db - self.target.Db().np, blocks=True)
        print("\n |n-n0|", np.linalg.norm(np.abs(dda_grid + ddb_grid) * w))

        v0 = np.zeros(int(self.nbf)*2)

        vp_array = optimizer.minimize(fun=self.lagr_WuYang, 
                                      x0=v0,
                                      jac=self.grad_WuYang,
                                      hess=self.hess_WuYang,
                                      method=opt_method,
                                      options={'gtol': 1e-8, 'disp': False},
                                      tol=None
                                      )

        Vks_a = contract("ijk,k->ij", self.three_overlap, vp_array.x[:self.nbf]) + self.initial_guess 
        Vks_b = contract("ijk,k->ij", self.three_overlap, vp_array.x[self.nbf:]) + self.initial_guess 
        Vks_a = psi4.core.Matrix.from_array(Vks_a)
        Vks_b = psi4.core.Matrix.from_array(Vks_b)

        self.frags[0].scf(  maxiter=0,
                            hamiltonian=["kinetic", "external", "xxxtra"], xfock_nm=[Vks_a, Vks_b],
                            get_ingredients=True)    
        
        _, _, _, _, _, _, potential = u_xc( self.target.Da(), self.target.Db(), 
                                            self.target.Ca(), self.target.Cb(), 
                                            self.target, self.frags[0].Vpot,
                                            ingredients=True, 
                                            orbitals=False, 
                                            vxc=None)

        target_hartree =              potential["vha"].copy()
        input_hartree = self.frags[0].potential["vha"].copy()

        self.t_hartree= target_hartree
        self.i_hartree = input_hartree

        dda_grid, _ = self.frags[0].basis_to_grid(self.frags[0].Da - self.target.Da().np, blocks=True)
        ddb_grid, w = self.frags[0].basis_to_grid(self.frags[0].Db - self.target.Db().np, blocks=True)
        print("|n-n0|", np.linalg.norm(np.abs(dda_grid + ddb_grid) * w))

        nocc = self.frags[0].nalpha + self.frags[0].nbeta
        vxc_a, _ = self.frags[0].basis_to_grid(vp_array.x[:self.nbf], blocks=True)
        vxc_b, _ = self.frags[0].basis_to_grid(vp_array.x[self.nbf:], blocks=True)
        vxc_a += ((nocc -1)/nocc)*target_hartree - input_hartree
        vxc_b += ((nocc -1)/nocc)*target_hartree  - input_hartree
        self.vxc_a = vxc_a.copy()
        self.vxc_b = vxc_b.copy()


        return







    #Other experimental Inversion methods

    def vp_wy_r(self, dd_a, dd_b):
        """ 
        Performs the Wu-Yang Method on the grid
        """

        dvp = np.zeros_like(self.molecule.Da)
        dd = dd_a + dd_b 

        #Bring grid information
        points_func = self.molecule.Vpot.properties()[0]
        
        #Calculate denominator
        for block in range(self.molecule.Vpot.nblocks()):
            grid_block = self.molecule.Vpot.get_block(block)
            points_func.compute_points(grid_block)
            npoints = grid_block.npoints()
            lpos = np.array(grid_block.functions_local_to_global())
            w = np.array(grid_block.w())
            phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

            x_a = np.zeros((npoints, npoints))
            x_b = np.zeros((npoints, npoints))

            for frag in self.frags:
                orb_a = frag.orbitals["alpha_r"]
                orb_b = frag.orbitals["beta_r"]
                
                for i_occ in range(0,frag.nalpha):
                    for i_vir in range(frag.nalpha, frag.nbf):
                        
                        den = frag.eigs_a.np[i_occ] - frag.eigs_a.np[i_vir]
                        num = np.zeros((npoints, npoints))

                        for r1 in range(npoints):
                            for r2 in range(npoints):
                                num[r1, r2] = orb_a[str(i_occ)][block][r1] * orb_a[str(i_vir)][block][r1] * orb_a[str(i_vir)][block][r2] * orb_a[str(i_occ)][block][r2]        
                        x_a += num / den

                        #Assume x_b = x_a

            dvp_block = np.zeros((npoints))
            for r1 in range(npoints):
                dvp_block += (1 / (x_a[r1, :] + x_a[r1, :])) * dd[block] * w  

            vtmp = contract('pb,p,p,pa->ab', phi, dvp_block, w, phi)
            dvp[(lpos[:, None], lpos)] += 0.5 * (vtmp + vtmp.T)

        return dvp, dvp

    def vp_zpm(self, dd_a, dd_b):
        """
        Performs the Zhao-Morrison-Parr Inversion
        Physical Review A, 50(3):2138, 1994.
        """
        dvp_a = (-0.5) * contract('imlj, ml->ij', self.molecule.mints.ao_eri().np, dd_a)
        dvp_b = (-0.5) * contract('imlj, ml->ij', self.molecule.mints.ao_eri().np, dd_b)
        return dvp_a, dvp_b

    def vp_dd(self, dd_a, dd_b):
        """
        Performs density difference inversion. 
        That is dvp = beta * (dd_a + dd_b)
        """
        dvp_a = dd_a 
        dvp_b = dd_b
        return dvp_a, dvp_b 

    #Methods for Ou Carter Inversion

    def get_epsilon_ks(self, eigs_a0=None, eigs_b0=None):
        """
        Obtains average local electron energy 
        10.1021/acs.jctc.8b00717
        
        From equation 15
        """

        na = self.molecule.nalpha
        nb = self.molecule.nbeta
        orb_a = self.molecule.orbitals["alpha_r"]
        orb_b = self.molecule.orbitals["beta_r"]

        eigs_a = self.molecule.eigs_a.np
        eigs_b = self.molecule.eigs_b.np

        if eigs_a0 is not None:
            #Displace energy according to step 3 of 2.2.3
            energy_displacer_a = eigs_a[self.molecule.nalpha] - eigs_a0[self.molecule.nalpha]
            energy_displacer_b = eigs_b[self.molecule.nalpha] - eigs_b0[self.molecule.nalpha]

            eigs_a += energy_displacer_a
            eigs_b += energy_displacer_b

        da = self.molecule.ingredients["density"]["da"]
        db = self.molecule.ingredients["density"]["db"]

        epsilon_ks_a = []        
        for block in range(self.nblocks):
            num = np.zeros_like(orb_a["0"][block])
            for i_occ in range(na):
                num += eigs_a[:na][i_occ] * np.abs(orb_a[str(i_occ)][block])**2
            num /= da[block]
            epsilon_ks_a.append(num)

        epsilon_ks_b = []        
        for block in range(self.nblocks):
            num = np.zeros_like(orb_b["0"][block])
            for i_occ in range(nb):
                num += eigs_b[:nb][i_occ] * np.abs(orb_b[str(i_occ)][block])**2
            num /= db[block]
            epsilon_ks_b.append(num)
        
        return [np.array(epsilon_ks_a), np.array(epsilon_ks_b)]

    def get_vext_tilde(self):
        """
        Obtains the numerically evaluated external potential solved for via KSDFT
        with any nonhybrid XC functional.
        10.1021/acs.jctc.8b00717
        From equation 22
        """

        e_ks = self.get_epsilon_ks()
        e_ks = e_ks[0] + e_ks[1]
        tau     = self.molecule.ingredients["tau"]["tau_a"] + self.molecule.ingredients["tau"]["tau_a"] 
        density = self.molecule.ingredients["density"]["da"] + self.molecule.ingredients["density"]["db"]
        vha     = self.molecule.potential["vha"]
        vxc     = self.molecule.potential["vxc_a"]

        vext_tilde = e_ks - tau/density - vha - vxc

        return vext_tilde

    def carter_staroverov(self, target_wfn, Vpot, max_iter=50):
        """
        LDA guess density-to-potential inversion
        10.1021/acs.jctc.8b00717

        Parameters
        ----------
        target_wfn : psi4.wfn
            psi4.wfn from target calculation

        Returns
        -------
        vxc_eff : np.array
            "exact" vxc from target density at basis accuracy
        """

        Da_target = target_wfn.Da_subset("AO")
        Db_target = target_wfn.Db_subset("AO")
        Ca_target = target_wfn.Ca_subset("AO", "ALL").np
        Cb_target = target_wfn.Cb_subset("AO", "ALL").np

        _, _, _, ingredients, _, _, potential = u_xc(Da_target, Db_target, Ca_target, Cb_target,
                                                               target_wfn, Vpot, True, True)

        #Ingredients from target system
        n  = ingredients["density"]["da"] + ingredients["density"]["db"]
        g  = (ingredients["gradient"]["da_x"].copy() + ingredients["gradient"]["db_x"].copy())**2
        g += (ingredients["gradient"]["da_y"].copy() + ingredients["gradient"]["db_y"].copy())**2
        g += (ingredients["gradient"]["da_z"].copy() + ingredients["gradient"]["db_z"].copy())**2
        l  = (ingredients["laplacian"]["la_x"].copy() 
            +ingredients["laplacian"]["la_y"].copy() 
            +ingredients["laplacian"]["la_z"].copy())
        l += (ingredients["laplacian"]["lb_x"].copy() 
             +ingredients["laplacian"]["lb_y"].copy() 
             +ingredients["laplacian"]["lb_z"].copy())
        t  = ingredients["tau"]["tau_a"] + ingredients["tau"]["tau_b"]
        vha = potential["vha"]
        vext_tilde = self.get_vext_tilde()

        #Fix orbital energies 
        orb_a = self.molecule.eigs_a.np.copy()
        orb_b = self.molecule.eigs_b.np.copy()

        #Components for Initial guess
        t_ks = self.molecule.ingredients["tau"]["tau_a"].copy() + self.molecule.ingredients["tau"]["tau_b"].copy()
        n_ks = self.molecule.ingredients["density"]["da"].copy() + self.molecule.ingredients["density"]["db"].copy()

        #Shift orbitals energies as Step 3
        epsilon_ks = self.get_epsilon_ks(orb_a, orb_b)
        epsilon_ks = epsilon_ks[0] + epsilon_ks[1]

        #Initial Guess | Equation 23
        vks_eff = 0.25 * l/n - g/(8*np.abs(n)**2) + epsilon_ks - t_ks/n_ks - vext_tilde - vha

        #Plot current vks_eff
        self.molecule.axis_plot_r([vks_eff], xrange=[-8,8])

        for i in range(max_iter):

            #Solve self consistently for new vks_eff
            self.molecule.scf(vks=vks_eff, get_ingredients=True, get_orbitals=True, get_matrices=True)

            #Update compotents for vks_eff
            t_ks = self.molecule.ingredients["tau"]["tau_a"].copy() + self.molecule.ingredients["tau"]["tau_b"].copy()
            n_ks = self.molecule.ingredients["density"]["da"].copy() + self.molecule.ingredients["density"]["db"].copy()

            #Shift orbitals energies as Step 3
            epsilon_ks = self.get_epsilon_ks(orb_a, orb_b)
            epsilon_ks = epsilon_ks[0] + epsilon_ks[1]

            vks_eff = 0.25 * l/n - g/(8*np.abs(n)**2) + epsilon_ks - t_ks/n_ks - vext_tilde - vha

            #Plot current vks_eff
            self.molecule.axis_plot_r([vks_eff], xrange=[-8,8])


