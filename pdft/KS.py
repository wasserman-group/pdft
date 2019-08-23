"""
KS.py
Gets KS quantities for a unrestricted KS-DFT Calculation
Defines a fragment class based on a Psi4 geometry calculation. 
"""

import psi4
import numpy as np
import pandas as pd


class KS:
    def __init__(self, wfn):
        self.wfn = wfn
        self.Vpot = wfn.V_potential()
        self.Da = np.array(wfn.Da())
        self.Db = np.array(wfn.Db())
    


    def xc(self, functional='lsda'):
        """
        Returns xc energy

        Parameters
        ----------
        KS: Wavefunction object
        functional: string, default: True
            XC functiona. 
        
        Retruns
        -------
        xc_e = float
            Exchange correlation energy for given functional
        """

        if functional == 'lsda':

            D = np.array(self.Da)
            V = np.zeros_like(D)
            Vpot = self.Vpot
            points_func = Vpot.properties()[0]
            superfunc = Vpot.functional()
            xc_e = 0.0

            for b in range(Vpot.nblocks()):
        
                # Obtain block information
                block = Vpot.get_block(b)
                points_func.compute_points(block)
                npoints = block.npoints()
                lpos = np.array(block.functions_local_to_global())
                
                # Obtain the grid weight
                w = np.array(block.w())

                # Compute phi!
                phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
                
                # Build a local slice of D
                lD = D[(lpos[:, None], lpos)]
                
                # Copmute rho
                rho = 2.0 * np.einsum('pm,mn,pn->p', phi, lD, phi)

                inp = {}
                inp["RHO_A"] = psi4.core.Vector.from_array(rho)
                
                # Compute the kernel
                ret = superfunc.compute_functional(inp, -1)
                
                # Compute the XC energy
                vk = np.array(ret["V"])[:npoints]
                xc_e += np.einsum('a,a->', w, vk)
                    
                # Compute the XC derivative.
                v_rho_a = np.array(ret["V_RHO_A"])[:npoints]
                Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho_a, w, phi)

                # Add the temporary back to the larger array by indexing, ensure it is symmetric
                V[(lpos[:, None], lpos)] += 0.5 * (Vtmp + Vtmp.T)

        return xc_e


    def kinetic(self, spin):
        """
        Returns kinetic matrix and kinetic energy

        Parameters
        ----------
        KS: Wavefunction object
        spin: string, "alpha"/"beta"
            alpha or beta spin 

        
        Returns
        -------
        Kinetic matrix 
        Kinetic energy
        """
        mints = psi4.core.MintsHelper(self.wfn.basisset())
        S = np.asarray(mints.ao_overlap())

        I = np.asarray(mints.ao_eri())
        T = np.asarray(mints.ao_kinetic())
        
        if spin == 'alpha':
            D = self.Da
            kinetic_energy = np.einsum('pq,pq->', T, D)


        if spin == 'beta':
            D = self.Db
            kinetic_energy = np.einsum('pq,pq->', T, D)

        return kinetic_energy, T

    def external(self, spin):
        """
        Returns potential matrix and potential energy

        Parameters
        ----------
        KS: Wavefunction object
        spin: string, "alpha"/"beta"
            alpha or beta spin 

        
        Returns
        -------
        Potential matrix 
        Potential energy

        """
        mints = psi4.core.MintsHelper(self.wfn.basisset())
        S = np.asarray(mints.ao_overlap())

        I = np.asarray(mints.ao_eri())
        V = np.asarray(mints.ao_potential())
        
        if spin == 'alpha':
            D = self.Da
            potential_energy = np.einsum('pq,pq->', V, D)


        if spin == 'beta':
            D = self.Db
            potential_energy = np.einsum('pq,pq->', V, D)

        return potential_energy, V
    

    def coulomb(self, spin):
        """
        Returns coulomb matrix and coulomb energy. 
        Requires global Psi4 option:
        'save_jk' set to True

        Parameters
        ----------
        KS: Wavefunction object
        spin: string, "alpha"/"beta"
            alpha or beta spin 

        
        Returns
        -------
        Coulomb matrix 
        Coulomb energy

        """
        J = self.wfn.jk().J()[0]
        
        if spin == 'alpha':
            D = self.Da
            coulomb_energy = np.einsum('pq,pq->', J, D)


        if spin == 'beta':
            D = self.Db
            coulomb_energy = np.einsum('pq,pq->', J, D)

        return coulomb_energy, J






