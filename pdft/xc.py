import psi4
import numpy as np


def functional_factory(method, restricted, deriv=1, points=500000):
    """
    Obtains and initializes Superfunctional object

    Parameters
    ----------
    
    method: str
        Density Functional, from libxc or psi4
    
    restricted: bool
        Restricted/Unrestricted functional

    deriv: int
        sets required derivatives

    points: int
        max number of points in grid

    Returns
    -------

    functional: psi4.core.SuperFunctional
        requested density functional
    """

    method = method.lower()

    dict_functionals = {}
    dict_functionals.update(psi4.driver.proc.dft.libxc_functionals.functional_list)
    dict_functionals.update(psi4.driver.proc.dft.lda_functionals.functional_list)
    dict_functionals.update(psi4.driver.proc.dft.gga_functionals.functional_list)
    dict_functionals.update(psi4.driver.proc.dft.mgga_functionals.functional_list)
    dict_functionals.update(psi4.driver.proc.dft.hyb_functionals.functional_list)
    dict_functionals.update(psi4.driver.proc.dft.dh_functionals.functional_list)
    
    functional = psi4.driver.dft.build_superfunctional_from_dictionary(dict_functionals[method], points, deriv, restricted)
    functional[0].allocate()
    
    return functional[0]

def xc(D, Vpot):
    """
    Calculates the exchange correlation energy and exchange correlation
    potential to be added to the KS matrix for a restricted calculation

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
    Vnm = np.zeros((nbf, nbf))
    
    total_e = 0.0
    
    points_func = Vpot.properties()[0]
    func = Vpot.functional()

    e_xc = 0.0
    
    # First loop over the outer set of blocks
    for b in range(Vpot.nblocks()):
        
        # Obtain general grid information
        block = Vpot.get_block(b)
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())
        w = np.array(block.w())

        #Compute phi/rho
        phi   = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
        #rho   = np.array(points_func.point_values()["RHO_A"])[:npoints]

        #GGA components
        if func.is_gga() is True:
            phi_x =  np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
            phi_y =  np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
            phi_z =  np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]

            rho_x =  np.array(points_func.point_values()["RHO_AX"])[:npoints]
            rho_y =  np.array(points_func.point_values()["RHO_AY"])[:npoints]
            rho_z =  np.array(points_func.point_values()["RHO_AZ"])[:npoints]
            gamma = np.array(points_func.point_values()["GAMMA_AA"])[:npoints]

        #meta components
        if func.is_meta() is True:
            tau = np.array(points_func.point_values()["TAU_A"])[:npoints]

        #Obtain Kernel
        ret = func.compute_functional(points_func.point_values(), -1)

        #Compute the XC energy
        vk = np.array(ret["V"])[:npoints]
        e_xc += np.einsum("a,a->", w, vk, optimize=True)
        #Compute the XC derivative
        v_rho_a = np.array(ret["V_RHO_A"])[:npoints]        
        Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho_a, w, phi, optimize=True)

        if func.is_gga() is True:
            v_gamma_aa = np.array(ret["V_GAMMA_AA"])[:npoints]
            Vtmp += 2.0 *np.einsum('pb,p,p,p,pa->ab', phi_x, v_gamma_aa, rho_x, w, phi, optimize=True)
            Vtmp += 2.0 *np.einsum('pb,p,p,p,pa->ab', phi_y, v_gamma_aa, rho_y, w, phi, optimize=True)
            Vtmp += 2.0 *np.einsum('pb,p,p,p,pa->ab', phi_z, v_gamma_aa, rho_z, w, phi, optimize=True)

        if func.is_meta() is True:
            v_tau_a = np.array(ret["V_TAU_A"])[:npoints]
            Vtmp += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_x, v_tau_a, w, phi_x, optimize=True)
            Vtmp += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_y, v_tau_a, w, phi_y, optimize=True)
            Vtmp += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_z, v_tau_a, w, phi_z, optimize=True)

        # Sum back to the correct place
        Vnm[(lpos[:, None], lpos)] += 0.5*(Vtmp + Vtmp.T)


    return e_xc, Vnm

def u_xc(D_a, D_b, Vpot):
    """
    Calculates the exchange correlation energy and exchange correlation
    potential to be added to the KS matrix for an unrestricted calculation

    Parameters
    ----------
    D_a: psi4.core.Matrix
        Alpha one-particle density matrix

    D_b: psi4.core.Matrix
        Beta one-particle density matrix
    
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

    nbf = D_a.shape[0]    
    V_a = np.zeros((nbf, nbf))
    V_b = np.zeros((nbf, nbf))
    
    total_e = 0.0
    
    points_func = Vpot.properties()[0]
    func = Vpot.functional()

    e_xc = 0.0
    
    # First loop over the outer set of blocks
    for b in range(Vpot.nblocks()):
        
        # Obtain general grid information
        block = Vpot.get_block(b)
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())
        w = np.array(block.w())

        #Compute phi/rho
        phi   = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
        rho_a   = np.array(points_func.point_values()["RHO_A"])[:npoints]
        rho_b   = np.array(points_func.point_values()["RHO_B"])[:npoints]

        #GGA components
        if func.is_gga() is True:
            phi_x = np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
            phi_y = np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
            phi_z = np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]

            rho_ax = np.array(points_func.point_values()["RHO_AX"])[:npoints]
            rho_ay = np.array(points_func.point_values()["RHO_AY"])[:npoints]
            rho_az = np.array(points_func.point_values()["RHO_AZ"])[:npoints]
            rho_bx = np.array(points_func.point_values()["RHO_BX"])[:npoints]
            rho_by = np.array(points_func.point_values()["RHO_BY"])[:npoints]
            rho_bz = np.array(points_func.point_values()["RHO_BZ"])[:npoints]

            gamma_aa = np.array(points_func.point_values()["GAMMA_AA"])[:npoints]
            gamma_ab = np.array(points_func.point_values()["GAMMA_AB"])[:npoints]
            gamma_bb = np.array(points_func.point_values()["GAMMA_BB"])[:npoints]

        #meta components
        if func.is_meta() is True:
            tau_a = np.array(points_func.point_values()["TAU_A"])[:npoints]
            tau_b = np.array(points_func.point_values()["TAU_B"])[:npoints]

        #Obtain Kernel
        ret = func.compute_functional(points_func.point_values(), -1)

        #Compute the XC energy
        vk = np.array(ret["V"])[:npoints]   
        e_xc += np.einsum("a,a->", w, vk, optimize=True)
        #Compute the XC derivative
        v_rho_a = np.array(ret["V_RHO_A"])[:npoints]  
        v_rho_b = np.array(ret["V_RHO_B"])[:npoints]   

        Vtmp_a = 1.0 * np.einsum('pb,p,p,pa->ab', phi, v_rho_a, w, phi, optimize=True)
        Vtmp_b = 1.0 * np.einsum('pb,p,p,pa->ab', phi, v_rho_b, w, phi, optimize=True)

        if func.is_gga() is True:

            v_gamma_aa = np.array(ret["V_GAMMA_AA"])[:npoints]
            v_gamma_ab = np.array(ret["V_GAMMA_AB"])[:npoints]
            v_gamma_bb = np.array(ret["V_GAMMA_BB"])[:npoints]

            xa = 2.0 * w * (v_gamma_aa * rho_ax + v_gamma_ab * rho_bx)
            ya = 2.0 * w * (v_gamma_aa * rho_ay + v_gamma_ab * rho_by)
            za = 2.0 * w * (v_gamma_aa * rho_az + v_gamma_ab * rho_bz)

            xb = 2.0 * w * (v_gamma_bb * rho_bx + v_gamma_ab * rho_ax)
            yb = 2.0 * w * (v_gamma_bb * rho_by + v_gamma_ab * rho_ay)
            zb = 2.0 * w * (v_gamma_bb * rho_bz + v_gamma_ab * rho_az)

            Vtmp_a += np.einsum('pb, p, pa->ab', phi_x, xa, phi, optimize=True)
            Vtmp_a += np.einsum('pb, p, pa->ab', phi_y, ya, phi, optimize=True)
            Vtmp_a += np.einsum('pb, p, pa->ab', phi_z, za, phi, optimize=True)

            Vtmp_b += np.einsum('pb, p, pa->ab', phi_x, xb, phi, optimize=True)
            Vtmp_b += np.einsum('pb, p, pa->ab', phi_y, yb, phi, optimize=True)
            Vtmp_b += np.einsum('pb, p, pa->ab', phi_z, zb, phi, optimize=True)


        if func.is_meta() is True:
            v_tau_a = np.array(ret["V_TAU_A"])[:npoints]
            v_tau_b = np.array(ret["V_TAU_B"])[:npoints]

            Vtmp_a += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_x, v_tau_a, w, phi_x, optimize=True)
            Vtmp_a += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_y, v_tau_a, w, phi_y, optimize=True)
            Vtmp_a += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_z, v_tau_a, w, phi_z, optimize=True)
    
            Vtmp_b += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_x, v_tau_b, w, phi_x, optimize=True)
            Vtmp_b += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_y, v_tau_b, w, phi_y, optimize=True)
            Vtmp_b += 0.5 * np.einsum( 'pb, p, p, pa -> ab' , phi_z, v_tau_b, w, phi_z, optimize=True)

        # Sum back to the correct place
        V_a[(lpos[:, None], lpos)] += 0.5 * (Vtmp_a + Vtmp_a.T)
        V_b[(lpos[:, None], lpos)] += 0.5 * (Vtmp_b + Vtmp_b.T)


    return e_xc, V_a, V_b