import psi4
import numpy as np
from opt_einsum import contract


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

def xc(D, Vpot, ingredients):
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
    if ingredients is True:
        points_func.set_ansatz(2)

    dfa_ingredients = {"d"   : [], 
                       "d_x" : [],
                       "d_y" : [],
                       "d_z" : [],
                       "d_xx" : [],
                       "d_yy" : [],
                       "d_zz" : [], 
                       "gamma" : [],
                       "tau" : [], 
                       "vxc" : []}

    grid = {"x" : [], "y" : [], "z" : [], "w" : []}
    func = Vpot.functional()

    e_xc = 0.0
    
    # First loop over the outer set of blocks
    for b in range(Vpot.nblocks()):
        
        # Obtain general grid information
        block = Vpot.get_block(b)
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())
        grid["x"].append(np.array(block.x()))
        grid["y"].append(np.array(block.y()))
        grid["z"].append(np.array(block.z()))
        w = np.array(block.w())
        grid["w"].append(w)

        #Compute phi/rho
        if points_func.ansatz() >= 0:
            phi   = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
            rho   = np.array(points_func.point_values()["RHO_A"])[:npoints]

            if ingredients is True:
                dfa_ingredients["d"].append(rho)
                
        #GGA components
        if points_func.ansatz() >= 1:
            phi_x =  np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
            phi_y =  np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
            phi_z =  np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]

            rho_x =  np.array(points_func.point_values()["RHO_AX"])[:npoints]
            rho_y =  np.array(points_func.point_values()["RHO_AY"])[:npoints]
            rho_z =  np.array(points_func.point_values()["RHO_AZ"])[:npoints]
            gamma = np.array(points_func.point_values()["GAMMA_AA"])[:npoints]

            if ingredients is True:
                dfa_ingredients["d_x"].append(rho_x)
                dfa_ingredients["d_y"].append(rho_y)
                dfa_ingredients["d_z"].append(rho_z)
                dfa_ingredients["gamma"].append(gamma)

        #meta components
        if points_func.ansatz() >= 2:
            tau = np.array(points_func.point_values()["TAU_A"])[:npoints]
            d_xx = np.array(points_func.point_values()["RHO_XX"])[:npoints]
            d_yy = np.array(points_func.point_values()["RHO_YY"])[:npoints]
            d_zz = np.array(points_func.point_values()["RHO_ZZ"])[:npoints]

            if ingredients is True:
                dfa_ingredients["tau"].append(tau)
                dfa_ingredients["d_xx"].append(d_xx)
                dfa_ingredients["d_yy"].append(d_yy)
                dfa_ingredients["d_zz"].append(d_zz)

        #Obtain Kernel
        ret = func.compute_functional(points_func.point_values(), -1)

        #Compute the XC energy
        vk = np.array(ret["V"])[:npoints]
        dfa_ingredients["vxc"].append(vk)
        e_xc += contract("a,a->", w, vk, optimize=True)
        #Compute the XC derivative
        v_rho_a = np.array(ret["V_RHO_A"])[:npoints]        
        Vtmp = contract('pb,p,p,pa->ab', phi, v_rho_a, w, phi, optimize=True)

        if func.is_gga() is True:
            v_gamma_aa = np.array(ret["V_GAMMA_AA"])[:npoints]
            Vtmp += 2.0 * contract('pb,p,p,p,pa->ab', phi_x, v_gamma_aa, rho_x, w, phi, optimize=True)
            Vtmp += 2.0 * contract('pb,p,p,p,pa->ab', phi_y, v_gamma_aa, rho_y, w, phi, optimize=True)
            Vtmp += 2.0 * contract('pb,p,p,p,pa->ab', phi_z, v_gamma_aa, rho_z, w, phi, optimize=True)

        if func.is_meta() is True:
            v_tau_a = np.array(ret["V_TAU_A"])[:npoints]
            Vtmp += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_x, v_tau_a, w, phi_x, optimize=True)
            Vtmp += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_y, v_tau_a, w, phi_y, optimize=True)
            Vtmp += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_z, v_tau_a, w, phi_z, optimize=True)

        # Sum back to the correct place
        Vnm[(lpos[:, None], lpos)] += 0.5*(Vtmp + Vtmp.T)


    return e_xc, Vnm, dfa_ingredients, grid

def u_xc(D_a, D_b, Ca, Cb, Vpot, ingredients):
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

    density   = {"da" : [],
                 "db" : []}

    gradient  = {"da_x" : [], 
                 "da_y" : [],
                 "da_z" : [],
                 "db_x" : [],
                 "db_y" : [],
                 "db_z" : []}

    laplacian = {"la_x" : [],
                 "la_y" : [],
                 "la_z" : [],
                 "lb_x" : [],
                 "lb_y" : [],
                 "lb_z" : []}

    gamma    =  {"g_aa" : [],
                 "g_ab" : [],
                 "g_bb" : []}

    tau       = {"tau_a" : [],
                 "tau_b" : []}

    vxc       = {"vxc" : []}

    grid      = {"x" : [],
                 "y" : [],
                 "z" : [],
                 "w" : []}

    orbitals_a   = {}
    orbitals_b   = {}
    orbitals_a_nm  = {}
    orbitals_b_nm  = {}

    orb_a_tmp = []
    orb_b_tmp = []

    for orb_j in range(nbf):
        orbitals_a[str(orb_j)]  = []
        orbitals_b[str(orb_j)]  = []
        orbitals_a_nm[str(orb_j)] = np.zeros((nbf, nbf))
        orbitals_b_nm[str(orb_j)] = np.zeros((nbf, nbf))

    
    total_e = 0.0
    
    points_func = Vpot.properties()[0]
    if ingredients is True:
        points_func.set_ansatz(2)

    func = Vpot.functional()

    e_xc = 0.0
    
    # First loop over the outer set of blocks
    for b in range(Vpot.nblocks()):
        
        # Obtain general grid information
        block = Vpot.get_block(b)
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())

        grid["x"].append(np.array(block.x()))
        grid["y"].append(np.array(block.y()))
        grid["z"].append(np.array(block.z()))
        w = np.array(block.w())
        grid["w"].append(w)

        #Compute phi/rho
        if points_func.ansatz() >= 0:
            phi     = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
            rho_a   = np.array(points_func.point_values()["RHO_A"])[:npoints]
            rho_b   = np.array(points_func.point_values()["RHO_B"])[:npoints]

            #for orb_i in range(len(Ca.np))
            #dfa_ingredients["phi"].append(phi)

            if ingredients is True:
                density["da"].append(rho_a)
                density["db"].append(rho_b)

        #GGA components
        if points_func.ansatz() >=1:
            phi_x = np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
            phi_y = np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
            phi_z = np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]

            phi_xx = np.array(points_func.basis_values()["PHI_XX"])[:npoints, :lpos.shape[0]]
            phi_yy = np.array(points_func.basis_values()["PHI_YY"])[:npoints, :lpos.shape[0]]
            phi_zz = np.array(points_func.basis_values()["PHI_ZZ"])[:npoints, :lpos.shape[0]]

            Da_reshaped = D_a.np[(lpos[:, None], lpos)]
            Db_reshaped = D_b.np[(lpos[:, None], lpos)]

            #Laplacian

            ###Laplacian with P4 matrices

            #phi_p4 = psi4.core.Matrix.from_array(phi)

            #phi_x_p4 = psi4.core.Matrix.from_array(phi_x)
            #phi_y_p4 = psi4.core.Matrix.from_array(phi_y)
            #phi_z_p4 = psi4.core.Matrix.from_array(phi_z)

            #phi_xx_p4 = psi4.core.Matrix.from_array(phi_xx)
            #phi_yy_p4 = psi4.core.Matrix.from_array(phi_yy)
            #phi_zz_p4 = psi4.core.Matrix.from_array(phi_zz)

            #Da_p4 = psi4.core.Matrix.from_array(Da_reshaped)
            #Db_p4 = psi4.core.Matrix.from_array(Db_reshaped)

            sandwich  = contract('pm, mn, pn ->p', phi, Da_reshaped, phi_xx, optimize=True)
            sandwich += 2* contract('pm, mn, pn ->p', phi_x, Da_reshaped, phi_x, optimize=True)
            sandwich += contract('pm, mn, pn ->p', phi, Da_reshaped, phi_xx, optimize=True)
            laplacian["la_x"].append(sandwich)

            sandwich  = contract('pm, mn, pn ->p', phi, Da_reshaped, phi_yy, optimize=True)
            sandwich += 2* contract('pm, mn, pn ->p', phi_y, Da_reshaped, phi_y, optimize=True)
            sandwich += contract('pm, mn, pn ->p', phi, Da_reshaped, phi_yy, optimize=True)
            laplacian["la_y"].append(sandwich)

            sandwich  = contract('pm, mn, pn ->p', phi, Da_reshaped, phi_zz, optimize=True)
            sandwich += 2* contract('pm, mn, pn ->p', phi_z, Da_reshaped, phi_z, optimize=True)
            sandwich += contract('pm, mn, pn ->p', phi, Da_reshaped, phi_zz, optimize=True)
            laplacian["la_z"].append(sandwich)

            sandwich  = contract('pm, mn, pn ->p', phi, Db_reshaped, phi_xx, optimize=True)
            sandwich += 2* contract('pm, mn, pn ->p', phi_x, Db_reshaped, phi_x, optimize=True)
            sandwich += contract('pm, mn, pn ->p', phi, Db_reshaped, phi_xx, optimize=True)
            laplacian["lb_x"].append(sandwich)

            sandwich  = contract('pm, mn, pn ->p', phi, Db_reshaped, phi_yy, optimize=True)
            sandwich += 2* contract('pm, mn, pn ->p', phi_y, Db_reshaped, phi_y, optimize=True)
            sandwich += contract('pm, mn, pn ->p', phi, Db_reshaped, phi_yy, optimize=True)
            laplacian["lb_y"].append(sandwich)

            sandwich  = contract('pm, mn, pn ->p', phi, Db_reshaped, phi_zz, optimize=True)
            sandwich += 2* contract('pm, mn, pn ->p', phi_z, Db_reshaped, phi_z, optimize=True)
            sandwich += contract('pm, mn, pn ->p', phi, Db_reshaped, phi_zz, optimize=True)
            laplacian["lb_z"].append(sandwich)

            # dfa_ingredients["l_ax"].append(contract('pm, mn, pn ->p', phi + 2 * phi_x + phi, Da_reshaped,  phi_xx + phi_x + phi_xx, optimize=True))
            # dfa_ingredients["l_ay"].append(contract('pm, mn, pn ->p', phi + 2 * phi_y + phi, Da_reshaped,  phi_yy + phi_x + phi_yy, optimize=True))
            # dfa_ingredients["l_az"].append(contract('pm, mn, pn ->p', phi + 2 * phi_z + phi, Da_reshaped,  phi_zz + phi_x + phi_zz, optimize=True))

            # dfa_ingredients["l_bx"].append(contract('pm, mn, pn ->p', phi + 2 * phi_x + phi, Db_reshaped,  phi_xx + phi_x + phi_xx, optimize=True))
            # dfa_ingredients["l_by"].append(contract('pm, mn, pn ->p', phi + 2 * phi_y + phi, Db_reshaped,  phi_yy + phi_x + phi_yy, optimize=True))
            # dfa_ingredients["l_bz"].append(contract('pm, mn, pn ->p', phi + 2 * phi_z + phi, Db_reshaped,  phi_zz + phi_x + phi_zz, optimize=True))

            rho_ax = np.array(points_func.point_values()["RHO_AX"])[:npoints]
            rho_ay = np.array(points_func.point_values()["RHO_AY"])[:npoints]
            rho_az = np.array(points_func.point_values()["RHO_AZ"])[:npoints]
            rho_bx = np.array(points_func.point_values()["RHO_BX"])[:npoints]
            rho_by = np.array(points_func.point_values()["RHO_BY"])[:npoints]
            rho_bz = np.array(points_func.point_values()["RHO_BZ"])[:npoints]

            gamma_aa = np.array(points_func.point_values()["GAMMA_AA"])[:npoints]
            gamma_ab = np.array(points_func.point_values()["GAMMA_AB"])[:npoints]
            gamma_bb = np.array(points_func.point_values()["GAMMA_BB"])[:npoints]

            if ingredients is True:
                gradient["da_x"].append(rho_ax)
                gradient["da_y"].append(rho_ay)
                gradient["da_z"].append(rho_az)
                gradient["db_x"].append(rho_bx)
                gradient["db_y"].append(rho_by)
                gradient["db_z"].append(rho_bz)
                gamma["g_aa"].append(gamma_aa)
                gamma["g_ab"].append(gamma_ab) 
                gamma["g_bb"].append(gamma_bb)

        #meta components
        if points_func.ansatz() >= 2:
            tau_a = np.array(points_func.point_values()["TAU_A"])[:npoints]
            tau_b = np.array(points_func.point_values()["TAU_B"])[:npoints]

            #lap_a = np.array(points_func.point_values()["LAPL_RHO_A"])[:npoints]
            #lap_b = np.array(points_func.point_values()["LAPL_RHO_B"])[:npoints]

            if ingredients is True:
                tau["tau_a"].append(tau_a)
                tau["tau_b"].append(tau_b)

        #Obtain Kernel
        ret = func.compute_functional(points_func.point_values(), -1)

        #Compute the XC energy
        vk = np.array(ret["V"])[:npoints]
        vxc["vxc"].append(vk)
        e_xc += contract("a,a->", w, vk, optimize=True)
        #Compute the XC derivative
        v_rho_a = np.array(ret["V_RHO_A"])[:npoints]  
        v_rho_b = np.array(ret["V_RHO_B"])[:npoints]   

        Vtmp_a = 1.0 * contract('pb,p,p,pa->ab', phi, v_rho_a, w, phi, optimize=True)
        Vtmp_b = 1.0 * contract('pb,p,p,pa->ab', phi, v_rho_b, w, phi, optimize=True)

        #Compute orbitals
        for i_orb in range(nbf):

            orb_a = contract('nm,pm->np', Ca.np[None,i_orb], phi, optimize=True)[0,:]
            orb_b = contract('nm,pm->np', Cb.np[None,i_orb], phi, optimize=True)[0,:]

            orbitals_a[str(i_orb)].append(orb_a)
            orbitals_b[str(i_orb)].append(orb_b)

            orb_a_tmp.append(1.0 * contract('pb,p,p,pa->ab', phi, orb_a, w, phi, optimize=True))
            orb_b_tmp.append(1.0 * contract('pb,p,p,pa->ab', phi, orb_b, w, phi, optimize=True))

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

            Vtmp_a += contract('pb, p, pa->ab', phi_x, xa, phi, optimize=True)
            Vtmp_a += contract('pb, p, pa->ab', phi_y, ya, phi, optimize=True)
            Vtmp_a += contract('pb, p, pa->ab', phi_z, za, phi, optimize=True)

            Vtmp_b += contract('pb, p, pa->ab', phi_x, xb, phi, optimize=True)
            Vtmp_b += contract('pb, p, pa->ab', phi_y, yb, phi, optimize=True)
            Vtmp_b += contract('pb, p, pa->ab', phi_z, zb, phi, optimize=True)

        if func.is_meta() is True:
            v_tau_a = np.array(ret["V_TAU_A"])[:npoints]
            v_tau_b = np.array(ret["V_TAU_B"])[:npoints]

            Vtmp_a += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_x, v_tau_a, w, phi_x, optimize=True)
            Vtmp_a += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_y, v_tau_a, w, phi_y, optimize=True)
            Vtmp_a += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_z, v_tau_a, w, phi_z, optimize=True)
    
            Vtmp_b += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_x, v_tau_b, w, phi_x, optimize=True)
            Vtmp_b += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_y, v_tau_b, w, phi_y, optimize=True)
            Vtmp_b += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_z, v_tau_b, w, phi_z, optimize=True)

        # Sum back to the correct place
        V_a[(lpos[:, None], lpos)] += 0.5 * (Vtmp_a + Vtmp_a.T)
        V_b[(lpos[:, None], lpos)] += 0.5 * (Vtmp_b + Vtmp_b.T)

        for orb_j in range(nbf):
            orbitals_a_nm[str(orb_j)][(lpos[:, None], lpos)] += 0.5 * (orb_a_tmp[orb_j] + orb_a_tmp[orb_j].T)
            orbitals_b_nm[str(orb_j)][(lpos[:, None], lpos)] += 0.5 * (orb_b_tmp[orb_j] + orb_b_tmp[orb_j].T)


    dfa_ingredients = {"density"  : density,
                       "gradient" : gradient,
                       "laplacian": laplacian,
                       "gamma"    : gamma,
                       "tau"      : tau,
                       "vxc"      : vxc,
                       "grid"     : grid,
                       "orbitals_a" : orbitals_a,
                       "orbitals_b" : orbitals_b,
                       "orbitals_a_nm" : orbitals_a_nm,
                       "orbitals_b_nm" : orbitals_b_nm}

    return e_xc, V_a, V_b, dfa_ingredients, grid