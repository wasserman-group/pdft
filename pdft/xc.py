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

def xc(D, C,
       wfn, Vpot,
       ingredients, orbitals):
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

    potential = { "vxc_a"  : [],
                  "vxc_b"  : [], 
                  "vext"   : [],
                  "vha"    : []}

    grid      = {"x" : [],
                 "y" : [],
                 "z" : [],
                 "w" : []}

    orbitals_a = {}
    orbitals_a_mn = {}
    if orbitals  is True:
        orbitals_a    = { str(i_orb) : [] for i_orb in range(nbf) } 
        orbitals_a_mn = { str(i_orb) : np.zeros((nbf, nbf)) for i_orb in range(nbf) }
    
    points_func = Vpot.properties()[0]
    if ingredients is True:
        points_func.set_ansatz(2)
    func = Vpot.functional()

    e_xc = 0.0

    #Geometry information for Vext
    mol_dict = wfn.molecule().to_schema(dtype='psi4')
    natoms = len(mol_dict["elem"])
    indx = [i for i in range(natoms) if wfn.molecule().charge(i) != 0.0]
    natoms = len(indx)
    #Atomic numbers and Atomic positions
    zs = [mol_dict["elez"][i] for i in indx]
    rs = [wfn.molecule().geometry().np[i] for i in indx]
    
    # First loop over the outer set of blocks
    for b in range(Vpot.nblocks()):
        
        # Obtain general grid information
        block = Vpot.get_block(b)
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())
        w = np.array(block.w())

        #Store Grid
        if ingredients is True:
                    
            x =  np.array(block.x())
            y =  np.array(block.y())
            z =  np.array(block.z())

            grid["x"].append(x)
            grid["y"].append(y)
            grid["z"].append(z)
            grid["w"].append(w)

        #Compute Hartree External
            #External
            vext_block = np.zeros(npoints)
            for atom in range(natoms):
                vext_block += -1.0 * zs[atom] / np.sqrt((x-rs[atom][0])**2 + (y-rs[atom][1])**2 + (z-rs[atom][2])**2)
            potential["vext"].append(vext_block)

            #Esp 
            grid_block = np.array((x,y,z)).T
            grid_block = psi4.core.Matrix.from_array(grid_block)
            esp_block = psi4.core.ESPPropCalc(wfn).compute_esp_over_grid_in_memory(grid_block).np
            potential["vha"].append(-1.0 * esp_block - vext_block)

        #Compute phi/rho
        if points_func.ansatz() >= 0:
            phi   = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
            rho   = np.array(points_func.point_values()["RHO_A"])[:npoints]

            if ingredients is True:
                density["da"].append(rho/2.0)
                density["db"].append(rho/2.0)

        #Compute Orbitals
        if orbitals is True:
            for i_orb in range(nbf):
                Ca_local = C[lpos, i_orb]
                orb_a = contract('m, pm -> p', Ca_local.T, phi)
                orbitals_a[str(i_orb)].append(orb_a)
                orb_a_tmp = contract('pb,p,p,pa->ab', phi, orb_a, w, phi)
                orbitals_a_mn[str(i_orb)][(lpos[:, None], lpos)] += 0.5 * (orb_a_tmp + orb_a_tmp.T)

        #GGA components
        if points_func.ansatz() >= 1:
            phi_x =  np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
            phi_y =  np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
            phi_z =  np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]

            rho_x =  np.array(points_func.point_values()["RHO_AX"])[:npoints]
            rho_y =  np.array(points_func.point_values()["RHO_AY"])[:npoints]
            rho_z =  np.array(points_func.point_values()["RHO_AZ"])[:npoints]
            gamma_aa = np.array(points_func.point_values()["GAMMA_AA"])[:npoints]

            if ingredients is True:
                gradient["da_x"].append(rho_x/2.0)
                gradient["da_y"].append(rho_y/2.0)
                gradient["da_z"].append(rho_z/2.0)
                gradient["db_x"].append(rho_x/2.0)
                gradient["db_y"].append(rho_y/2.0)
                gradient["db_z"].append(rho_z/2.0)
                #Gamma/4 from g_aa, g_bb, g_ab, g_ba
                gamma["g_aa"].append(gamma_aa/4.0)

        #meta components
        if points_func.ansatz() >= 2:
            tau_a = np.array(points_func.point_values()["TAU_A"])[:npoints]
            d_xx = np.array(points_func.point_values()["RHO_XX"])[:npoints]
            d_yy = np.array(points_func.point_values()["RHO_YY"])[:npoints]
            d_zz = np.array(points_func.point_values()["RHO_ZZ"])[:npoints]

            if ingredients is True:
                tau["tau_a"].append(tau_a/2.0)
                tau["tau_b"].append(tau_a/2.0)
                #Laplacian missing for restricted
                laplacian["la_x"].append(d_xx/2.0)
                laplacian["la_y"].append(d_yy/2.0)
                laplacian["la_z"].append(d_zz/2.0)
                laplacian["la_x"].append(d_xx/2.0)
                laplacian["la_y"].append(d_yy/2.0)
                laplacian["la_z"].append(d_zz/2.0)

        #Obtain Kernel
        ret = func.compute_functional(points_func.point_values(), -1)

        #Compute the XC energy
        vk = np.array(ret["V"])[:npoints]
        e_xc += contract("a,a->", w, vk)
        #Compute the XC derivative
        v_rho_a = np.array(ret["V_RHO_A"])[:npoints]  
        v_rho_a_dict = v_rho_a.copy()    
        Vtmp = contract('pb,p,p,pa->ab', phi, v_rho_a, w, phi)

        if func.is_gga() is True:
            v_gamma_aa = np.array(ret["V_GAMMA_AA"])[:npoints]
            Vtmp_gga  = 2.0 * contract('pb,p,p,p,pa->ab', phi_x, v_gamma_aa, rho_x, w, phi)
            Vtmp_gga += 2.0 * contract('pb,p,p,p,pa->ab', phi_y, v_gamma_aa, rho_y, w, phi)
            Vtmp_gga += 2.0 * contract('pb,p,p,p,pa->ab', phi_z, v_gamma_aa, rho_z, w, phi)

            Vtmp += Vtmp_gga
            v_rho_a_dict += contract('pm,mn,pn->p', phi, Vtmp_gga, phi)

        if func.is_meta() is True:
            v_tau_a = np.array(ret["V_TAU_A"])[:npoints]
            Vtmp_meta  = 0.5 * contract( 'pb, p, p, pa -> ab' , phi_x, v_tau_a, w, phi_x)
            Vtmp_meta += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_y, v_tau_a, w, phi_y)
            Vtmp_meta += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_z, v_tau_a, w, phi_z)

            Vtmp += Vtmp_meta
            v_rho_a_dict = contract('pm,mn,pn->p', phi, Vtmp_meta, phi)
        

        potential["vxc_a"].append(v_rho_a_dict)
        potential["vxc_b"].append(v_rho_a_dict)  

        # Sum back to the correct place
        Vnm[(lpos[:, None], lpos)] += 0.5*(Vtmp + Vtmp.T)

    for i_key in potential.keys():
        potential[i_key] = np.array(potential[i_key])

    for i_key in density.keys():
        density[i_key] = np.array(density[i_key])

    for i_key in gradient.keys():
        gradient[i_key] = np.array(gradient[i_key])

    for i_key in laplacian.keys():
        laplacian[i_key] = np.array(laplacian[i_key])

    for i_key in tau.keys():
        tau[i_key] = np.array(tau[i_key])

    for i_key in orbitals_a.keys():
        orbitals_a[i_key] = np.array(orbitals_a[i_key])

    for i_key in gamma.keys():
        gamma[i_key] = np.array(gamma[i_key])

    density_ingredients = {"density"  : density,
                           "gradient" : gradient,
                           "laplacian": laplacian,
                           "gamma"    : gamma,
                           "tau"      : tau,}

    orbital_dictionary = {"alpha_r"    : orbitals_a, 
                          "beta_r"     : orbitals_a,
                          "alpha_mn"   : orbitals_a_mn,
                          "beta_mn"    : orbitals_a_mn}

    return e_xc, Vnm, density_ingredients, orbital_dictionary, grid, potential

def u_xc(D_a, D_b, Ca, Cb, 
        wfn, Vpot,
        ingredients, orbitals):
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

    potential = { "vxc_a"  : [],
                  "vxc_b"  : [], 
                  "vext"   : [],
                  "vha"    : []}

    grid      = {"x" : [],
                 "y" : [],
                 "z" : [],
                 "w" : []}

    orbitals_a   = {}
    orbitals_b   = {}
    orbitals_a_mn  = {}
    orbitals_b_mn  = {}

    if orbitals is True:
        orbitals_a    = { str(i_orb) : [] for i_orb in range(nbf) } 
        orbitals_a_mn = { str(i_orb) : np.zeros((nbf, nbf)) for i_orb in range(nbf) }
        orbitals_b    = { str(i_orb) : [] for i_orb in range(nbf) } 
        orbitals_b_mn = { str(i_orb) : np.zeros((nbf, nbf)) for i_orb in range(nbf) }
    
    total_e = 0.0
    
    points_func = Vpot.properties()[0]
    #if ingredients is True :
    points_func.set_ansatz(2)

    func = Vpot.functional()
    e_xc = 0.0

    #Geometry information for Vext
    mol_dict = wfn.molecule().to_schema(dtype='psi4')
    natoms = len(mol_dict["elem"])
    indx = [i for i in range(natoms) if wfn.molecule().charge(i) != 0.0]
    natoms = len(indx)
    #Atomic numbers and Atomic positions
    zs = [mol_dict["elez"][i] for i in indx]
    rs = [wfn.molecule().geometry().np[i] for i in indx]
    
    # First loop over the outer set of blocks
    for b in range(Vpot.nblocks()):
        
        # Obtain general grid information
        block = Vpot.get_block(b)
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())
        w = np.array(block.w())

        if ingredients is True:
                    
            x =  np.array(block.x())
            y =  np.array(block.y())
            z =  np.array(block.z())

            grid["x"].append(x)
            grid["y"].append(y)
            grid["z"].append(z)
            grid["w"].append(w)

        #Compute Hartree External
            #External
            vext_block = np.zeros(npoints)
            for atom in range(natoms):
                vext_block += -1.0 * zs[atom] / np.sqrt((x-rs[atom][0])**2 + (y-rs[atom][1])**2 + (z-rs[atom][2])**2)
            potential["vext"].append(vext_block)

            #Esp 
            grid_block = np.array((x,y,z)).T
            grid_block = psi4.core.Matrix.from_array(grid_block)
            esp_block = psi4.core.ESPPropCalc(wfn).compute_esp_over_grid_in_memory(grid_block).np
            potential["vha"].append(-1.0 * esp_block - vext_block)

        #Compute phi/rho
        if points_func.ansatz() >= 0:
            phi      = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
            rho_a    = np.array(points_func.point_values()["RHO_A"])[:npoints]
            rho_b    = np.array(points_func.point_values()["RHO_B"])[:npoints]

            if ingredients is True:
                density["da"].append(rho_a)
                density["db"].append(rho_b)

        #Compute Orbitals:
        if orbitals is True:
            for i_orb in range(nbf):
                Ca_local = Ca[lpos, i_orb]
                orb_a = contract('m, pm -> p', Ca_local, phi)
                orbitals_a[str(i_orb)].append(orb_a)
                orb_a_tmp = contract('pb,p,p,pa->ab', phi, orb_a, w, phi)
                orbitals_a_mn[str(i_orb)][(lpos[:, None], lpos)] += 0.5 * (orb_a_tmp + orb_a_tmp.T)

                Cb_local = Cb[lpos, i_orb]
                orb_b = contract('m, pm -> p', Cb_local, phi)
                orbitals_b[str(i_orb)].append(orb_b)
                orb_b_tmp = contract('pb,p,p,pa->ab', phi, orb_b, w, phi)
                orbitals_b_mn[str(i_orb)][(lpos[:, None], lpos)] += 0.5 * (orb_b_tmp + orb_b_tmp.T)

        #GGA components
        if points_func.ansatz() >=1:
            phi_x = np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
            phi_y = np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
            phi_z = np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]

            phi_xx = np.array(points_func.basis_values()["PHI_XX"])[:npoints, :lpos.shape[0]]
            phi_yy = np.array(points_func.basis_values()["PHI_YY"])[:npoints, :lpos.shape[0]]
            phi_zz = np.array(points_func.basis_values()["PHI_ZZ"])[:npoints, :lpos.shape[0]]

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

                Da_reshaped = D_a.np[(lpos[:, None], lpos)]
                Db_reshaped = D_b.np[(lpos[:, None], lpos)]

                #Laplacian
                sandwich  = contract('pm, mn, pn ->p', phi, Da_reshaped, phi_xx)
                sandwich += 2* contract('pm, mn, pn ->p', phi_x, Da_reshaped, phi_x)
                sandwich += contract('pm, mn, pn ->p', phi, Da_reshaped, phi_xx)
                laplacian["la_x"].append(sandwich)

                sandwich  = contract('pm, mn, pn ->p', phi, Da_reshaped, phi_yy)
                sandwich += 2* contract('pm, mn, pn ->p', phi_y, Da_reshaped, phi_y)
                sandwich += contract('pm, mn, pn ->p', phi, Da_reshaped, phi_yy)
                laplacian["la_y"].append(sandwich)

                sandwich  = contract('pm, mn, pn ->p', phi, Da_reshaped, phi_zz)
                sandwich += 2* contract('pm, mn, pn ->p', phi_z, Da_reshaped, phi_z)
                sandwich += contract('pm, mn, pn ->p', phi, Da_reshaped, phi_zz)
                laplacian["la_z"].append(sandwich)

                sandwich  = contract('pm, mn, pn ->p', phi, Db_reshaped, phi_xx)
                sandwich += 2* contract('pm, mn, pn ->p', phi_x, Db_reshaped, phi_x)
                sandwich += contract('pm, mn, pn ->p', phi, Db_reshaped, phi_xx)
                laplacian["lb_x"].append(sandwich)

                sandwich  = contract('pm, mn, pn ->p', phi, Db_reshaped, phi_yy)
                sandwich += 2* contract('pm, mn, pn ->p', phi_y, Db_reshaped, phi_y)
                sandwich += contract('pm, mn, pn ->p', phi, Db_reshaped, phi_yy)
                laplacian["lb_y"].append(sandwich)

                sandwich  = contract('pm, mn, pn ->p', phi, Db_reshaped, phi_zz)
                sandwich += 2* contract('pm, mn, pn ->p', phi_z, Db_reshaped, phi_z)
                sandwich += contract('pm, mn, pn ->p', phi, Db_reshaped, phi_zz)
                laplacian["lb_z"].append(sandwich)

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

            if ingredients is True:
                tau["tau_a"].append(tau_a)
                tau["tau_b"].append(tau_b)

        #Obtain Kernel
        ret = func.compute_functional(points_func.point_values(), -1)
        #Compute the XC energy
        vk = np.array(ret["V"])[:npoints]
        e_xc += contract("a,a->", w, vk)
        #Compute the XC derivative
        v_rho_a = np.array(ret["V_RHO_A"])[:npoints]  
        v_rho_b = np.array(ret["V_RHO_B"])[:npoints]   
        potential["vxc_a"].append(v_rho_a)
        potential["vxc_b"].append(v_rho_b)

        Vtmp_a = contract('pb,p,p,pa->ab', phi, v_rho_a, w, phi)
        Vtmp_b = contract('pb,p,p,pa->ab', phi, v_rho_b, w, phi)

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

            Vtmp_a += contract('pb, p, pa->ab', phi_x, xa, phi)
            Vtmp_a += contract('pb, p, pa->ab', phi_y, ya, phi)
            Vtmp_a += contract('pb, p, pa->ab', phi_z, za, phi)

            Vtmp_b += contract('pb, p, pa->ab', phi_x, xb, phi)
            Vtmp_b += contract('pb, p, pa->ab', phi_y, yb, phi)
            Vtmp_b += contract('pb, p, pa->ab', phi_z, zb, phi)

        if func.is_meta() is True:
            v_tau_a = np.array(ret["V_TAU_A"])[:npoints]
            v_tau_b = np.array(ret["V_TAU_B"])[:npoints]

            Vtmp_a += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_x, v_tau_a, w, phi_x)
            Vtmp_a += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_y, v_tau_a, w, phi_y)
            Vtmp_a += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_z, v_tau_a, w, phi_z)
    
            Vtmp_b += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_x, v_tau_b, w, phi_x)
            Vtmp_b += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_y, v_tau_b, w, phi_y)
            Vtmp_b += 0.5 * contract( 'pb, p, p, pa -> ab' , phi_z, v_tau_b, w, phi_z)

        # Sum back to the correct place
        V_a[(lpos[:, None], lpos)] += 0.5 * (Vtmp_a + Vtmp_a.T)
        V_b[(lpos[:, None], lpos)] += 0.5 * (Vtmp_b + Vtmp_b.T)

    for i_key in potential.keys():
        potential[i_key] = np.array(potential[i_key])

    for i_key in density.keys():
        density[i_key] = np.array(density[i_key])

    for i_key in gradient.keys():
        gradient[i_key] = np.array(gradient[i_key])

    for i_key in laplacian.keys():
        laplacian[i_key] = np.array(laplacian[i_key])

    for i_key in tau.keys():
        tau[i_key] = np.array(tau[i_key])

    for i_key in orbitals_a.keys():
        orbitals_a[i_key] = np.array(orbitals_a[i_key])

    for i_key in orbitals_b.keys():
        orbitals_b[i_key] = np.array(orbitals_b[i_key])

    for i_key in gamma.keys():
        gamma[i_key] = np.array(gamma[i_key])

    dfa_ingredients = {"density"  : density,
                       "gradient" : gradient,
                       "laplacian": laplacian,
                       "gamma"    : gamma,
                       "tau"      : tau}

    orbital_dictionary = {"alpha_r"  : orbitals_a,
                          "beta_r"   : orbitals_b,
                          "alpha_mn" : orbitals_a_mn,
                          "beta_mn"  : orbitals_b_mn}
    
    return e_xc, V_a, V_b, dfa_ingredients, orbital_dictionary, grid, potential