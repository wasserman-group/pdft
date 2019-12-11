"""
tools.py

Different tools for pdft
"""


def basis_to_grid(mol, mat):
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

    points_func = mol.Vpot.properties()[0]
    superfunc = mol.Vpot.functional()

    full_phi, fullx, fully, fullz, fullw, full_mat = [], [], [], [], [], []
    frag_phi, frag_w, frag_mat, frag_pos = [],[],[],[]

    # Loop Over Blocks
    for l_block in range(mol.Vpot.nblocks()):

        # Obtain general grid information
        l_grid = mol.Vpot.get_block(l_block)

        l_w = np.array(l_grid.w())
        frag_w.append(l_w)
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())

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
        frag_phi.append(phi)
        
        l_mat = mat[(lpos[:, None], lpos)]
        
        mat_r = np.einsum('pm,mn,pn->p', phi, l_mat, phi, optimize=True)
        frag_mat.append(mat_r[:l_npoints])

        for i in range(len(mat_r)):
            full_mat.append(mat_r[i])
            
        for i in range(len(phi)):
            full_phi.append(phi[i])
            
    x, y, z, w = np.array(fullx), np.array(fully), np.array(fullz), np.array(fullw)
    full_mat = np.array(full_mat)
    full_phi = np.array(full_phi)
        
#    return [x,y,z,w], phi_plot, mat_plot, 
#    return frag_mat, frag_w, frag_phi, frag_pos, full_phi, full_mat
    return frag_phi, frag_pos, full_mat



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
    
    for block in range(len(phi)):
        appended = np.zeros((len(phi[block]), nbf**2))
        for points in range(len(phi[block])):
            appendelements = np.zeros((1, nbf))
            appendelements[0, frag_pos[block]] = phi[block][points,:]
            appended[points, :] = np.squeeze((appendelements.T.dot(appendelements)).reshape(nbf ** 2, 1))
        appended = appended.reshape(len(phi[block]), nbf ** 2)
        basis_grid_matrix = np.append(basis_grid_matrix, appended, axis=0)
            
    mat = np.linalg.lstsq(np.array(basis_grid_matrix), f, rcond=-1e-16)
    mat = mat[0].reshape(nbf, nbf)
    mat = 0.5 * (mat + mat.T)
    
    return mat


    def get_orbitals(mol, C, ndocc):
        """
        Gets orbitals on the grid

        Parameters
        ----------
        mol: pdft.Molecule

        C: Psi4.core.Matrix
            Molecular orbitals coefficient matrix
        
        ndocc: int
            Number of occupied orbitals
        """
    orbitals = []
    for i in range(ndocc):
        orb_i = np.zeros_like(C)
        orb_i[:,i] = C.np[:,i]
        phi, pos, orb_g = basis_to_grid(mol, orb_i)
        orbitals.append(orb_g)
        