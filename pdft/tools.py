"""
tools.py

Different tools for pdft.
"""

import numpy as np
import psi4
import pdft.cubeprop
from opt_einsum import contract


def basis_to_spherical_grid(coeff, Vpot):
    """
    Turns a basis represented function to its representation on Lebedev-Laikov spherical quadratures grid.

    Parameters
    ----------
    coeff: Numpy array
        basis representation of the function

    Vpot: psi4.core.VBase
        psi4 V_potential(), contains the information of the spherical grid.

    Returns
    -------
    value_on_grid = List
        Values on the grid.

    [x,y,z,w] = List
        Positions and integral weights of the grid points.

    """

    points_func = Vpot.properties()[0]

    full_phi, fullx, fully, fullz, fullw, value_on_grid = [], [], [], [], [], []

    # Matrix representation, e.g. density matrices
    if coeff.ndim == 2:
        # Loop Over Blocks
        for l_block in range(Vpot.nblocks()):
            # Obtain general grid information
            l_grid = Vpot.get_block(l_block)

            l_w = np.array(l_grid.w())
            l_x = np.array(l_grid.x())
            l_y = np.array(l_grid.y())
            l_z = np.array(l_grid.z())

            fullx.extend(l_x)
            fully.extend(l_y)
            fullz.extend(l_z)
            fullw.extend(l_w)

            l_npoints = l_w.shape[0]
            points_func.compute_points(l_grid)

            # Recompute to l_grid
            lpos = np.array(l_grid.functions_local_to_global())
            points_func.compute_points(l_grid)
            nfunctions = lpos.shape[0]

            phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]

            l_coeff = coeff[(lpos[:, None], lpos)]
            block_value = contract('pm,mn,pn->p', phi, l_coeff, phi)

            value_on_grid.extend(block_value)

            full_phi.extend(phi)

        x, y, z = np.array(fullx), np.array(fully), np.array(fullz)
        value_on_grid = np.array(value_on_grid)
        full_w = np.array(fullw)

    # Vector representation, e.g. vxc, vp
    elif coeff.ndim == 1:
        for l_block in range(Vpot.nblocks()):
            # Obtain general grid information
            l_grid = Vpot.get_block(l_block)

            l_w = np.array(l_grid.w())
            l_x = np.array(l_grid.x())
            l_y = np.array(l_grid.y())
            l_z = np.array(l_grid.z())

            fullx.extend(l_x)
            fully.extend(l_y)
            fullz.extend(l_z)
            fullw.extend(l_w)

            l_npoints = l_w.shape[0]
            points_func.compute_points(l_grid)

            # Recompute to l_grid
            lpos = np.array(l_grid.functions_local_to_global())
            points_func.compute_points(l_grid)
            nfunctions = lpos.shape[0]

            phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]

            l_coeff = coeff[lpos]
            block_value = contract('pm,m->p', phi, l_coeff)
            # frag_mat.append(mat_r)

            value_on_grid.extend(block_value)

            full_phi.extend(phi)

        x, y, z = np.array(fullx), np.array(fully), np.array(fullz)
        value_on_grid = np.array(value_on_grid)
        full_w = np.array(fullw)

    return value_on_grid, [x, y, z, full_w]


def spherical_grid_to_fock(f, Vpot):
    """
    For a given function value on the spherical grid, f, return the Fock matrix by quadrature integral.
    Parameters
    ----------
    f: Numpy array
        Function value on the grid

    Vpot: psi4.core.VBase
        psi4 V_potential(), contains the information of the spherical grid.

    Returns
    -------
    V: Numpy array
        corresponding Fock Matrix.
    """

    V = np.zeros((Vpot.basis().nbf(),Vpot.basis().nbf()))
    points_func = Vpot.properties()[0]

    i = 0
    # Loop over the blocks
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

        Vtmp = contract('pb,p,p,pa->ab', phi, f[i:i+npoints], w, phi)

        # Add the temporary back to the larger array by indexing, ensure it is symmetric
        V[(lpos[:, None], lpos)] += 0.5 * (Vtmp + Vtmp.T)

        i += npoints
    assert i == f.shape[0], "Did not run through all the points. %i %i" %(i, f.shape[0])
    return V


def axis_plot(function, ax, Vpot=None,
                xyz=None, axis="x", threshold=1e-5,
                label=None, color=None, ls=None, lw=None
                ):
    """
    For a given function value on a grid, function, add a ax.line to plot it's value alone x/y/z axis.

    Parameters
    ----------
    function: Numpy array
        Function value on the grid.

    ax: matplotlib.axes._subplots.AxesSubplot
        The Axes for matplotlib.

    Vpot: psi4.core.VBase
        psi4 V_potential(), contains the information of the spherical grid.
        When Vpot is given for the grid information, function value should be the
        corresponding vaules on the spherical grid.
        One of Vpot and xyz has to be not None. If both are given, xyz has the priority.


    xyz: list of Numpy arrays
        Grid information. Can be used for function value on any grid.
        One of Vpot and xyz has to be not None. If both are given, xyz has the priority.

    threshold: 1e-5
        Threshold to determine if a grid point is on a axis.

    Returns
    -------

    Example
    -------
    f,ax = plt.subplots(1,1)
    ax.scatter(f, wfn.V_potential())
    f.show()
    """

    if xyz is not None:
        x, y, z = xyz
    else:
        x, y, z = Vpot.get_np_xyzw()[:-1]

    assert np.shape(x)[0] == np.shape(function)[0], "The lengths of function and grid don't match."

    if axis == "z":
        x_is_zero_filter = np.isclose(abs(x), 0, atol=threshold)
        y_is_zero_filter = np.isclose(abs(y), 0, atol=threshold)
        order = np.argsort(z[x_is_zero_filter & y_is_zero_filter])

        ax.plot(z[x_is_zero_filter & y_is_zero_filter][order], function[x_is_zero_filter & y_is_zero_filter][order],
                label=label, color=color, ls=ls, lw=lw)

    elif axis == "y":

        x_is_zero_filter = np.isclose(abs(x), 0, atol=threshold)

        z_is_zero_filter = np.isclose(abs(z), 0, atol=threshold)

        order = np.argsort(y[x_is_zero_filter & z_is_zero_filter])

        ax.plot(y[x_is_zero_filter & z_is_zero_filter][order], function[x_is_zero_filter & z_is_zero_filter][order],
                label=label, color=color, ls=ls, lw=lw)

    elif axis == "x":

        y_is_zero_filter = np.isclose(abs(y), 0, atol=threshold)

        z_is_zero_filter = np.isclose(abs(z), 0, atol=threshold)

        order = np.argsort(x[y_is_zero_filter & z_is_zero_filter])

        ax.plot(x[y_is_zero_filter & z_is_zero_filter][order], function[y_is_zero_filter & z_is_zero_filter][order],
                label=label, color=color, ls=ls, lw=lw)
    return


def fouroverlap(wfn, geometry, basis, mints):
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
        Four overlap tensor. Not optimized.
    """
    aux_basis = psi4.core.BasisSet.build(geometry, "DF_BASIS_SCF", "",
                                         "JKFIT", basis)
    S_Pmn = np.squeeze(mints.ao_3coverlap(aux_basis, wfn.basisset(),
                                          wfn.basisset()))
    S_PQ = np.array(mints.ao_overlap(aux_basis, aux_basis))
    S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-12)
    d_mnQ = contract('Pmn,PQ->mnQ', S_Pmn, S_PQinv)
    S_densityfitting = contract('Pmn,PQ,Qrs->mnrs', S_Pmn, S_PQinv, S_Pmn)
    return S_densityfitting, d_mnQ, S_Pmn, S_PQ

def basis_to_cubic_grid(coeff, wfn, L, D, write_file=False, title=None):
    """
    Turns a basis represented function to its representation on a cubic grid.

    Parameters
    ----------
    coeff: Numpy array
        basis representation of the function. Can be 1D or 2D.

    wfn: psi4.core.UHF
        psi4 wavefunction, contains the information of the grid.

    L: list of 3 numbers
        Length extended to all 6 direction of the given geometry.

    D: list of 3 numbers
        Separation of the points in 3 directions.

    E.g. for H2 separated by 1 bohr on x axis.
    L = [1.0, 1.0, 1.0]
    D = [0.1, 0.1, 0.1]
    will give a 3*2*2 cubic with [30,20,20] points in each direction.

    write_file: Bool, True
        If False, will compute the value in memory.
        If True, will return a cubic file.
        The cubic file can be read by:
        value, cube_info = libpdft.cubeprop.cube_to_array("Water_Density.cube")

    title: String, None
        The title of the cubic file.

    Returns
    -------
    value = Numpy array
        Values on the grid.

    xyzw = List
        Positions and integral weights of the grid points.

    """
    # O is the very corner. N is the number of points.
    O, N = pdft.cubeprop.build_grid(wfn, L, D)
    block, points, nxyz, npoints = pdft.cubeprop.populate_grid(wfn, O, N, D)
    if coeff.ndim == 1:
        value, xyzw = pdft.cubeprop.compute_density_vector_basis(wfn, O, N, D, npoints, points, nxyz, block, coeff,
                                                        xyzw=True, write_file=write_file, name=title)
    elif coeff.ndim == 2:
        value, xyzw = pdft.cubeprop.compute_density_matrix_basis(wfn, O, N, D, npoints, points, nxyz, block, coeff,
                                                        xyzw=True, write_file=write_file, name=title)

    if not write_file:
        return value, xyzw

