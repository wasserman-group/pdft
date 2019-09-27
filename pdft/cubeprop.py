""""
cubeprop.py

"""

import psi4
import matplotlib.pyplot as plt
import numpy as np
import os

def _getline(cube):
    """
    Read a line from cube file where first field is an int 
    and the remaining fields are floats.
    
    Parameters
    
    ----------
    cube: file object of the cube file
    
    Returns
    
    -------
    (int, list<float>)
    
    From Gist: aditya95sriram/cubetools.py
    """
    l = cube.readline().strip().split()
    return int(l[0]), map(float, l[1:])

def read_cube(fname):
    """ 
    Read cube file into numpy array
    
    Parameters
    ----------
    fname: filename of cube file
        
    Returns 
    --------
    (data: np.array, metadata: dict)
    
    From Gist: aditya95sriram/cubetools.py
    """
    meta = {}
    with open(fname, 'r') as cube:
        cube.readline(); cube.readline()  # ignore comments
        natm, meta['org'] = _getline(cube)
        nx, meta['xvec'] = _getline(cube)
        ny, meta['yvec'] = _getline(cube)
        nz, meta['zvec'] = _getline(cube)
        meta['atoms'] = [_getline(cube) for i in range(natm)]
        data = np.zeros((nx*ny*nz))
        idx = 0
        for line in cube:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1
    data = np.reshape(data, (nx, ny, nz))
    return data, meta

class Cube():

    def __init__(self, wfn):
        """
        Wavefunction *must not* come from a pdft.Molecule object
        """
        self.wfn = wfn
        
    def get_density(self, which_density,delete_cubefile=True):
        """
        Gets density from wfn object

        Parameters
        ----------

        which_density: str
            Density to be computed. Options are: 'Da', 'Db', 'Ds', 'Dt'

        delete_cubefile: Boolean
            Deletes the computed cubefiles. Good idea to keep it as True, 
            since volumetric information can be very expensive with small spacings

        Returns
        ----------

        D: Numpy array
            Array of volumetric information
        
        """
    
        psi4.set_options({'cubeprop_tasks':['density']})
        
        psi4.cubeprop(self.wfn)
        
        if which_density == 'Da':
            D, D_info = read_cube('Da.cube')
            
        elif which_density == 'Db':
            D, D_info = read_cube('Db.cube')
            
        elif which_density == 'Ds':
            D, D_info = read_cube('Ds.cube')
            
        elif which_density == 'Dt':
            D, D_info = read_cube('Dt.cube')
            
        
        if delete_cubefile == True:
            os.remove('Da.cube')
            os.remove('Db.cube')
            os.remove('Ds.cube')
            os.remove('Dt.cube')
        
        print(D.shape)
                    
        return D

    def plot_density(self, D, direction, slice):
        """
        Plots density from numpy array

        Parameters
        ----------
        Matrix: NumPy array
            Matrix from a scf calculation. Must match the basis set of the wfn object
        
        direction: int
            Perpendicular direction in which the slice will be shown

        direction: int
            index of numpy array that will be plotted. Need to be contained in 
            cube.shape of each direction
        
        """

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))

        if direction == 0:
            ax.imshow(D[slice,:,:], interpolation='bicubic')

        if direction == 1:
            ax.imshow(D[:,slice,:], interpolation='bicubic')

        if direction == 2:
            ax.imshow(D[:,:,slice], interpolation='bicubic') 

        #        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
#        print(D.shape)
#        for i in range(0):
#            for j in range(0):
            
#        ax.imshow(D[60,:,:], interpolation="nearest")

#        for i in range(D.shape[1]):
#            for j in range(D.shape[2]):
#                if np.isclose(D[60, i, j], iso, 0.3e-1) == True:
#                    if (i+j) % 0 == 0: 
#                    ax.scatter(j,i, color="white")


    def plot_matrix(self, matrix, direction, slice):
        """
        Plots matrix from numpy array. 
        (Warning! This overwrites information from wavefunction)
        
        Parameters
        ----------
        Matrix: NumPy array
            Matrix from a scf calculation. Must match the basis set of the wfn object
        
        direction: int
            Perpendicular direction in which the slice will be shown

        direction: int
            index of numpy array that will be plotted. Need to be contained in 
            cube.shape of each direction
        
        """
        
        self.wfn.Da().copy(matrix)
        psi4.cubeprop(self.wfn)
        cube, _ = read_cube('Da.cube')

        print(cube.shape)

        os.remove('Da.cube')
        os.remove('Db.cube')
        os.remove('Ds.cube')
        os.remove('Dt.cube')

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
        if direction == 0:
            ax.imshow(cube[slice,:,:], interpolation='bicubic')

        if direction == 1:
            ax.imshow(cube[:,slice,:], interpolation='bicubic')

        if direction == 2:
            ax.imshow(cube[:,:,slice], interpolation='bicubic') 







        