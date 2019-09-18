### Sample input

This is a sample calculation to find vp from fragment densities and target density

```
import psi4
import pdft

#Define individual Fragments

Li = psi4.geometry("""
0 2 
Li 0.0 0.0 2.0
@H 0.0 0.0 0.0
symmetry c1
unit bohr
""")

H= psi4.geometry("""
0 2
@Li 0.0 0.0 2.0 
H 0.0 0.0 0.0 
symmetry c1
unit bohr
""")

#Define full molecule

LiH = psi4.geometry("""
0 1
Li 0.0 0.0 2.0 
H 0.0 0.0 0.0 
symmetry c1
unit bohr
""")

#Define pdft fragments and full molecule

f1 = pdft.Molecule(H, "STO-3G", "SVWN")
f2 = pdft.Molecule(Li, "STO-3G", "SVWN", mints=f1.mints)
mol = pdft.Molecule(LiH, "STO-3G", "SVWN", mints=f1.mints)
#Notice how that since we are using dummy atoms and same basis set, we can share the mints object

#Define the Embedding object
LiH = pdft.Embedding([f1, f2], mol)




```
