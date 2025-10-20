# GenApp-Shape2SAS
  
See full documantation at the main GitHub page:
[github.com/andreashlarsen/Shape2SAS](https://github.com/andreashlarsen/shape2sas)

This is the GitHub for the web application, which calls the main script

## How to run the program

via the web application:
[somo.chem.utk.edu](https://somo.chem.utk.edu/shape2sas/)

## Files that are not descriped in the main GitHub

### modules/shape2sas.json
GUI input/output

### bin/shape2sas_wrapper.py
python wrapper.
<<<<<<< HEAD
takes input from GUI, send to the main script `shape2sas.py` functions and return output to GUI.

## Contact
Andreas Haahr Larsen
Emre Brookes
=======
takes input from GUI, send to functions and return output to GUI.

### bin/helpfunctions.py
the engine.
contains a functions that, e.g.:
- generates points from user input
- calculates p(r)
- calculates I(q)
- plot results
- make 3D and 2D representations of generated structures

### other files
all other files are:
- genapp-specific (related to the GUI) and should not be altered
- old versions of the above

### Cite
Andreas H. Larsen, Emre Brookes, Martin C. Pedersen and Jacob J.K. Kirkensgaard (2023)
Journal of Applied Crystallography 56, 1287-1294
Shape2SAS: a web application to simulate small-angle scattering data and pair distance distributions from user-defined shapes
https://doi.org/10.1107/S1600576723005848

## Contact
Andreas Haahr Larsen
andreas.larsen(at)nbi.ku.dk
