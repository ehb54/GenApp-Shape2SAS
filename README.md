# GenApp-Shape2SAS

calulating the pair distance distribution function, p(r), and the scattering intensity, I(q), from user-defined shapes

the program is strongly inspired by McSim (written by Steen Hansen, see citation below), but completely rewritten by Andreas Haahr Larsen    

## How to run the program

via the web GUI: https://somo.chem.utk.edu/shape2sas/

## Files

### modules/shape2sas.json
GUI

### bin/shape2sas.py
python wrapper.   
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
