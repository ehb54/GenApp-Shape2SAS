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
please cite:     
Steen Hansen (1990)    
Journal of Applied Crystallography 23, 344-346     
Calculation of small-angle scattering profiles using Monte Carlo simulation    

Alexey Savelyev, Emre Brookes (2019)    
Future Generation Computer Systems 94, 929-936    
GenApp: Extensible tool for rapid generation of web and native GUI applications    

and please acknowledge:    
Andreas Haahr Larsen who wrote the program and GUI    
Emre Brookes, who maintains GenApp and supports development of new programs and features   

## Contact
Andreas Haahr Larsen    
andreas.larsen(at)sund.ku.dk  
