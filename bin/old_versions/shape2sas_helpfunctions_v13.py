import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma,j0
from typing import Tuple, List, Any
#from dataclasses import dataclass
from fast_histogram import histogram1d #histogram1d from fast_histogram is faster than np.histogram (https://pypi.org/project/fast-histogram/) 
import inspect
import sys
import re
import warnings
from dataclasses import dataclass, field


################################ Data classes ################################

Vector2D = Tuple[np.ndarray, np.ndarray]
Vector3D = Tuple[np.ndarray, np.ndarray, np.ndarray]
Vector4D = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def sinc(x) -> np.ndarray:
    """
    function for calculating sinc = sin(x)/x
    numpy.sinc is defined as sinc(x) = sin(pi*x)/(pi*x)
    """
    return np.sinc(x / np.pi)   

### Subunits

class Sphere:
    aliases = ["sphere","ball","sph"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 1:
            print("\nERROR: subunit sphere needs 1 dimension, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()
        self.R = dimensions[0]

    def getVolume(self) -> float:
        """Returns the volume of a sphere"""
        return (4 / 3) * np.pi * self.R**3
    
    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a sphere"""

        Volume = self.getVolume()
        Volume_max = (2*self.R)**3 ###Box around sphere.
        Vratio = Volume_max/Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R, self.R, N)
        d = np.sqrt(x**2 + y**2 + z**2)

        idx = np.where(d < self.R) #save points inside sphere
        x_add,y_add,z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, 
                     x_eff: np.ndarray, 
                     y_eff: np.ndarray, 
                     z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a sphere"""

        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        idx = np.where(d > self.R)
        return idx

class HollowSphere:
    aliases = ["hollowsphere","shell"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 2:
            print("\nERROR: subunit hollow_sphere needs 2 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()
        self.R,self.r = dimensions

    def getVolume(self) -> float:
        """Returns the volume of a hollow sphere"""
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            return 4 * np.pi * self.R**2 #surface area of a sphere
        else: 
            return (4 / 3) * np.pi * (self.R**3 - self.r**3)

    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a hollow sphere"""
        Volume = self.getVolume()
        if self.r == self.R:
            #The hollow sphere is a shell
            phi = np.random.uniform(0,2 * np.pi, Npoints)
            costheta = np.random.uniform(-1, 1, Npoints)
            theta = np.arccos(costheta)

            x_add = self.R * np.sin(theta) * np.cos(phi)
            y_add = self.R * np.sin(theta) * np.sin(phi)
            z_add = self.R * np.cos(theta)
            return x_add, y_add, z_add
        Volume_max = (2*self.R)**3 ###Box around the sphere
        Vratio = Volume_max/Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R, self.R, N)
        d = np.sqrt(x**2 + y**2 + z**2)

        idx = np.where((d < self.R) & (d > self.r))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a hollow sphere"""

        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        if self.r > self.R:
             self.r, self.R = self.R, self.r
        if self.r == self.R:
            idx = np.where(d != self.R)
            return idx
        else:
            idx = np.where((d > self.R) | (d < self.r))
            return idx

class Cylinder:
    aliases = ["cylinder","rod","cyl"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 2:
            print("\nERROR: subunit cylinder needs 2 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()        
        self.R,self.l = dimensions

    def getVolume(self) -> float:
        """Returns the volume of a cylinder"""

        return np.pi * self.R**2 * self.l
    
    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a cylinder"""

        Volume = self.getVolume()
        Volume_max = 2 * self.R * 2 * self.R * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)
        d = np.sqrt(x**2 + y**2)
        idx = np.where(d < self.R)
        x_add,y_add,z_add = x[idx],y[idx],z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a cylinder"""

        d = np.sqrt(x_eff**2+y_eff**2)
        idx = np.where((d > self.R) | (abs(z_eff) > self.l / 2))
        return idx

class Ellipsoid:
    aliases = ["ellipsoid"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 3:
            print("\nERROR: subunit ellipsoid needs 3 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()    
        self.a,self.b,self.c = dimensions

    def getVolume(self) -> float:
        """Returns the volume of an ellipsoid"""
        return (4 / 3) * np.pi * self.a * self.b * self.c

    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of an ellipsoid"""
        Volume = self.getVolume()
        Volume_max = 2 * self.a * 2 * self.b * 2 * self.c
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.a, self.a, N)
        y = np.random.uniform(-self.b, self.b, N)
        z = np.random.uniform(-self.c, self.c, N)

        d2 = x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2
        idx = np.where(d2 < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """check for points within a ellipsoid"""

        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2 + z_eff**2 / self.c**2
        idx = np.where(d2 > 1)

        return idx

class EllipticalCylinder:
    aliases = ["ellipticalcylinder","ellipticalrod"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 3:
            print("\nERROR: subunit elliptical_cylinder needs 3 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()  
        self.a,self.b,self.l = dimensions

    def getVolume(self) -> float:
        """Returns the volume of an elliptical cylinder"""
        return np.pi * self.a * self.b * self.l

    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of an elliptical cylinder"""

        Volume = self.getVolume()
        Volume_max = 2 * self.a * 2 * self.b * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.a, self.a, N)
        y = np.random.uniform(-self.b, self.b, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)

        d2 = x**2 / self.a**2 + y**2 / self.b**2
        idx = np.where(d2 < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add 

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a Elliptical cylinder"""
        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2
        idx = np.where((d2 > 1) | (abs(z_eff) > self.l / 2))
        return idx

class Cube:
    aliases = ["cube","dice"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 1:
            print("\nERROR: subunit cube needs 1 dimension, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()  
        self.a = dimensions[0]

    def getVolume(self) -> float:
        """Returns the volume of a cube"""
        return self.a**3

    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a cube"""

        Volume = self.getVolume()

        N = Npoints
        x_add = np.random.uniform(-self.a, self.a, N)
        y_add = np.random.uniform(-self.a, self.a, N)
        z_add = np.random.uniform(-self.a, self.a, N)

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a cube"""

        idx = np.where((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
            (abs(z_eff) >= self.a/2) | ((abs(x_eff) <= self.b/2) 
            & (abs(y_eff) <= self.b/2) & (abs(z_eff) <= self.b/2)))
        
        return idx

class HollowCube:
    aliases = ["hollowcube","hollowdice"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 2:
            print("\nERROR: subunit hollow_cube needs 2 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()  
        self.a,self.b = dimensions

    def getVolume(self) -> float:
        """Returns the volume of a hollow cube"""
        if self.a < self.b:
            self.a, self.b = self.b, self.a
        if self.a == self.b:
            return 6 * self.a**2 #surface area of a cube
        else: 
            return (self.a - self.b)**3
    
    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a hollow cube"""

        Volume = self.getVolume()
        
        if self.a == self.b:
            #The hollow cube is a shell
            d = self.a / 2
            N = int(Npoints / 6)
            one = np.ones(N)
            
            #make each side of the cube at a time
            x_add, y_add, z_add = [], [], []
            for sign in [-1, 1]:
                x_add = np.concatenate((x_add, sign * one * d))
                y_add = np.concatenate((y_add, np.random.uniform(-d, d, N)))
                z_add = np.concatenate((z_add, np.random.uniform(-d, d, N)))
                
                x_add = np.concatenate((x_add, np.random.uniform(-d, d, N)))
                y_add = np.concatenate((y_add, sign * one * d))
                z_add = np.concatenate((z_add, np.random.uniform(-d, d, N)))

                x_add = np.concatenate((x_add, np.random.uniform(-d, d, N)))
                y_add = np.concatenate((y_add, np.random.uniform(-d, d, N)))
                z_add = np.concatenate((z_add, sign * one * d))
            return x_add, y_add, z_add
        
        Volume_max = self.a**3
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)

        x = np.random.uniform(-self.a / 2,self.a / 2, N)
        y = np.random.uniform(-self.a / 2,self.a / 2, N)
        z = np.random.uniform(-self.a / 2,self.a / 2, N)

        d = np.maximum.reduce([abs(x), abs(y), abs(z)])
        idx = np.where(d >= self.b / 2)
        x_add,y_add,z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a hollow cube"""

        if self.a < self.b:
            self.a, self.b = self.b, self.a
        
        if self.a == self.b:
            idx = np.where((abs(x_eff)!=self.a/2) | (abs(y_eff)!=self.a/2) | (abs(z_eff)!=self.a/2))
            return idx
        
        else: 
            idx = np.where((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
            (abs(z_eff) >= self.a/2) | ((abs(x_eff) <= self.b/2) 
            & (abs(y_eff) <= self.b/2) & (abs(z_eff) <= self.b/2)))

        return idx

class Cuboid:
    aliases = ["cuboid","brick"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 3:
            print("\nERROR: subunit hollow_cube needs 3 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()
        self.a,self.b,self.c = dimensions

    def getVolume(self) -> float:
        """Returns the volume of a cuboid"""
        return self.a * self.b * self.c
    
    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a cuboid"""
        Volume = self.getVolume()
        x_add = np.random.uniform(-self.a, self.a, Npoints)
        y_add = np.random.uniform(-self.b, self.b, Npoints)
        z_add = np.random.uniform(-self.c, self.c, Npoints)
        return x_add, y_add, z_add
    
    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a Cuboid"""
        idx = np.where((abs(x_eff) >= self.a/2) 
        | (abs(y_eff) >= self.b/2) | (abs(z_eff) >= self.c/2))
        return idx

class CylinderRing:
    aliases = ["cylinderring","ring","discring","hollowcylinder","hollowdisc","hollowrod"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 3:
            print("\nERROR: subunit CylinderRing needs 3 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()
        self.R,self.r,self.l = dimensions
    
    def getVolume(self) -> float:
        """Returns the volume of a cylinder ring"""
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            return 2 * np.pi * self.R * self.l #surface area of a cylinder
        else: 
            return np.pi * (self.R**2 - self.r**2) * self.l

    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a cylinder ring"""
        Volume = self.getVolume()
        if self.r == self.R:
            #The cylinder ring is a shell
            phi = np.random.uniform(0, 2 * np.pi, Npoints)
            x_add = self.R * np.cos(phi)
            y_add = self.R * np.sin(phi)
            z_add = np.random.uniform(-self.l / 2, self.l / 2, Npoints)
            return x_add, y_add, z_add
        Volume_max = 2 * self.R * 2 * self.R * self.l
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)
        d = np.sqrt(x**2 + y**2)
        idx = np.where((d < self.R) & (d > self.r))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a cylinder ring"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            idx = np.where((d != self.R) | (abs(z_eff) > self.l / 2))
            return idx
        else: 
            idx = np.where((d > self.R) | (d < self.r) | (abs(z_eff) > self.l / 2))
            return idx

class Torus:
    aliases = ["torus","toroid","doughnut"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 2:
            print("\nERROR: subunit Torus needs 2 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()
        self.R,self.r = dimensions

    def getVolume(self) -> float:
        """Returns the volume of a torus"""

        return 2 * np.pi**2 * self.r**2 * self.R

    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a torus"""
        Volume = self.getVolume()
        L = 2 * (self.R + self.r)
        l = 2 * self.r
        Volume_max = L*L*l
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-L/2, L/2, N)
        y = np.random.uniform(-L/2, L/2, N)
        z = np.random.uniform(-l/2, l/ 2, N)
        # equation: (R-sqrt(x**2+y**2))**2 + z**2 = (R-d)**2 + z**2 = r
        d = np.sqrt(x**2 + y**2)
        idx = np.where((self.R-d)**2 + z**2 < self.r**2)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a torus"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        idx = np.where((self.R-d)**2 + z_eff**2 > self.r**2)
        return idx
        
class Hyperboloid:
    aliases = ["hyperboloid", "hourglass", "coolingtower"]
    
    # https://mathworld.wolfram.com/One-SheetedHyperboloid.html
    # https://www.vcalc.com/wiki/hyperboloid-volume

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 3:
            print("\nERROR: subunit Hyperboloid needs 3 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()
        self.r,self.c,self,h = dimensions

    def getVolume(self) -> float:
        """Returns the volume of a hyperboloid"""
        return np.pi * 2*self.h * self.r**2 * ( 1 + (2*self.h)**2 / ( 12 * self.c**2 ) )

    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a hyperboloid"""
        #Volume = self.getVolume()
        L = 2 * self.h
        R = self.r * np.sqrt( 1 + L**2 / (4 * self.c**2 ) )
        Volume_max = 2*self.h * 2*R * 2*R
        Volume = np.pi * L * ( 2 * self.r**2 + R**2 ) / 3
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-R, R, N)
        y = np.random.uniform(-R, R, N)
        z = np.random.uniform(-self.h, self.h, N)
        idx = np.where(x**2/self.r**2 + y**2/self.r**2 - z**2/self.c**2 < 1.0)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a Hyperboloid"""
        idx = np.where(x_eff**2/self.r**2 + y_eff**2/self.r**2 - z_eff**2/self.c**2 > 1.0)
        return idx
    
class Superellipsoid:
    aliases = ["superellipsoid"]

    def __init__(self, dimensions: List[float]):
        if len(dimensions) != 4:
            print("\nERROR: subunit Superellipsoid needs 4 dimensions, but " + str(len(dimensions)) + ' dimensions were given: ' + str(dimensions) + '\n')
            exit()
        self.R,self.eps,self.t,self.s = dimensions

    @staticmethod
    def beta(a, b) -> float:
        """beta function"""
        return gamma(a) * gamma(b) / gamma(a + b)

    def getVolume(self) -> float:
        """Returns the volume of a superellipsoid"""
        return (8 / (3 * self.t * self.s) * self.R**3 * self.eps * 
                self.beta(1 / self.s, 1 / self.s) * self.beta(2 / self.t, 1 / self.t))
    
    def getPointDistribution(self, Npoints: int) -> Vector3D:
        """Returns the point distribution of a superellipsoid"""
        Volume = self.getVolume()
        Volume_max = 2 * self.R * self.eps * 2 * self.R * 2 * self.R
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R * self.eps, self.R * self.eps, N)
        d = ((np.abs(x)**self.s + np.abs(y)**self.s)**(self.t/ self.s) 
            + np.abs(z / self.eps)**self.t)
        idx = np.where(d < np.abs(self.R)**self.t)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add
    
    def checkOverlap(self, x_eff: np.ndarray, 
                           y_eff: np.ndarray, 
                           z_eff: np.ndarray) -> np.ndarray:
        """Check for points within a superellipsoid"""
        d = ((np.abs(x_eff)**self.s + np.abs(y_eff)**self.s)**(self.t / self.s) 
        + np.abs(z_eff / self.eps)**self.t)
        idx = np.where(d >= np.abs(self.R)**self.t)
        return idx

def Rotate(x,y,z,alpha,beta,gamma):
    """
    Simple Euler rotation
    input angles in degrees
    """
    a,b,g = np.radians(alpha),np.radians(beta),np.radians(gamma)
    ca,cb,cg = np.cos(a),np.cos(b),np.cos(g)
    sa,sb,sg = np.sin(a),np.sin(b),np.sin(g)
    x_rot = ( x * cg * cb + y * (cg * sb * sa - sg * ca) + z * (cg * sb * ca + sg * sa))
    y_rot = ( x * sg * cb + y * (sg * sb * sa + cg * ca) + z * (sg * sb * ca - cg * sa))
    z_rot = (-x * sb      + y * cb * sa                  + z * cb * ca)
    return x_rot, y_rot, z_rot

class GenerateAllPoints:
    def __init__(self, Npoints: int, 
                            com: List[List[float]], 
                        subunits: List[List[float]], 
                        dimensions: List[List[float]], 
                        rotation : List[List[float]], 
                        sld: List[float], 
                        exclude_overlap: bool):
        self.Npoints = Npoints
        self.com = com
        self.subunits = subunits
        self.Number_of_subunits = len(subunits)
        self.dimensions = dimensions
        self.rotation = rotation
        self.sld = sld
        self.exclude_overlap = exclude_overlap
        self.setAvailableSubunits()

    def setAvailableSubunits(self):
        """Dynamically build dictionary of aliases -> subunit classes"""
        current_module = sys.modules[__name__]
        classes = inspect.getmembers(current_module, inspect.isclass)
        self.subunitClasses = {}
        for _, cls in classes:
            if hasattr(cls, "aliases"):
                for alias in cls.aliases:
                    self.subunitClasses[alias.lower().replace("_", "").replace(" ", "")] = cls

    @staticmethod
    def AppendingPoints(x_new: np.ndarray, 
                          y_new: np.ndarray, 
                          z_new: np.ndarray,
                          sld_new: np.ndarray, 
                          x_add: np.ndarray, 
                          y_add: np.ndarray, 
                          z_add: np.ndarray, 
                          sld_add: np.ndarray) -> Vector4D:
        """append new points to vectors of point coordinates"""
        
        # add points to (x_new,y_new,z_new)
        if isinstance(x_new, int):
            # if these are the first points to append to (x_new,y_new,z_new)
            x_new = x_add
            y_new = y_add
            z_new = z_add
            sld_new = sld_add
        else:
            x_new = np.append(x_new, x_add)
            y_new = np.append(y_new, y_add)
            z_new = np.append(z_new, z_add)
            sld_new = np.append(sld_new, sld_add)

        return x_new, y_new, z_new, sld_new

    @staticmethod
    def onCheckOverlap(x: np.ndarray, 
                       y: np.ndarray, 
                       z: np.ndarray, 
                       p: np.ndarray, 
                       rotation: List[float], 
                       com: List[float], 
                       subunitClass: object, 
                       dimensions: List[float]):
        """check for overlap with previous subunits. 
        if overlap, the point is removed"""
        # shift back to origin
        x_eff,y_eff,z_eff = x-com[0],y-com[1],z-com[2]
        if sum(rotation) != 0:
            #rotate back to original orientation
            alpha, beta, gamma = rotation
            x_eff, y_eff, z_eff  = Rotate(x_eff,y_eff,z_eff,-alpha,-beta,-gamma)

        # then check overlaps
        idx = subunitClass(dimensions).checkOverlap(x_eff, y_eff, z_eff)
        x_add, y_add, z_add, sld_add = x[idx], y[idx], z[idx], p[idx]

        ## number of excluded points
        N_x = len(x) - len(idx[0])
        return x_add, y_add, z_add, sld_add, N_x

    def onGeneratingAllPointsSeparately(self) -> Vector3D:
        """Generating points for all subunits from each built model, but
        save them separately in their own list"""
        volume = []
        sum_vol = 0

        #Get volume of each subunit
        for i in range(self.Number_of_subunits):

            subunitClass = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")]
            v = subunitClass(self.dimensions[i]).getVolume()
            volume.append(v)
            sum_vol += v

        N, rho, N_exclude = [], [], []
        x_new, y_new, z_new, sld_new, volume_total = [], [], [], [], 0

        for i in range(self.Number_of_subunits):
            Npoints = int(self.Npoints * volume[i] / sum_vol)
            
            x_add, y_add, z_add = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")](self.dimensions[i]).getPointDistribution(Npoints)
            alpha, beta, gamma = self.rotation[i]
            com_x, com_y, com_z = self.com[i]

            # rotate and translate
            x_add, y_add, z_add = Rotate(x_add, y_add, z_add,alpha,beta,gamma)
            x_add, y_add, z_add = x_add+com_x,y_add+com_y,z_add+com_z
            
            #Remaining points
            N_subunit = len(x_add)
            rho_subunit = N_subunit / volume[i]
            sld_add = np.ones(N_subunit) * self.sld[i]

            #Check for overlap with previous subunits
            N_x_sum = 0
            if self.exclude_overlap:
                for j in range(i): 
                    x_add, y_add, z_add, sld_add, N_x = self.onCheckOverlap(x_add, y_add, z_add, sld_add, self.rotation[j],  
                                                    self.com[j], self.subunitClasses[self.subunits[j].lower().replace("_", "").replace(" ", "")], self.dimensions[j])
                    N_x_sum += N_x
    
            N.append(N_subunit)
            rho.append(rho_subunit)
            N_exclude.append(N_x_sum)
            fraction_left = (N_subunit-N_x_sum) / N_subunit
            volume_total += volume[i] * fraction_left

            x_new.append(x_add)
            y_new.append(y_add)
            z_new.append(z_add)
            sld_new.append(sld_add)
        
        #Show information about the model and its subunits
        N_remain = []
        for j in range(self.Number_of_subunits):
            srho = rho[j] * self.sld[j]
            N_remain.append(N[j] - N_exclude[j])
            print(f"        {N[j]} points for subunit {j}: {self.subunits[j]}")
            print(f"             Point density     : {rho[j]:.3e} (points per volume)")
            print(f"             Scattering density: {srho:.3e} (density times scattering length)")
            if self.exclude_overlap:
                print(f"             Excluded points   : {N_exclude[j]} (overlap region)")
            else:
                print(f"             Excluded points   : none - exclude overlap disabled")
            print(f"             Remaining points  : {N_remain[j]} (non-overlapping region)")
        N_total = sum(N_remain)
        print(f"        Total points in model: {N_total}")
        print(f"        Total volume of model: {volume_total:.3e} A^3")
        print(" ")

        return x_new, y_new, z_new, sld_new, volume_total

    def onGeneratingAllPoints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Generating points for all subunits from each built model"""
        volume = []
        sum_vol = 0
        #Get volume of each subunit
        for i in range(self.Number_of_subunits):
            subunitClass = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")]
            v = subunitClass(self.dimensions[i]).getVolume()
            volume.append(v)
            sum_vol += v
        
        N, rho, N_exclude = [], [], []
        x_new, y_new, z_new, sld_new, volume_total = 0, 0, 0, 0, 0

        #Generate subunits
        for i in range(self.Number_of_subunits):
            Npoints = int(self.Npoints * volume[i] / sum_vol)
            
            x_add, y_add, z_add = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")](self.dimensions[i]).getPointDistribution(self.Npoints)
            alpha, beta, gamma = self.rotation[i]
            com_x, com_y, com_z = self.com[i]

            # rotate and translate
            x_add, y_add, z_add = Rotate(x_add, y_add, z_add,alpha,beta,gamma)
            x_add, y_add, z_add = x_add+com_x,y_add+com_y,z_add+com_z

            #Remaining points
            N_subunit = len(x_add)
            rho_subunit = N_subunit / volume[i]
            sld_add = np.ones(N_subunit) * self.sld[i]

            #Check for overlap with previous subunits
            N_x_sum = 0
            if self.exclude_overlap:
                for j in range(i): 
                    x_add, y_add, z_add, sld_add, N_x = self.onCheckOverlap(x_add, y_add, z_add, sld_add, self.rotation[j],  
                                                    self.com[j], self.subunitClasses[self.subunits[j].lower().replace("_", "").replace(" ", "")], self.dimensions[j])
                    N_x_sum += N_x
            
            #Append points
            x_new, y_new, z_new, sld_new = self.AppendingPoints(x_new, y_new, z_new, sld_new, x_add, y_add, z_add, sld_add)

            N.append(N_subunit)
            rho.append(rho_subunit)
            N_exclude.append(N_x_sum)
            fraction_left = (N_subunit-N_x_sum) / N_subunit
            volume_total += volume[i] * fraction_left

        #Show information about the model and its subunits
        N_remain = []
        for j in range(self.Number_of_subunits):
            srho = rho[j] * self.sld[j]
            N_remain.append(N[j] - N_exclude[j])
            print(f"        {N[j]} points for subunit {j}: {self.subunits[j]}")
            print(f"             Point density     : {rho[j]:.3e} (points per volume)")
            print(f"             Scattering density: {srho:.3e} (density times scattering length)")
            print(f"             Excluded points   : {N_exclude[j]} (overlap region)")
            print(f"             Remaining points  : {N_remain[j]} (non-overlapping region)")

        N_total = sum(N_remain)
        print(f"        Total points in model: {N_total}")
        print(f"        Total volume of model: {volume_total:.3e} A^3")
        print(" ")

        return x_new, y_new, z_new, sld_new, volume_total

class WeightedPairDistribution:
    def __init__(self, x: np.ndarray, 
                       y: np.ndarray, 
                       z: np.ndarray, 
                       sld: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
        self.sld = sld #contrast
    
    def calc_all_dist(self) -> np.ndarray:
        """
        Calculate unique pairwise distances between 3D points.
        Returns a 1D float32 array of length N*(N-1)/2.
        """
        x = self.x.astype(np.float32, copy=False)
        y = self.y.astype(np.float32, copy=False)
        z = self.z.astype(np.float32, copy=False)
        N = len(x)

        out = np.empty(N * (N - 1) // 2, dtype=np.float32)

        k = 0
        for i in range(N - 1):
            dx = x[i] - x[i+1:]
            dy = y[i] - y[i+1:]
            dz = z[i] - z[i+1:]
            out[k : k + (N - i - 1)] = np.sqrt(dx*dx + dy*dy + dz*dz)
            k += N - i - 1

        return out
    
    def calc_all_contrasts(self) -> np.ndarray:
        """
        Calculate unique pairwise contrast products of p.
        Returns a 1D float32 array of length N*(N-1)/2,
        matching calc_all_dist().
        """
        sld = self.sld.astype(np.float32, copy=False)
        N = len(sld)

        # Preallocate result array (unique pairs only)
        out = np.empty(N * (N - 1) // 2, dtype=np.float32)

        # Fill it using triangular indexing without making an (N, N) array
        k = 0
        for i in range(N - 1):
            # multiply p[i] with all following elements at once
            out[k : k + (N - i - 1)] = sld[i] * sld[i+1:]
            k += N - i - 1

        return out

    @staticmethod
    def generate_histogram(dist: np.ndarray, Nbins: int, contrast: np.ndarray, r_max: float) -> Vector2D:
        """
        make histogram of point pairs, h(r), binned after pair-distances, r
        used for calculating scattering (fast Debye)

        input
        dist     : all pairwise distances
        Nbins    : number of bins in h(r)
        contrast : contrast of points
        r_max    : max distance to include in histogram

        output
        r        : distances of bins
        h    : histogram, weighted by contrast

        """

        h, bin_edges = np.histogram(dist, bins=Nbins, weights=contrast, range=(0,r_max)) 
        r = (bin_edges[:-1] + bin_edges[1:]) * 0.5

        return r, h
    
    @staticmethod
    def calc_Rg(r: np.ndarray, pr: np.ndarray) -> float:
        """ 
        calculate Rg from r and p(r)
        """
        sum_pr_r2 = np.sum(pr * r**2)
        sum_pr = np.sum(pr)
        Rg = np.sqrt(abs(sum_pr_r2 / sum_pr) / 2)

        return Rg

    def calc_hr(self, 
                dist: np.ndarray, 
                Nbins: int, 
                contrast: np.ndarray, 
                polydispersity: float) -> Vector2D:
        """
        calculate h(r)
        h(r) is the contrast-weighted histogram of distances, including self-terms (dist = 0)

        input: 
        dist      : all pairwise distances
        contrast  : all pair-wise contrast products
        polydispersity: relative polydispersity, float

        output:
        hr        : pair distance distribution function 
        """
    
        if dist.dtype != np.float32:
            dist = dist.astype(np.float32, copy=False)
        if contrast.dtype != np.float32:
            contrast = contrast.astype(np.float32, copy=False)

        ## make r range in h(r) histogram slightly larger than Dmax
        ratio_rmax_dmax = 1.05

        lognormal = False
        ## calc h(r) with/without polydispersity
        if polydispersity > 0.0:
            if lognormal:
                Dmax = np.amax(dist)*np.exp(3* polydispersity)
            else:
                Dmax = np.amax(dist) * (1 + 3 * polydispersity)
            r_max = Dmax * ratio_rmax_dmax
            r, hr_1 = self.generate_histogram(dist, Nbins, contrast, r_max)
            N_poly_integral = 25 # should be uneven to include 1 in factor_range (precalculated)
            hr  = np.zeros_like(hr_1, dtype=np.float32)
            #norm = 0.0
            if lognormal:
                log_factors = np.linspace(-3*polydispersity, 3*polydispersity, N_poly_integral, dtype=np.float32)
                factor_range = np.exp(log_factors)
            else:
                factor_range = 1 + np.linspace(-3, 3, N_poly_integral, dtype=np.float32) * polydispersity
            res_range = (1.0 - factor_range) / polydispersity
            if lognormal:
                w_range = np.exp(-(np.log(factor_range))**2 / (2*polydispersity**2)) / (factor_range * polydispersity * np.sqrt(2*np.pi))
            else:
                w_range = np.exp(-0.5*res_range**2)
            vol2_range = factor_range**6
            norm_range = w_range*vol2_range
            for i,factor_d in enumerate(factor_range):
                if factor_d == 1.0:
                    hr += hr_1
                    #norm += 1.0
                else:
                    # calculate in the same bins so histograms can be added
                    dhr = histogram1d(dist * factor_d, bins=Nbins, weights=contrast, range=(0,r_max))
                    #res = (1.0 - factor_d) / polydispersity
                    #w = np.exp(-res**2 / 2.0) # weight: normal distribution
                    #vol2 = (factor_d**3)**2 # weight: relative volumen squared, because larger particles scatter more
                    #dnorm = w * vol2
                    #norm += dnorm
                    #hr += dhr * dnorm
                    hr += dhr * norm_range[i]
            norm = np.sum(norm_range)
            hr /= norm
        else:
            Dmax = np.amax(dist)
            r_max = Dmax * ratio_rmax_dmax
            r, hr = self.generate_histogram(dist, Nbins, contrast, r_max)

        # print Dmax
        print(f"        Dmax: {Dmax:.3e} A")

        return r, hr

    def calc_pr(self, Nbins: int, polydispersity: float) -> Vector3D:
        """
        calculate p(r)
        p(r) is the contrast-weighted histogram of distances, without the self-terms (dist = 0)

        input: 
        dist      : all pairwise distances
        contrast  : all pair-wise contrast products
        polydispersity: boolian, True or False

        output:
        pr        : pair distance distribution function
        """
        print('        calculating distances...')
        dist = self.calc_all_dist()
        print('        calculating contrasts...')
        contrast = self.calc_all_contrasts()

        ## calculate pr
        #idx_nonzero = np.where(dist > 0.0) #  nonzero elements
        #r, pr = self.calc_hr(dist[idx_nonzero], Nbins, contrast[idx_nonzero], polydispersity)
        print('        calculating pr...')
        r, pr = self.calc_hr(dist, Nbins, contrast, polydispersity)

        ## normalize so pr_max = 1
        pr_norm = pr / np.amax(pr)

        ## calculate Rg
        Rg = self.calc_Rg(r, pr_norm)
        print(f"        Rg  : {Rg:.3e} A")

        #returned N values after generating
        pr /= len(self.x)**2 #NOTE: N_total**2

        return r, pr, pr_norm 
    
    @staticmethod
    def save_pr(Nbins: int,
                r: np.ndarray, 
                pr_norm: np.ndarray, 
                Model: str):
        """
        save p(r) to textfile
        """

        with open('pr_%s.dat' % Model,'w') as f:
            f.write('# %-17s %-17s\n' % ('r','p(r)'))
            for i in range(Nbins):
                f.write('  %-17.5e %-17.5e\n' % (r[i], pr_norm[i]))


class StructureDecouplingApprox:
    def __init__(self, q: np.ndarray, 
                 x_new: np.ndarray, 
                 y_new: np.ndarray, 
                 z_new: np.ndarray, 
                 sld_new: np.ndarray):
        self.q = q
        self.x_new = x_new
        self.y_new = y_new
        self.z_new = z_new
        self.sld_new = sld_new

    def calc_com_dist(self) -> np.ndarray:
        """ 
        calc contrast-weighted com distance
        """
        w = np.abs(self.sld_new)

        if np.sum(w) == 0:
            w = np.ones(len(self.x_new))

        x_com, y_com, z_com = np.average(self.x_new, weights=w), np.average(self.y_new, weights=w), np.average(self.z_new, weights=w)
        dx, dy, dz = self.x_new - x_com, self.y_new - y_com, self.z_new - z_com
        com_dist = np.sqrt(dx**2 + dy**2 + dz**2)

        return com_dist

    def calc_A00(self) -> np.ndarray:
        """
        calc zeroth order sph harm, for decoupling approximation
        """
        d_new = self.calc_com_dist()
        M = len(self.q)
        A00 = np.zeros(M)
        
        for i in range(M):
            qr = self.q[i] * d_new

            A00[i] = sum(self.sld_new * sinc(qr))
        A00 = A00 / A00[0] # normalise, A00[0] = 1

        return A00

    def decoupling_approx(self, Pq: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        modify structure factor with the decoupling approximation
        for combining structure factors with non-spherical (or polydisperse) particles

        see, for example, Larsen et al 2020: https://doi.org/10.1107/S1600576720006500
        and refs therein

        input
        q
        x,y,z,p    : coordinates and contrasts
        Pq         : form factor
        S          : structure factor

        output
        S_eff      : effective structure factor, after applying decoupl. approx
        """
        A00 = self.calc_A00()
        const = 1e-3 # add constant in nominator and denominator, for stability (numerical errors for small values dampened)
        Beta = (A00**2 + const) / (Pq + const)
        S_eff = 1 + Beta * (S - 1)
        return S_eff


### structure factor classes

class HardSphereStructure(StructureDecouplingApprox):
    S_aliases = ["hardsphere","hs"]

    def __init__(self, q: np.ndarray, 
                 x_new: np.ndarray, 
                 y_new: np.ndarray, 
                 z_new: np.ndarray, 
                 sld_new: np.ndarray, 
                 par: List[float]):
        super(HardSphereStructure, self).__init__(q, x_new, y_new, z_new, sld_new)
        self.q = q
        self.x_new = x_new
        self.y_new = y_new
        self.z_new = z_new
        self.sld_new = sld_new
        if len(par) != 2:
            print("\nERROR: structure factor hard-sphere needs 2 parameters, but " + str(len(par)) + ' parameters were given: ' + str(par) + '\n')
            exit()  
        self.conc,self.R_HS = par

    # def calc_S_HS_(self) -> np.ndarray:
    #     """
    #     calculate the hard-sphere structure factor
    #     calls function calc_G()

    #     input
    #     q       : momentum transfer
    #     eta     : volume fraction
    #     R       : estimation of the hard-sphere radius

    #     output
    #     S_HS    : hard-sphere structure factor
    #     """
    #     if self.conc > 0.0:
    #         A = 2 * self.R_HS * self.q 
    #         G = self.calc_G(A, self.conc)
    #         S_HS = 1 / (1 + 24 * self.conc * G / A) # percus-yevick approximation
    #     else:                         
    #         S_HS = np.ones(len(self.q))

    #     return S_HS
    
    # @staticmethod
    # def calc_G(A: np.ndarray, eta: float) -> np.ndarray:
    #     """ 
    #     calculate G in the hard-sphere potential
    #     A  : 2*R*q
    #     eta: volume fraction
    #     """
    #     a = (1 + 2 * eta)**2 / (1 - eta)**4
    #     b = -6 * eta * (1 + eta / 2)**2/(1 - eta)**4 
    #     c = eta * a / 2
    #     sinA = np.sin(A)
    #     cosA = np.cos(A)
    #     A2,A3,A4,A5 = A**2,A**3,A**4,A**5
    #     fa = sinA - A * cosA
    #     fb = 2 * A * sinA + (2 - A2) * cosA-2
    #     fc = -A4 * cosA + 4 * ((3 * A2 - 6) * cosA + (A3 - 6 * A) * sinA + 6)
    #     G = a * fa / A2 + b * fb / A3 + c * fc / A5
    #     return G

    def calc_S_HS(self) -> np.ndarray:
        """
        Calculate the hard-sphere structure factor using the Percus-Yevick approximation.
        Implements the stable version with Taylor expansion for small A = 2*R*q.
        adapted directly from the sasview code
        """
        if self.conc <= 0.0:
            return np.ones(len(self.q))

        vf = self.conc
        R = self.R_HS
        q = self.q
        X = np.abs(2.0 * R * q)  # same as A in your earlier code

        # Precompute constants
        denom = (1.0 - vf)
        if denom < 1e-12:  # avoid division by zero
            return np.ones_like(q)

        Xinv = 1.0 / denom
        D = Xinv * Xinv
        A = (1.0 + 2.0 * vf) * D
        A *= A
        B = (1.0 + 0.5 * vf) * D
        B *= B
        B *= -6.0 * vf
        G = 0.5 * vf * A

        # Cutoffs
        cutoff_tiny = 5e-6
        cutoff_series = 0.05  # corresponds to CUTOFFHS in C code

        S_HS = np.empty_like(q)

        for i, x in enumerate(X):
            if x < cutoff_tiny:
                # limit q -> 0
                S_HS[i] = 1.0 / A
            elif x < cutoff_series:
                # Taylor series expansion
                x2 = x * x
                # Equivalent to the FF expression in the C code
                FF = (8.0 * A + 6.0 * B + 4.0 * G
                    + (-0.8 * A - B / 1.5 - 0.5 * G
                        + (A / 35.0 + 0.0125 * B + 0.02 * G) * x2) * x2)
                S_HS[i] = 1.0 / (1.0 + vf * FF)
            else:
                # Normal expression
                x2 = x * x
                x4 = x2 * x2
                s, c = np.sin(x), np.cos(x)
                # FF expression refactored from the C code
                FF = ((G * ((4.0 * x2 - 24.0) * x * s
                            - (x4 - 12.0 * x2 + 24.0) * c
                            + 24.0) / x2
                    + B * (2.0 * x * s - (x2 - 2.0) * c - 2.0)) / x
                    + A * (s - x * c)) / x
                S_HS[i] = 1.0 / (1.0 + 24.0 * vf * FF / x2)

        return S_HS

    def structure_eff(self, Pq: np.ndarray) -> np.ndarray:
        S = self.calc_S_HS()
        S_eff = self.decoupling_approx(Pq, S)
        return S_eff 

class Aggregation(StructureDecouplingApprox):
    S_aliases = ["aggregation","aggr","frac2D"]

    def __init__(self, q: np.ndarray, 
                 x_new: np.ndarray, 
                 y_new: np.ndarray, 
                 z_new: np.ndarray, 
                 sld_new: np.ndarray, 
                 par: List[float]):
        super(Aggregation, self).__init__(q, x_new, y_new, z_new, sld_new)
        self.q = q
        self.x_new = x_new
        self.y_new = y_new
        self.z_new = z_new
        self.sld_new = sld_new
        if len(par) != 3:
            print("\nERROR: structure factor aggregation needs 3 parameters, but " + str(len(par)) + ' parameters were given: ' + str(par) + '\n')
            exit()  
        self.Reff,self.Naggr,self.fracs_aggr = par

    def calc_S_aggr(self) -> np.ndarray:
        """
        calculates fractal aggregate structure factor with dimensionality 2

        S_{2,D=2} in Larsen et al 2020, https://doi.org/10.1107/S1600576720006500

        input 
        q      :
        Naggr  : number of particles per aggregate
        Reff   : effective radius of one particle 

        output
        S_aggr :
        """
        qR = self.q * self.Reff
        S_aggr = 1 + (self.Naggr - 1)/(1 + qR**2 * self.Naggr / 3)
        return S_aggr
    
    def structure_eff(self, Pq: np.ndarray) -> np.ndarray:
        """Return effective structure factor for aggregation"""

        S = self.calc_S_aggr()
        S_eff = self.decoupling_approx(Pq, S)
        S_eff = (1 - self.fracs_aggr) + self.fracs_aggr * S_eff
        return S_eff

class NoStructure(StructureDecouplingApprox):
    S_aliases = ["none","no","one","unity"]

    def __init__(self, q: np.ndarray, 
                 x_new: np.ndarray, 
                 y_new: np.ndarray, 
                 z_new: np.ndarray, 
                 sld_new: np.ndarray, 
                 par: Any):
        super(NoStructure, self).__init__(q, x_new, y_new, z_new, sld_new)
        self.q = q
        if len(par) != 0:
            print("\nERROR:  structure factor none needs 0 parameters, but " + str(len(par)) + ' parameters were given: ' + str(par) + '\n')
            exit()  
        
    def structure_eff(self, Pq: Any) -> np.ndarray:
        """Returns unity, no structure factor"""
        return np.ones(len(self.q))

class StructureFactor:

    @staticmethod
    def setAvailableStructureFactors():
        """Dynamically build dictionary of Structure factor aliases (S_aliases) -> structure factor classes"""
        current_module = sys.modules[__name__]
        classes = inspect.getmembers(current_module, inspect.isclass)
        registry = {}
        for _, cls in classes:
            if hasattr(cls, "S_aliases"):
                for alias in cls.S_aliases:
                    registry[alias.lower().replace("_", "").replace(" ", "")] = cls
        return registry
        
    def __init__(self, q: np.ndarray, 
                 x_new: np.ndarray, 
                 y_new: np.ndarray, 
                 z_new: np.ndarray, 
                 sld_new: np.ndarray,
                 Stype: str,
                 par=None):
        self.q = q
        self.x_new = x_new
        self.y_new = y_new
        self.z_new = z_new
        self.sld_new = sld_new
        self.Stype = Stype.lower().replace("_", "").replace("-", "").replace(" ", "")
        self.par = par
        self.registry = StructureFactor.setAvailableStructureFactors()

    def getStructureFactor(self):
        """Return chosen structure factor"""
        if self.Stype in self.registry:
            cls = self.registry[self.Stype]
            return cls(self.q, self.x_new, self.y_new, self.z_new, self.sld_new, self.par)
        else:
            raise ValueError(f"Structure factor '{self.Stype}' does not exist. Choose from {list(self.registry.keys())}")

    @staticmethod
    def save_S(q: np.ndarray, S_eff: np.ndarray, Model: str):
        """ 
        save S to file
        """
        with open('Sq_%s.dat' % Model,'w') as f:
            f.write('# Structure factor, S(q), used in: I(q) = P(q)*S(q)\n')
            f.write('# Default: S(q) = 1.0\n')
            f.write('# %-17s %-17s\n' % ('q','S(q)'))
            for (q_i, S_i) in zip(q, S_eff):
                f.write('  %-17.5e%-17.5e\n' % (q_i, S_i))

class ITheoretical:
    def __init__(self, q: np.ndarray):
        self.q = q

    def calc_Pq(self, r: np.ndarray, pr: np.ndarray, conc: float, volume_total: float) -> Vector2D:
        """
        calculate form factor, P(q), and forward scattering, I(0), using pair distribution, p(r) 
        """
        ## calculate P(q) and I(0) from p(r)
        I0, Pq = 0, 0
        for (r_i, pr_i) in zip(r, pr):
            I0 += pr_i
            qr = self.q * r_i
            Pq += pr_i * sinc(qr)
    
        # normalization, P(0) = 1
        if I0 == 0:
            I0 = 1E-5
        elif I0 < 0:
            I0 = abs(I0)
        Pq /= I0

        # make I0 scale with volume fraction (concentration) and 
        # volume squared and scale so default values gives I(0) of approx unity

        I0 *= conc * volume_total * 1E-4

        return I0, Pq
    
    def calc_Iq(self, Pq: np.ndarray, 
                S_eff: np.ndarray, 
                sigma_r: float) -> np.ndarray:
        """
        calculates intensity
        """

        ## save structure factor to file
        #self.save_S(self.q, S_eff, Model)

        ## multiply formfactor with structure factor
        I = Pq * S_eff

        ## interface roughness (Skar-Gislinge et al. 2011, DOI: 10.1039/c0cp01074j)
        if sigma_r > 0.0:
            roughness = np.exp(-(self.q * sigma_r)**2 / 2)
            I *= roughness

        return I

    def save_I(self, I: np.ndarray, Model: str):
        """Save theoretical intensity to file"""

        with open('Iq_%s.dat' % Model,'w') as f:
            f.write('# Theoretical SAS data\n')
            f.write('# %-12s %-12s\n' % ('q','I'))
            for i in range(len(I)):
                f.write('  %-12.5e %-12.5e\n' % (self.q[i], I[i]))

class IExperimental:
    def __init__(self, 
                 q: np.ndarray, 
                 I0: np.ndarray, 
                 I: np.ndarray, 
                 exposure: float):
        self.q = q
        self.I0 = I0
        self.I = I
        self.exposure = exposure

    def simulate_data(self) -> Vector2D:
        """
        Simulate SAXS data using calculated scattering and empirical expression for sigma

        input
        q,I      : calculated scattering, normalized
        I0       : forward scattering
        #noise   : relative noise (scales the simulated sigmas by a factor)
        exposure : exposure (in arbitrary units) - affects the noise level of data

        output
        sigma    : simulated noise
        Isim     : simulated data

        data is also written to a file
        """

        ## simulate exp error
        #input, sedlak errors (https://doi.org/10.1107/S1600576717003077)
        #k = 5000000
        #c = 0.05
        #sigma = noise*np.sqrt((I+c)/(k*q))

        ## simulate exp error, other approach, also sedlak errors

        # set constants
        k = 4500
        c = 0.85

        # convert from intensity units to counts
        scale = self.exposure
        I_sed = scale * self.I0 * self.I

        # make N
        N = k * self.q # original expression from Sedlak2017 paper

        qt = 1.4 # threshold - above this q value, the linear expression do not hold
        a = 3.0 # empirical constant 
        b = 0.6 # empirical constant
        idx = np.where(self.q > qt)
        N[idx] = k * qt * np.exp(-0.5 * ((self.q[idx] - qt) / b)**a)

        # make I(q_arb)
        q_max = np.amax(self.q)
        q_arb = 0.3
        if q_max <= q_arb:
           I_sed_arb = I_sed[-2]
        else: 
            idx_arb = np.where(self.q > q_arb)[0][0]
            I_sed_arb = I_sed[idx_arb]

        # calc variance and sigma
        v_sed = (I_sed + 2 * c * I_sed_arb / (1 - c)) / N
        sigma_sed = np.sqrt(v_sed)

        # rescale
        #sigma = noise * sigma_sed/scale
        sigma = sigma_sed / scale

        ## simulate data using errors
        mu = self.I0 * self.I
        Isim = np.random.normal(mu, sigma)

        return Isim, sigma

    def save_Iexperimental(self, Isim: np.ndarray, sigma: np.ndarray, Model: str):
        with open('Isim_%s.dat' % Model,'w') as f:
            f.write('# Simulated SAS data with noise\n')
            f.write('# sigma generated using Sedlak et al, k=100000, c=0.55, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)\n')
            f.write('# %-12s %-12s %-12s\n' % ('q','I','sigma'))
            for i in range(len(Isim)):
                f.write('  %-12.5e %-12.5e %-12.5e\n' % (self.q[i], Isim[i], sigma[i]))


def get_max_dimension(x_list: np.ndarray, y_list: np.ndarray, z_list: np.ndarray) -> float:
    """
    find max dimensions of n models
    used for determining plot limits
    """

    max_x,max_y,max_z = 0, 0, 0
    for i in range(len(x_list)):
        tmp_x = np.amax(abs(x_list[i]))
        tmp_y = np.amax(abs(y_list[i]))
        tmp_z = np.amax(abs(z_list[i]))
        if tmp_x>max_x:
            max_x = tmp_x
        if tmp_y>max_y:
            max_y = tmp_y
        if tmp_z>max_z:
            max_z = tmp_z

    max_l = np.amax([max_x,max_y,max_z])

    return max_l


def plot_2D(x_list: np.ndarray, 
            y_list: np.ndarray, 
            z_list: np.ndarray, 
            sld_list: np.ndarray, 
            Models: np.ndarray, 
            high_res: bool,
            colors: List[str]) -> None:
    """
    plot 2D-projections of generated points (shapes) using matplotlib:
    positive contrast in red (Model 1) or blue (Model 2) or yellow (Model 3) or green (Model 4)
    zero contrast in grey
    negative contrast in black

    input
    (x_list,y_list,z_list) : coordinates of simulated points
    sld_list               : excess scattering length densities (contrast) of simulated points
    Model                  : Model number

    output
    plot                   : points<Model>.png

    """

    ## figure settings
    markersize = 0.5
    max_l = get_max_dimension(x_list, y_list, z_list)*1.1
    lim = [-max_l, max_l]

    for x,y,z,p,Model,color in zip(x_list,y_list,z_list,sld_list,Models,colors):

        ## find indices of positive, zero and negatative contrast
        idx_neg = np.where(p < 0.0)
        idx_pos = np.where(p > 0.0)
        idx_nul = np.where(p == 0.0)

        f,ax = plt.subplots(1,3,figsize=(12,4))

        ## plot, perspective 1
        ax[0].plot(x[idx_pos], z[idx_pos], linestyle='none', marker='.', markersize=markersize, color=color)
        ax[0].plot(x[idx_neg], z[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[0].plot(x[idx_nul], z[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')
        ax[0].set_xlim(lim)
        ax[0].set_ylim(lim)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('z')
        ax[0].set_title('pointmodel, (x,z), "front"')

        ## plot, perspective 2
        ax[1].plot(y[idx_pos], z[idx_pos], linestyle='none', marker='.', markersize=markersize, color=color) 
        ax[1].plot(y[idx_neg], z[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[1].plot(y[idx_nul], z[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')
        ax[1].set_xlim(lim)
        ax[1].set_ylim(lim)
        ax[1].set_xlabel('y')
        ax[1].set_ylabel('z')
        ax[1].set_title('pointmodel, (y,z), "side"')

        ## plot, perspective 3
        ax[2].plot(x[idx_pos], y[idx_pos], linestyle='none', marker='.', markersize=markersize, color=color) 
        ax[2].plot(x[idx_neg], y[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[2].plot(x[idx_nul], y[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')    
        ax[2].set_xlim(lim)
        ax[2].set_ylim(lim)
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('y')
        ax[2].set_title('pointmodel, (x,y), "bottom"')
    
        plt.tight_layout()
        if high_res:
            plt.savefig('points_%s.png' % Model,dpi=600)
        else:
            plt.savefig('points_%s.png' % Model)
        plt.close()

def plot_results(q: np.ndarray, 
                 r_list: List[np.ndarray], 
                 pr_list: List[np.ndarray], 
                 I_list: List[np.ndarray], 
                 Isim_list: List[np.ndarray], 
                 sigma_list: List[np.ndarray], 
                 S_list: List[np.ndarray], 
                 name_list: List[str], 
                 scales: List[float], 
                 xscale_lin: bool, 
                 high_res: bool,
                 colors: List[str]) -> None:
    """
    plot results for all models, using matplotlib:
    - p(r) 
    - calculated formfactor, P(r) on log-log or log-lin scale
    - simulated noisy data on log-log or log-lin scale

    """
    fig, ax = plt.subplots(1,3,figsize=(12,4))

    zo = 1
    for (r, pr, I, Isim, sigma, S, model_name, scale,color) in zip (r_list, pr_list, I_list, Isim_list, sigma_list, S_list, name_list, scales, colors):
        ax[0].plot(r,pr,zorder=zo, color=color, label='p(r), %s' % model_name)

        if scale > 1: 
            ax[2].errorbar(q,Isim*scale,yerr=sigma*scale,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s, scaled by %d' % (model_name,scale),zorder=1/zo)
        else:
            ax[2].errorbar(q,Isim*scale,yerr=sigma*scale,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s' % model_name,zorder=zo)

        if S[0] != 1.0 or S[-1] != 1.0:
            ax[1].plot(q, S, linestyle='--', color=color, zorder=0, label=r'$S(q)$, %s' % model_name)
            ax[1].plot(q, I,color=color, zorder=zo, label=r'$I(q)=P(q)S(q)$, %s' % model_name)
            ax[1].set_ylabel(r'$I(q)=P(q)S(q)$')
        else:
            ax[1].plot(q, I, zorder=zo, color=color, label=r'$P(q)=I(q)/I(0)$, %s' % model_name)
            ax[1].set_ylabel(r'$P(q)=I(q)/I(0)$')
        zo += 1

    ## figure settings, p(r)
    ax[0].set_xlabel(r'$r$ [$\mathrm{\AA}$]')
    ax[0].set_ylabel(r'$p(r)$')
    ax[0].set_title('pair distance distribution function')
    ax[0].legend(frameon=False)

    ## figure settings, calculated scattering
    if not xscale_lin:
        ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[1].set_title('normalized scattering, no noise')
    ax[1].legend(frameon=False)

    ## figure settings, simulated scattering
    if not xscale_lin:
        ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[2].set_ylabel(r'$I(q)$ [a.u.]')
    ax[2].set_title('simulated scattering, with noise')
    ax[2].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    if high_res:
        plt.savefig('plot.png', dpi=600)
    else:
        plt.savefig('plot.png')
    plt.close()
    
def plot_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list, scales, high_res, colors):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    zo = 1
    for (d, G, Gsim, sigmaG, model_name, scale, color) in zip (delta_list, G_list, Gsim_list, sigma_G_list, name_list, scales, colors):

        ax[0].plot(d, G, zorder=zo, color=color,label=r'$G$, %s' % model_name)
        #ax[0].set_ylabel(r'$G$ [a.u.]')
        ax[0].set_ylabel(r'$G(\delta)$ [cm$^{-1}$]')
        ax[0].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[0].set_title('theoretical SESANS, no noise')
        ax[0].legend(frameon=False)
                         
        if scale > 1: 
            ax[1].errorbar(d,Gsim*scale,yerr=sigmaG*scale,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s, scaled by %d' % (model_name,scale),zorder=1/zo)
        else:
            ax[1].errorbar(d,Gsim*scale,yerr=sigmaG*scale,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s' % model_name,zorder=zo)
        ax[1].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[1].set_ylabel(r'$\ln(P)/(t\lambda^2)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
        ax[1].set_title('simulated SESANS, with noise')
        ax[1].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    if high_res:
        plt.savefig('sesans.png', dpi=600)
    else:
        plt.savefig('sesans.png')
    plt.close()

def save_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list):

    for (d, G, Gsim, sigmaG, model_name) in zip (delta_list, G_list, Gsim_list, sigma_G_list, name_list):
        model_name 
        with open('G_%s.ses' % model_name,'w') as f:
            f.write('# Theoretical SESANS data\n')
            f.write('# %-12s %-12s\n' % ('delta','G'))
            for i in range(len(d)):
                f.write('  %-12.5e %-12.5e\n' % (d[i], G[i]))
        
        with open('Gsim_%s.ses' % model_name,'w') as f:
            f.write('# Simulated SESANS data, with noise\n')
            f.write('# %-12s %-12s %-12s\n' % ('delta','G','sigma_G'))
            for i in range(len(d)):
                f.write('  %-12.5e %-12.5e %-12.5e\n' % (d[i], Gsim[i], sigmaG[i]))

def generate_pdb(x_list: List[np.ndarray], 
                 y_list: List[np.ndarray], 
                 z_list: List[np.ndarray], 
                 sld_list: List[np.ndarray], 
                 Model_list: List[str]) -> None:
    """
    Generates a visualisation file in PDB format with the simulated points (coordinates) and contrasts
    ONLY FOR VISUALIZATION!
    Each bead is represented as a dummy atom
    Carbon, C : positive contrast
    Hydrogen, H : zero contrast
    Oxygen, O : negateive contrast
    information of accurate contrasts not included, only sign
    IMPORTANT: IT WILL NOT GIVE THE CORRECT RESULTS IF SCATTERING IS CACLLUATED FROM THIS MODEL WITH E.G. CRYSOL, PEPSI-SAXS, FOXS, CAPP OR THE LIKE!
    """

    for (x,y,z,p,Model) in zip(x_list, y_list, z_list, sld_list, Model_list):
        with open('%s.pdb' % Model,'w') as f:
            f.write('TITLE    POINT SCATTER FOR MODEL: %s\n' % Model)
            f.write('REMARK   GENERATED WITH Shape2SAS\n')
            f.write('REMARK   EACH BEAD REPRESENTED BY DUMMY ATOM\n')
            f.write('REMARK   CARBON, C : POSITIVE EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   HYDROGEN, H : ZERO EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   OXYGEN, O : NEGATIVE EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   ACCURATE SCATTERING LENGTH DENSITY INFORMATION NOT INCLUDED\n')
            f.write('REMARK   OBS: WILL NOT GIVE CORRECT RESULTS IF SCATTERING IS CALCULATED FROM THIS MODEL WITH E.G CRYSOL, PEPSI-SAXS, FOXS, CAPP OR THE LIKE!\n')
            f.write('REMARK   ONLY FOR VISUALIZATION, E.G. WITH PYMOL\n')
            f.write('REMARK    \n')
            for i in range(len(x)):
                if p[i] > 0:
                    atom = 'C'
                elif p[i] == 0:
                    atom = 'H'
                else:
                    atom = 'O'
                f.write('ATOM  %6i %s   ALA A%6i  %8.3f%8.3f%8.3f  1.00  0.00           %s \n'  % (i,atom,i,x[i],y[i],z[i],atom))
            f.write('END')


def check_unique(A_list: List[float]) -> bool:
    """
    if all elements in a list are unique then return True, else return False
    """
    unique = True
    N = len(A_list)
    for i in range(N):
        for j in range(N):
            if i != j:
                if A_list[i] == A_list[j]:
                    unique = False

    return unique

def calc_G_sesans(q,delta,I) -> np.ndarray:
    """
    Calculated projected correlation function for SESANS from Hankel Transform of I(q)
    """

    # Init empty G(delta)
    G = np.empty(len(delta), dtype=float)

    # calculate G(delta) from I(q)
    for i, delta_i in enumerate(delta):
        dq_int = q[1] - q[0]
        G[i] = 1 / 2 / np.pi * np.sum(dq_int * q * I * j0(delta_i * q))

    return G

def simulate_data_sesans(self) -> Vector2D:
    """
    Simulate SESANS data using calculated scattering and estimate for sigma

    input
    delta, G: spin-echo lengths and corresponding theoretical G_delta
    sesans_noise: baseline noise level

    output
    sesans_sigma: simulated errors
    Gsim: simulated data

    data is also written to a file
    """

    # Compute baseline noise as sesans_noise % of min(G-G(0))
    noise_baseline = self.sesans_noise * np.abs(np.min(self.G - self.G[0]))

    # Compute delta-dependent noise as function of baseline noise
    m = 1/50000 # 1/50000 adds a baseline worth of noise per 5 micrometers of additional spin echo length (delta)
    sesans_sigma = np.linspace(noise_baseline, noise_baseline * (1 + m * (self.delta[-1] - self.delta[0])), np.size(self.delta))

    # Simulate SESANS data using errors
    lnPsim = np.random.normal((self.G - self.G[0]), sesans_sigma)

    return lnPsim, sesans_sigma

def str2bool(v):
    """
    Function to circumvent the argparse default behaviour 
    of not taking False inputs, when default=True.
    """
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def separate_string(arg):
    arg = re.split('[ ,]+', arg)
    return [str(i) for i in arg]

def float_list(arg):
    """
    Function to convert a string to a list of floats.
    Note that this function can interpret numbers with scientific notation 
    and negative numbers.

    input:
        arg: string, input string

    output:
        list of floats
    """
    arg = re.sub(r'\s+', ' ', arg.strip())
    arg = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", arg)
    return [float(i) for i in arg]

def parse_3_floats(s):
    # accept both commas and spaces
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected exactly 3 floats (x,y,z), got {len(parts)}: {s}"
        )
    return [float(x) for x in parts]

def check_3Dinput(input: list, default: list, name: str, N_subunits: int, i: int):
    """
    Function to check if 3D vector input matches 
    in lenght with the number of subunits

    input:
        input: list of floats, input values
        default: list of floats, default values

    output:
        list of floats
    """
    try:
        inputted = input[i]
        if len(inputted) != N_subunits:
            warnings.warn(f"The number of subunits and {name} do not match. Using {default}")
            inputted = default * N_subunits
    except:
        inputted = default * N_subunits
        #warnings.warn(f"Could not find {name}. Using default {default}.")

    return inputted

def check_input(input: float, default: float, name: str, i: int):
    """
    Function to check if input is given, 
    if not, use default value.

    input:
        input: float, input value
        default: float, default value
        name: string, name of the input

    output:
        float
    """
    try:
        inputted = input[i]
    except:
        inputted = default
        #warnings.warn(f"Could not find {name}. Using default {default}.")

    return inputted

@dataclass
class ModelPointDistribution:
    """
    Point distribution of a model
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sld: np.ndarray #scattering length density for each point
    volume_total: float

@dataclass
class SimulationParameters:
    """
    Class containing parameters for the simulation and default parameters
    """

    qmin: float = 0.001
    qmax: float =  0.5
    qpoints: int = 400
    prpoints: int = 100
    Npoints: int = 8000
    model_name: List[str] = field(default_factory=lambda: ['Model_1'])

@dataclass
class ModelSystem:
    """
    Parameters for the system
    """

    PointDistribution: ModelPointDistribution
    Stype: str = field(default_factory=lambda: "None") #structure factor
    par: List[float] = field(default_factory=np.ndarray) #parameters for structure factor
    polydispersity: float = 0.0 #polydispersity
    conc: float = 0.02 #concentration, as volume fraction
    sigma_r: float = 0.0 #interface roughness

@dataclass
class TheoreticalScatteringCalculation:
    """Class containing parameters for simulating
    scattering for a given model system"""

    System: ModelSystem
    Calculation: SimulationParameters

@dataclass
class TheoreticalScattering:
    """Class containing parameters for
    theoretical scattering"""

    q: np.ndarray
    I0: np.ndarray
    I: np.ndarray
    S_eff: np.ndarray
    r: np.ndarray #pair distance distribution
    pr: np.ndarray #pair distance distribution
    pr_norm: np.ndarray #normalized pair distance distribution

@dataclass
class SimulateScattering:
    """Class containing parameters for
    simulating scattering"""

    q: np.ndarray = field(default_factory=np.ndarray)
    I0: np.ndarray = field(default_factory=np.ndarray)
    I: np.ndarray = field(default_factory=np.ndarray)
    exposure: float = 500

@dataclass
class SimulatedScattering:
    """Class containing parameters for
    simulated scattering"""

    I_sim: np.ndarray
    q: np.ndarray
    I_err: np.ndarray

def getTheoreticalScattering(scalc: TheoreticalScatteringCalculation) -> TheoreticalScattering:
    """Calculate theoretical scattering for a given model."""
    sys = scalc.System
    prof = sys.PointDistribution
    calc = scalc.Calculation
    x = np.concatenate(prof.x)
    y = np.concatenate(prof.y)
    z = np.concatenate(prof.z)
    p = np.concatenate(prof.sld)

    r, pr, pr_norm = WeightedPairDistribution(x, y, z, p).calc_pr(calc.prpoints, sys.polydispersity)

    print('        calculating scattering...')
    q = np.linspace(calc.qmin, calc.qmax, calc.qpoints)
    I_theory = ITheoretical(q)
    I0, Pq = I_theory.calc_Pq(r, pr, sys.conc, prof.volume_total)

    S_class = StructureFactor(q, x, y, z, p, sys.Stype, sys.par)
    S_eff = S_class.getStructureFactor().structure_eff(Pq)

    I = I_theory.calc_Iq(Pq, S_eff, sys.sigma_r)

    return TheoreticalScattering(q=q, I=I, I0=I0, S_eff=S_eff, r=r, pr=pr, pr_norm=pr_norm)

def getSimulatedScattering(scalc: SimulateScattering) -> SimulatedScattering:
    """Simulate scattering for a given theoretical scattering."""

    Isim_class = IExperimental(scalc.q, scalc.I0, scalc.I, scalc.exposure)
    I_sim, I_err = Isim_class.simulate_data()


    return SimulatedScattering(I_sim=I_sim, q=scalc.q, I_err=I_err)

def getPointDistribution(subunit_type,sld,dimensions,com,rotation,exclude_overlap,Npoints):
    x_new, y_new, z_new, sld_new, volume_total = GenerateAllPoints(Npoints, com, subunit_type, dimensions, rotation, sld, exclude_overlap).onGeneratingAllPointsSeparately()
    return ModelPointDistribution(x=x_new, y=y_new, z=z_new, sld=sld_new, volume_total=volume_total)

def simulate_sesans(delta,G,error):
    """
    Simulate SESANS data using calculated scattering and estimate for sigma

    input
    delta, G: spin-echo lengths and theoretical G(delta)
    error: relative error

    output
    sesans_sigma: simulated errors
    lnPsim: simulated data

    """
    # Compute baseline noise as sesans_noise % of min(G-G(0))
    noise_baseline = error * np.abs(np.min(G - G[0]))
    # Compute delta-dependent noise as function of baseline noise
    m = 1/50000 # 1/50000 adds a baseline worth of noise per 5 micrometers of additional spin echo length (delta)
    d_delta = delta[-1] - delta[0]
    sesans_sigma = np.linspace(noise_baseline, noise_baseline * (1 + m * d_delta), len(delta))
    # pick random points using mean and sigma
    lnPsim = np.random.normal((G - G[0]), sesans_sigma)
    return lnPsim,sesans_sigma

def save_points(x,y,z,sld,model_nam):
        with open('points_%s.txt' % model_nam,'w') as f:
            f.write('# x y z sld\n')
            for xi,yi,zi,s in zip(x,y,z,sld):
                f.write('%f %f %f %f\n' % (xi,yi,zi,s))
