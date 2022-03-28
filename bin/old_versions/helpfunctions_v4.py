import numpy as np

def sinc(x):
    """
    function for calculating sinc = sin(x)/x
    numpy.sinc is defined as sinc(x) = sin(pi*x)/(pi*x)
    """
    return np.sinc(x/np.pi)   

def get_volume(model,a,b,c):
    """ 
    calculates volume for a given object
    """
    Volume = 0.0

    if model in ['sphere','hollow_sphere']:
        SHELL=False
        if model == 'hollow_sphere':
            if b > a:
                R,r = b,a
            elif b == a:
                SHELL=True
                R = a # do not use r
            else:
                R,r = a,b
        else:
            R,r = a,0
        if SHELL:
            Area = 4*np.pi*R**2
            Volume = Area # need a volume for calculating rho
        else:
            Volume = 4*np.pi*(R**3-r**3)/3
        
    if model == 'ellipsoid':
        Volume = 4*np.pi*a*b*c/3
        
    if model in ['cylinder','disc']:
        l = c
        Volume = np.pi*a*b*l

    if model in ['cube','hollow_cube']:
        SHELL = False
        if model == 'hollow_cube':
            if b > a:
                a,b = b,a
            elif b == a:
                SHELL = True
        else:
            b = 0
        if SHELL:
            Area = 6*a**2
            Volume = Area # need a volume for calculating rho
        else:
            Volume = a**3 - b**3

    if model == 'cuboid':
        Volume = a*b*c

    if model in ['cyl_ring','disc_ring']:
        SHELL = False
        if b > a:
            R,r,l = b,a,c
        elif b == a:
            SHELL = True
            R,l = a,c # do not use r
        else:
            R,r,l = a,b,c
        if SHELL:
            Area      = 2*np.pi*R*l
            Volume    = Area # need a volume for calculating rho
        else:
            Volume = np.pi*(R**2-r**2)*l
    
    return Volume

def genpoints(x_com,y_com,z_com,model,a,b,c,p,Npoints):
    """ 
    generate random uniformly distributed points (x,y,z) in user-defined shape
    model defines the shape in which the points should be placed
    a,b,c are model parameters
    x_com,y_com,z_com are the com position
    Npoints is the number of points to generate
    """
    
    if model in ['sphere','hollow_sphere']:
        SHELL=False
        if model == 'hollow_sphere':
            if b > a:
                R,r = b,a
            elif b == a:
                SHELL=True
                R = a # do not use r
            else:
                R,r = a,b
        else:
            R,r = a,0
        if SHELL:
            Area = 4*np.pi*R**2
            Volume = Area # need a volume for calculating rho later
            phi      = np.random.uniform(0,2*np.pi,Npoints)
            costheta = np.random.uniform(-1,1,Npoints)
            theta    = np.arccos(costheta)
            x = R*np.sin(theta)*np.cos(phi)
            y = R*np.sin(theta)*np.sin(phi)
            z = R*np.cos(theta)
            x_add,y_add,z_add = x,y,z
        else:
            Volume = 4*np.pi*(R**3-r**3)/3
            Volume_max = (2*R)**3
            Vratio = Volume_max/Volume
            N = int(Vratio*Npoints)
            x = np.random.uniform(-R,R,N)
            y = np.random.uniform(-R,R,N)
            z = np.random.uniform(-R,R,N)
            d = np.sqrt(x**2+y**2+z**2)
            idx = np.where((d<R)&(d>r))
            x_add,y_add,z_add = x[idx],y[idx],z[idx]

    if model == 'ellipsoid':
        Volume = 4*np.pi*a*b*c/3
        Volume_max = 2*a*2*b*2*c
        Vratio = Volume_max/Volume
        N = int(Vratio*Npoints)
        x = np.random.uniform(-a,a,N)
        y = np.random.uniform(-b,b,N)
        z = np.random.uniform(-c,c,N)
        d2 = x**2/a**2 + y**2/b**2 + z**2/c**2
        idx = np.where(d2<1)
        x_add,y_add,z_add = x[idx],y[idx],z[idx]

    if model in ['cylinder','disc']:
        l = c
        Volume = np.pi*a*b*l
        Volume_max = 2*a*2*b*l
        Vratio = Volume_max/Volume
        N = int(Vratio*Npoints)
        x = np.random.uniform(-a,a,N)
        y = np.random.uniform(-b,b,N)
        z = np.random.uniform(-l/2,l/2,N)
        d2 = x**2/a**2 + y**2/b**2
        idx = np.where(d2<1)
        x_add,y_add,z_add = x[idx],y[idx],z[idx]

    if model in ['cube','hollow_cube']:
        SHELL = False
        if model == 'hollow_cube':
            if b > a:
                a,b = b,a
            elif b == a:
                SHELL = True
        else:
            b = 0
        if SHELL:
            Area = 6*a**2
            Volume = Area # need a volume for calculating rho later
            
            d = a/2
            N = int(Npoints/6)
            one = np.ones(N)
            
            #make each side of the cube at a time
            x,y,z = [],[],[]
            for sign in [-1,1]:
                x = np.concatenate((x,sign*one*d))
                y = np.concatenate((y,np.random.uniform(-d,d,N)))
                z = np.concatenate((z,np.random.uniform(-d,d,N)))
                
                x = np.concatenate((x,np.random.uniform(-d,d,N)))
                y = np.concatenate((y,sign*one*d))
                z = np.concatenate((z,np.random.uniform(-d,d,N)))

                x = np.concatenate((x,np.random.uniform(-d,d,N)))
                y = np.concatenate((y,np.random.uniform(-d,d,N)))
                z = np.concatenate((z,sign*one*d))
            
            x_add,y_add,z_add = x,y,z
        else:
            Volume_max = a**3
            Volume = a**3 - b**3
            Vratio = Volume_max/Volume
            N = int(Vratio*Npoints)
            x = np.random.uniform(-a/2,a/2,N)
            y = np.random.uniform(-a/2,a/2,N)
            z = np.random.uniform(-a/2,a/2,N)
            d = np.maximum.reduce([abs(x),abs(y),abs(z)])
            idx = np.where(d>=b/2)
            x_add,y_add,z_add = x[idx],y[idx],z[idx]

    if model == 'cuboid':
        Volume = a*b*c
        N = Npoints
        x = np.random.uniform(-a/2,a/2,N)
        y = np.random.uniform(-b/2,b/2,N)
        z = np.random.uniform(-c/2,c/2,N)
        x_add,y_add,z_add = x,y,z

    if model in ['cyl_ring','disc_ring']:
        SHELL = False
        if b > a:
            R,r,l = b,a,c
        elif b == a:
            SHELL = True
            R,l = a,c # do not use r
        else:
            R,r,l = a,b,c
        if SHELL:
            Area      = 2*np.pi*R*l
            Volume    = Area # need a volume for calculating rho later
            Nside     = Npoints

            # side
            phi = np.random.uniform(0,2*np.pi,Nside)
            x = R*np.cos(phi)
            y = R*np.sin(phi)
            z = np.random.uniform(-l/2,l/2,Nside)

            x_add,y_add,z_add = x,y,z
        else:
            Volume = np.pi*(R**2-r**2)*l
            Volume_max = (2*R)**2*l
            Vratio = Volume_max/Volume
            N = int(Vratio*Npoints)
            x = np.random.uniform(-R,R,N)
            y = np.random.uniform(-R,R,N)
            z = np.random.uniform(-l/2,l/2,N)
            d = np.sqrt(x**2 + y**2)
            idx = np.where((d<R)&(d>r))
            x_add,y_add,z_add = x[idx],y[idx],z[idx]

    ## shift by center of mass
    x_add += x_com
    y_add += y_com
    z_add += +z_com

    ## calculate point density
    N_add = len(x_add)
    rho = N_add/Volume
    
    ## make contrast vector
    p_add = np.ones(N_add)*p
     
    return x_add,y_add,z_add,p_add,N_add,rho 

def append_points(x_new,y_new,z_new,p_new,x_add,y_add,z_add,p_add):
    """
    append new points to vectors of point coordinates
    """

    ## add points to (x_new,y_new,z_new)
    if isinstance(x_new,int):
        # if these are the first points to append to (x_new,y_new,z_new)
        x_new = x_add
        y_new = y_add
        z_new = z_add
        p_new = p_add
    else:
        x_new = np.append(x_new,x_add)
        y_new = np.append(y_new,y_add)
        z_new = np.append(z_new,z_add)
        p_new = np.append(p_new,p_add)

    return x_new,y_new,z_new,p_new

def check_overlap(x,y,z,p,x_com,y_com,z_com,model,a,b,c):
    """
    check for overlap with previous models. 
    if overlap, the point is removed
    """
   
    ## declare variable idx
    idx = []
    
    ## effective coordinates, shifted by (x_com,y_com,z_com)
    x_eff,y_eff,z_eff = x-x_com,y-y_com,z-z_com
    
    ## find indices of excluded points, depending on model
    if model in ['sphere','hollow_sphere']:
        SHELL=False
        if model == 'hollow_sphere':
            if b > a:
                R,r = b,a
            elif b == a:
                SHELL=True
                R = a # do not use r
            else:
                R,r = a,b
        else:
            R,r = a,0
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        if SHELL:
            idx = np.where(d!=R)
        else:
            idx = np.where((d>R) | (d<r))

    if model == 'ellipsoid':
        d2 = x_eff**2/a**2 + y_eff**2/b**2 + z_eff**2/c**2
        idx = np.where(d2>1)

    if model in ['cylinder','disc']:
        d2 = x_eff**2/a**2 + y_eff**2/b**2
        idx = np.where((d2>1) | (abs(z_eff)>c/2))

    if model in ['cube','hollow_cube']:
        SHELL = False
        if model == 'hollow_cube':
            if b > a:
                a,b = b,a
            elif b == a:
                SHELL = True
        else:
            b = 0
        if SHELL:
            idx = np.where((abs(x_eff)!=a/2) | (abs(y_eff)!=a/2) | (abs(z_eff)!=a/2)) 
        else:
            idx = np.where((abs(x_eff)>=a/2) | (abs(y_eff)>=a/2) | (abs(z_eff)>=a/2) | ((abs(x_eff)<=b/2) & (abs(y_eff)<=b/2) & (abs(z_eff)<=b/2)))

    if model == 'cuboid':
        idx = np.where((abs(x_eff)>=a/2) | (abs(y_eff)>=b/2) | (abs(z_eff)>=c/2))

    if model in ['cyl_ring','disc_ring']:
        d = np.sqrt(x_eff**2 + y_eff**2)
        SHELL = False
        if b > a:
            R,r,l = b,a,c
        elif b == a:
            SHELL = True
            R,l = a,c # do not use r
        else:
            R,r,l = a,b,c
        if SHELL:
            idx = np.where((d!=R) | (abs(z_eff)>l/2))
        else:
            idx = np.where((d>R) | (d<r) | (abs(z_eff)>l/2))

    ## exclude points
    x_add,y_add,z_add,p_add = x[idx],y[idx],z[idx],p[idx]
    
    ## number of excluded points
    N_x = len(x)-len(idx[0])
    
    return x_add,y_add,z_add,p_add,N_x

def calc_dist(x):
    """
    calculate all distances between points in an array
    """
    # mesh this array so that you will have all combinations
    m,n = np.meshgrid(x,x,sparse=True)
    # get the distance via the norm
    return abs(m-n)

def calc_Rg(r,pr):
    """ 
    calculate Rg from r and p(r)
    """
    sum_pr_r2 = np.sum(pr*r**2)
    sum_pr = np.sum(pr)
    Rg = np.sqrt(abs(sum_pr_r2/sum_pr)/2)
    return Rg

def calc_S(q,R,eta):
    """
    calculate the hard-sphere potential
    q  : momentum transfer
    R  : hard-sphere radius
    eta: volume fraction
    """
    A = 2*R*q 
    #A = R*q # according to Pedersen1997
    G = calc_G(A,eta)
    S = 1/(1 + 24*eta*G/A)
    return S

def calc_G(A,eta):
    """ 
    calculate G in the hard-sphere potential
    A  : 2*R*q
    q  : momentum transfer
    R  : hard-sphere radius
    eta: volume fraction
    """
    a = (1+2*eta)**2/(1-eta)**4
    b = -6*eta*(1+eta/2)**2/(1-eta)**4 
    #b = -6*eta*(1+eta/2)**2/(1-eta)**2 # according to Pedersen1997
    c = eta * a/2
    sinA = np.sin(A)
    cosA = np.cos(A)
    fa = sinA-A*cosA
    fb = 2*A*sinA+(2-A**2)*cosA-2
    fc = -A**4*cosA + 4*((3*A**2-6)*cosA+(A**3-6*A)*sinA+6)
    G = a*fa/A**2 + b*fb/A**3 + c*fc/A**5
    return G
