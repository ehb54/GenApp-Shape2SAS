import numpy as np
import matplotlib.pyplot as plt
from fast_histogram import histogram1d #histogram1d from fast_histogram is faster than np.histogram (https://pypi.org/project/fast-histogram/) 

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

def gen_points(x_com,y_com,z_com,model,a,b,c,p,Npoints):
    """ 
    generate random uniformly distributed points (x,y,z) in user-defined shape
    model defines the shape in which the points should be placed
    
    input:
    a,b,c             : geometrical model parameters
    p                 : excess scattering length density (contrast)
    x_com,y_com,z_com : centre-of-mass position
    Npoints           : attempt to generate this number of points
    
    output: 
    x_add,y_add,z_add : coordinates of simulated points
    p_add             : contrast of each point
    N_add             : points generated
    rho               : point density
    """
    
    if model in ['sphere','hollow_sphere']:
        """
        a  : (outer) radius
        b  : (only hollow_sphere) inner radius
        
        a shell if b=a (only hollow_sphere)

        """
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
        """
        a,b,c : axes of tri-axial elllipsoid
        """
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
        """
        a,b : axes of elliptical cross section
        l   : cylinder length / disc height
        """
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
        """
        a  : side length
        b  : (hollow cube only) inner side length
        
        a shell if b=a

        """
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
        """
        a,b,c : side lengths
        """
        Volume = a*b*c
        N = Npoints
        x = np.random.uniform(-a/2,a/2,N)
        y = np.random.uniform(-b/2,b/2,N)
        z = np.random.uniform(-c/2,c/2,N)
        x_add,y_add,z_add = x,y,z

    if model in ['cyl_ring','disc_ring']:
        """
        a  : outer radius
        b  : inner radius
        l  : length

        a shell if b=a

        """
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

def gen_all_points(Number_of_models,x_com,y_com,z_com,model,a,b,c,p,exclude_overlap):
    """
    generate points from a collection of objects
    calling gen_points() for each object

    input:
    x_com,y_com,z_com : center of mass coordinates
    a,b,c             : model params (see GUI)
    p                 : contrast for each object
    exclude_overlap   : if True, points is removed from overlapping regions (True/False)

    output:
    N         : number of points in each model (before optional exclusion, see N_exclude)
    rho       : density of points for each model
    N_exclude : number of points excluded from each model (due to overlapping regions)
    x_new,y_new,z_new : coordinates of generated points
    p_new     : contrasts for each point 
    volume    : volume of each object
    """

    ## Total number of points should not exceed N_points_max (due to memory limitations). This number is system-dependent
    N_points_max = 5000        
    
    ## calculate volume and sum of volume (for calculating the number of points in each shape)
    volume = []
    sum_vol = 0
    for i in range(Number_of_models):
        v = get_volume(model[i],a[i],b[i],c[i])
        volume.append(v)
        sum_vol += v

    ## generate points
    N,rho,N_exclude = [],[],[]
    x_new,y_new,z_new,p_new = 0,0,0,0
    for i in range(Number_of_models):
        if model[i] != 'none': 
            Npoints = int(N_points_max * volume[i]/sum_vol)
            
            ## generate points
            x_add,y_add,z_add,p_add,N_model,rho_model = gen_points(x_com[i],y_com[i],z_com[i],model[i],a[i],b[i],c[i],p[i],Npoints)

            ## exclude overlap region (optional)
            N_x_sum = 0
            if exclude_overlap:
                for j in range(i):
                    x_add,y_add,z_add,p_add,N_x = check_overlap(x_add,y_add,z_add,p_add,x_com[j],y_com[j],z_com[j],model[j],a[j],b[j],c[j])
                    N_x_sum += N_x
            
            ## append points to vector
            x_new,y_new,z_new,p_new = append_points(x_new,y_new,z_new,p_new,x_add,y_add,z_add,p_add)

            ## append to lists
            N.append(N_model)
            rho.append(rho_model)
            N_exclude.append(N_x_sum)
        else:
            N.append(0)
            rho.append(0.0)
            N_exclude.append(0)
    return N,rho,N_exclude,volume,x_new,y_new,z_new,p_new

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
    
    ## find indices of non-excluded points, depending on model
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

    if model == 'none':
        ## do not exclude any points
        x_add,y_add,z_add,p_add = x,y,z,p 
        N_x = 0
    else:
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
    dist = abs(m-n) 

    return dist

def calc_all_dist(x_new,y_new,z_new):
    """ 
    calculate all pairwise distances
    calls calc_dist() for each set of coordinates: x,y,z
    does a square sum of coordinates
    convert from matrix to 
    """
    square_sum = 0.0
    for arr in [x_new,y_new,z_new]:
        square_sum += calc_dist(arr)**2
    d = np.sqrt(square_sum)
    # convert from matrix to array
    # reshape is slightly faster than flatten() and ravel()
    dist = d.reshape(-1)
    # reduce precision, for computational speed
    dist = dist.astype('float32')

    return dist

def calc_all_contrasts(p_new):
    """
    calculate all pairwise contrast products
    p_new: all contrasts 
    """
    dp = np.outer(p_new,p_new)
    contrast = dp.reshape(-1)
    contrast = contrast.astype('float32')
    
    return contrast

def generate_histogram(dist,contrast,r_max,Nbins):
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
    histo    : histogram, weighted by contrast

    """

    histo,bin_edges = np.histogram(dist,bins=Nbins,weights=contrast,range=(0,r_max)) 
    dr = bin_edges[2]-bin_edges[1]
    r = bin_edges[0:-1]+dr/2
    
    return r,histo

def calc_Rg(r,pr):
    """ 
    calculate Rg from r and p(r)
    """
    
    sum_pr_r2 = np.sum(pr*r**2)
    sum_pr = np.sum(pr)
    Rg = np.sqrt(abs(sum_pr_r2/sum_pr)/2)
    
    return Rg

def calc_S_HS(q,eta,R):
    """
    calculate the hard-sphere structure factor
    calls function calc_G()
    
    input
    q       : momentum transfer
    eta     : volume fraction
    R       : estimation of the hard-sphere radius
    
    output
    S_HS    : hard-sphere structure factor
    """

    if eta > 0.0:
        A = 2*R*q 
        G = calc_G(A,eta)
        S_HS = 1/(1 + 24*eta*G/A)
    else:
        S_HS = np.ones(len(q))

    return S_HS

def calc_G(A,eta):
    """ 
    calculate G in the hard-sphere potential
    
    input
    A  : 2*R*q
    q  : momentum transfer
    R  : hard-sphere radius
    eta: volume fraction
    
    output:
    G  
    """

    a = (1+2*eta)**2/(1-eta)**4
    b = -6*eta*(1+eta/2)**2/(1-eta)**4 
    c = eta * a/2
    sinA = np.sin(A)
    cosA = np.cos(A)
    fa = sinA-A*cosA
    fb = 2*A*sinA+(2-A**2)*cosA-2
    fc = -A**4*cosA + 4*((3*A**2-6)*cosA+(A**3-6*A)*sinA+6)
    G = a*fa/A**2 + b*fb/A**3 + c*fc/A**5
    
    return G

def generate_q(qmin,qmax,Nq):
    """
    generate q-vector
    equidistance on linear scale
    """

    q = np.linspace(qmin,qmax,Nq)

    return q

def save_S(q,S,Model):
    """ 
    save S to file
    """

    with open('Sq%s.dat' % Model,'w') as f:
        f.write('# Structure factor, S(q), used in: I(q) = P(q)*S(q)\n')
        f.write('# Default: S(q) = 1.0\n')
        f.write('# %-17s %-17s\n' % ('q','S(q)'))
        for (q_i,S_i) in zip(q,S):
            f.write('  %-17.5e%-17.5e\n' % (q_i,S_i))

def calc_S_aggr(q,Reff,Naggr):
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
    
    qR = q*Reff
    S_aggr = 1 + (Naggr-1)/(1+qR**2*Naggr/3)

    return S_aggr

def calc_com_dist(x_new,y_new,z_new,p_new):
    """ 
    calc contrast-weighted com distance
    """
    w = np.abs(p_new)
    if np.sum(w) == 0:
        w = np.ones(len(x_new))
    x_com,y_com,z_com = np.average(x_new,weights=w),np.average(y_new,weights=w),np.average(z_new,weights=w)
    dx,dy,dz = x_new-x_com,y_new-y_com,z_new-z_com
    com_dist = np.sqrt(dx**2+dy**2+dz**2)
    
    return com_dist

def calc_A00(q,x_new,y_new,z_new,p_new):
    """
    calc zeroth order sph harm, for decoupling approximation
    """
    d_new = calc_com_dist(x_new,y_new,z_new,p_new)
    M = len(q)
    A00 = np.zeros(M)
    for i in range(M):
        qr = q[i]*d_new
        A00[i] = sum(p_new*sinc(qr))
    A00 = A00/A00[0] # normalise, A00[0] = 1

    return A00

def calc_Pq(q,r,pr):
    """
    calculate form factor using histogram
    """
    ## calculate formfactor P(q) from p(r)
    Pq = 0.0
    for (r_i,pr_i) in zip(r,pr):
        qr = q*r_i
        Pq += pr_i*sinc(qr)
    Pq /= np.amax(Pq) # normalization

    return Pq

def decoupling_approx(q,x_new,y_new,z_new,p_new,Pq,S):
    """
    modify structure factor with the decoupling approximation
    for combining structure factors with non-spherical (or polydisperse) objects
    
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
    A00 = calc_A00(q,x_new,y_new,z_new,p_new)
    const = 1e-3 # add constant in nominator and denominator, for stability (numerical errors for small values dampened)
    Beta = (A00**2+const)/(Pq+const)
    S_eff = 1 + Beta*(S-1)
    
    return S_eff

def calc_Iq(q,Pq,S_eff,sigma_r,Model):
    """
    calculates intensity
    """

    ## save structure factor to file
    save_S(q,S_eff,Model)
    
    ## multiply formfactor with structure factor
    I = Pq*S_eff
     
    ## interface roughness (Skar-Gislinge et al. 2011, DOI: 10.1039/c0cp01074j)
    if sigma_r > 0.0:
        roughness = np.exp(-(q*sigma_r)**2/2)
        I *= roughness

    ## save intensity to file
    with open('Iq%s.dat' % Model,'w') as f:
        f.write('# %-17s %-17s\n' % ('q','I(q)'))
        for (q_i,I_i) in zip(q,I):
            f.write('  %-17.5e %-17.5e\n' % (q_i,I_i))
    
    return I

def simulate_data(q,I,noise,Model):
    """
    simulate data using calculated scattering and empirical expression for sigma

    input
    q,I    : calculated scattering
    noise  : relative noise (scales the simulated sigmas by a factor)
    Model  : is it Model 1 or Model 2 (see the GUI)

    output
    sigma  : simulated noise
    Isim   : simulated data

    data is also written to a file
    """

    ## simulate exp error
    #input, sedlak errors (https://doi.org/10.1107/S1600576717003077)
    k = 5000000
    c = 0.05
    mu = I
    sigma = noise*np.sqrt((mu+c)/(k*q))

    ## simulate data using errors
    Isim = np.random.normal(mu,sigma)

    ## save to file
    with open('Isim%s.dat' % Model,'w') as f:
        f.write('# Simulated data\n')
        f.write('# sigma generated using Sedlak et al, k=100000, c=0.55, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)\n')
        f.write('# %-12s %-12s %-12s\n' % ('q','I','sigma'))
        for i in range(len(Isim)):
            f.write('  %-12.5e %-12.5e %-12.5e\n' % (q[i],Isim[i],sigma[i]))
    
    return Isim,sigma

def calc_hr(dist,Nbins,contrast,polydispersity,Model):
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

    ## make r range in h(r) histogram slightly larger than Dmax
    ratio_rmax_dmax = 1.05

    ## calc h(r) with/without polydispersity
    if polydispersity > 0.0:
        Dmax = np.amax(dist) * (1+3*polydispersity)
        r_max = Dmax*ratio_rmax_dmax
        N_poly_integral = 7
        r,hr_1 = generate_histogram(dist,contrast,r_max,Nbins)
        hr = 0.0
        factor_range = 1 + np.linspace(-3,3,N_poly_integral)*polydispersity
        for factor_d in factor_range:
            if factor_d == 1.0:
                hr += hr_1
            else:
                dhr = histogram1d(dist*factor_d,bins=Nbins,weights=contrast,range=(0,r_max))
                res = (1.0-factor_d)/polydispersity
                w = np.exp(-res**2/2.0) # weight: normal distribution
                vol = factor_d**3 # weight: relative volume, because larger particles scatter more
                hr += dhr*w*vol**2
    else:
        Dmax = np.amax(dist)
        r_max = Dmax*ratio_rmax_dmax
        r,hr = generate_histogram(dist,contrast,r_max,Nbins)
    
    ## normalize so hr_max = 1 
    hr /= np.amax(hr) 
    
    ## calculate Rg
    Rg = calc_Rg(r,hr)

    return r,hr,Dmax,Rg

def calc_pr(dist,Nbins,contrast,polydispersity,Model):
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

    ## calculate pr
    idx_nonzero = np.where(dist>0.0) #  nonzero elements
    r,pr,Dmax,Rg = calc_hr(dist[idx_nonzero],Nbins,contrast[idx_nonzero],polydispersity,Model)

    ## save p(r) to textfile
    with open('pr%s.dat' % Model,'w') as f:
        f.write('# %-17s %-17s\n' % ('r','p(r)'))
        for i in range(Nbins):
            f.write('  %-17.5e %-17.5e\n' % (r[i],pr[i]))

    return r,pr,Dmax,Rg

def get_max_dimension(x1,y1,z1,x2,y2,z2):
    """
    find max dimensions of 2 models 
    used for determining plot limits
    """
    
    max_x = np.amax([np.amax(abs(x1)),np.amax(abs(x2))])
    max_y = np.amax([np.amax(abs(y1)),np.amax(abs(y2))])
    max_z = np.amax([np.amax(abs(z1)),np.amax(abs(z2))])
    max_l = np.amax([max_x,max_y,max_z])

    return max_l

def plot_2D(x_new,y_new,z_new,p_new,max_dimension,Model):
    """
    plot 2D-projections of generated points (shapes) using matplotlib:
    positive contrast in red/blue (Model 1/Model 2)
    zero contrast in grey
    negative contrast in green
    
    input
    (x_new,y_new,z_new) : coordinates of simulated points
    p_new               : excess scattering length density (contrast) of simulated points
    max_dimension       : max dimension of previous model (for plot limits)
    Model               : Model number (1 or 2) from GUI

    output
    plot      : points_<Model>.png

    """
    
    ## find max dimensions of model
    max_x = np.amax(abs(x_new))
    max_y = np.amax(abs(y_new))
    max_z = np.amax(abs(z_new))
    max_l = np.amax([max_x,max_y,max_z,max_dimension])*1.1
    lim = [-max_l,max_l]

    ## find indices of positive, zero and negatative contrast
    idx_neg = np.where(p_new<0.0)
    idx_pos = np.where(p_new>0.0)
    idx_nul = np.where(p_new==0.0)
    
    ## figure settings
    markersize = 0.5
    if Model == '':
        color = 'red'
    elif Model == '_2':
        color = 'blue'

    f,ax = plt.subplots(1,3,figsize=(12,4))
    
    ## plot, perspective 1
    ax[0].plot(x_new[idx_pos],z_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color=color)
    ax[0].plot(x_new[idx_neg],z_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    ax[0].plot(x_new[idx_nul],z_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('z')
    ax[0].set_title('pointmodel, (x,z), "front"')

    ## plot, perspective 2
    ax[1].plot(y_new[idx_pos],z_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color=color)
    ax[1].plot(y_new[idx_neg],z_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    ax[1].plot(y_new[idx_nul],z_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')    
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('z')
    ax[1].set_title('pointmodel, (y,z), "side"')

    ## plot, perspective 3
    ax[2].plot(x_new[idx_pos],y_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color=color)
    ax[2].plot(x_new[idx_neg],y_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    ax[2].plot(x_new[idx_nul],y_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')    
    ax[2].set_xlim(lim)
    ax[2].set_ylim(lim)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title('pointmodel, (x,y), "bottom"')
    
    plt.tight_layout()
    plt.savefig('points%s.png' % Model)
    plt.close()

def plot_results(q,r,pr,I,Isim,sigma,S,xscale_log):
    """
    plot results using matplotlib:
    - p(r) 
    - calculated scattering
    - simulated data with noise

    Shape2SAS uses this function if there is only 1 Model, else plot_results_combined() is used

    """
   
    ## plot settings
    fig,ax = plt.subplots(1,3,figsize=(12,4))
    color = 'red'

    ## plot p(r)
    ax[0].plot(r,pr,color=color,label='p(r), monodisperse')
    ax[0].set_xlabel('r [Angstrom]')
    ax[0].set_ylabel('p(r)')
    ax[0].set_title('pair distance distribution function')

    ## plot calculated scattering
    if xscale_log:
        ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('q [1/Angstrom]')
    ax[1].set_ylabel('I(q)')
    ax[1].set_title('calculated scattering, without noise')
    if S[0] != 1.0 or S[-1] != 1.0:
        ax[1].plot(q,S,color='black',label='S(q)')
        ax[1].plot(q,I,color=color,label='I(q) = P(q)*S(q)')
    else:
       ax[1].plot(q,I,color=color,label='I(q)')
    ax[1].legend(frameon=False)

    ## plot simulated scattering 
    if xscale_log:
        ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('q [1/Angstrom]')
    ax[2].set_ylabel('I(q)')
    ax[2].set_title('simulated scattering, with noise')
    ax[2].errorbar(q,Isim,yerr=sigma,linestyle='none',marker='.',color='firebrick',label='I(q), simulated',zorder=0)
    ax[2].legend(frameon=False)

    ## figure settings
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

def plot_results_combined(q,r1,pr1,I1,Isim1,sigma1,S1,r2,pr2,I2,Isim2,sigma2,S2,xscale_log,scale_Isim):
    """
    plot results (combined = Model 1 and Model 2), using matplotlib:
    - p(r) 
    - calculated formfactor, P(r) on log-log and lin-lin scale
    - simulated noisy data on log-log and lin-lin scale

    Shape2SAS uses this function if there is 2 Models are opted for in GUI, else plot_results() is used

    """

    fig,ax = plt.subplots(1,3,figsize=(12,4))

    for (r,pr,I,Isim,sigma,S,model,col,col_sim,line,scale,zo) in zip ([r1,r2],[pr1,pr2],[I1,I2],[Isim1,Isim2],[sigma1,sigma2],[S1,S2],[1,2],['red','blue'],['firebrick','royalblue'],['-','--'],[1,scale_Isim],[1,2]):
        ax[0].plot(r,pr,linestyle=line,color=col,zorder=zo,label='p(r), Model %d' % model)
        if scale > 1: 
            ax[2].errorbar(q,Isim*scale,yerr=sigma*scale,linestyle='none',marker='.',color=col_sim,label='Isim(q), Model %d, scaled by %d' % (model,scale),zorder=1/zo)
        else:
            ax[2].errorbar(q,Isim*scale,yerr=sigma*scale,linestyle='none',marker='.',color=col_sim,label='Isim(q), Model %d' % model,zorder=zo)
        if S[0] != 1.0 or S[-1] != 1.0:
            ax[1].plot(q,S,linestyle=line,color='black',label='S(q), Model %d' % model,zorder=0)
            ax[1].plot(q,I,linestyle=line,color=col,zorder=zo,label='I(q)=P(q)*S(q), Model %d' % model)
        else:
            ax[1].plot(q,I,linestyle=line,color=col,zorder=zo,label='I(q), Model %d' % model)

    ## figure settings, p(r)
    ax[0].set_xlabel('r [Angstrom]')
    ax[0].set_ylabel('p(r)')
    ax[0].set_title('pair distance distribution function')

    ## figure settings, calculated scattering
    if xscale_log:
        ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('q [1/Angstrom]')
    ax[1].set_ylabel('I(q)')
    ax[1].set_title('calculated scattering, without noise')
    ax[1].legend(frameon=False)

    ## figure settings, simulated scattering
    if xscale_log:
        ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('q [1/Angstrom]')
    ax[2].set_ylabel('I(q)')
    ax[2].set_title('simulated scattering, with noise')
    ax[2].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig('plot_combined.png')
    plt.close()

def generate_pdb(x_new,y_new,z_new,p_new,Model):
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

    with open('model%s.pdb' % Model,'w') as f:
        f.write('TITLE    POINT SCATTER MODEL\n')
        f.write('REMARK   GENERATED WITH McSIM\n')
        f.write('REMARK   EACH BEAD REPRESENTED BY DUMMY ATOM\n')
        f.write('REMARK   CARBON, C : POSITIVE EXCESS SCATTERING LENGTH\n')
        f.write('REMARK   HYDROGEN, H : ZERO EXCESS SCATTERING LENGTH\n')
        f.write('REMARK   OXYGEN, O : NEGATIVE EXCESS SCATTERING LENGTH\n')
        f.write('REMARK   ACCURATE SCATTERING LENGTH DENSITY INFORMATION NOT INCLUDED\n')
        f.write('REMARK   OBS: WILL NOT GIVE CORRECT RESULTS IF SCATTERING IS CALCULATED FROM THIS MODEL WITH E.G CRYSOL, PEPSI-SAXS, FOXS, CAPP OR THE LIKE!\n')
        f.write('REMARK   ONLY FOR VISUALIZATION, E.G. WITH PYMOL\n')
        f.write('REMARK    \n')
        for i in range(len(x_new)):
            if p_new[i] > 0:
                atom = 'C'
            elif p_new[i] == 0:
                atom = 'H'
            else:
                atom = 'O'
            f.write('ATOM  %5i  %s   ALA A%4i    %8.3f%8.3f%8.3f  1.00  0.00           %s \n'  % (i,atom,i,x_new[i],y_new[i],z_new[i],atom))
        f.write('END')
