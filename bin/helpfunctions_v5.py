import numpy as np
import matplotlib.pyplot as plt

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

def generate_hr(dist,polydispersity,Nbins,contrast):
    """
    make histogram of point pairs, h(r), binned after pair-distances, r
    used for calculating scattering (fast Debye)
    dist     : all pairwise distances
    Nbins    : number of bins in h(r)
    contrast : contrast of points
    """
    # find nonzero elements
    #idx_nonzero = np.where(dist>0.0)
    Dmax = np.amax(dist)
    Dmax_poly = Dmax*(1+3*polydispersity)
    hr,bin_edges = np.histogram(dist,bins=Nbins,weights=contrast,range=(0,Dmax_poly)) 
    dr = bin_edges[2]-bin_edges[1]
    r = bin_edges[0:-1]+dr/2
    #del bin_edges,dr
    
    return r,hr,Dmax,Dmax_poly

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
    if eta > 0.0:
        S = calc_S(q,r_eff,eta)
        A = 2*R*q 
        G = calc_G(A,eta)
        S = 1/(1 + 24*eta*G/A)
    else:
        S = np.ones(len(q))

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
    c = eta * a/2
    sinA = np.sin(A)
    cosA = np.cos(A)
    fa = sinA-A*cosA
    fb = 2*A*sinA+(2-A**2)*cosA-2
    fc = -A**4*cosA + 4*((3*A**2-6)*cosA+(A**3-6*A)*sinA+6)
    G = a*fa/A**2 + b*fb/A**3 + c*fc/A**5
    return G

def simulate_data(polydispersity,I_poly,S,I,noise,q):

    ## simulate exp error
    #input, sedlak errors (https://doi.org/10.1107/S1600576717003077)
    k = 10000
    c = 0.65
    if polydispersity > 0.0:
        mu = I_poly*S
    else:
        mu = I*S
    sigma = noise*np.sqrt((mu+c)/(k*q))

    ##pseudo-rebin
    Nrebin = 10 # keep every Nth point
    mu = mu[::Nrebin]
    qsim = q[::Nrebin]
    sigma = sigma[::Nrebin]/np.sqrt(Nrebin)

    ## simulate data using errors
    Isim = np.random.normal(mu,sigma)

    ## save to file
    with open('Isim.d','w') as f:
        f.write('# Simulated data\n# sigma generated using Sedlak et al, k=10000, c=0.85, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)\n# q I sigma\n')
        for i in range(len(Isim)):
            f.write('%f %f %f\n' % (qsim[i],Isim[i],sigma[i]))
    
    return qsim,Isim,sigma

def plot_2D(x_new,y_new,z_new,p_new):
    """
    plot 2D-projections of generated points (shapes) using matplotlib:
    positive contrast in green
    zero contrast in grey
    negative contrast in red

    (x_new,y_new,z_new) : coordinates of simulated points
    p_new               : excess scattering length density (contrast) of simulated points
    """

    ## find plot limits
    max_x = np.amax(abs(x_new))
    max_y = np.amax(abs(y_new))
    max_z = np.amax(abs(z_new))
    max_l = np.amax([max_x,max_y,max_z])
    lim = [-max_l,max_l]

    ## figure
    #indices of negatative contrast
    idx_neg = np.where(p_new<0.0)
    idx_pos = np.where(p_new>0.0)
    idx_nul = np.where(p_new==0.0)
    plt.figure(figsize=(15,10))
    markersize = 3
    p1 = plt.subplot(2,3,1)
    p1.plot(x_new[idx_pos],z_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color='red')
    p1.plot(x_new[idx_neg],z_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    p1.plot(x_new[idx_nul],z_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')
    p1.set_xlim(lim)
    p1.set_ylim(lim)
    p1.set_xlabel('x')
    p1.set_ylabel('z')
    p1.set_title('pointmodel, (x,z), "front"')

    p2 = plt.subplot(2,3,2)
    p2.plot(y_new[idx_pos],z_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color='red')
    p2.plot(y_new[idx_neg],z_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    p2.plot(y_new[idx_nul],z_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')    
    p2.set_xlim(lim)
    p2.set_ylim(lim)
    p2.set_xlabel('y')
    p2.set_ylabel('z')
    p2.set_title('pointmodel, (y,z), "side"')

    p3 = plt.subplot(2,3,3)
    p3.plot(x_new[idx_pos],y_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color='red')
    p3.plot(x_new[idx_neg],y_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    p3.plot(x_new[idx_nul],y_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')    
    p3.set_xlim(lim)
    p3.set_ylim(lim)
    p3.set_xlabel('x')
    p3.set_ylabel('y')
    p3.set_title('pointmodel, (x,y), "bottom"')

def plot_results(r,pr,pr_poly,q,I,I_poly,S,qsim,Isim,sigma,polydispersity,eta):
    """
    plot results using matplotlib:
    - p(r) 
    - calculated formfactor, P(r) on log-log and lin-lin scale
    - simulated noisy data on log-log and lin-lin scale
    """

    ## plot p(r)
    p4 = plt.subplot(2,3,4)
    #p4.plot(r,hr,color='green')
    p4.plot(r,pr,color='red',label='p(r), monodisperse')
    p4.set_xlabel('r [Angstrom]')
    p4.set_ylabel('p(r)')
    p4.set_title('Pair distribution funciton, p(r)')
    if polydispersity > 0.0:
        p4.plot(r,pr_poly,linestyle='--',color='grey',label='p(r), polydisperse')
        p4.legend()

    ## plot scattering, log-log
    p5 = plt.subplot(2,3,5)
    p5.set_xscale('log')
    p5.set_yscale('log')
    p5.set_xlabel('q [1/Angstrom]')
    p5.set_ylabel('I(q)')
    p5.set_title('Scattering, log-log scale')
    if polydispersity > 0.0 and eta == 0.0:
        p5.plot(q,I,color='red',label='P(q), monodisperse')
        p5.plot(q,I_poly,linestyle='--',color='grey',label='P(q), polydisperse')
    elif polydispersity == 0.0 and eta > 0.0:
        p5.plot(q,I,linestyle='-',color='red',label='P(q)')
        p5.plot(q,S,linestyle='-',color='black',label='S(q)')
        p5.plot(q,I*S,linestyle='--',color='blue',label='P(q)*S(q)')
    elif polydispersity > 0.0 and eta > 0.0:
        p5.plot(q,I,linestyle='-',color='red',label='P(q) monodisperse')
        p5.plot(q,I_poly,linestyle='-',color='grey',label='P(q), polydisperse')
        p5.plot(q,S,linestyle='-',color='black',label='S(q)')
        p5.plot(q,I*S,linestyle='--',color='blue',label='P(q)*S(q), monodisperse')
        p5.plot(q,I_poly*S,linestyle='--',color='green',label='P(q)*S(q), polydisperse')
    else:
        p5.plot(q,I,color='red',label='P(q)')
    p5.errorbar(qsim,Isim,yerr=sigma,linestyle='none',marker='.',color='lightgrey',label='I(q), simulated',zorder=0)
    p5.legend()

    ## plot scattering, lin-log
    p6 = plt.subplot(2,3,6)
    p6.set_yscale('log')
    p6.set_xlabel('q')
    p6.set_ylabel('I(q)')
    p6.set_title('Scattering, lin-log scale')
    if polydispersity > 0.0 and eta == 0.0:
        p6.plot(q,I,color='red',label='P(q), monodisperse')
        p6.plot(q,I_poly,linestyle='--',color='grey',label='P(q), polydisperse')
    elif polydispersity == 0.0 and eta > 0.0:
        p6.plot(q,I,linestyle='-',color='red',label='P(q)')
        p6.plot(q,S,linestyle='-',color='black',label='S(q)')
        p6.plot(q,I*S,linestyle='--',color='blue',label='P(q)*S(q)')
    elif polydispersity > 0.0 and eta > 0.0:
        p6.plot(q,I,linestyle='-',color='red',label='P(q) monodisperse')
        p6.plot(q,I_poly,linestyle='-',color='grey',label='P(q), polydisperse')
        p6.plot(q,S,linestyle='-',color='black',label='S(q)')
        p6.plot(q,I*S,linestyle='--',color='blue',label='P(q)*S(q), monodisperse')
        p6.plot(q,I_poly*S,linestyle='--',color='green',label='P(q)*S(q), polydisperse')
    else:
        p6.plot(q,I,color='red',label='P(q)')
    p6.errorbar(qsim,Isim,yerr=sigma,linestyle='none',marker='.',color='lightgrey',label='I(q), simulated',zorder=0)
    p6.legend()

    ## figure settings
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

def generate_pdb(x_new,y_new,z_new,p_new):
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

    with open('model.pdb','w') as f:
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
