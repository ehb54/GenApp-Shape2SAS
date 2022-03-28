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

def gen_all_points(Number_of_models,x_com,y_com,z_com,model,a,b,c,p,exclude_overlap):
    """
    generate points from a collection of shapes
    calling gen_points() for each shape

    input:
    x_com,y_com,z_com : center of mass coordinates
    a,b,c  : model params (see GUI)
    p      : contrast for each model
    exclude_overlap : if True, points is removed from overlapping regions (True/False)

    output:
    N         : number of points in each model (before optional exclusion, see N_exclude)
    rho       : density of points for each model
    N_exclude : number of points excluded from each model (due to overlapping regions)
    x_new,y_new,z_new : coordinates of generated points
    p_new     : contrasts for each point 
    volume    : volume of each model 
    """

    ## Total number of points should not exceed N_points_max (due to memory limitations). This number is system-dependent
    N_points_max = 4000        
    
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
    dist = abs(m-n) 
    
    return dist

def calc_all_dist(x_new,y_new,z_new):
    """ 
    calculate all pairwise distances
    """
    square_sum = 0.0
    for arr in [x_new,y_new,z_new]:
        square_sum += calc_dist(arr)**2
    d = np.sqrt(square_sum)
    dist = d.reshape(-1)  # reshape is slightly faster than flatten() and ravel()
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
    q       : momentum transfer
    R       : estimation of the hard-sphere radius
    eta     : volume fraction
    """

    if eta > 0.0:
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

def calc_Iq(qmin,qmax,Nq,Nbins,dist,contrast,polydispersity,eta,sigma_r):
    
    """
    calculates intensity using histogram, h(r) - like p(r) but with self-terms
    """

    ## make h(r): histogram of all distances weighted with contrasts
    r,hr,Dmax,Dmax_poly = generate_hr(dist,polydispersity,Nbins,contrast)
    Rg = calc_Rg(r,hr) 
    
    ## generate q
    M = 10*Nq
    q = np.linspace(qmin,qmax,M)

    # monodisperse intensity
    I = 0.0 
    for i in range(Nbins):
        qr = q*r[i]
        I += hr[i]*sinc(qr)
    I /= np.amax(I)
    del hr

    # polydisperse intensity
    N_poly_integral = 9 # number of steps in polydispersity integral
    if polydispersity > 0.0:
        I_poly = 0.0
        factor_range = 1 + np.linspace(-3,3,N_poly_integral)*polydispersity
        for factor_d in factor_range:
            # histogram1d is faster than np.histogram (used in generate_hr()), but does not provide r. r is not needed here.
            dhr = histogram1d(dist*factor_d,bins=Nbins,weights=contrast,range=(0,Dmax_poly))
            res = (1.0-factor_d)/polydispersity
            w = np.exp(-res**2/2.0) # weight: normal distribution
            vol = factor_d**3 # weight: relative volume
            dI = 0.0
            for i in range(Nbins):
                qr = q*r[i]
                dI += dhr[i]*sinc(qr)
            dI /= np.amax(dI)
            I_poly += dI*w*vol**2
        I_poly /= np.amax(I_poly)
        del dhr
    else:
        I_poly = I

    ## estimate hard-sphere radius by non-contrast weighted Rg (assume spherical shape)
    hist_no_contrast = histogram1d(dist,bins=Nbins,range=(0,Dmax_poly))
    Rg_no_contrast = calc_Rg(r,hist_no_contrast)
    R_HS = np.sqrt(5./3.*Rg_no_contrast)

    ## calculate (hard-sphere) structure factor 
    S = calc_S(q,R_HS,eta)
    
    ## interface roughness (Skar-Gislinge et al. DOI: 10.1039/c0cp01074j)
    if sigma_r > 0.0:
        roughness = np.exp(-(q*sigma_r)**2/2)
        I *= roughness
        I_poly *= roughness

    ## save all intensities to textfile
    with open('Iq.d','w') as f:
        f.write('# %-17s %-17s %-17s %-17s\n' % ('q','I(q)','I(q) polydisperse','S(q)'))
        for i in range(M):
            f.write('  %-17.5e %-17.5e %-17.5e %-17.5e\n' % (q[i],I[i],I_poly[i],S[i]))
    
    return Dmax,Dmax_poly,I_poly,S,I,q,r,Rg_no_contrast

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
        f.write('# Simulated data\n')
        f.write('# sigma generated using Sedlak et al, k=10000, c=0.85, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)\n')
        f.write('# %-12s %-12s %-12s\n' % ('q','I','sigma'))
        for i in range(len(Isim)):
            f.write('  %-12.5e %-12.5e %-12.5e\n' % (qsim[i],Isim[i],sigma[i]))
    
    return qsim,Isim,sigma

def calc_pr(dist,Nbins,contrast,Dmax_poly,polydispersity,r):
    """
    calculate p(r)
    p(r) is the contrast-weighted histogram of distances, without the self-terms (dist = 0)
    due to lack of self-terms it is not used to calc scattering (fast Debye by histogram), but used for structural interpretation
    
    input: 
    dist      : all pairwise distances
    contrast  : all pair-wise contrast products
    Dmax_poly : Dmax for polydisperse ensemble
    polydispersity: boolian, True or False
    r         : pair distances of bins

    output:
    pr        : pair distance distribution function (PDDF) for monodisperse shape
    pr_poly   : PDDF for polydisperse ensemble
    """
    ## remove non-zero elements (tr for truncate)
    idx_nonzero = np.where(dist>0.0)
    dist_tr = dist[idx_nonzero]
    del dist # less memory consumption 

    contrast_tr = contrast[idx_nonzero]
    del contrast # less memory consumption 
    
    # calculate monodisperse p(r)
    pr = histogram1d(dist_tr,bins=Nbins,weights=contrast_tr,range=(0,Dmax_poly))
    
    ## calculate polydisperse p(r)
    N_poly_integral = 9
    if polydispersity > 0.0:
        pr_poly = 0.0
        factor_range = 1 + np.linspace(-3,3,N_poly_integral)*polydispersity
        for factor_d in factor_range:
            dpr = histogram1d(dist_tr*factor_d,bins=Nbins,weights=contrast_tr,range=(0,Dmax_poly))
            res = (1.0-factor_d)/polydispersity
            w = np.exp(-res**2/2.0) # weight: normal distribution
            vol = factor_d**3 # weight: relative volume, because larger particles scatter more
            pr_poly += dpr*w*vol**2
    else:
        pr_poly = pr
    
    ## normalize so pr_max = 1 
    pr /= np.amax(pr) 
    pr_poly /= np.amax(pr_poly)
    
    ## save p(r) to textfile
    with open('pr.d','w') as f:
        f.write('# %-17s %-17s %-17s\n' % ('r','p(r)','p_polydisperse(r)'))
        for i in range(Nbins):
            f.write('  %-17.5e %-17.5e %-17.5e\n' % (r[i],pr[i],pr_poly[i]))
    
    return pr,pr_poly

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

    ## find indices of positive, zero and negatative contrast
    idx_neg = np.where(p_new<0.0)
    idx_pos = np.where(p_new>0.0)
    idx_nul = np.where(p_new==0.0)
    
    ## figure settings
    plt.figure(figsize=(15,10))
    markersize = 3

    ## plot, perspective 1
    p1 = plt.subplot(2,3,1)
    p1.plot(x_new[idx_pos],z_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color='red')
    p1.plot(x_new[idx_neg],z_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    p1.plot(x_new[idx_nul],z_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')
    p1.set_xlim(lim)
    p1.set_ylim(lim)
    p1.set_xlabel('x')
    p1.set_ylabel('z')
    p1.set_title('pointmodel, (x,z), "front"')

    ## plot, perspective 2
    p2 = plt.subplot(2,3,2)
    p2.plot(y_new[idx_pos],z_new[idx_pos],linestyle='none',marker='.',markersize=markersize,color='red')
    p2.plot(y_new[idx_neg],z_new[idx_neg],linestyle='none',marker='.',markersize=markersize,color='green')
    p2.plot(y_new[idx_nul],z_new[idx_nul],linestyle='none',marker='.',markersize=markersize,color='grey')    
    p2.set_xlim(lim)
    p2.set_ylim(lim)
    p2.set_xlabel('y')
    p2.set_ylabel('z')
    p2.set_title('pointmodel, (y,z), "side"')

    ## plot, perspective 3
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
