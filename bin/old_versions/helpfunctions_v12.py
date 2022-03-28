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

def generate_histogram(dist,contrast,Dmax,Nbins):
    """
    make histogram of point pairs, h(r), binned after pair-distances, r
    used for calculating scattering (fast Debye)
    dist     : all pairwise distances
    Nbins    : number of bins in h(r)
    contrast : contrast of points
    """
    histo,bin_edges = np.histogram(dist,bins=Nbins,weights=contrast,range=(0,Dmax)) 
    dr = bin_edges[2]-bin_edges[1]
    r = bin_edges[0:-1]+dr/2
    
    return r,histo

def calc_Dmax(dist,polydispersity):
    Dmax = np.amax(dist)
    Dmax *= (1+3*polydispersity)

    return Dmax

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

    
#def calc_Iq(qmin,qmax,Nq,Nbins,dist,contrast,polydispersity,eta,sigma_r,Model):
#def calc_Iq(r,pr,qmin,qmax,Nq,Nbins,dist,contrast,polydispersity,eta,sigma_r,Model):
def calc_Iq(r,pr,qmin,qmax,Nq,dist,Dmax,eta,sigma_r,Model):
    """
    calculates intensity using histogram
    """
    ## make h(r): histogram of all distances weighted with contrasts
    #Dmax = calc_Dmax(dist,polydispersity)
    #r,hr,Dmax = generate_histogram(dist,contrast,Dmax,Nbins)
    
    ## generate q
    M = 100*Nq
    q = np.linspace(qmin,qmax,M)

    I = 0.0
    """
    if polydispersity > 0.0:
        N_poly_integral = 9 # number of steps in polydispersity integral
        factor_range = 1 + np.linspace(-3,3,N_poly_integral)*polydispersity
        for factor_d in factor_range:
            # histogram1d is faster than np.histogram (used in generate_histogram()), but does not provide r. r is not needed here.
            dpr = histogram1d(dist*factor_d,bins=Nbins,weights=contrast,range=(0,Dmax))
            res = (1.0-factor_d)/polydispersity
            w = np.exp(-res**2/2.0) # weight: normal distribution
            vol = factor_d**3 # weight: relative volume
            dI = 0.0
            for i in range(Nbins):
                qr = q*r[i]
                dI += dpr[i]*sinc(qr)
            dI /= np.amax(dI)
            I += dI*w*vol**2
        del dpr
    else:
        for i in range(Nbins):
            qr = q*r[i]
            I += pr[i]*sinc(qr)
    """

    for (r_i,pr_i) in zip(r,pr):
        qr = q*r_i
        I += pr_i*sinc(qr)
    
    ## normalization
    I /= np.amax(I)

    ## estimate hard-sphere radius by non-contrast weighted Rg (assume spherical shape)
    hist_no_contrast = histogram1d(dist,bins=len(r),range=(0,Dmax))
    Rg_no_contrast = calc_Rg(r,hist_no_contrast)
    R_HS = np.sqrt(5./3.*Rg_no_contrast)

    ## calculate (hard-sphere) structure factor 
    S = calc_S(q,R_HS,eta)
    I *= S
    
    ## interface roughness (Skar-Gislinge et al. DOI: 10.1039/c0cp01074j)
    if sigma_r > 0.0:
        roughness = np.exp(-(q*sigma_r)**2/2)
        I *= roughness

    ## save intensity to file
    with open('Iq%s.d' % Model,'w') as f:
        f.write('# %-17s %-17s\n' % ('q','I(q)'))
        for i in range(M):
            f.write('  %-17.5e %-17.5e\n' % (q[i],I[i]))
    
    ## save structure factor to file
    with open('Sq%s.d' % Model,'w') as f:
        f.write('# Structure factor, S(q), used in: I(q) = P(q)*S(q)\n')
        f.write('# Default: S(q) = 1.0)\n')
        f.write('# %-17s %-17s\n' % ('q','S(q)'))
        for i in range(M):
            f.write('  %-17.5e%-17.5e\n' % (q[i],S[i]))    

    return q,I

def simulate_data(q,I,noise,Model):

    ## simulate exp error
    #input, sedlak errors (https://doi.org/10.1107/S1600576717003077)
    k = 100000
    c = 0.55
    mu = I
    sigma = noise*np.sqrt((mu+c)/(k*q))

    ##pseudo-rebin
    Nrebin = 100 # keep every Nth point
    mu = mu[::Nrebin]
    qsim = q[::Nrebin]
    sigma = sigma[::Nrebin]/np.sqrt(Nrebin)

    ## simulate data using errors
    Isim = np.random.normal(mu,sigma)

    ## save to file
    with open('Isim%s.d' % Model,'w') as f:
        f.write('# Simulated data\n')
        f.write('# sigma generated using Sedlak et al, k=100000, c=0.55, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)\n')
        f.write('# %-12s %-12s %-12s\n' % ('q','I','sigma'))
        for i in range(len(Isim)):
            f.write('  %-12.5e %-12.5e %-12.5e\n' % (qsim[i],Isim[i],sigma[i]))
    
    return qsim,Isim,sigma

def calc_hr(dist,Nbins,contrast,polydispersity,Model):
    """
    calculate h(r)
    h(r) is the contrast-weighted histogram of distances, including self-terms (dist = 0)
    
    input: 
    dist      : all pairwise distances
    contrast  : all pair-wise contrast products
    polydispersity: boolian, True or False

    output:
    hr        : pair distance distribution function 
    """
    
    ## calculate p(r)
    Dmax = calc_Dmax(dist,polydispersity)
    r,hr = generate_histogram(dist,contrast,Dmax,Nbins)

    ## calculate p(r)
    if polydispersity > 0.0:
        N_poly_integral = 9
        hr = 0.0
        factor_range = 1 + np.linspace(-3,3,N_poly_integral)*polydispersity
        for factor_d in factor_range:
            dhr = histogram1d(dist*factor_d,bins=Nbins,weights=contrast,range=(0,Dmax*1.5))
            res = (1.0-factor_d)/polydispersity
            w = np.exp(-res**2/2.0) # weight: normal distribution
            vol = factor_d**3 # weight: relative volume, because larger particles scatter more
            hr += dhr*w*vol**2
    else:
        hr = histogram1d(dist,bins=Nbins,weights=contrast,range=(0,Dmax*1.5))
    
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

    ## remove non-zero elements (tr for truncate)
    idx_nonzero = np.where(dist>0.0)
    dist_tr = dist[idx_nonzero]
    del dist # less memory consumption
    contrast_tr = contrast[idx_nonzero]
    del contrast # less memory consumption

    ## calculate pr
    r,pr,Dmax,Rg = calc_hr(dist_tr,Nbins,contrast_tr,polydispersity,Model)

    ## save p(r) to textfile
    with open('pr%s.d' % Model,'w') as f:
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
    positive contrast in red/blue
    zero contrast in grey
    negative contrast in green

    (x_new,y_new,z_new) : coordinates of simulated points
    p_new               : excess scattering length density (contrast) of simulated points
    max_dimension       : max dimension of previous model (for plot limits)
    """
    
    ## find max dimensions of model
    max_x = np.amax(abs(x_new))
    max_y = np.amax(abs(y_new))
    max_z = np.amax(abs(z_new))
    max_l = np.amax([max_x,max_y,max_z,max_dimension])
    lim = [-max_l,max_l]

    ## find indices of positive, zero and negatative contrast
    idx_neg = np.where(p_new<0.0)
    idx_pos = np.where(p_new>0.0)
    idx_nul = np.where(p_new==0.0)
    
    ## figure settings
    markersize = 3
    if Model == '':
        color = 'red'
    elif Model == '_2':
        color = 'blue'

    f,ax = plt.subplots(1,3,figsize=(15,5))
    
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


def plot_results(r,pr,q,I,qsim,Isim,sigma):
    """
    plot results using matplotlib:
    - p(r) 
    - calculated formfactor, P(r) on log-log and lin-lin scale
    - simulated noisy data on log-log and lin-lin scale
    """
   
    ## plot settings
    fig,ax = plt.subplots(1,3,figsize=(15,5))
    color = 'red'

    ## plot p(r)
    ax[0].plot(r,pr,color=color,label='p(r), monodisperse')
    ax[0].set_xlabel('r [Angstrom]')
    ax[0].set_ylabel('p(r)')
    ax[0].set_title('Pair distribution funciton, p(r)')

    ## plot scattering, log-log
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('q [1/Angstrom]')
    ax[1].set_ylabel('I(q)')
    ax[1].set_title('Scattering, log-log scale')
    ax[1].plot(q,I,color=color,label='P(q)')
    ax[1].errorbar(qsim,Isim,yerr=sigma,linestyle='none',marker='.',color='lightgrey',label='I(q), simulated',zorder=0)
    ax[1].legend()

    ## plot scattering, lin-log
    ax[2].set_yscale('log')
    ax[2].set_xlabel('q [1/Angstrom]')
    ax[2].set_ylabel('I(q)')
    ax[2].set_title('Scattering, lin-log scale')
    ax[2].plot(q,I,color=color,label='P(q)')
    ax[2].errorbar(qsim,Isim,yerr=sigma,linestyle='none',marker='.',color='lightgrey',label='I(q), simulated',zorder=0)
    ax[2].legend()

    ## figure settings
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

def plot_results_combined(r1,pr1,q1,I1,qsim1,Isim1,sigma1,r2,pr2,q2,I2,qsim2,Isim2,sigma2):
    """
    plot results (combined = Model 1 and Model 2), using matplotlib:
    - p(r) 
    - calculated formfactor, P(r) on log-log and lin-lin scale
    - simulated noisy data on log-log and lin-lin scale
    """

    fig,ax = plt.subplots(1,3,figsize=(15,5))

    for (r,pr,q,I,qsim,Isim,sigma,model,col,col_sim,line,scale,zo) in zip ([r1,r2],[pr1,pr2],[q1,q2],[I1,I2],[qsim1,qsim2],[Isim1,Isim2],[sigma1,sigma2],[1,2],['red','blue'],['pink','skyblue'],['-','--'],[1,100],[2,3]):
        ax[0].plot(r,pr,linestyle=line,color=col,zorder=zo,label='p(r), Model %d' % model)
        ax[1].plot(q,I*scale,linestyle=line,color=col,zorder=zo,label='P(q), Model %d' % model)
        ax[2].plot(q,I*scale,linestyle=line,color=col,label='P(q), Model %d, scaled' % model)
        ax[1].errorbar(qsim,Isim*scale,yerr=sigma*scale,linestyle='none',marker='.',color=col_sim,label='I(q), simulated',zorder=zo-2)
        ax[2].errorbar(qsim,Isim*scale,yerr=sigma*scale,linestyle='none',marker='.',color=col_sim,label='I(q), simulated',zorder=zo-2)

    ## plot p(r)
    ax[0].set_xlabel('r [Angstrom]')
    ax[0].set_ylabel('p(r)')
    ax[0].set_title('Pair distribution funciton, p(r)')

    ## plot scattering, log-log
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('q [1/Angstrom]')
    ax[1].set_ylabel('I(q)')
    ax[1].set_title('Scattering, log-log scale')
    ax[1].legend()

    ## plot scattering, lin-log
    ax[2].set_yscale('log')
    ax[2].set_xlabel('q [1/Angstrom]')
    ax[2].set_ylabel('I(q)')
    ax[2].set_title('Scattering, lin-log scale')
    ax[2].legend()

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
