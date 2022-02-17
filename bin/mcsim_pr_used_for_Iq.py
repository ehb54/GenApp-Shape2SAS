#!/usr/bin/python3

import json
import io
import sys
import os
import socket # for sending progress messages to textarea
from genapp3 import genapp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess
from helpfunctions import sinc,get_volume,genpoints,calc_dist,calc_Rg,calc_S
import time
from fast_histogram import histogram1d #histogram1d from fast_histogram is faster than np.histogram (https://pypi.org/project/fast-histogram/) 
import gc # garbage collector for freeing memory
from sys import getsizeof

if __name__=='__main__':
    
    ## time
    start_total = time.time()
    
    ## read global Json input
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    qmin = float(json_variables['qmin'])
    qmax = float(json_variables['qmax'])
    Nq = int(json_variables['qpoints']) # number of points in (simulated) q
    noise = float(json_variables['noise'])
    Nbins = int(json_variables['prpoints']) # number of points in p(r)
    polydispersity = float(json_variables['polydispersity'])
    eta = float(json_variables['eta']) # volume fraction
    sigma_r = float(json_variables['sigma_r']) # interface roughness
    folder = json_variables['_base_directory'] # output folder dir
    N_poly_integral = 9 # number of steps in polydispersity integral

    ## setup  messaging
    message = genapp(json_variables)
    
    ## read model parameters
    Number_of_models = 5
    model,a,b,c,p,x,y,z = [],[],[],[],[],[],[],[]
    for i in range(Number_of_models) :
        number = i+1
        model_name = 'model%d' % number
        a_name = 'a%d' % number
        b_name = 'b%d' % number
        c_name = 'c%d' % number
        p_name = 'p%d' % number
        x_name = 'x%d' % number
        y_name = 'y%d' % number
        z_name = 'z%d' % number
        model.append(json_variables[model_name])
        a.append(float(json_variables[a_name]))
        try:
            b.append(float(json_variables[b_name]))
            if model[i] in ['hollow_sphere','hollow_cube','cyl_ring','disc_ring']:
                if b[i] == a[i]:
                    message.udpmessage({"_textarea":"!!WARNING!!\n! As b = a, you get a %s with an infinitely thin shell\n" % model[i]})
                    message.udpmessage({"_textarea":"! In principle, for shells,  the density is inf, but in the program, volume has been set equal to the area"})
                if b[i] > a[i]:
                    message.udpmessage({"_textarea":"!!WARNING!!\n! b > a in %s not possible, setting b = a and a = b\n" % model[i] })
        except:
            if model[i] in ['hollow_sphere','hollow_cube']:
                b.append(0.5*a[i])
            else:
                b.append(a[i])
        try:
            c.append(float(json_variables[c_name]))
        except:
            if model[i] in ['ellipsoid','cylinder','cuboid','cyl_ring']:
                c.append(4*a[i])
            elif model[i] in ['disc','disc_ring']:
                c.append(0.5*a[i])
            else:
                c.append(0.0)
        try: 
            p.append(float(json_variables[p_name]))
        except:
            p.append(1.0)
        try:
            x.append(float(json_variables[x_name]))
        except:
            x.append(0.0)
        try:
            y.append(float(json_variables[y_name]))
        except:
            y.append(0.0)
        try:
            z.append(float(json_variables[z_name]))
        except:
            z.append(0.0)
#    message.udpmessage({"_textarea":"a = [%f,%f,%f,%f]\n" % (a[0],a[1],a[2],a[3]) })
#    message.udpmessage({"_textarea":"b = [%f,%f,%f,%f]\n" % (b[0],b[1],b[2],b[3]) })
#    message.udpmessage({"_textarea":"c = [%f,%f,%f,%f]\n" % (c[0],c[1],c[2],c[3]) })
#    message.udpmessage({"_textarea":"p = [%f,%f,%f,%f]\n" % (p[0],p[1],p[2],p[3]) })
#    message.udpmessage({"_textarea":"x = [%f,%f,%f,%f]\n" % (x[0],x[1],x[2],x[3]) })
#    message.udpmessage({"_textarea":"y = [%f,%f,%f,%f]\n" % (y[0],y[1],y[2],y[3]) })
#    message.udpmessage({"_textarea":"z = [%f,%f,%f,%f]\n" % (z[0],z[1],z[2],z[3]) })

    ## adjust number of points, total is no more than 3,500
    count = 0
    sum_vol = 0
    r_eff_sum = 0
    volume = []
    for i in range(Number_of_models):
        v = get_volume(model[i],a[i],b[i],c[i])
        r = (3*v/(4*np.pi))**(1./3.)
        #message.udpmessage({"_textarea":"\n\n    i: %d, v: %1.2f, r: %1.2f\n" % (i,v,r)})
        volume.append(v)
        sum_vol += v
        if model[i] != 'none': 
            count += 1
            r_eff_sum += r
    r_eff = r_eff_sum/count
    Npoints_max = 4000
    #message.udpmessage({"_textarea":"\n\n    count: %d\n" % (count)})
    
    ## generate points
    start_points = time.time()
    message.udpmessage({"_textarea":"\n# Generating and plotting points\n" })
    Nsum = 0
    x_new,y_new,z_new,p_new = 0,0,0,0
    for i in range(Number_of_models):
        if model[i] != 'none': 
            Npoints = int(Npoints_max * volume[i]/sum_vol)
            #message.udpmessage({"_textarea":"i,x[i],y[i],z[i],model[i],a[i],b[i],c[i],p[i],Npoints = %d,%f,%f,%f,%s,%f,%f,%f,%f,%d\n" % (i,x[i],y[i],z[i],model[i],a[i],b[i],c[i],p[i],Npoints) })
            x_new,y_new,z_new,p_new,N,rho = genpoints(x_new,y_new,z_new,p_new,x[i],y[i],z[i],model[i],a[i],b[i],c[i],p[i],Npoints)
            srho = rho*p[i]
            message.udpmessage({"_textarea":"    generating %d points for model %d: %s\n" % (N,i+1,model[i]) })
            message.udpmessage({"_textarea":"       point density      : %1.2e (points per volume)\n" % rho}) 
            message.udpmessage({"_textarea":"       scattering density : %1.2e (density times scattering length)\n" % srho})
            Nsum += N
    
    message.udpmessage({"_textarea":"    total number of points: %d\n" % Nsum})

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

    ## generate pdb file with points
    with open('model.pdb','w') as f:
        f.write('TITLE    POINT SCATTER MODEL\n')
        f.write('REMARK   GENERATED WITH McSIM\n')
        f.write('REMARK   EACH BEAD REPRESENTED BY DUMMY ATOM\n')
        f.write('REMARK   CARBON, C: POSITIVE EXCESS SCATTERING LENGTH\n')
        f.write('REMARK   CARBON, H: ZERO EXCESS SCATTERING LENGTH\n')
        f.write('REMARK   CARBON, O: NEGATIVE EXCESS SCATTERING LENGTH\n')
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

    ## end time for point generation
    time_points = time.time()-start_points
    message.udpmessage({"_textarea":"    time points: %1.2f\n" % time_points})
    
    ## calculate all distances
    start_dist = time.time()
    message.udpmessage({"_textarea":"\n# Calculating distances...\n"})
    
    square_sum = 0.0
    for arr in [x_new,y_new,z_new]:
        square_sum += calc_dist(arr)**2
    d = np.sqrt(square_sum)
    dist = d.reshape(-1)  # reshape is slightly faster than flatten() and ravel()
    #message.udpmessage({"_textarea":"    time dist: %s\n" %dist.dtype})
    #message.udpmessage({"_textarea":"    time dist: %d\n" %getsizeof(dist)})
    dist = dist.astype('float32')
    #message.udpmessage({"_textarea":"    time dist: %s\n" %dist.dtype})
    #message.udpmessage({"_textarea":"    time dist: %d\n" %getsizeof(dist)})
    del x_new,y_new,z_new,square_sum,d  # delete unnecessary data (reduce memory usage)

    ## calculate all contrasts
    dp = np.outer(p_new,p_new)
    contrast = dp.reshape(-1)
    contrast = contrast.astype('float32')
    del p_new,dp

    time_dist = time.time() - start_dist
    message.udpmessage({"_textarea":"    time dist: %1.2f\n" % time_dist})

    ## make histograms and calculate scattering
    start_pr = time.time()
    message.udpmessage({"_textarea":"\n# Making p(r) (weighted histogram)..."})
    
    ## generate q
    M = 10*Nq
    q = np.linspace(qmin,qmax,M)

    ## make h(r)
    idx_nonzero = np.where(dist>0.0)
    Dmax = np.amax(dist)
    Dmax_poly = Dmax*(1+3*polydispersity)
    hr,bin_edges = np.histogram(dist,bins=Nbins,weights=contrast,range=(0,Dmax_poly)) 
    dr = bin_edges[2]-bin_edges[1]
    r = bin_edges[0:-1]+dr/2
    del bin_edges,dr

    ## make p(r)
    
    ## remove non-zero elements (tr for truncate)
    dist_tr = dist[idx_nonzero]
    #message.udpmessage({"_textarea":"    type dist_tr: %s\n" %dist_tr.dtype})
    #message.udpmessage({"_textarea":"    size dist_tr: %d\n" %getsizeof(dist_tr)})
    del dist
    contrast_tr = contrast[idx_nonzero]
    del contrast
    pr = histogram1d(dist_tr,bins=Nbins,weights=contrast_tr,range=(0,Dmax_poly))

    # monodisperse intensity
    I = 0.0 
    for i in range(Nbins):
        qr = q*r[i]
        I += pr[i]*sinc(qr)
    I /= np.amax(I)
    del hr

    # polydisperse intensity
    if polydispersity > 0.0:
        I_poly = 0.0
        factor_range = 1 + np.linspace(-3,3,N_poly_integral)*polydispersity
        for factor_d in factor_range:
            dhr = histogram1d(dist*factor_d,bins=Nbins,weights=contrast,range=(0,Dmax_poly))
            res = (1.0-factor_d)/polydispersity
            w = np.exp(-res**2/2.0) # give weight according to normal distribution
            #vol = factor_d**3 # relative volume
            #hr_poly += dhr*(w*vol)**2
            dI = 0.0
            for i in range(Nbins):
                qr = q*r[i]
                dI += dhr[i]*sinc(qr)
            dI /= np.amax(dI)
            I_poly += w*dI
            message.udpmessage({"_textarea":"."})
        I_poly /= np.amax(I_poly)
        del dhr
    else:
        I_poly = I

    ## structure factor 
    if eta > 0.0:
        #message.udpmessage({"_textarea":"\n\n    R: %1.2f, eta: %1.2f\n" % (r_eff,eta)})
        S = calc_S(q,r_eff,eta)
    else:
        S = np.ones(len(q))
    
    ## interface roughness (Skar-Gislinge et al. DOI: 10.1039/c0cp01074j)
    if sigma_r > 0.0:
        roughness = np.exp(-(q*sigma_r)**2/2)
        I *= roughness
        I_poly *= roughness

    ## save all intensities to textfile
    with open('Iq.d','w') as f:
        f.write('# q I(q) I(q) polydisperse S(q)\n')
        for i in range(M):
            f.write('%f %f %f %f\n' % (q[i],I[i],I_poly[i],S[i]))

    ## simulate exp error
    #input, sedlak errors (https://doi.org/10.1107/S1600576717003077)
    k = 10000
    c = 0.85
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

    
    if polydispersity > 0.0:
        pr_poly = 0.0
        factor_range = 1 + np.linspace(-3,3,N_poly_integral)*polydispersity
        for factor_d in factor_range:
            dpr = histogram1d(dist_tr*factor_d,bins=Nbins,weights=contrast_tr,range=(0,Dmax_poly))
            res = (1.0-factor_d)/polydispersity
            w = np.exp(-res**2/2.0) # give weight according to normal distribution
            vol = factor_d**3 # give weight according to (relative) volume square
            pr_poly += dpr*w*vol
            message.udpmessage({"_textarea":"."})
    else:
        pr_poly = pr
    
    message.udpmessage({"_textarea":".\n"})
    del dist_tr,contrast_tr
   
    ## normalize so pr_max = 1 
    pr /= np.amax(pr) 
    pr_poly /= np.amax(pr_poly)


    ## save p(r) to textfile
    with open('pr.d','w') as f:
        #f.write('#  r p(r) p(r) polydisperse\n')
        f.write('#  r p(r) p_polydisperse(r)\n')
        for i in range(Nbins):
            f.write('%f %f %f\n' % (r[i],pr[i],pr_poly[i]))

    # calculate Rg
    Rg = calc_Rg(r,pr)
    Rg_poly = calc_Rg(r,pr_poly)
    #Rg_poly = calc_Rg(r,pr)

    time_pr = time.time() - start_pr
    message.udpmessage({"_textarea":"    Dmax              = %1.2f\n" % Dmax})
    message.udpmessage({"_textarea":"    Rg                = %1.2f\n" % Rg})
    if polydispersity > 0.0:
        message.udpmessage({"_textarea":"    Dmax polydisperse = %1.2f\n" % Dmax_poly})
        message.udpmessage({"_textarea":"    Rg polydisperse   = %1.2f\n" % Rg_poly})
    message.udpmessage({"_textarea":"    time p(r)         : %1.2f sec\n" % time_pr})

    ## generating plots, pdb and other outputs
    start_output = time.time()
    message.udpmessage({"_textarea":"\n# Making plots of p(r) and I(q)...\n"})

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

    ## compress output files to zip file
    os.system('zip results.zip pr.d Iq.d Isim.d model.pdb plot.png')

    time_output = time.time()-start_output
    message.udpmessage({"_textarea":"    time output: %1.2f sec\n" % time_output}) 

    ## time
    time_total = time.time()-start_total
    message.udpmessage({"_textarea":"\n# Finished succesfully.\n    time total: %1.2f sec\n" % time_total}) 

    ## generate output
    output = {} # create an empty python dictionary
    output["pr"] = "%s/pr.d" % folder
    output["Iq"] = "%s/Iq.d" % folder
    output["Isim"] = "%s/Isim.d" % folder
    output["pdb"] = "%s/model.pdb" % folder
    output["fig"] = "%s/plot.png" % folder
    output["zip"] = "%s/results.zip" % folder
    output["Dmax"] = "%1.2f" % Dmax
    output["Rg"] = "%1.2f" % Rg
    if polydispersity > 0.0:
        output["Dmax_poly"] = "%1.2f" % Dmax_poly 
        output["Rg_poly"] = "%1.2f" % Rg_poly
    else:
        output["Dmax_poly"] = "N/A"
        output["Rg_poly"] = "N/A"

    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";

    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output
