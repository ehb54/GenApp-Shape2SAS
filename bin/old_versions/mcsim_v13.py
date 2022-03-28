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
from helpfunctions import * 
import time
from fast_histogram import histogram1d #histogram1d from fast_histogram is faster than np.histogram (https://pypi.org/project/fast-histogram/) 
import gc # garbage collector for freeing memory
from sys import getsizeof

if __name__=='__main__':
    
    ## time
    start_total = time.time()
    
    ## read global Json input (input from GUI)
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

    ## read checkboxes and related input
    # the Json input for checkboxes only exists if boxes are checked
    # therefore, I use try-except to import
    try:
        dummy = json_variables['exclude_overlap']
        exclude_overlap = True
    except:
        exclude_overlap = False

    ## setup  messaging in GUI
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
                    message.udpmessage({"_textarea":"! In principle, for shells,  the density is inf, but in the program, volume has been set equal to the area\n"})
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

    ## adjust number of points,so total is no more than Npoints_max (due to memory limitations)
    Npoints_max = 4000
    
    ## calculate volume and sum of volume (for calculating the number of points in each shape)
    volume = []
    sum_vol = 0
    for i in range(Number_of_models):
        v = get_volume(model[i],a[i],b[i],c[i])
        volume.append(v)
        sum_vol += v
   
    ## calculate effective radius (used in structure factor)
    count = 0
    r_eff_sum = 0
    for i in range(Number_of_models):
        r = (3*v/(4*np.pi))**(1./3.)
        if model[i] != 'none':
            count += 1
            r_eff_sum += r
    r_eff = r_eff_sum/count

    ## generate points
    start_points = time.time()
    message.udpmessage({"_textarea":"\n# Generating and plotting points\n" })
    Nsum = 0
    x_new,y_new,z_new,p_new = 0,0,0,0
    for i in range(Number_of_models):
        if model[i] != 'none': 
            Npoints = int(Npoints_max * volume[i]/sum_vol)
            #message.udpmessage({"_textarea":"i,x[i],y[i],z[i],model[i],a[i],b[i],c[i],p[i],Npoints = %d,%f,%f,%f,%s,%f,%f,%f,%f,%d\n" % (i,x[i],y[i],z[i],model[i],a[i],b[i],c[i],p[i],Npoints) }) # use this message for debugging
            
            ## generate points
            x_add,y_add,z_add,p_add,N,rho = genpoints(x[i],y[i],z[i],model[i],a[i],b[i],c[i],p[i],Npoints)
          
            ## exclude overlap region (optional)
            if exclude_overlap:
                N_x_total = 0
                for j in range(i):
                    x_add,y_add,z_add,p_add,N_x = check_overlap(x_add,y_add,z_add,p_add,x[j],y[j],z[j],model[j],a[j],b[j],c[j])
                    N_x_total += N_x
            
            ## append points to vector
            x_new,y_new,z_new,p_new = append_points(x_new,y_new,z_new,p_new,x_add,y_add,z_add,p_add)

            ## print densities
            srho = rho*p[i]
            message.udpmessage({"_textarea":"    generating %d points for model %d: %s\n" % (N,i+1,model[i]) })
            message.udpmessage({"_textarea":"       point density      : %1.2e (points per volume)\n" % rho}) 
            message.udpmessage({"_textarea":"       scattering density : %1.2e (density times scattering length)\n" % srho})
            if exclude_overlap:
                message.udpmessage({"_textarea":"       excluded points    : %d (overlap region)\n" % N_x_total})
            Nsum += N
    
    message.udpmessage({"_textarea":"    total number of points: %d\n" % Nsum})

    ## plot 2D projections
    plot_2D(x_new,y_new,z_new,p_new)

    ## generate pdb file with points
    generate_pdb(x_new,y_new,z_new,p_new)

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
    dist = dist.astype('float32')
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
    
    ## make h(r)
    r,hr,Dmax,Dmax_poly = generate_hr(dist,polydispersity,Nbins,contrast)
    """
    Dmax = np.amax(dist)
    Dmax_poly = Dmax*(1+3*polydispersity)
    hr,bin_edges = np.histogram(dist,bins=Nbins,weights=contrast,range=(0,Dmax_poly)) 
    dr = bin_edges[2]-bin_edges[1]
    r = bin_edges[0:-1]+dr/2
    del bin_edges,dr
    """

    # find indices of nonzero distances
    idx_nonzero = np.where(dist>0.0)
    
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

    ## calculate (hard-sphere) structure factor 
    S = calc_S(q,r_eff,eta)
    
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

    ## simulate data
    qsim,Isim,sigma = simulate_data(polydispersity,I_poly,S,I,noise,q)

    ## make p(r)
    
    ## remove non-zero elements (tr for truncate)
    dist_tr = dist[idx_nonzero]
    del dist
    contrast_tr = contrast[idx_nonzero]
    del contrast
    pr = histogram1d(dist_tr,bins=Nbins,weights=contrast_tr,range=(0,Dmax_poly))
    
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

    ## send output messages
    time_pr = time.time() - start_pr
    message.udpmessage({"_textarea":"    Dmax              = %1.2f\n" % Dmax})
    message.udpmessage({"_textarea":"    Rg                = %1.2f\n" % Rg})
    if polydispersity > 0.0:
        message.udpmessage({"_textarea":"    Dmax polydisperse = %1.2f\n" % Dmax_poly})
        message.udpmessage({"_textarea":"    Rg polydisperse   = %1.2f\n" % Rg_poly})
    message.udpmessage({"_textarea":"    time p(r)         : %1.2f sec\n" % time_pr})

    ## start time for output
    start_output = time.time()

    ## generate plots of p(r) and I(q) 
    message.udpmessage({"_textarea":"\n# Making plots of p(r) and I(q)...\n"})
    plot_results(r,pr,pr_poly,q,I,I_poly,S,qsim,Isim,sigma,polydispersity,eta)

    ## compress output files to zip file
    os.system('zip results.zip pr.d Iq.d Isim.d model.pdb plot.png')

    time_output = time.time()-start_output
    message.udpmessage({"_textarea":"    time output: %1.2f sec\n" % time_output}) 

    ## total time
    time_total = time.time()-start_total
    message.udpmessage({"_textarea":"\n# Finished succesfully.\n    time total: %1.2f sec\n" % time_total}) 

    ## generate output to GUI
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

    ## send output to GUI
    print( json.dumps(output) ) # convert dictionary to json and output
