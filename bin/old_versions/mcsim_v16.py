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

   ################### IMPORT INPUT FROM GUI #####################################

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

    ################### GENERATE POINTS IN USER-DEFINED SHAPES #####################################

    ## timing
    start_points = time.time()
    message.udpmessage({"_textarea":"\n# Generating and plotting points\n" })

    ## generate points
    N,rho,N_exclude,volume,x_new,y_new,z_new,p_new = gen_all_points(Number_of_models,x,y,z,model,a,b,c,p,exclude_overlap)

    ## vizualization part 1: plot 2D projections
    plot_2D(x_new,y_new,z_new,p_new)

    ## vizualization part 2: generate pdb file with points
    generate_pdb(x_new,y_new,z_new,p_new)

    ## end time for point generation
    time_points = time.time()-start_points

    ## output
    N_remain = []
    for i in range(Number_of_models):
        if model[i] != 'none':
            srho = rho[i]*p[i]
            N_remain.append(N[i] - N_exclude[i])
            message.udpmessage({"_textarea":"    generating %d points for model %d: %s\n" % (N[i],i+1,model[i]) })
            message.udpmessage({"_textarea":"       point density      : %1.2e (points per volume)\n" % rho[i]})
            message.udpmessage({"_textarea":"       scattering density : %1.2e (density times scattering length)\n" % srho})
            if exclude_overlap:
                message.udpmessage({"_textarea":"       excluded points    : %d (overlap region)\n" % N_exclude[i]})
                message.udpmessage({"_textarea":"       remaining points   : %d (non-overlapping region)\n" % N_remain[i]})
    N_total = np.sum(N_remain)
    message.udpmessage({"_textarea":"    total number of points: %d\n" % np.sum(N_total)})
    message.udpmessage({"_textarea":"    time, points: %1.2f\n" % time_points})

    ################### CALCULATE PAIR-WISE DISTANCES FOR ALL POINTS  #####################################
    
    ## timing
    start_dist = time.time()
    message.udpmessage({"_textarea":"\n# Calculating distances...\n"})

    ## calculate all pair-wise distances in COMPOSED object
    dist = calc_all_dist(x_new,y_new,z_new)

    ## calculate all pair-wise contrasts
    contrast = calc_all_contrasts(p_new)

    ## delete unnecessary data (reduce memory usage)
    del x_new,y_new,z_new,p_new
    
    time_dist = time.time() - start_dist
    message.udpmessage({"_textarea":"    time dist: %1.2f\n" % time_dist})

    ################### CALCULATE I(q) using histogram  #####################################
    
    ## timing
    start_pr = time.time()
    message.udpmessage({"_textarea":"\n# Making p(r) (weighted histogram) and I(q) (intensity)...\n"})
    
    ## calculate intensity 
    Dmax,Dmax_poly,I_poly,S,I,q,r,Rg_no_contrast = calc_Iq(qmin,qmax,Nq,Nbins,dist,contrast,polydispersity,eta,sigma_r)

    ## simulate data
    qsim,Isim,sigma = simulate_data(polydispersity,I_poly,S,I,noise,q)

    ################### CALCULATE p(r) #####################################
    
    # calculate p(r) 
    pr,pr_poly = calc_pr(dist,Nbins,contrast,Dmax_poly,polydispersity,r)
    
    # calculate Rg
    Rg = calc_Rg(r,pr)
    Rg_poly = calc_Rg(r,pr_poly)

    ## output
    time_pr = time.time() - start_pr
    message.udpmessage({"_textarea":"    Dmax              = %1.2f\n" % Dmax})
    message.udpmessage({"_textarea":"    Rg                = %1.2f\n" % Rg})
    if polydispersity > 0.0:
        message.udpmessage({"_textarea":"    Dmax polydisperse = %1.2f\n" % Dmax_poly})
        message.udpmessage({"_textarea":"    Rg polydisperse   = %1.2f\n" % Rg_poly})
    message.udpmessage({"_textarea":"    time p(r)         : %1.2f sec\n" % time_pr})

    ################### GENERATE OUTPUT TO GUI  #####################################
    
    ## start time for output generation
    start_output = time.time()

    ## generate plots of p(r) and I(q) 
    message.udpmessage({"_textarea":"\n# Making plots of p(r) and I(q)...\n"})
    plot_results(r,pr,pr_poly,q,I,I_poly,S,qsim,Isim,sigma,polydispersity,eta)

    ## compress output files to zip file
    os.system('zip results.zip pr.d Iq.d Isim.d model.pdb plot.png')

    ## structure output to GUI
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

    ## timing for output generation
    time_output = time.time()-start_output
    message.udpmessage({"_textarea":"    time output: %1.2f sec\n" % time_output}) 
    
    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";
    
    ## total time
    time_total = time.time()-start_total
    message.udpmessage({"_textarea":"\n# Finished succesfully.\n    time total: %1.2f sec\n" % time_total})

    ## send output to GUI
    print( json.dumps(output) ) # convert dictionary to json and output

    
