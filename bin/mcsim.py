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
    
    pd1 = float(json_variables['polydispersity'])
    eta1 = float(json_variables['eta']) # volume fraction
    R_HS1 = float(json_variables['r_hs']) # hard-sphere radius
    sr1 = float(json_variables['sigma_r']) # interface roughness
    
    pd2 = float(json_variables['polydispersity_2'])
    eta2 = float(json_variables['eta_2']) # volume fraction
    R_HS2 = float(json_variables['r_hs_2']) # hard-sphere radius
    sr2 = float(json_variables['sigma_r_2']) # interface roughness

    scale_Isim = float(json_variables['scale_Isim']) # scale simulated intensity of Model 2

    folder = json_variables['_base_directory'] # output folder dir

    ## read checkboxes and related input
    # the Json input for checkboxes only exists if boxes are checked
    # therefore, I use try-except to import
    try:
        dummy = json_variables['exclude_overlap']
        exclude_overlap = True
    except:
        exclude_overlap = False
    try:
        dummy = json_variables['exclude_overlap_2']
        exclude_overlap_2 = True
    except:
        exclude_overlap_2 = False
    try:
        dummy = json_variables['xscale_lin']
        xscale_log = False
    except:
        xscale_log = True

    ## setup  messaging in GUI
    message = genapp(json_variables)    
    output = {} # create an empty python dictionary

    ## generate q vector
    q = generate_q(qmin,qmax,Nq)
    
    ## check if Model 2 is included
    Number_of_objects = 5
    count_objects_Model2 = 0
    for i in range(Number_of_objects):
        model_name = 'model%d%s' % (i+1,'_2')
        model = json_variables[model_name]
        if model != 'none':
            count_objects_Model2 += 1

    if count_objects_Model2 >= 1:
        Models = ['','_2']
        pds    = [pd1,pd2]
        etas   = [eta1,eta2]
        R_HSs  = [R_HS1,R_HS2]
        srs    = [sr1,sr2]
    else:
        Models = ['']
        pds    = [pd1]
        etas   = [eta1]
        R_HSs  = [R_HS1]
        srs    = [sr1]

    for (Model,polydispersity,eta,sigma_r,R_HS) in zip(Models,pds,etas,srs,R_HSs):
       
        ## print model number to stdout
        if Model == '':
            model_number = 1
        elif Model == '_2':
            model_number = 2
        message.udpmessage({"_textarea":"\n#####################################################\n" })
        message.udpmessage({"_textarea":"##########   MODEL %s   ##############################\n" % model_number })
        message.udpmessage({"_textarea":"#####################################################\n" })
        
        ## read model parameters
        model,a,b,c,p,x,y,z = [],[],[],[],[],[],[],[]
        for i in range(Number_of_objects) :
            number = i+1
            model_name = 'model%d%s' % (number,Model)
            a_name = 'a%d%s' % (number,Model)
            b_name = 'b%d%s' % (number,Model)
            c_name = 'c%d%s' % (number,Model)
            p_name = 'p%d%s' % (number,Model)
            x_name = 'x%d%s' % (number,Model)
            y_name = 'y%d%s' % (number,Model)
            z_name = 'z%d%s' % (number,Model)
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
        #message.udpmessage({"_textarea":"\n# folder: %s \n" % folder })

        ## generate points
        N,rho,N_exclude,volume,x_new,y_new,z_new,p_new = gen_all_points(Number_of_objects,x,y,z,model,a,b,c,p,exclude_overlap)

        ## vizualization: generate pdb file with points
        generate_pdb(x_new,y_new,z_new,p_new,Model)

        ## end time for point generation
        time_points = time.time()-start_points

        ## output
        N_remain = []
        for i in range(Number_of_objects):
            if model[i] != 'none':
                srho = rho[i]*p[i]
                N_remain.append(N[i] - N_exclude[i])
                message.udpmessage({"_textarea":"    generating %d points for object %d: %s\n" % (N[i],i+1,model[i]) })
                message.udpmessage({"_textarea":"       point density      : %1.2e (points per volume)\n" % rho[i]})
                message.udpmessage({"_textarea":"       scattering density : %1.2e (density times scattering length)\n" % srho})
                if exclude_overlap:
                    message.udpmessage({"_textarea":"       excluded points    : %d (overlap region)\n" % N_exclude[i]})
                    message.udpmessage({"_textarea":"       remaining points   : %d (non-overlapping region)\n" % N_remain[i]})
            else:
                N_remain.append(0)
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
        
        ## timing
        time_dist = time.time() - start_dist
        message.udpmessage({"_textarea":"    time dist: %1.2f\n" % time_dist})
        
        ################### CALCULATE p(r) #####################################

        ## timing
        start_pr = time.time()
        message.udpmessage({"_textarea":"\n# Making p(r) (weighted histogram)...\n"})

        ## calculate p(r) 
        r,pr,Dmax,Rg = calc_pr(dist,Nbins,contrast,polydispersity,Model)
    
        ## send output to GUI
        message.udpmessage({"_textarea":"    Dmax              = %1.2f\n" % Dmax})
        message.udpmessage({"_textarea":"    Rg                = %1.2f\n" % Rg})
        
        ## timing
        time_pr = time.time() - start_pr
        message.udpmessage({"_textarea":"    time p(r): %1.2f sec\n" % time_pr})

        ################### CALCULATE I(q) using histogram  #####################################
        
        ## timing
        start_Iq = time.time()
        message.udpmessage({"_textarea":"\n# Calculating intensity, I(q)...\n"})

        ## calculate structure factor
        S = calc_S(q,R_HS,eta,Model)

        ## calculate intensity 
        I = calc_Iq(q,r,pr,S,sigma_r,Model)

        ## simulate data
        Isim,sigma = simulate_data(q,I,noise,Model)

        ## timing
        time_Iq = time.time() - start_Iq
        message.udpmessage({"_textarea":"    time I(q): %1.2f sec\n" % time_Iq})
        
        ################### OUTPUT to GUI #####################################

        output["pr%s" % Model] = "%s/pr%s.d" % (folder,Model)
        output["Iq%s" % Model] = "%s/Iq%s.d" % (folder,Model)
        output["Isim%s" % Model] = "%s/Isim%s.d" % (folder,Model)
        output["pdb%s" % Model] = "%s/model%s.pdb" % (folder,Model)
        output["Dmax%s" % Model] = "%1.2f" % Dmax
        output["Rg%s" % Model] = "%1.2f" % Rg

        ## save variables for combined plots
        if Model == '':
            r1,pr1,I1,Isim1,sigma1,S1 = r,pr,I,Isim,sigma,S
            x1,y1,z1,p1 = x_new,y_new,z_new,p_new
            if count_objects_Model2 >= 1:
                # delete unnecessary data (reduce memory usage)
                del x_new,y_new,z_new,p_new

    ################### GENERATING PLOTS  #####################################
    
    ## start time for output generation
    start_output = time.time()
    
    ## generate plots of p(r) and I(q) 
    message.udpmessage({"_textarea":"\n# Making plots of p(r) and I(q)...\n"})
    
    ## plot 2D projections
    if count_objects_Model2 >= 1:
        max_dimension = get_max_dimension(x1,y1,z1,x_new,y_new,z_new)
        for (x,y,z,p,Model) in zip([x1,x_new],[y1,y_new],[z1,z_new],[p1,p_new],Models):
            plot_2D(x,y,z,p,max_dimension,Model)
    else:
        plot_2D(x_new,y_new,z_new,p_new,0,Model)
    
    ## plot p(r) and I(q)
    if count_objects_Model2 >= 1:
        plot_results_combined(q,r1,pr1,I1,Isim1,sigma1,S1,r,pr,I,Isim,sigma,S,xscale_log,scale_Isim)
        output["fig"] = "%s/plot_combined.png" % folder
    else:
        plot_results(q,r1,pr1,I1,Isim1,sigma1,S1,xscale_log)
        output["fig"] = "%s/plot.png" % folder
    
    ## compress (zip) results for output
    for Model in Models:
        output["points%s" % Model] = "%s/points%s.png" % (folder,Model)
        os.system('zip results%s.zip pr%s.d Iq%s.d Sq%s.d Isim%s.d model%s.pdb points%s.png plot_combined.png' % (Model,Model,Model,Model,Model,Model,Model))
        output["zip%s" % Model] = "%s/results%s.zip" % (folder,Model)

    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";
    
    ## timing for output generation
    time_output = time.time()-start_output
    message.udpmessage({"_textarea":"    time plots: %1.2f sec\n" % time_output}) 
    
    ## total time
    time_total = time.time()-start_total
    message.udpmessage({"_textarea":"\n# Finished succesfully.\n    time total: %1.2f sec\n" % time_total})

    ## send output to GUI
    print( json.dumps(output) ) # convert dictionary to json and output

    
