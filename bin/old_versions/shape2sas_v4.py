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
    Npoints = int(json_variables['Npoints']) # max number of points per model
    
    try:
        dummy = json_variables['include_model_1']
        pd1 = float(json_variables['polydispersity'])
        Stype1 = json_variables['S'] # type of structure factore 
        sr1 = float(json_variables['sigma_r']) # interface roughness
        label1 = json_variables['label_model']
        include1 = 1
    except:
        pd1 = 0.0
        Stype1 = 'None'
        sr1 = 0.0
        label1 = 'Model 1'
        include1 = 0
    try:
        dummy = json_variables['include_model_2']
        pd2 = float(json_variables['polydispersity_2'])
        Stype2 = json_variables['S_2'] # type of structure factore 
        sr2 = float(json_variables['sigma_r_2']) # interface roughness
        label2 = json_variables['label_model_2']
        include2 = 1 
    except:
        pd2 = 0.0
        Stype2 = 'None'
        sr2 = 0.0
        label2 = 'Model 2'
        include2 = 0

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
    if Stype1 == 'HS':
        eta1 = float(json_variables['eta']) # volume fraction
        R_HS1 = float(json_variables['r_hs']) # hard-sphere radius
    else:
        eta1,R_HS1 = 0,0
    if Stype1 == 'Aggr':
        frac1 = float(json_variables['frac']) # fraction of particles in aggregated form
        Reff1 = float(json_variables['R_eff']) # effective radius per particle in aggregate
        Naggr1 = float(json_variables['N_aggr']) # number of particles per aggregate
    else:
        frac1,Reff1,Naggr1 = 0,0,0
    if Stype2 == 'HS':
        eta2 = float(json_variables['eta_2']) # volume fraction
        R_HS2 = float(json_variables['r_hs_2']) # hard-sphere radius
    else:
        eta2,R_HS2 = 0,0
    if Stype2 == 'Aggr':
        frac2 = float(json_variables['frac_2']) # fraction of particles in aggregated form
        Reff2 = float(json_variables['R_eff_2']) # effective radius per particle in aggregate
        Naggr2 = float(json_variables['N_aggr_2']) # number of particles per aggregate
    else:
        frac2,Reff2,Naggr2 = 0,0,0

    ## setup  messaging in GUI
    message = genapp(json_variables)    
    output = {} # create an empty python dictionary

    ## generate q vector
    q = generate_q(qmin,qmax,Nq)
    
    Models_all = ['','_2']
    pds_all    = [pd1,pd2]
    Stypes_all = [Stype1,Stype2]
    etas_all   = [eta1,eta2]
    R_HSs_all  = [R_HS1,R_HS2]
    srs_all    = [sr1,sr2]
    fracs_all  = [frac1,frac2]
    Naggrs_all = [Naggr1,Naggr2]
    Reffs_all  = [Reff1,Reff2]
    labels_all = [label1,label2]
    includes   = [include1,include2]

    ## check if Model 2 is included
    Max_number_of_subunits = 5
    count_subunits = [0,0]
    Models,pds,Stypes,etas,R_HSs,srs,fracs,Naggrs,Reffs,labels = [],[],[],[],[],[],[],[],[],[]
    for j in range(len(includes)):
        if includes[j]:
            for i in range(Max_number_of_subunits):
                model_name = 'model%d_%d' % (i+1,j+1)
                model = json_variables[model_name]
                if model != 'none':
                    count_subunits[j] += 1
            if count_subunits[j]:
                Models.append(Models_all[j])
                pds.append(pds_all[j])
                Stypes.append(Stypes_all[j])
                etas.append(etas_all[j])
                R_HSs.append(R_HSs_all[j])
                srs.append(srs_all[j])
                fracs.append(fracs_all[j])
                Naggrs.append(Naggrs_all[j])
                Reffs.append(Reffs_all[j])
                labels.append(labels_all[j])
        else:
            count_subunits[j] = 0
    """
    if count_objects_Model[1] >= 1:
        Models = ['','_2']
        pds    = [pd1,pd2]
        Stypes = [Stype1,Stype2]
        etas   = [eta1,eta2]
        R_HSs  = [R_HS1,R_HS2]
        srs    = [sr1,sr2]
        fracs  = [frac1,frac2]
        Naggrs = [Naggr1,Naggr2]
        Reffs  = [Reff1,Reff2]
        labels = [label1,label2]
    else:
        Models = ['']
        pds    = [pd1]
        Stypes = [Stype1]
        etas   = [eta1]
        R_HSs  = [R_HS1]
        fracs  = [frac1]
        Naggrs = [Naggr1]
        Reffs  = [Reff1]
        srs    = [sr1]
        labels = [label1]
    """

    #for (Model,polydispersity,Stype,eta,R_HS,frac,Naggr,Reff,sigma_r) in zip(Models,pds,Stypes,etas,R_HSs,fracs,Naggrs,Reffs,srs):
    for (Model,polydispersity,Stype,eta,R_HS,frac,Naggr,Reff,sigma_r,label) in zip(Models,pds,Stypes,etas,R_HSs,fracs,Naggrs,Reffs,srs,labels):
       
        ## print model number to stdout
        if Model == '':
            model_number = 1
        elif Model == '_2':
            model_number = 2
        message.udpmessage({"_textarea":"\n#####################################################\n" })
        message.udpmessage({"_textarea":"##########   MODEL: %s   ##########################\n" % label })
        message.udpmessage({"_textarea":"#####################################################\n" })
        
        ## read model parameters
        model,a,b,c,p,x,y,z = [],[],[],[],[],[],[],[]
        for i in range(Max_number_of_subunits) :
            number = i+1
            model_name = 'model%d_%d' % (number,model_number)
            a_name = 'a%d_%d' % (number,model_number)
            b_name = 'b%d_%d' % (number,model_number)
            c_name = 'c%d_%d' % (number,model_number)
            p_name = 'p%d_%d' % (number,model_number)
            x_name = 'x%d_%d' % (number,model_number)
            y_name = 'y%d_%d' % (number,model_number)
            z_name = 'z%d_%d' % (number,model_number)
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
        message.udpmessage({"_textarea":"\n# Generating points\n" })

        ## generate points
        N,rho,N_exclude,volume_total,x_new,y_new,z_new,p_new = gen_all_points(Max_number_of_subunits,Npoints,x,y,z,model,a,b,c,p,exclude_overlap)

        ## vizualization: generate pdb file with points
        generate_pdb(x_new,y_new,z_new,p_new,Model)

        ## end time for point generation
        time_points = time.time()-start_points

        ## output
        N_remain = []
        for i in range(Max_number_of_subunits):
            if model[i] != 'none':
                srho = rho[i]*p[i]
                N_remain.append(N[i] - N_exclude[i])
                message.udpmessage({"_textarea":"    generating %d points for subunit %d: %s\n" % (N[i],i+1,model[i]) })
                message.udpmessage({"_textarea":"       point density      : %1.2e (points per volume)\n" % rho[i]})
                message.udpmessage({"_textarea":"       scattering density : %1.2e (density times scattering length)\n" % srho})
                if exclude_overlap:
                    message.udpmessage({"_textarea":"       excluded points    : %d (overlap region)\n" % N_exclude[i]})
                    message.udpmessage({"_textarea":"       remaining points   : %d (non-overlapping region)\n" % N_remain[i]})
            else:
                N_remain.append(0)
        N_total = np.sum(N_remain)
        message.udpmessage({"_textarea":"    total number of points: %d\n" % N_total})
        message.udpmessage({"_textarea":"    total volume          : %1.1f A^3\n" % volume_total})
        message.udpmessage({"_textarea":"    time, points          : %1.2f sec\n" % time_points})

        ################### CALCULATE PAIR-WISE DISTANCES FOR ALL POINTS  #####################################
    
        ## timing
        start_dist = time.time()
        message.udpmessage({"_textarea":"\n# Calculating distances...\n"})

        ## calculate all pair-wise distances in model (composed of all subunits) 
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
        r,pr,pr_norm,Dmax,Rg = calc_pr(dist,Nbins,contrast,polydispersity,Model)
        pr /= N_total**2 # make p(r) independent on number of points per model

        ## send output to GUI
        message.udpmessage({"_textarea":"    Dmax : %1.2f A\n" % Dmax})
        message.udpmessage({"_textarea":"    Rg   : %1.2f A\n" % Rg})
        
        ## timing
        time_pr = time.time() - start_pr
        message.udpmessage({"_textarea":"    time p(r): %1.2f sec\n" % time_pr})

        ################### CALCULATE I(q) using histogram  #####################################
        
        ## timing
        start_Iq = time.time()
        message.udpmessage({"_textarea":"\n# Calculating intensity, I(q)...\n"})
        if eta > 0:
            message.udpmessage({"_textarea":"    eta               = %1.2f\n" % eta})
            message.udpmessage({"_textarea":"    R_HS              = %1.2f\n" % R_HS})
        if frac > 0:
            message.udpmessage({"_textarea":"    frac              = %1.2f\n" % frac})
            message.udpmessage({"_textarea":"    Reff              = %1.2f\n" % Reff})
            message.udpmessage({"_textarea":"    Naggr             = %1.2f\n" % Naggr})
        if sigma_r > 0:
            message.udpmessage({"_textarea":"    sigma_r           = %1.2f\n" % sigma_r})

        ## calculate forward scattering and form factor
        I0,Pq = calc_Pq(q,r,pr)
        
        I0 *= (volume_total)**2*1E-12 # make I0 scale with volume squared and scale to realistic numbers
        message.udpmessage({"_textarea":"    I(0): %1.2e\n" % I0}) 

        ## calculate structure factor
        if Stype == 'HS':
            # hard sphere structure factor
            S = calc_S_HS(q,eta,R_HS)
        elif Stype == 'Aggr':
            # aggregate structure factor: fractal aggregate with dimensionality 2
            S = calc_S_aggr(q,Reff,Naggr)
        else:
            S = np.ones(len(q))
        # decoupling approx
        S_eff = decoupling_approx(q,x_new,y_new,z_new,p_new,Pq,S)
        #S_eff = S
        # fraction of aggregates
        if Stype == 'Aggr':
            S_eff = (1-frac) + frac*S_eff

        ## calculate normalised intensity = P(q)*S(q) 
        I = calc_Iq(q,Pq,S_eff,sigma_r,Model)
         
        ## simulate data
        Isim,sigma = simulate_data(q,I,I0,noise,Model)

        ## timing
        time_Iq = time.time() - start_Iq
        message.udpmessage({"_textarea":"    time I(q): %1.2f sec\n" % time_Iq})
        
        ################### OUTPUT to GUI #####################################

        output["pr%s" % Model] = "%s/pr%s.dat" % (folder,Model)
        output["Iq%s" % Model] = "%s/Iq%s.dat" % (folder,Model)
        output["Isim%s" % Model] = "%s/Isim%s.dat" % (folder,Model)
        output["pdb%s" % Model] = "%s/model%s.pdb" % (folder,Model)
        output["Dmax%s" % Model] = "%1.2f" % Dmax
        output["Rg%s" % Model] = "%1.2f" % Rg
        output["pdb_jmol%s" % Model] = "%s/model%s.pdb" % (folder,Model)
        #output["pdbshow"] = folder + "/" + "model.pdb"

        ## save variables for combined plots
        if Model == '':
            r1,pr_norm1,I1,Isim1,sigma1,S_eff1 = r,pr_norm,I,Isim,sigma,S_eff
            x1,y1,z1,p1 = x_new,y_new,z_new,p_new
            if count_subunits[1] >= 1:
                # delete unnecessary data (reduce memory usage)
                del x_new,y_new,z_new,p_new

    ################### GENERATING PLOTS  #####################################

    message.udpmessage({"_textarea":"\n#####################################################\n" })
    message.udpmessage({"_textarea":"##########   PLOTS AND OUTPUT  ######################\n"})
    message.udpmessage({"_textarea":"#####################################################\n" })

    ## start time for output generation
    start_output = time.time()
    
    ## generate plots of p(r) and I(q) 
    message.udpmessage({"_textarea":"\n# Making plots of p(r) and I(q)...\n"})
    
    ## plot 2D projections
    if count_subunits[1] >= 1:
        max_dimension = get_max_dimension(x1,y1,z1,x_new,y_new,z_new)
        for (x,y,z,p,Model) in zip([x1,x_new],[y1,y_new],[z1,z_new],[p1,p_new],Models):
            plot_2D(x,y,z,p,max_dimension,Model)
    else:
        plot_2D(x_new,y_new,z_new,p_new,0,Model)
    
    ## plot p(r) and I(q)
    if count_subunits[1] >= 1:
        plot_results_combined(q,r1,pr_norm1,I1,Isim1,sigma1,S_eff1,r,pr_norm,I,Isim,sigma,S_eff,xscale_log,scale_Isim,label1,label2)
        output["fig"] = "%s/plot_combined.png" % folder
    else:
        plot_results(q,r1,pr_norm1,I1,Isim1,sigma1,S_eff1,xscale_log)
        output["fig"] = "%s/plot.png" % folder
    
    ## compress (zip) results for output
    for Model in Models:
        output["points%s" % Model] = "%s/points%s.png" % (folder,Model)
        os.system('zip results%s.zip pr%s.dat Iq%s.dat Sq%s.dat Isim%s.dat model%s.pdb points%s.png plot_combined.png' % (Model,Model,Model,Model,Model,Model,Model))
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

