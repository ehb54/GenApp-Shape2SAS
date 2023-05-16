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

    ################# HARD-CODED VARIABLES #######################################
    
    N_Models = 4
    Models_all = ['','_2','_3','_4']
    color_all = ['red','blue','orange','forestgreen']
    color2_all= ['firebrick','royalblue','darkorange','darkgreen']
    
    ################# GUI SETUP ##################################################
    
    ## read json variables
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    
    ## output folder
    folder = json_variables['_base_directory'] # output folder dir

    ## setup  messaging in GUI
    message = genapp(json_variables)
    output = {} # create an empty python dictionary

   ################### IMPORT INPUT FROM GUI #####################################

    ## global input (from GUI)
    qmin = float(json_variables['qmin'])
    qmax = float(json_variables['qmax'])
    Nq = int(json_variables['qpoints']) # number of points in (simulated) q
    exposure = float(json_variables['exposure'])
    Nbins = int(json_variables['prpoints']) # number of points in p(r)
    Npoints = int(json_variables['Npoints']) # max number of points per model

    ## generate q vector
    q = generate_q(qmin,qmax,Nq)

    ## plot options
    try:
        dummy = json_variables['xscale_lin']
        xscale_log = False
    except:
        xscale_log = True
    
    ## read Model-related input (from GUI)
    label           = []
    exclude_overlap = []
    subunit_type    = []
    N_subunits      = []
    a               = []
    b               = []
    b               = []
    p               = []
    x               = []
    y               = []
    z               = []
    pd              = []
    Stype           = []
    R_HS            = []
    frac_aggr       = []
    R_aggr          = []
    N_aggr          = []
    conc            = []
    sigma_r         = []
    include         = [0,0,0,0]

    r_list,pr_norm_list,I_list,Isim_list,sigma_list,S_eff_list,x_list,y_list,z_list,p_list,color_list,color2_list,Model_list,scale_list,label_list = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    for i in range(N_Models):
    #for i in range(4):
        n = i+1
        try:
        #if 1:
            dummy = json_variables['include_model_%d' % n]
            label = json_variables['label_model_%d' % n] # name of model
            try:
                dummy = json_variables['exclude_overlap_%d' % n]
                exclude_overlap = 1
            except:
                exclude_overlap = 0
            subunit_type = json_variables['model1_%d' % n]
            N_subunits = len(subunit_type)
            a_s = json_variables['a1_%d' % n]
            b_s = json_variables['b1_%d' % n]
            c_s = json_variables['c1_%d' % n]
            p_s = json_variables['p1_%d' % n]
            x_s = json_variables['x1_%d' % n]
            y_s = json_variables['y1_%d' % n]
            z_s = json_variables['z1_%d' % n]
            # loop over subunits to assign float values to subunits dimensions, contrast and com position
            a,b,c,p,x,y,z = [],[],[],[],[],[],[]
            for j in range(N_subunits):
                try:
                    a.append(float(a_s[j]))
                except:
                    a.append(50.0)
                try:
                    b.append(float(b_s[j]))
                    if subunit_type[j] in ['hollow_sphere','hollow_cube','cyl_ring','disc_ring']:
                        if b[j] == a[j]:
                            message.udpmessage({"_textarea":"!!WARNING!!\n! As b = a, you get a %s with an infinitely thin shell\n" % subunit_type[j]})
                            message.udpmessage({"_textarea":"! In principle, for shells,  the density is inf, but in the program, volume has been set equal to the area\n"})
                        if b[j] > a[j]:
                            message.udpmessage({"_textarea":"!!WARNING!!\n! b > a in %s not possible, setting b = a and a = b\n" % subunit_type[j] })
                except:
                    if subunit_type[j] in ['hollow_sphere','hollow_cube']:
                        b.append(0.5*a[j])
                    else:
                        b.append(a[j])
                try:
                    c.append(float(c_s[j]))
                except:
                    if subunit_type[j] in ['ellipsoid','cylinder','cuboid','cyl_ring']:
                        c.append(4*a[j])
                    elif subunit_type[j] in ['disc','disc_ring']:
                        c.append(0.5*a[j])
                    else:
                        c.append(0.0)
                try:
                    p.append(float(p_s[j]))
                except:
                    p.append(1.0)
                try:
                    x.append(float(x_s[j]))
                except:
                    x.append(0.0)
                try:
                    y.append(float(y_s[j]))
                except:
                    y.append(0.0)
                try:
                    z.append(float(z_s[j]))
                except:
                    z.append(0.0)

            pd = float(json_variables['polydispersity_%d' % n])
            Stype = json_variables['S_%d' % n]
            if Stype == 'HS':
                R_HS = float(json_variables['r_hs_%d' % n]) # hard-sphere radius
            if Stype == 'Aggr':
                fracs_aggr = float(json_variables['frac_%d' % n]) # fraction of particles in aggregated form
                R_aggr = float(json_variables['R_eff_%d' % n]) # effective radius per particle in aggregate
                N_aggr = float(json_variables['N_aggr_%d' % n]) # number of particles per aggregate
            conc = float(json_variables['conc_%d' % n]) # volume fraction (concentration)
            sigma_r = float(json_variables['sigma_r_%d' % n]) # interface roughness
            scale = float(json_variables['scale%d' % n]) # in the plot, scale simulated intensity of Model n
            include[i] = 1

            # print to stdout
            message.udpmessage({"_textarea":"\n#####################################################\n" })
            message.udpmessage({"_textarea":"##########   MODEL %d: %s   #####################\n" % (n,label) })
            message.udpmessage({"_textarea":"#####################################################\n" })

            ################### GENERATE POINTS IN USER-DEFINED SHAPES #####################################

            ## timing
            start_points = time.time()
            message.udpmessage({"_textarea":"\n# Generating points...\n" })

            ## generate points
            N,rho,N_exclude,volume_total,x_new,y_new,z_new,p_new = gen_all_points(N_subunits,Npoints,x,y,z,subunit_type,a,b,c,p,exclude_overlap)

            ## vizualization: generate pdb file with points
            Model = Models_all[i]
            generate_pdb(x_new,y_new,z_new,p_new,Model)

            ## output
            N_remain = []
            for j in range(N_subunits):
                srho = rho[j]*p[j]
                N_remain.append(N[j] - N_exclude[j])
                message.udpmessage({"_textarea":"    generating %d points for subunit %d: %s\n" % (N[j],j+1,subunit_type[j]) })
                message.udpmessage({"_textarea":"       point density      : %1.2e (points per volume)\n" % rho[j]})
                message.udpmessage({"_textarea":"       scattering density : %1.2e (density times scattering length)\n" % srho})
                if exclude_overlap:
                    message.udpmessage({"_textarea":"       excluded points    : %d (overlap region)\n" % N_exclude[j]})
                    message.udpmessage({"_textarea":"       remaining points   : %d (non-overlapping region)\n" % N_remain[j]})
            N_total = np.sum(N_remain)
            message.udpmessage({"_textarea":"    total number of points: %d\n" % N_total})
            message.udpmessage({"_textarea":"    total volume          : %1.1f A^3\n" % volume_total})

            ## end time for point generation
            time_points = time.time()-start_points
            message.udpmessage({"_textarea":"    time, points          : %1.2f sec\n" % time_points})

            ################### CALCULATE PAIR-WISE DISTANCES FOR ALL POINTS  #####################################

            ## timing
            start_dist = time.time()
            message.udpmessage({"_textarea":"\n# Calculating distances...\n"})

            ## calculate all pair-wise distances in particle (composed of all subunits) 
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
            r,pr,pr_norm,Dmax,Rg = calc_pr(dist,Nbins,contrast,pd,Model)
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
            message.udpmessage({"_textarea":"    volume fraction :  %1.2f\n" % conc})
            if sigma_r > 0:
                message.udpmessage({"_textarea":"    sigma_r :  %1.2f\n" % sigma_r})

            ## calculate forward scattering and form factor
            I0,Pq = calc_Pq(q,r,pr)

            I0 *= conc*volume_total*1E-4 # make I0 scale with volume fraction (concentration) and volume squared and scale so default values gives I(0) of approx unity
            message.udpmessage({"_textarea":"    I(0) :  %1.2e\n" % I0})

            ## calculate structure factor
            if Stype == 'HS':
                # hard sphere structure factor
                S = calc_S_HS(q,conc,R_HS)
                message.udpmessage({"_textarea":"    hard-sphere radius :  %1.2f\n" % R_HS})
            elif Stype == 'Aggr':
                # aggregate structure factor: fractal aggregate with dimensionality 2
                S = calc_S_aggr(q,Reff,Naggr)
                message.udpmessage({"_textarea":"    fraction of aggregates :  %1.2f\n" % frac})
                message.udpmessage({"_textarea":"    effective radius of aggregates :  %1.2f\n" % Reff})
                message.udpmessage({"_textarea":"    particles per aggregate :  %1.2f\n" % Naggr})
            else:
                S = np.ones(len(q))
            # decoupling approx
            S_eff = decoupling_approx(q,x_new,y_new,z_new,p_new,Pq,S)

            # fraction of aggregates
            if Stype == 'Aggr':
                S_eff = (1-frac) + frac*S_eff

            ## calculate normalised intensity = P(q)*S(q)
            I = calc_Iq(q,Pq,S_eff,sigma_r,Model)

            ## simulate data
            Isim,sigma = simulate_data(q,I,I0,exposure,Model)

            ## timing
            time_Iq = time.time() - start_Iq
            message.udpmessage({"_textarea":"    time I(q) :  %1.2f sec\n" % time_Iq})

            ################### OUTPUT to GUI #####################################
            
            output["pr%s" % Model] = "%s/pr%s.dat" % (folder,Model)
            output["Iq%s" % Model] = "%s/Iq%s.dat" % (folder,Model)
            output["Isim%s" % Model] = "%s/Isim%s.dat" % (folder,Model)
            output["pdb%s" % Model] = "%s/model%s.pdb" % (folder,Model)
            output["Dmax%s" % Model] = "%1.2f" % Dmax
            output["Rg%s" % Model] = "%1.2f" % Rg
            output["pdb_jmol%s" % Model] = "%s/model%s.pdb" % (folder,Model)

            ## save variables for combined plots
            r_list.append(r)
            pr_norm_list.append(pr_norm)
            I_list.append(I)
            Isim_list.append(Isim)
            sigma_list.append(sigma)
            S_eff_list.append(S_eff)
            x_list.append(x_new)
            y_list.append(y_new)
            z_list.append(z_new)
            p_list.append(p_new)
            color_list.append(color_all[i])
            color2_list.append(color2_all[i])
            Model_list.append(Models_all[i])
            scale_list.append(scale)
            label_list.append(label)
            del x_new,y_new,z_new,p_new

        except:
            pass

    ## check input
    if np.sum(include) == 0:
        message.udpmessage({"_textarea":"\n############################################################\n" })
        message.udpmessage({"_textarea":"##########   You need to include at least 1 model ##########\n" })
        message.udpmessage({"_textarea":"############################################################\n" })
        exit()

    """
    count_subunits = [0,0,0,0]
    #Models,pds,Stypes,etas,R_HSs,srs,fracs,Naggrs,Reffs,labels,colors,colors2,scales,concs,excludes = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    Models,pds,Stypes,R_HSs,srs,fracs,Naggrs,Reffs,labels,colors,colors2,scales,concs,excludes = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for j in range(4):
        if include[j]:
            model_name = 'model1_%d' % (j+1)
            model = json_variables[model_name]
            Number_of_subunits = len(model)
            #for i in range(Max_number_of_subunits):
            for i in range(Number_of_subunits):
                #model_name = 'model%d_%d' % (i+1,j+1)
                #model = json_variables[model_name]
                if model[i] != 'none':
                    count_subunits[j] += 1
            if count_subunits[j]:
                Models.append(Models_all[j])
                pds.append(pds_all[j])
                Stypes.append(Stypes_all[j])
                #etas.append(etas_all[j])
                R_HSs.append(R_HSs_all[j])
                srs.append(srs_all[j])
                fracs.append(fracs_all[j])
                Naggrs.append(Naggrs_all[j])
                Reffs.append(Reffs_all[j])
                labels.append(labels_all[j])
                colors.append(colors_all[j])
                colors2.append(colors2_all[j])
                scales.append(scales_all[j])
                concs.append(concs_all[j])
                excludes.append(exclude_overlaps_all[j])
        else:
            count_subunits[j] = 0

    r_list,pr_norm_list,I_list,Isim_list,sigma_list,S_eff_list,x_list,y_list,z_list,p_list = [],[],[],[],[],[],[],[],[],[]
    #for (Model,polydispersity,Stype,eta,R_HS,frac,Naggr,Reff,sigma_r,label,exclude_overlap,conc) in zip(Models,pds,Stypes,etas,R_HSs,fracs,Naggrs,Reffs,srs,labels,excludes,concs):
    for (Model,polydispersity,Stype,R_HS,frac,Naggr,Reff,sigma_r,label,exclude_overlap,conc) in zip(Models,pds,Stypes,R_HSs,fracs,Naggrs,Reffs,srs,labels,excludes,concs):
        

        ## print model to stdout
        if Model == '':
            model_number = 1
        elif Model == '_2':
            model_number = 2
        elif Model == '_3':
            model_number = 3
        elif Model == '_4':
            model_number = 4
        message.udpmessage({"_textarea":"\n#####################################################\n" })
        message.udpmessage({"_textarea":"##########   MODEL: %s   ########################\n" % label })
        message.udpmessage({"_textarea":"#####################################################\n" })
        
        ## read model parameters
        #model,a,b,c,p,x,y,z = [],[],[],[],[],[],[],[]
        model_name = 'model1_%d' % (model_number)
        a_name = 'a1_%d' % (model_number)
        b_name = 'b1_%d' % (model_number)
        c_name = 'c1_%d' % (model_number)
        p_name = 'p1_%d' % (model_number)
        x_name = 'x1_%d' % (model_number)
        y_name = 'y1_%d' % (model_number)
        z_name = 'z1_%d' % (model_number)
        model = json_variables[model_name]
        message.udpmessage({"_textarea":"%s" % model })
        a = json_variables[a_name]
        message.udpmessage({"_textarea":"%s" % a })
        b = json_variables[b_name]
        message.udpmessage({"_textarea":"%s" % b })
        c = json_variables[c_name]
        message.udpmessage({"_textarea":"%s" % c })
        p = float(json_variables[p_name])
        x = json_variables[x_name]
        y = json_variables[y_name]
        z = json_variables[z_name]

        for i in range(Max_number_of_subunits):
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
        message.udpmessage({"_textarea":"    volume fraction :  %1.2f\n" % conc})
        if sigma_r > 0:
            message.udpmessage({"_textarea":"    sigma_r :  %1.2f\n" % sigma_r})

        ## calculate forward scattering and form factor
        I0,Pq = calc_Pq(q,r,pr)
        
        I0 *= conc*volume_total*1E-4 # make I0 scale with volume fraction (concentration) and volume squared and scale so default values gives I(0) of approx unity
        message.udpmessage({"_textarea":"    I(0) :  %1.2e\n" % I0}) 

        ## calculate structure factor
        if Stype == 'HS':
            # hard sphere structure factor
            S = calc_S_HS(q,conc,R_HS)
            message.udpmessage({"_textarea":"    hard-sphere radius :  %1.2f\n" % R_HS})
        elif Stype == 'Aggr':
            # aggregate structure factor: fractal aggregate with dimensionality 2
            S = calc_S_aggr(q,Reff,Naggr)
            message.udpmessage({"_textarea":"    fraction of aggregates :  %1.2f\n" % frac})
            message.udpmessage({"_textarea":"    effective radius of aggregates :  %1.2f\n" % Reff})
            message.udpmessage({"_textarea":"    particles per aggregate :  %1.2f\n" % Naggr})
        else:
            S = np.ones(len(q))
        # decoupling approx
        S_eff = decoupling_approx(q,x_new,y_new,z_new,p_new,Pq,S)

        # fraction of aggregates
        if Stype == 'Aggr':
            S_eff = (1-frac) + frac*S_eff

        ## calculate normalised intensity = P(q)*S(q) 
        I = calc_Iq(q,Pq,S_eff,sigma_r,Model)
         
        ## simulate data
        #Isim,sigma = simulate_data(q,I,I0,noise,Model)
        Isim,sigma = simulate_data(q,I,I0,exposure,Model)

        ## timing
        time_Iq = time.time() - start_Iq
        message.udpmessage({"_textarea":"    time I(q) :  %1.2f sec\n" % time_Iq})
        
        ################### OUTPUT to GUI #####################################

        output["pr%s" % Model] = "%s/pr%s.dat" % (folder,Model)
        output["Iq%s" % Model] = "%s/Iq%s.dat" % (folder,Model)
        output["Isim%s" % Model] = "%s/Isim%s.dat" % (folder,Model)
        output["pdb%s" % Model] = "%s/model%s.pdb" % (folder,Model)
        output["Dmax%s" % Model] = "%1.2f" % Dmax
        output["Rg%s" % Model] = "%1.2f" % Rg
        output["pdb_jmol%s" % Model] = "%s/model%s.pdb" % (folder,Model)

        ## save variables for combined plots
        r_list.append(r)
        pr_norm_list.append(pr_norm)
        I_list.append(I)
        Isim_list.append(Isim)
        sigma_list.append(sigma)
        S_eff_list.append(S_eff)
        x_list.append(x_new)
        y_list.append(y_new)
        z_list.append(z_new)
        p_list.append(p_new)
        del x_new,y_new,z_new,p_new

    """
    ################### GENERATING PLOTS  #####################################

    message.udpmessage({"_textarea":"\n#####################################################\n" })
    message.udpmessage({"_textarea":"##########   PLOTS AND OUTPUT  ######################\n"})
    message.udpmessage({"_textarea":"#####################################################\n" })

    ## start time for output generation
    start_output = time.time()
    
    ## generate plots of p(r) and I(q) 
    message.udpmessage({"_textarea":"\n# Making plots of p(r) and I(q)...\n"})
    
    ## plot 2D projections
    plot_2D(x_list,y_list,z_list,p_list,color_list,Model_list)

    ## plot p(r) and I(q)
    plot_results_combined(q,r_list,pr_norm_list,I_list,Isim_list,sigma_list,S_eff_list,label_list,color_list,color2_list,scale_list,xscale_log)
    output["fig"] = "%s/plot.png" % folder
    
    ## compress (zip) results for output
    for Model in Model_list:
        output["points%s" % Model] = "%s/points%s.png" % (folder,Model)
        os.system('zip results%s.zip pr%s.dat Iq%s.dat Sq%s.dat Isim%s.dat model%s.pdb points%s.png plot.png' % (Model,Model,Model,Model,Model,Model,Model))
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

