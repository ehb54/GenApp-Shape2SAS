#!/usr/bin/python3

import time
import argparse
from sys import argv
import numpy as np
import shutil
from shape2sas_helpfunctions import *

# current version
version = 2.4

if __name__ == "__main__":
       
    ## remove existing log file
    open('shape2sas.log','w').close()

    ### welcome message
    printt('#######################################################################################')
    printt('RUNNING shape2sas.py version %s \n - for instructions type: python shape2sas.py -h' % version)
    command = "python shape2sas.py"
    for aa in argv[1:]:
        if ' ' in aa:
            command += " \"%s\"" % aa
        else:
            command += " %s" % aa
    printt('command used: %s' % command)
    printt('#######################################################################################')

    # timing
    start_total = time.time()

    ### input values 
    parser = argparse.ArgumentParser(description='Shape2SaS - calculates small-angle scattering from a given shape defined by the user.')
      
    # Mandatory inputs
    parser.add_argument('-s', '--subunit', type=separate_string, nargs='+', action='extend',
                       help='Type of subunits for each model.')
    parser.add_argument('-d', '--dimension', type=float_list, nargs='+', action='append',
                        help='dimensions of subunits for each model.')
    
    # Optional model-dependent inputs:
    parser.add_argument('-m', '--model_name', nargs='+', action='extend',
                        help='Name of model.')
    parser.add_argument('-sld', '--sld', type=float, nargs='+', action='append',
                        help='excess scattering length density or contrast.')
    parser.add_argument('-pd', '--polydispersity', type=float, nargs='+', action='extend',
                        help='Polydispersity of subunits for each model.')
    parser.add_argument('-com', '--com', type=float_list, nargs='+', action='append',
                         help='displacement for each subunits in each model.')
    parser.add_argument('-rot', '--rotation', type=float_list, nargs='+', action='append',
                         help='rotation for each subunits in each model.')
    parser.add_argument('-sigmar', '--sigma_r', type=float, nargs='+', action='extend',
                        help='interface roughness for each model.')
    parser.add_argument('-c', '--conc', type=float, nargs='+', action='extend',
                        help='volume fraction concentration.')
    parser.add_argument('-exclude', '--exclude_overlap', type=str2bool, nargs='+', action='extend',
                        help='bool to exclude overlap.')

    # Optional structure factor related inputs
    parser.add_argument('-S', '--S', type=str, nargs='+', action='extend',
                        help='structure factor: None/HS/aggregation in each model.')
    parser.add_argument('-Sp', '--S_par', type=float_list, nargs='+', action='append',
                        help='parameters of structure factor for each model.')

    # Optional general inputs
    parser.add_argument('-qmin', '--qmin', type=float, default=SimulationParameters.qmin, 
                        help='Minimum q-value for the scattering curve.')
    parser.add_argument('-qmax', '--qmax', type=float, default=SimulationParameters.qmax, 
                        help='Maximum q-value for the scattering curve.')
    parser.add_argument('-Nq', '--qpoints', type=int, default=SimulationParameters.qpoints, 
                        help='Number of points in q.')
    parser.add_argument('-Np', '--prpoints', type=int, default=SimulationParameters.prpoints, 
                        help='Number of points in the pair distance distribution function.')
    parser.add_argument('-N', '--Npoints', type=int, default=SimulationParameters.Npoints, 
                        help='Number of simulated points per model.')
    parser.add_argument('-expo', '--exposure', type=float, default=500, 
                        help='Exposure time in arbitrary units.')

    # Optional plot options
    parser.add_argument('-lin', '--xscale_lin', action='store_true', default=False, 
                        help='include flag (no input) to make q scale linear instead of logarithmic.')
    parser.add_argument('-hres', '--high_res', action='store_true', default=False, 
                        help='include flag (no input) to output high resolution plot.')

    # Optional SESANS-related options (Shape2SESANS)
    parser.add_argument('-ss', '--sesans', action='store_true', default=False,
                        help='Calculate SESANS data from the SAS data.')
    parser.add_argument('-sse', '--sesans_error', type=float, default=0.02, 
                        help='Baseline SESANS error relative to max signal.')
    parser.add_argument('-Nd', '--deltapoints', type=int, default=150,
                        help='Number of points in delta.')
    
    args = parser.parse_args()

    ################################ Read input values ################################

    if args.sesans:
        # make extended q-range for sesans
        Sim_par = SimulationParameters(qmin=1e-6, qmax=0.1, qpoints=20000, prpoints=args.prpoints, Npoints=args.Npoints)
    else:
        Sim_par = SimulationParameters(qmin=args.qmin, qmax=args.qmax, qpoints=args.qpoints, prpoints=args.prpoints, Npoints=args.Npoints)
    
    # read subunit type(s)
    if args.subunit is None:
        raise argparse.ArgumentError(args.subunit, "No subunit type was given as an input.")
    
    # read dimensions
    if args.dimension is None:
        raise argparse.ArgumentError(args.dimension, "No dimensions were given as an input.")
    for subunit, dimension in zip(args.subunit, args.dimension):
         if len(subunit) != len(dimension):
            raise argparse.ArgumentTypeError("Mismatch between number subunit types (%d) and dimensions lists (%d)." % (len(subunit),len(dimension)))
    
    # read number of models
    num_models = len(args.subunit)
    if num_models == 1:
        printt(f"Simulating {num_models} model...")
    else: 
        printt(f"Simulating {num_models} models...")

    # prepare lists (if several models are simulated simultaneously)
    r_list, pr_norm_list, I_list, Isim_list, sigma_list, S_eff_list = [], [], [], [], [], [] 
    x_list, y_list, z_list, sld_list, model_filename_list, name_list = [], [], [], [], [], []
    if args.sesans:
        delta_list,G_list,Gsim_list,sigma_G_list = [],[],[],[]

    # loop over models
    for i in range(num_models):
        
        # read model name for model i
        model_name = check_input(args.model_name, f"Model {i}", "model name", i)
        if model_name in name_list:
            #model names should be unique
            model_name += '_' + str(i+1)  

        printt(" ")
        printt(f"    Generating points for Model: " + model_name)

        # read subunit and dimensions for model i
        N_subunits = len(args.subunit[i])

        #read for SLD, COM, and rotation for model i
        sld = check_3Dinput(args.sld, [1.0], "SLD", N_subunits, i)
        com = check_3Dinput(args.com, [[0, 0, 0]], "COM", N_subunits, i)
        rotation = check_3Dinput(args.rotation, [[0, 0, 0]], "rotation", N_subunits, i)

        #read exclude overlap input
        exclude_overlap = check_input(args.exclude_overlap, True, "exclude_overlap", i)

        #make point cloud
        Distr = getPointDistribution(args.subunit[i],sld,args.dimension[i],com,rotation,exclude_overlap,args.Npoints)
        
        ################################# Calculate Theoretical I(q) #################################
        printt(" ")
        printt("    Calculating intensity, I(q)...")

        # read polydispersity and concentration values (or use default values)
        pd = check_input(args.polydispersity, 0.0, "polydispersity", i)
        conc = check_input(args.conc, 0.02, "concentration", i)

        #read structure factor (default None) and related parameters
        Stype = check_input(args.S, 'None', "Structure type", i)
        try:
            S_par = args.S_par[i][0]
        except:
            S_par = []

        #read interface roughness (default none)
        sigma_r = check_input(args.sigma_r, 0.0, "sigma_r", i)

        #calculate theoretical scattering
        Theo_calc = TheoreticalScatteringCalculation(System=ModelSystem(PointDistribution=Distr, 
                                                                        Stype=Stype, par=S_par, 
                                                                        polydispersity=pd, conc=conc, 
                                                                        sigma_r=sigma_r), 
                                                                        Calculation=Sim_par)
        Theo_I = getTheoreticalScattering(Theo_calc)

        #save models
        model_filename = "_".join(model_name.split()) # remove whitespace
        WeightedPairDistribution.save_pr(args.prpoints, Theo_I.r, Theo_I.pr, model_filename)
        StructureFactor.save_S(Theo_I.q, Theo_I.S_eff, model_filename)
        ITheoretical(Theo_I.q).save_I(Theo_I.I, model_filename)

        #save points
        save_points(np.concatenate(Distr.x), np.concatenate(Distr.y), np.concatenate(Distr.z), np.concatenate(Distr.sld), model_filename)

        ######################################### Simulate I(q) ##########################################
        exposure = args.exposure
        Sim_calc = SimulateScattering(q=Theo_I.q, I0=Theo_I.I0, I=Theo_I.I, exposure=exposure)
        Sim_I = getSimulatedScattering(Sim_calc)

        # Save simulated I(q) using IExperimental
        Isim_class = IExperimental(q=Sim_I.q, I0=Theo_I.I0, I=Theo_I.I, exposure=exposure)
        Isim_class.save_Iexperimental(Sim_I.I_sim, Sim_I.I_err, model_filename)

        ######################################### save data for plots ##########################################
        x_list.append(np.concatenate(Distr.x))
        y_list.append(np.concatenate(Distr.y))
        z_list.append(np.concatenate(Distr.z))
        sld_list.append(np.concatenate(Distr.sld))

        r_list.append(Theo_I.r)
        pr_norm_list.append(Theo_I.pr_norm)
        I_list.append(Theo_I.I)
        S_eff_list.append(Theo_I.S_eff)

        Isim_list.append(Sim_I.I_sim)
        sigma_list.append(Sim_I.I_err)

        model_filename_list.append(model_filename)
        name_list.append(model_name)

        ######################################### SESANS ##########################################

        if args.sesans:
            # make spin echo length (delta) range (x-axis in SESANS)
            delta = np.linspace(0, 3 * np.max(Theo_I.r), args.deltapoints)
            G = calc_G_sesans(Theo_I.q,delta,Theo_I.I)

            # simulate noisy sesans data         
            G_sim,sigma_G = simulate_sesans(delta,G,args.sesans_error)
            
            # append to list (in case of multiple models)
            delta_list.append(delta)
            G_list.append(G)
            Gsim_list.append(G_sim)
            sigma_G_list.append(sigma_G)


    printt(" ")
    printt("Generating plots...")
    colors = ['blue','red','green','orange','purple','cyan','magenta','black','grey','pink','forrestgreen']

    #plot 2D projections
    for m in model_filename_list:
        print("    2D projection: points_" + m + ".png ...")
    plot_2D(x_list, y_list, z_list, sld_list, model_filename_list, args.high_res, colors)
    
    #3D vizualization: generate pdb file with points
    for m in model_filename_list:
        print("    3D models: " + m + ".pdb ...")
    generate_pdb(x_list, y_list, z_list, sld_list, model_filename_list)
    
    #plot p(r) and I(q)
    print("    plot pr and Iq and Isim: plot.png ...")
    plot_results(Theo_I.q, r_list, pr_norm_list, I_list, Isim_list, 
                 sigma_list, S_eff_list, name_list, args.xscale_lin, args.high_res, colors)

    #plot and save sesans
    if args.sesans:
        plot_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list, args.high_res, colors)
        save_sesans(delta_list, G_list, Gsim_list, sigma_G_list, model_filename_list)

    time_total = time.time() - start_total
    printt(" ")
    printt("Simulation successfully completed.")
    printt("    Total run time: " + str(round(time_total, 1)) + " seconds.")
    printt(" ")

    # close log file and copy into model directories
    #f_out.close()
    for model_filename in model_filename_list:
        shutil.copy('shape2sas.log','%s/%s.log' % (model_filename,model_filename))
        if args.high_res:
            shutil.copy('plot.pdf','%s/plot_%s.pdf' % (model_filename,model_filename))
        else:
            shutil.copy('plot.png','%s/plot_%s.png' % (model_filename,model_filename))
        if args.sesans:
            if args.high_res:
                shutil.copy('sesans.pdf','%s/sesans_%s.pdf' % (model_filename,model_filename))
            else:
                shutil.copy('sesans.png','%s/sesans_%s.png' % (model_filename,model_filename))