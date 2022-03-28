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
from helpfunctions import sinc,genpoints,calc_dist,calc_Rg
import time

if __name__=='__main__':

    ## time
    start_total = time.time()
    
    ## read Json input
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    qmin = float(json_variables['qmin'])
    qmax = float(json_variables['qmax'])
    Nbins = int(json_variables['prpoints']) # number of points in p(r)
    polydispersity = float(json_variables['polydispersity'])
    folder = json_variables['_base_directory'] # output folder dir
    
    i = 1
    model_name = 
    model = json_variables['model1']
    a1 = float(json_variables['a1']),float(json_variables['b1']),float(json_variables['c1']),float(json_variables['p1'])
    try:
        b1 = float(json_variables['b1'])
    except:
        b1 = a1
        ,float(json_variables['c1']),float(json_variables['p1'])
    a1,b1,c1,p1 = float(json_variables['a1']),float(json_variables['b1']),float(json_variables['c1']),float(json_variables['p1'])
    x1,y1,z1 = float(json_variables['x1']),float(json_variables['y1']),float(json_variables['z1'])

    model2 = json_variables['model2']
    a2,b2,c2,p2 = float(json_variables['a2']),float(json_variables['b2']),float(json_variables['c2']),float(json_variables['p2'])
    x2,y2,z2 = float(json_variables['x2']),float(json_variables['y2']),float(json_variables['z2'])

    model3 = json_variables['model3']
    a3,b3,c3,p3 = float(json_variables['a3']),float(json_variables['b3']),float(json_variables['c3']),float(json_variables['p3'])
    x3,y3,z3 = float(json_variables['x3']),float(json_variables['y3']),float(json_variables['z3'])

    model4 = json_variables['model4']
    a4,b4,c4,p4 = float(json_variables['a4']),float(json_variables['b4']),float(json_variables['c4']),float(json_variables['p4'])
    x4,y4,z4 = float(json_variables['x4']),float(json_variables['y4']),float(json_variables['z4'])

    ## setup  messaging
    message = genapp(json_variables)

    ## adjust number of points, to maximum is 10,000
    count = 0
    if model1 != 'none': count += 1
    if model2 != 'none': count += 1
    if model3 != 'none': count += 1
    if model4 != 'none': count += 1 
    Npoints_max = 5000
    Npoints = int(Npoints_max/count)
    
    ## generate points
    start_points = time.time()
    message.udpmessage({"_textarea":"generating points\n" })
    Nsum = 0
    x_new,y_new,z_new,p_new = 0,0,0,0
    if model1 != 'none': 
        x_new,y_new,z_new,p_new,N,rho = genpoints(x_new,y_new,z_new,p_new,x1,y1,z1,model1,a1,b1,c1,p1,Npoints)
        srho = rho*p1
        message.udpmessage({"_textarea":"    generating %d points for model 1: %s\n" % (N,model1) })
        message.udpmessage({"_textarea":"        point density      : %1.2e (points per volume)\n" % rho}) 
        message.udpmessage({"_textarea":"        scattering density : %1.2e (density times scattering length)\n" % srho})
        Nsum += N
    if model2 != 'none': 
        x_new,y_new,z_new,p_new,N,rho = genpoints(x_new,y_new,z_new,p_new,x2,y2,z3,model2,a2,b2,c2,p2,Npoints)
        srho = rho*p2
        message.udpmessage({"_textarea":"    generating %d points for model 2: %s\n" % (N,model2) })
        message.udpmessage({"_textarea":"        point density      : %1.2e (points per volume)\n" % rho}) 
        message.udpmessage({"_textarea":"        scattering density : %1.2e (density times scattering length)\n" % srho})
        Nsum += N
    if model3 != 'none': 
        x_new,y_new,z_new,p_new,N,rho = genpoints(x_new,y_new,z_new,p_new,x3,y3,z3,model3,a3,b3,c3,p3,Npoints)
        srho = rho*p3
        message.udpmessage({"_textarea":"    generating %d points for model 3: %s\n" % (N,model3) })
        message.udpmessage({"_textarea":"        point density      : %1.2e (points per volume)\n" % rho}) 
        message.udpmessage({"_textarea":"        scattering density : %1.2e (density times scattering length)\n" % srho})
        Nsum += N
    if model4 != 'none': 
        x_new,y_new,z_new,p_new,N,rho = genpoints(x_new,y_new,z_new,p_new,x4,y4,z4,model4,a4,b4,c4,p4,Npoints)
        srho = rho*p4
        message.udpmessage({"_textarea":"    generating %d points for model 4: %s\n" % (N,model4) })
        message.udpmessage({"_textarea":"        point density      : %1.2e (points per volume)\n" % rho}) 
        message.udpmessage({"_textarea":"        scattering density : %1.2e (density times scattering length)\n" % srho})
        Nsum += N
    message.udpmessage({"_textarea":"    total number of points: %d\n" % Nsum})
    time_points = time.time()-start_points
    message.udpmessage({"_textarea":"    time points: %1.2f\n" % time_points})

    ## find plot limits
    max_x = np.amax(abs(x_new))
    max_y = np.amax(abs(y_new))
    max_z = np.amax(abs(z_new))
    max_l = np.amax([max_x,max_y,max_z])
    lim = [-max_l,max_l]

    ## figure
    plt.figure(figsize=(15,10))
    markersize = 3
    p1 = plt.subplot(2,3,1)
    p1.plot(x_new,z_new,linestyle='none',marker='.',markersize=markersize,color='red')
    p1.set_xlim(lim)
    p1.set_ylim(lim)
    p1.set_xlabel('x')
    p1.set_ylabel('z')
    p1.set_title('pointmodel, (x,z), "front"')

    p2 = plt.subplot(2,3,2)
    p2.plot(y_new,z_new,linestyle='none',marker='.',markersize=markersize,color='red')
    p2.set_xlim(lim)
    p2.set_ylim(lim)
    p2.set_xlabel('y')
    p2.set_ylabel('z')
    p2.set_title('pointmodel, (y,z), "side"')

    p3 = plt.subplot(2,3,3)
    p3.plot(x_new,y_new,linestyle='none',marker='.',markersize=markersize,color='red')
    p3.set_xlim(lim)
    p3.set_ylim(lim)
    p3.set_xlabel('x')
    p3.set_ylabel('y')
    p3.set_title('pointmodel, (x,y), "bottom"')

    ## calculate all distances
    start_dist = time.time()
    message.udpmessage({"_textarea":"calculating distances...\n"})
    dx = calc_dist(x_new)
    dy = calc_dist(y_new)
    dz = calc_dist(z_new)
    dp = np.outer(p_new,p_new)
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    dist = d.flatten()
    contrast = dp.flatten()
    time_dist = time.time() - start_dist
    message.udpmessage({"_textarea":"    time dist: %1.2f\n" % time_dist})

    ## make histograms p(r)
    start_pr = time.time()
    message.udpmessage({"_textarea":"making p(r) (weighted histogram)..."})
    idx_nonzero = np.where(dist>0.0)
    Dmax = np.amax(dist)

    sigma = polydispersity
    factor_min = np.max([0,1-3*sigma])
    factor_max = 1+3*sigma
    Dmax_poly = Dmax*factor_max
    hr,bin_edges = np.histogram(dist,bins=Nbins,weights=contrast,range=(0,Dmax_poly)) 
    pr,bin_edges = np.histogram(dist[idx_nonzero],bins=Nbins,weights=contrast[idx_nonzero],range=(0,Dmax_poly))
    if polydispersity > 0.0:
        hr_poly = 0.0
        pr_poly = 0.0
        factor_range = np.linspace(factor_min,factor_max,7)
        for factor_d in factor_range:
            message.udpmessage({"_textarea":"." % factor_d})
            #message.udpmessage({"_textarea":"factor_d = %1.2f \n" % factor_d})
            dhr,bin_edges = np.histogram(dist*factor_d,bins=Nbins,weights=contrast,range=(0,Dmax_poly))
            dpr,bin_edges = np.histogram(dist[idx_nonzero]*factor_d,bins=Nbins,weights=contrast[idx_nonzero],range=(0,Dmax_poly)) 
            res = (1.0-factor_d)/sigma
            w = np.exp(-res**2/2.0) # give weight according to normal distribution
            #message.udpmessage({"_textarea":"w = %1.2f \n" % w})
            hr_poly += dhr*w
            pr_poly += dpr*w  
        #hr_poly /= sigma*np.sqrt(2*np.pi) # normalization
        #pr_poly /= sigma*np.sqrt(2*np.pi) # normalization
    else:
        hr_poly = hr
        pr_poly = pr
    dr = bin_edges[2]-bin_edges[1]
    r = bin_edges[0:-1]+dr/2
    pr /= np.amax(pr)
    pr_poly /= np.amax(pr_poly)
    message.udpmessage({"_textarea":".\n"})

    ## save p(r) to textfile
    with open('pr.d','w') as f:
        f.write('#  r p(r) p(r) polydisperse\n')
        for i in range(Nbins):
            f.write('%f %f %f\n' % (r[i],pr[i],pr_poly[i]))

    # calculate Rg
    Rg = calc_Rg(r,pr)
    Rg_poly = calc_Rg(r,pr_poly)

    time_pr = time.time() - start_pr
    message.udpmessage({"_textarea":"    Dmax              = %1.2f\n" % Dmax})
    message.udpmessage({"_textarea":"    Rg                = %1.2f\n" % Rg})
    if polydispersity > 0.0:
        message.udpmessage({"_textarea":"    Dmax polydisperse = %1.2f\n" % Dmax_poly})
        message.udpmessage({"_textarea":"    Rg polydisperse   = %1.2f\n" % Rg_poly})
    message.udpmessage({"_textarea":"    time p(r)         : %1.2f sec\n" % time_pr})

    ## calculate scattering
    start_scat = time.time()
    message.udpmessage({"_textarea":"calculating scattering...\n"})
    M = 1000
    q = np.linspace(qmin,qmax,M)

    I,I_poly = 0.0,0.0
    for i in range(Nbins):
        qr = q*r[i]
        I += hr[i]*sinc(qr)
        I_poly += hr_poly[i]*sinc(qr)
    I /= np.amax(I)
    I_poly /= np.amax(I_poly)

    ## save intensity to textfile
    with open('Iq.d','w') as f:
        f.write('# q I(q) I(q) polydisperse\n')
        for i in range(M):
            f.write('%f %f %f\n' % (q[i],I[i],I_poly[i]))

    time_scat = time.time() - start_scat
    message.udpmessage({"_textarea":"    time I(q): %1.2f sec\n" % time_scat})
   
    # generating plots, pdb and other outputs
    start_output = time.time()
    message.udpmessage({"_textarea":"making plots and pdb file etc...\n"})

    ## plot p(r)
    p4 = plt.subplot(2,3,4)
    #p4.plot(r,hr,color='green')
    p4.plot(r,pr,color='red',label='monodisperse')
    p4.set_xlabel('r')
    p4.set_ylabel('p(r)')
    p4.set_title('Pair distribution funciton, p(r)')
    if polydispersity > 0.0:
        p4.plot(r,pr_poly,linestyle='--',color='grey',label='polydisperse')
        p4.legend()

    ## plot scattering
    p5 = plt.subplot(2,3,5)
    p5.plot(q,I,color='red',label='monodisperse')
    p5.set_xscale('log')
    p5.set_yscale('log')
    p5.set_xlabel('q')
    p5.set_ylabel('I(q)')
    p5.set_title('Scattering, log-log scale')
    if polydispersity > 0.0:
        p5.plot(q,I_poly,linestyle='--',color='grey',label='polydisperse')
        p5.legend()

    ## plot scattering
    p6 = plt.subplot(2,3,6)
    p6.plot(q,I,color='red',label='monodisperse')
    p6.set_yscale('log')
    p6.set_xlabel('q')
    p6.set_ylabel('I(q)')
    p6.set_title('Scattering, lin-log scale')
    if polydispersity > 0.0:
        p6.legend()
        p6.plot(q,I_poly,linestyle='--',color='grey',label='polydisperse')

    ## figure settings
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

    ## generate pdb file
    with open('model.pdb','w') as f:
        f.write('TITLE    POINT SCATTER MODEL\n')
        f.write('REMARK   GENERATED WITH McSIM\n')
        f.write('REMARK   EACH BEAD REPRESENTED BY DUMMY CARBON ATOM\n')
        f.write('REMARK   SCATTERING LENGTH DENSITY INFORMATION NOT INCLUDED\n')
        f.write('REMARK   ONLY FOR VISUALIZATION, E.G. WITH PYMOL\n')
        f.write('REMARK    \n') 
        for i in range(len(x_new)):
            f.write('ATOM  %5i  C   ALA A%4i    %8.3f%8.3f%8.3f  1.00  0.00           C \n'  % (i,i,x_new[i],y_new[i],z_new[i]))
        f.write('END')

    ## compress output files to zip file
    os.system('zip results.zip pr.d Iq.d model.pdb plot.png')

    time_output = time.time()-start_output
    message.udpmessage({"_textarea":"    time output: %1.2f sec\n" % time_output}) 

    ## time
    time_total = time.time()-start_total
    message.udpmessage({"_textarea":"time total: %1.2f sec\n" % time_total}) 

    ## generate output
    output = {} # create an empty python dictionary
    output["pr"] = "%s/pr.d" % folder
    output["Iq"] = "%s/Iq.d" % folder
    output["pdb"] = "%s/model.pdb" % folder
    output["fig"] = "%s/plot.png" % folder
    output["zip"] = "%s/results.zip" % folder
    output["Dmax"] = "%1.2f" % Dmax
    output["Rg"] = "%1.2f" % Rg
    if polydispersity > 0.0:
        output["Dmax_poly"] = "%1.2f" % Dmax_poly 
        output["Rg_poly"] = "%1.2f" % Rg_poly
    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";

    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output


