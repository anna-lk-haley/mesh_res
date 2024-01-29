import sys
import os
import csv
import pyvista as pv
import vtk
import numpy as np
import math as math
from bsl.dataset import Dataset
from bsl import common as cc
from pathlib import Path 
from scipy.spatial import cKDTree
import multiprocess as mp
import h5py

mu = 0.0035

def integrate_data(data):
    """Integrate point and cell data.

    Area or volume is also provided in point data.

    This filter uses the VTK `vtkIntegrateAttributes
    <https://vtk.org/doc/nightly/html/classvtkIntegrateAttributes.html>`_

    Returns
    -------
    UnstructuredGrid
        Mesh with 1 point and 1 vertex cell with integrated data in point
        and cell data.
    """

    alg = vtk.vtkIntegrateAttributes()
    alg.SetInputData(data)
    alg.Update()
    return pv.wrap(alg.GetOutput())

def average_WSS(dd):
    dd.surf.point_arrays['wss_avg'] = np.zeros(dd(idx=0, array='wss').shape)
    #do about 100 tsteps for each
    step = 1 #math.floor(len(dd.up_files)/100)
    tsteps = len(list(range(0,len(dd.wss_files), step)))
    for idx in range(0,len(dd.wss_files), step):
        #if idx % 10 == 0:
        #    print(idx)
        dd.surf.point_arrays['wss_avg'] += dd(idx=idx, array='wss')/tsteps
    return dd

def average_WSSG(dd, outfolder, case_name):
    gradient = dd.surf.compute_derivative(scalars='wss_avg', gradient=True)
    dd.surf.point_arrays['wssg_avg'] = gradient.point_arrays['gradient']
    dd.surf.save(str(outfolder)+'/'+ case_name + '_data.vtp')
    return dd

def L2_norm(outfolder, case_names, writer2):
    wss_avg_files = [s + '_data.vtp' for s in case_names] #to preserve the ordering
    high_surf = pv.read(sorted(outfolder.glob('*_high_data.vtp'))[0])
    l2_high = np.sum(high_surf.point_arrays['wss_avg']**2)
    l2_high_gradient = np.sum(high_surf.point_arrays['wssg_avg']**2)
    newlist = ['DOFs','WSS L2 norm', 'WSSG L2 norm']
    writer2.writerow(newlist)

    for file in wss_avg_files:
        surf = pv.read(str(outfolder)+'/'+file)
        dofs = len(surf.points)
        #have to interpolate to high surf
        interpolated = high_surf.sample(surf)
        l2_norm = np.sum((interpolated.point_arrays['wss_avg']-high_surf.point_arrays['wss_avg'])**2)/l2_high
        l2_gradient_norm = np.sum((interpolated.point_arrays['wssg_avg']-high_surf.point_arrays['wssg_avg'])**2)/l2_high_gradient
        writer2.writerow([dofs, l2_norm, l2_gradient_norm])

def get_u(u_file):
    with h5py.File(u_file, 'r') as hf:
        val = np.array(hf['Solution']['u'])
    return val

def compute_Ve(case_name, tsteps, mesh, up_files, output):
    Ve_avg = 0
    #do about 100 tsteps for each to start
    for idx, u_file in enumerate(up_files):
        if idx % 10 == 0:
            print(idx)

        mesh.point_arrays['u'] = get_u(u_file)

        mesh = mesh.compute_derivative(scalars="u", gradient=True, qcriterion=False, faster=False)
        J = mesh.point_arrays['gradient'].reshape(-1, 3, 3)
        D = J + np.transpose(J, axes=(0,2,1))

        mesh.point_arrays['Ve'] = np.sum(np.sum(D*D, axis = 2), axis=1) #double dot product
        volume_integrated = integrate_data(mesh)

        Ve = volume_integrated['Ve'][0]*1E-3 #because the derivatives are in mm
        Ve_avg += Ve/tsteps

    output.put(Ve_avg)

def compute_Ve_interp(case_name, tsteps, mesh, mesh_i, up_files, output):
    Ve_avg = 0
    #do about 100 tsteps for each to start
    for idx, u_file in enumerate(up_files):
        if idx % 10 == 0:
            print(idx)

        mesh.point_arrays['u'] = get_u(u_file)
        if len(mesh_i.points) == len(mesh.points): #same mesh do nothing
            newmesh=mesh
        else: #needs to be interpolated
            newmesh = mesh_i.sample(mesh)
        newmesh = newmesh.compute_derivative(scalars="u", gradient=True, qcriterion=False, faster=False)
        J = newmesh.point_arrays['gradient'].reshape(-1, 3, 3)
        D = J + np.transpose(J, axes=(0,2,1))

        newmesh.point_arrays['Ve'] = np.sum(np.sum(D*D, axis = 2), axis=1) #double dot product
        volume_integrated = integrate_data(mesh)

        Ve = volume_integrated['Ve'][0]*1E-3 #because the derivatives are in mm
        Ve_avg += Ve/tsteps

    output.put(Ve_avg)

if __name__ == "__main__":

    outfolder = Path(('convergence_data/case_{}'.format(sys.argv[1])))
    if not outfolder.exists():
        outfolder.mkdir(parents=True, exist_ok=True)
    
    #loop through each case in case folder
    folder = 'cases/case_{}'.format(sys.argv[1])
    case_names = [ name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name)) ]

    file2 = str(outfolder)+'/case_{}_wssg_convergence.csv'.format(sys.argv[1])
    file_2p = Path(file2)
    if not file_2p.exists():
        outfile2 = open(file2, 'w', encoding='UTF8', newline='')
        writer2 = csv.writer(outfile2) #writer for wss & wssg
        for case_name in case_names:
            results = folder+'/'+ case_name + '/results/'
            results_folder = Path((results + os.listdir(results)[0])) #results folder eg. results/art_
            print(results + os.listdir(results)[0])
            main_folder = Path(results_folder).parents[1]
            dd = Dataset(results_folder)
            splits = case_name.split('_')
            seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]
            vtu_file = Path(main_folder/ ('mesh/' + seg_name + '.vtu'))
            dd = dd.assemble_mesh().assemble_surface(mesh_file=vtu_file) 

            #WSS averages at each node
            dd = average_WSS(dd)
            #WSS gradients at each node
            dd = average_WSSG(dd, outfolder, case_name) 
        #get L2 norms of WSS and WSSG
        L2_norm(outfolder, case_names, writer2)
        print('completed wss convergence')

    #it seems dumb but actually having two loops here is better
    file1 = str(outfolder)+'/case_{}_Ve_convergence.csv'.format(sys.argv[1])
    file_1p = Path(file1)
    if not file_1p.exists():
        outfile1 = open(file1, 'w', encoding='UTF8', newline='')
        writer = csv.writer(outfile1) #writer for Ve
        writer.writerow(['Case name', 'Ve', 'tsteps']) 
        #Parallel implementation of Ve calculation
        for case_name in case_names:
            results = folder+'/'+ case_name + '/results/'
            results_folder = Path((results + os.listdir(results)[0])) #results folder eg. results/art_
            print(results + os.listdir(results)[0])
            main_folder = Path(results_folder).parents[1]
            dd = Dataset(results_folder)
            splits = case_name.split('_')
            seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]
            vtu_file = Path(main_folder/ ('mesh/' + seg_name + '.vtu'))
            dd = dd.assemble_mesh().assemble_surface(mesh_file=vtu_file) 
            centerline_file = Path(main_folder /('PTSeg'+ splits[1] + '_cl_centerline_mapped.vtp'))
            #Ve_avg = compute_Ve(case_name = case_name, dd=dd, writer=writer)
            Ve_avg = 0
            #divide up_files into 40 procs
            num = math.floor(len(dd.up_files)/39)
            up_files = []
            for i in range(39):
                up_files.append(dd.up_files[i*num:(i+1)*num-1])
            up_files.append(dd.up_files[39*num:-1]) #the remaining list of files

            output = mp.Queue()
            tsteps = len(dd.up_files)
            processes = [mp.Process(target=compute_Ve, args=(case_name, tsteps, dd.mesh, up_files[x], output)) for x in range(40)]
            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()
                Ve_avg += output.get()
            Ve_avg = mu*Ve_avg
            newlist = ["{}".format(case_name.split('_')[-1]), Ve_avg, tsteps]
            writer.writerow(newlist)
        print('completed Ve convergence')
    

    file2 = str(outfolder)+'/case_{}_Ve_interp_convergence.csv'.format(sys.argv[1])
    file_2p = Path(file2)
    if not file_2p.exists():
        outfile2 = open(file2, 'w', encoding='UTF8', newline='')
        writer = csv.writer(outfile2) #writer for Ve
        writer.writerow(['Case name', 'Ve_interpolated']) 

        #get interpolation mesh
        interp_case = [s for s in case_names if "high" in s][0]
        results_interp = folder+'/'+ interp_case + '/results/'
        results_folder_interp = Path((results_interp + os.listdir(results_interp)[0])) #results folder eg. results/art_
        main_folder_interp = Path(results_folder_interp).parents[1]
        di = Dataset(results_folder_interp)
        splits_interp = interp_case.split('_')
        seg_name_interp = 'PTSeg'+ splits_interp[1] +'_' + splits_interp[-1]
        vtu_file_interp = Path(main_folder_interp/ ('mesh/' + seg_name_interp + '.vtu'))
        di = di.assemble_mesh().assemble_surface(mesh_file=vtu_file_interp) 

        #Parallel implementation of Ve calculation
        for case_name in case_names:
            results = folder+'/'+ case_name + '/results/'
            results_folder = Path((results + os.listdir(results)[0])) #results folder eg. results/art_
            print(results + os.listdir(results)[0])
            main_folder = Path(results_folder).parents[1]
            dd = Dataset(results_folder)
            splits = case_name.split('_')
            seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]
            vtu_file = Path(main_folder/ ('mesh/' + seg_name + '.vtu'))
            dd = dd.assemble_mesh().assemble_surface(mesh_file=vtu_file) 
            #Ve_avg = compute_Ve(case_name = case_name, dd=dd, writer=writer)
            Ve_avg = 0
            #divide up_files into 40 procs
            num = math.floor(len(dd.up_files)/39)
            up_files = []
            for i in range(39):
                up_files.append(dd.up_files[i*num:(i+1)*num-1])
            up_files.append(dd.up_files[39*num:-1]) #the remaining list of files

            output = mp.Queue()
            tsteps = len(dd.up_files)
            processes = [mp.Process(target=compute_Ve_interp, args=(case_name, tsteps, dd.mesh, di.mesh, up_files[x], output)) for x in range(40)]
            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()
                Ve_avg += output.get()
            Ve_avg = mu*Ve_avg
            newlist = ["{}".format(case_name.split('_')[-1]), Ve_avg]
            writer.writerow(newlist)
        print('completed Ve interpolated convergence')
    


    