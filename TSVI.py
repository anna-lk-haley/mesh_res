import sys
import os
import csv
import pyvista as pv
import numpy as np
from numpy.ctypeslib import as_ctypes
import math as math
from pathlib import Path 
import multiprocess as mp
import h5py

class Dataset():
    """ Load BSL-specific data and common ops. 
    """
    def __init__(self, folder, file_glob_key=None, mesh_glob_key=None):
        """ Init the dataset.

        Args:
            folder (path): a folder with h5 data and mesh files from the BSL solver.
            file_glob_key (str): key for globbing h5 files. 
            mesh_glob_key (str): key for globbing mesh h5 file.
        """
        self.folder = Path(folder)

        if mesh_glob_key is None:
            mesh_glob_key = '*h5'

        wss_folder = (folder / 'wss_files')
        if wss_folder.exists():
            wss_glob_key = '*_curcyc_*wss.h5'
            self.wss_files = sorted(wss_folder.glob(wss_glob_key),
                key=self._get_ts)

    def _get_ts(self, h5_file):
        """ Given a simulation h5_file, get ts. """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])
    
    def assemble_surface(self, mesh_file):
        """ Create PolyData from h5 mesh file. 

        Args:
            mesh_file
        """
        # assert self.mesh_file.exists(), 'mesh_file does not exist.'
        if mesh_file.suffix == '.h5':
            with h5py.File(mesh_file, 'r') as hf:
                points = np.array(hf['Mesh']['Wall']['coordinates'])
                cells = np.array(hf['Mesh']['Wall']['topology'])

                cell_type = np.ones((cells.shape[0], 1), dtype=int) * 3
                cells = np.concatenate([cell_type, cells], axis = 1)
                self.surf = pv.PolyData(points, cells)
        return self

def get_wss(wss_file, array='wss'):
    if array == 'wss':
        with h5py.File(wss_file, 'r') as hf:
            val = np.array(hf['Computed']['wss'])
    else:
        with h5py.File(wss_file, 'r') as hf:
            val = np.array(hf[array])
    return val

def WSSDivergence(dd, outfolder, tsteps, wss_files, idx, divwss_avg = None):
    surf=dd.surf.copy()

    if divwss_avg is not None: #should be an array when calculating the rms
        surf.point_arrays['div_wss_avg']=divwss_avg
        surf.point_arrays['div_wss_sqr']=np.zeros((len(surf.points),))
    else: #we want to calculate the avg
        surf.point_arrays['div_wss_avg']=np.zeros((len(surf.points),))

    for i, wss_file in enumerate(wss_files):
        ts = dd._get_ts(wss_file)
        file_old = str(outfolder) + '/divWSS_{}.h5'.format(ts)
        if divwss_avg is not None:
            surf.point_arrays['div_wss_sqr'] += (get_wss(file_old, 'DivWSS')-surf.point_arrays['div_wss_avg'])**2 
        else:
            if not Path(file_old).exists():
                #compute normalized wss
                wss = get_wss(wss_file)
                normalize = np.linalg.norm(wss, axis=1) 
                surf.point_arrays['wss'] = wss/normalize
                #compute gradients
                grad = surf.compute_derivative(scalars="wss", gradient=True, qcriterion=False, faster=False)
                surf.point_arrays['div_wss'] = grad.point_arrays['gradient'][:,0]+grad.point_arrays['gradient'][:,5]+grad.point_arrays['gradient'][:,8]
                surf.point_arrays['div_wss_avg'] += surf.point_arrays['div_wss']/tsteps
                
                #write to h5
                f = h5py.File(str(outfolder) + '/divWSS_{}.h5'.format(ts), 'w')
                f.create_dataset('DivWSS', data = surf.point_arrays['div_wss'])
            else:
                surf.point_arrays['div_wss'] = get_wss(file_old, 'DivWSS')
                surf.point_arrays['div_wss_avg'] += surf.point_arrays['div_wss']/tsteps
    
    if divwss_avg is None:
        f_proc = h5py.File(str(outfolder) + '/divWSS_avg_{}.h5'.format(idx), 'w')
        f_proc.create_dataset('DivWSS_avg', data = surf.point_arrays['div_wss_avg'])
    else:
        f_proc = h5py.File(str(outfolder) + '/divWSS_sqr_{}.h5'.format(idx), 'w')
        f_proc.create_dataset('DivWSS_sqr', data = surf.point_arrays['div_wss_sqr'])

if __name__ == "__main__":
    case = sys.argv[1]
    outfolder = Path(('convergence_data/case_{}/divwss'.format(case)))

    folder = 'cases/case_{}'.format(case)
    case_names = [ name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name)) ]
    
    for case_name in case_names:
        results = folder+'/'+ case_name + '/results/'
        results_folder = Path((results + os.listdir(results)[0])) 
        dd = Dataset(results_folder)
        main_folder = Path(results_folder).parents[1]
        splits = case_name.split('_')
        seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]
        h5_file = Path(main_folder/ ('data/' + seg_name + '.h5'))
        dd=dd.assemble_surface(mesh_file=h5_file)
        num = math.floor(len(dd.wss_files)/39)
        wss_files = []
        for i in range(39):
            wss_files.append(dd.wss_files[i*num:(i+1)*num-1])
        wss_files.append(dd.wss_files[39*num:-1]) #the remaining list of files

        tsteps = len(dd.wss_files)
        outfolder_case = outfolder / (case_name)
        if not outfolder_case.exists():
            outfolder_case.mkdir(parents=True, exist_ok=True)
        #first time get the average
        file_avg = str(outfolder_case) + '/divWSS_avg.h5'
        if not Path(file_avg).exists():
            processes = [mp.Process(target=WSSDivergence, args=(dd, outfolder_case, tsteps, wss_files[x], x)) for x in range(40)]
            # Run processes
            divwss_avg = np.zeros((len(dd.surf.points),))
            for p in processes:
                p.start()
            #
            # Exit the completed processes
            for p in processes:
                p.join()
                
            for idx in range(40):
                hf = h5py.File(str(outfolder_case) + '/divWSS_avg_{}.h5'.format(idx), 'r')
                divwss_avg += np.array(hf['DivWSS_avg'])

            f2 = h5py.File(file_avg, 'w')
            f2.create_dataset('DivWSS_avg', data = divwss_avg)
        else:
            hf = h5py.File(file_avg, 'r')
            divwss_avg = np.array(hf['DivWSS_avg'])
   
        #second time get the rms
        rms_file = str(outfolder_case) + '/divWSS_rms.h5'
        if not Path(rms_file).exists():
            #output2 = mp.Array('f', len(dd.surf.points) )
            processes_2 = [mp.Process(target=WSSDivergence, args=(dd, outfolder_case, tsteps, wss_files[x], x , divwss_avg)) for x in range(40)]
            # Run processes
            for p in processes_2:
                p.start()

            divwss_sqr = np.zeros((len(dd.surf.points),))
            # Exit the completed processes
            for p in processes_2:
                p.join()

            for idx in range(40):
                hf = h5py.File(str(outfolder_case) + '/divWSS_sqr_{}.h5'.format(idx), 'r')
                divwss_sqr += np.array(hf['DivWSS_sqr'])

            divwss_rms = np.sqrt(divwss_sqr)/tsteps
            f2 = h5py.File(rms_file, 'w')
            f2.create_dataset('DivWSS_rms', data = divwss_rms)
        
    #get L2 norms
    high_case = [s for s in case_names if "high" in s][0]
    high_wss = str(outfolder) + '/' + high_case + '/divWSS_rms.h5'
    divw_rms_high = get_wss(high_wss, 'DivWSS_rms')
    l2_high = np.sum(divw_rms_high**2)
    l2_TVSI_file = str(outfolder) + '/TSVI_L2.csv'
    outfile = open(l2_TVSI_file, 'w', encoding='UTF8', newline='')
    writer = csv.writer(outfile) #writer for wss & wssg
    writer.writerow(['Name','TSVI L2 Norm'])
    
    for case_name in case_names:
        outfolder_case = outfolder / (case_name)
        case_wss = str(outfolder_case) + '/divWSS_rms.h5'
        
        results = folder+'/'+ case_name + '/results/'
        results_folder = Path((results + os.listdir(results)[0])) 
        dd2 = Dataset(results_folder)
        main_folder = Path(results_folder).parents[1]

        results_high = folder+'/'+ high_case + '/results/'
        results_folder_high = Path((results_high + os.listdir(results_high)[0])) 
        dd = Dataset(results_folder_high)
        main_folder_high = Path(results_folder_high).parents[1]

        splits = case_name.split('_')
        high_name = 'PTSeg'+ splits[1] +'_high'
        high_file = Path(main_folder_high/ ('data/' + high_name + '.h5'))
        dd=dd.assemble_surface(mesh_file=high_file)

        seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]
        h5_file = Path(main_folder/ ('data/' + seg_name + '.h5'))
        dd2=dd2.assemble_surface(mesh_file=h5_file)
        
        #interpolate onto high mesh
        dd2.surf.point_arrays['divw_rms'] = get_wss(case_wss, 'DivWSS_rms')
        dd.surf = dd.surf.sample(dd2.surf) 
        l2_norm = np.sqrt(np.sum((divw_rms_high-dd.surf.point_arrays['divw_rms'])**2))/l2_high
        writer.writerow([case_name,l2_norm])
        



