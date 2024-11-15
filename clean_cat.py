#!/usr/bin/env python3

import glob
import time
import os
import gc
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import argparse
from astropy.table import Table
from numba import njit
import asdf

from fast_cksum.cksum_io import CksumWriter
from tools.compute_dist import dist

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['catalog_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus/halo_light_cones/"
#DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/new_lc_halos/"
DEFAULTS['z_start'] = 0.1
DEFAULTS['z_stop'] = 2.5

def save_asdf(table, filename, header, cat_lc_dir):
    """
    Save light cone catalog
    """
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]
        
    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }
    
    # save the data and close file
    output_file = asdf.AsdfFile(data_tree)
    output_file.write_to(os.path.join(cat_lc_dir, filename+".asdf"))
    output_file.close()

@njit
def fast_avg(vel, npout):
    """
    Compute the average position/velocity for each halo given an array containing the particle positions/velocities 
    for all halos and another array with the number of particles per halo
    """
    nstart = 0
    v_int = np.zeros((len(npout), 3), dtype=np.float32)
    for i in range(len(npout)):
        if npout[i] == 0: continue
        v = vel[nstart:nstart+npout[i]]

        s = np.array([0., 0., 0.])
        for k in range(npout[i]):
            for j in range(3):
                s[j] += v[k][j]
        for j in range(3):
            s[j] /= (npout[i])

        v_int[i] = s
        nstart += npout[i]

    return v_int

def vrange(starts, stops):
    """
    Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:
        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])
    """
    stops = np.asarray(stops)
    l = stops - starts # Lengths of each range.
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    
def compress_asdf(asdf_fn, table, header):
    """
    Given the file name of the asdf file, the table and the header, compress the table info and save as `asdf_fn' 
    """
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]

    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }
    
    # set compression options here
    # OLD SETTINGS
    """
    asdf.compression.set_compression_options(typesize="auto", shuffle="shuffle", asdf_block_size=12*1024**2, blocksize=3*1024**2, nthreads=4)
    with asdf.AsdfFile(data_tree) as af, CksumWriter(str(asdf_fn)) as fp: # where data_tree is the ASDF dict tree structure
        af.write_to(fp, all_array_compression="blsc")
    """
    # NEW SETTINGS
    compression_kwargs=dict(typesize="auto", shuffle="shuffle", compression_block_size=12*1024**2, blosc_block_size=3*1024**2, nthreads=4)
    with asdf.AsdfFile(data_tree) as af, CksumWriter(str(asdf_fn)) as fp: # where data_tree is the ASDF dict tree structure
        af.write_to(fp, all_array_compression='blsc', compression_kwargs=compression_kwargs)

    
def extract_redshift(fn):
    """
    Extract the redshift value from the file name
    """
    red = float(str(fn).split('z')[-1][:5])
    return red

def float_trunc(a, zerobits):
    """
    Set the least significant <zerobits> bits to zero in a numpy float32 or float64 array.
    Do this in-place. Also return the updated array.
    Maximum values of 'nzero': 51 for float64; 22 for float32.
    """
    at = a.dtype
    assert at == np.float64 or at == np.float32 or at == np.complex128 or at == np.complex64
    if at == np.float64 or at == np.complex128:
        assert zerobits <= 51
        mask = 0xffffffffffffffff - (1 << zerobits) + 1
        bits = a.view(np.uint64)
        bits &= mask
    elif at == np.float32 or at == np.complex64:
        assert zerobits <= 22
        mask = 0xffffffff - (1 << zerobits) + 1
        bits = a.view(np.uint32)
        bits &= mask
    return a

def clean_cat(z_current, cat_lc_dir, want_subsample_B):
    # load the halo light cone catalog
    halo_fn = cat_lc_dir / ("z%4.3f"%z_current) / "halo_info_lc.asdf"
    with asdf.open(halo_fn, lazy_load=True, copy_arrays=True) as f:
        halo_header = f['header']
        table_halo = f['data']

    # simulation parameters
    Lbox = halo_header['BoxSize']
    origins = np.array(halo_header['LightConeOrigins']).reshape(-1,3)
    print("Lbox = ", Lbox)

    # load the particles light cone catalog
    parts_fn = cat_lc_dir / ("z%4.3f"%z_current) / "pid_rv_lc.asdf"
    with asdf.open(parts_fn, lazy_load=True, copy_arrays=True) as f:
        parts_header = f['header']
        table_parts = f['data']

    # parse the halo positions, npstart, npoutA and halo ids (can reduce data usage with del's)
    halo_pos = table_halo['pos_interp']
    halo_index = table_halo['index_halo']
    halo_npout = table_halo['npoutA']
    halo_npoutA = halo_npout.copy()
    if want_subsample_B:
        halo_npout += table_halo['npoutB']
    halo_origin = (table_halo['origin'])%3
    
    # if we are removing the edges get rid of halos 10 Mpc/h off the x, y and z edges
    remove_edges = True
    if remove_edges:
        str_edges = ""
        # find halos that are near the edges
        offset = 10.
        x_min = -Lbox/2.+offset
        y_min = -Lbox/2.+offset
        z_min = -Lbox/2.+offset
        x_max = Lbox/2.-offset
        if origins.shape[0] == 1: # true of only the huge box where the origin is at the center
            y_max = Lbox/2.-offset
            z_max = Lbox/2.-offset
        else:
            y_max = 3./2*Lbox # what about when we cross the 1000. boundary
            z_max = 3./2*Lbox

        # define mask that picks away from the edges
        halo_mask = (halo_pos[:, 0] >= x_min) & (halo_pos[:, 0] < x_max)
        halo_mask &= (halo_pos[:, 1] >= y_min) & (halo_pos[:, 1] < y_max)
        halo_mask &= (halo_pos[:, 2] >= z_min) & (halo_pos[:, 2] < z_max)

        print("spatial masking = ", np.sum(halo_mask)*100./len(halo_mask))
    else:
        str_edges = "_all"
        halo_mask = np.ones(halo_pos.shape[0], dtype=bool)


    # figure out how many origins for the given redshifts
    unique_origins = np.unique(halo_origin)
    print("unique origins = ", unique_origins)

    # start an empty boolean array which will have "True" for only unique halos
    halo_mask_extra = np.zeros(halo_pos.shape[0], dtype=bool)

    # origin 1 relates to z direction while origin 2 relates to y direction
    origin_xyz_dic = {1: 2, 2: 1}

    # add to the halo mask requirement that halos be unique (for a given origin)
    for origin in unique_origins:
        # skip the original box
        if origin == 0: continue

        # boolean array masking halos at this origin
        mask_origin = halo_origin == origin

        # halo indices for this origin
        halo_inds = np.arange(len(halo_mask), dtype=int)[mask_origin]

        # reorder halos outwards (in order of z for origin 2 and y for origin 1)
        i_sort = np.argsort((halo_pos[:, origin_xyz_dic[origin]])[halo_inds])
        halo_inds = halo_inds[i_sort]

        # find unique halo indices (already for specific origins)
        _, inds = np.unique(halo_index[halo_inds], return_index=True)
        halo_mask_extra[halo_inds[inds]] = True

        # how many halos were left
        print("non-unique masking %d = "%origin, len(inds)*100./np.sum(mask_origin))

    # additionally remove halos that are repeated on the borders (0 & 1 and 0 & 2)
    for key in origin_xyz_dic.keys():

        # select calos in the original box (cond1) and halos living in box 1 (z < Lbox/2+10.) or box 2 (y < Lbox/2+10.)
        cond1 = np.arange(len(halo_mask), dtype=int)[halo_origin == 0]
        cond2 = np.arange(len(halo_mask), dtype=int)[(halo_origin == key) & (halo_pos[:, origin_xyz_dic[key]] < Lbox/2.+offset)]

        # forget about this if there are no halos with origin = 0
        if np.sum(cond1) == 0: continue

        # combine the conditions above
        halo_inds = np.hstack((cond1, cond2))
        _, inds = np.unique(halo_index[halo_inds], return_index=True)

        
        # overwrite the information about the halos living in 0 or 1 (z < Lbox/2+10.) and then 0 or 2 (y < Lbox/2+10.)
        halo_mask_extra[halo_inds] = False
        halo_mask_extra[halo_inds[inds]] = True

        # how many halos were left
        print("non-unique masking extra %d = "%key, len(inds)*100./len(halo_inds))

        # because of the continue statement above, this means that the only origin is 0, so executing this once is enough
        if len(unique_origins) == 1:
            break
        
    # add the extra mask coming from the uniqueness requirement
    halo_mask &= halo_mask_extra        

    # repeat halo mask npout times to get a mask for the particles
    parts_mask = np.repeat(halo_mask, halo_npout)
    print("particle masking from halos = ", np.sum(parts_mask)*100./len(parts_mask))

    # halo indices of the particles
    halo_inds = np.arange(len(halo_mask), dtype=int)
    parts_halo_inds = np.repeat(halo_inds, halo_npout)

    # number of unique hosts of particles belonging to halos near edges or repeated
    num_uni_hosts = len(np.unique(parts_halo_inds[parts_mask]))
    print("unique parts hosts, filtered halos = ", num_uni_hosts, np.sum(halo_mask))
    assert num_uni_hosts <= np.sum(halo_mask), "number of unique particle hosts must be less than or equal to number of halos in the mask"

    # add to the particle mask, particles whose pid equals 0 (i.e. not matched)
    parts_mask_extra = table_parts['pid'] != 0
    perc_before = np.sum(parts_mask)*100./len(parts_mask)
    parts_mask &= parts_mask_extra
    perc_after = np.sum(parts_mask)*100./len(parts_mask)
    print("pid =/= 0 masking all percent = ", np.sum(parts_mask_extra)*100./len(parts_mask))
    print("pid == 0 masking w/o edges percent = ", perc_before-perc_after)
    print("number of particles missing w/o edges = ", (perc_before-perc_after)/100.*len(parts_mask))
    
    # filter out the host halo indices of the particles left after removing halos near edges, non-unique halos and particles that were not matched
    parts_halo_inds = parts_halo_inds[parts_mask]

    # we can now count how many particles were left per halo and indicate the starting index and the count in the npstart and npout (note that this is A and B)
    uni_halo_inds, inds, counts = np.unique(parts_halo_inds, return_index=True, return_counts=True)
    print("how many halos' lives did you ruin? = ", num_uni_hosts - len(inds)) # sometimes we would have gotten rid of all particles in a halo (very rare)
    table_halo['npstartA'][:] = -999
    table_halo['npoutA'][:] = 0
    table_halo['npstartA'][uni_halo_inds] = inds
    table_halo['npoutA'][uni_halo_inds] = counts

    # apply the mask to the particles and to the halos
    for key in table_parts.keys():
        table_parts[key] = table_parts[key][parts_mask]
    for key in table_halo.keys():
        table_halo[key] = table_halo[key][halo_mask]

    # check for whether the npouts add up to the number of particles; whether we got rid of all pid == 0; whether we got rid of all non-unique halos
    assert np.sum(table_halo['npoutA']) == len(table_parts['pid']), "different number of particles and npout expectation"
    assert np.sum(table_parts['pid'] == 0) == 0, "still some particles with pid == 0"
    for key in origin_xyz_dic.keys():
        condition = (key == (table_halo['origin'])%3) | (0 == (table_halo['origin'])%3)
        assert len(np.unique(table_halo['index_halo'][condition])) == np.sum(condition), "still some non-unique halos left %d vs. %d"%(len(np.unique(table_halo['index_halo'][condition])), np.sum(condition))

    # check for whether the particles stray too far away from their halos
    halo_pos = table_halo['pos_interp']
    parts_halo_pos = np.repeat(halo_pos, table_halo['npoutA'], axis=0)
    #parts_dist = parts_halo_pos - parts_pos
    #parts_dist = np.sqrt(np.sum(parts_dist**2, axis=1))
    parts_dist = dist(parts_halo_pos, table_parts['pos'])
    print("min dist = ", np.min(parts_dist))
    print("max dist = ", np.max(parts_dist))

    # adding average velocity and position from subsample A (and B)
    halo_pos_avg = fast_avg(table_parts['pos'], table_halo['npoutA'])
    halo_vel_avg = fast_avg(table_parts['vel'], table_halo['npoutA'])

    # scaling down to only record the A subsample
    halo_npoutA = halo_npoutA[halo_mask]
    mask_lost = halo_npoutA > table_halo['npoutA']
    print("halos that now have fewer particles left than the initial subsample A = ", np.sum(mask_lost))
    halo_npoutA[mask_lost] = table_halo['npoutA'][mask_lost]
    starts = table_halo['npstartA'].astype(int)
    stops = starts + halo_npoutA.astype(int)
    parts_inds = vrange(starts, stops)

    # record the particles and the halos
    table_halo['npoutA'] = halo_npoutA
    halo_npstartA = np.zeros(len(halo_npoutA), dtype=table_halo['npstartA'].dtype)
    halo_npstartA[1:] = np.cumsum(halo_npoutA)[:-1]
    table_halo['npstartA'] = halo_npstartA
    for key in table_parts.keys():
        table_parts[key] = table_parts[key][parts_inds]
        #table_parts = Table(table_parts)

    # add columns for the averaged position and velocity
    table_halo = {field: np.array(table_halo[field]) for field in table_halo}
    table_halo['pos_avg'] = np.empty(halo_pos_avg.shape, dtype=np.float32)
    table_halo['vel_avg'] = np.empty(halo_vel_avg.shape, dtype=np.float32)
    table_halo['pos_avg'][:] = halo_pos_avg
    table_halo['vel_avg'][:] = halo_vel_avg

    # remove B subsample references
    if want_subsample_B:
        table_halo.pop('npoutB')
        table_halo.pop('npstartB')
        #table_halo.remove_column(f'npoutB')
        #table_halo.remove_column(f'npstartB')

    '''
    # save asdf without compression or truncation
    save_asdf(table_parts, "lc"+str_edges+"_pid_rv", parts_header, cat_lc_dir / ("z%4.3f"%z_current))
    save_asdf(table_halo, "lc"+str_edges+"_halo_info", halo_header, cat_lc_dir / ("z%4.3f"%z_current))
    '''

    # knock out last few digits: 4 bits of the pos, the lowest 12 bits of vel
    table_parts['pos'] = float_trunc(table_parts['pos'], 4)
    table_parts['vel'] = float_trunc(table_parts['vel'], 12)
    #table_parts['redshift'] = float_trunc(table_parts['redshift'], 12)
    table_parts = {field: np.array(table_parts[field]) for field in table_parts}
    
    # condense the asdf file
    halo_fn_new = cat_lc_dir / ("z%4.3f"%z_current) / ("lc"+str_edges+"_halo_info.asdf")
    compress_asdf(str(halo_fn_new), table_halo, halo_header)
    parts_fn_new = cat_lc_dir / ("z%4.3f"%z_current) / ("lc"+str_edges+"_pid_rv.asdf")
    compress_asdf(str(parts_fn_new), table_parts, parts_header)

def main(sim_name, z_start, z_stop, catalog_parent, want_subsample_B=True):
    """
    Main function: this script is for cleaning up the final halo light cone catalogs: in particular, 
    we remove the halos and their particles from the edges; we also remove the repeated halos (and their
    particles) from the box(es). Special care is taken when we have repeated halos on the boundary
    between two boxes (0 and 1 or 2), where we need to order halos in order of their y (for origin = 3)
    and z (for origin = 2). We then find the unique halos for origin 1 and origin 2 (and combine the
    halo indices). We find the unique halos in 0 or 1 (z < Lbox/2+10.) erasing any previous information about
    the halos living there (so that we avoid the case that a halo was unique in 1, but is not unique anymore
    because it appears in 0 and 1). We dind the unique halos in 0 or 1 (y < Lbox/2+10) erasing any previous 
    information abotu the halos living there (so that we avoid the case that a halo was unique in 2, but is not
    unique anymore because it appears in 0 and 2).  Next, we deal with the particles, where we remove all 
    particles for which we couldn't find matches. We compute the average position of the halo based on the A
    and B particle sumbsamples and finally, we scale down the particle subsamples to only include subsample A.
    Everything is compressed and floats are truncated for the particles.
    """
    # location of the light cone catalogs
    catalog_parent = Path(catalog_parent)
    
    # directory where we have saved the final outputs from merger trees and halo catalogs
    cat_lc_dir = catalog_parent / "halo_light_cones"/ sim_name

    # list all available redshifts
    sim_slices = sorted(cat_lc_dir.glob('z*'))
    
    redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
    print("redshifts = ",redshifts)

    # loop through all available redshifts
    for z_current in redshifts:
        print("current redshift = ", z_current)
        
        # skip the redshift if not between the desired start and stop
        if (z_current < z_start) or (z_current > z_stop): continue

        # clean the current catalog and save it as a compressed asdf
        clean_cat(z_current, cat_lc_dir, want_subsample_B)

    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    # parser arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)    
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_stop'])
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--want_subsample_B', help='If this option is called, will only work with subsample A and exclude B', action='store_false')
    args = vars(parser.parse_args())
    main(**args)
