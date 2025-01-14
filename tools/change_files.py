#!/usr/bin/env python3

import glob
import time
import os
import gc
from pathlib import Path

import asdf
import numpy as np
import argparse

from fast_cksum.cksum_io import CksumWriter

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/light_cone_catalog/"
DEFAULTS['new_catalog_parent'] = "/global/cscratch1/sd/boryanah/new_lc_halos/"
DEFAULTS['compaso_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus"
DEFAULTS['z_start'] = 0.1
DEFAULTS['z_stop'] = 2.5


def rewrite_asdf(old_file, new_file):
    """
    Given old astropy based file, rewrite new file as numpy array
    """
    old_file = str(old_file)
    new_file = str(new_file)
    with asdf.open(old_file, memmap=False) as oldaf:
        newtree = dict(header=oldaf['header'].copy())
        newtree['data'] = {c: np.asarray(oldaf['data'][c]) for c in oldaf['data']}

    # set compression options here
    # old compression options
    """
    asdf.compression.set_compression_options(typesize="auto", shuffle="shuffle", asdf_block_size=12*1024**2, blocksize=3*1024**2, nthreads=4)
    newaf = asdf.AsdfFile(newtree)
    with CksumWriter(new_file) as fp:
        newaf.write_to(fp, all_array_compression='blsc')
    newaf.close()
    """
    # new compression settings
    compression_kwargs=dict(typesize="auto", shuffle="shuffle", compression_block_size=12*1024**2, blosc_block_size=3*1024**2, nthreads=4)
    with asdf.AsdfFile(newtree) as af, CksumWriter(new_file) as fp:
        af.write_to(fp, all_array_compression='blsc', compression_kwargs=compression_kwargs)
    
def extract_redshift(fn):
    """
    Extract the redshift value from the file name
    """
    red = float(str(fn).split('z')[-1][:5])
    return red

def main(sim_name, z_start, z_stop, compaso_parent, catalog_parent, new_catalog_parent):

    # location of the light cone catalogs
    catalog_parent = Path(catalog_parent)

    # location of the new light cone catalogs
    new_catalog_parent = Path(new_catalog_parent)

    # location of the compaso catalogs
    compaso_parent = Path(compaso_parent)
    
    # old and new directory where we save the final outputs from merger trees and halo catalogs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones"
    new_cat_lc_dir = new_catalog_parent / sim_name / "halos_light_cones"
    
    # standard redshift names
    compaso_dir = compaso_parent / sim_name / "halos"
    sim_slices = sorted(compaso_dir.glob('z*'))
    new_redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
    print("new redshifts = ", new_redshifts)
    
    # old redshift names
    sim_slices = sorted(cat_lc_dir.glob('z*'))
    redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
    print("redshifts = ", redshifts)
    
    # loop through all available redshifts (there should be more of new but both start at z = 0.1, so fine)
    for i in range(len(redshifts)):
        z_old = redshifts[i]
        z_new = new_redshifts[i]
        print("old, new redshift = ", z_old, z_new)

        # create new directory
        new_dir = new_cat_lc_dir / (f'z{z_new:.3f}')
        os.makedirs(new_dir, exist_ok=True)
        
        # skip the redshift if not between the desired start and stop
        if (z_old < z_start) or (z_old > z_stop): continue

        # change the halo info file
        old_file = cat_lc_dir / (f'z{z_old:.3f}') / 'lc_halo_info.asdf'
        new_file = new_dir / 'lc_halo_info.asdf'
        rewrite_asdf(old_file, new_file)

        # change the particle file
        old_file = cat_lc_dir / (f'z{z_old:.3f}') / 'lc_pid_rv.asdf'
        new_file = new_dir / 'lc_pid_rv.asdf'
        rewrite_asdf(old_file, new_file)
        quit()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    # parser arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)    
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_stop'])
    parser.add_argument('--compaso_parent', help='Halo catalog directory', default=(DEFAULTS['compaso_parent']))
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--new_catalog_parent', help='New light cone catalog directory', default=(DEFAULTS['new_catalog_parent']))
    args = vars(parser.parse_args())
    main(**args)

