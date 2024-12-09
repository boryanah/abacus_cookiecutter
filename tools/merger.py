from pathlib import Path

import asdf
import numpy as np
from astropy.table import Table
from numba import jit
import re

def extract_superslab(fn: Path):
    # looks like "associations_z0.100.0.asdf"
    return int(re.search(r"(\d+)\.asdf", fn.name).group(1))
    
def extract_redshift(fn):
    # looks like "associations_z0.100.0.asdf.minified" or "associations_z0.100.0.asdf"
    redshift = float('.'.join(fn.split("z")[-1].split('.')[:2]))
    return redshift

def get_zs_from_headers(snap_names):
    '''
    Read redshifts from merger tree files
    '''
    
    zs = np.empty(len(snap_names))
    for i, snap_name in enumerate(snap_names):
        with asdf.open(snap_name) as f:
            zs[i] = f["header"]["Redshift"]
    return zs

def get_one_header(merger_dir):
    '''
    Get an example header by looking at one association
    file in a merger directory
    '''

    # choose one of the merger tree files
    fn = list(merger_dir.glob('associations*.asdf'))[0]
    with asdf.open(fn) as af:
        header = af['header']
    return header

def unpack_inds(halo_ids):
    '''
    Unpack indices in Sownak's format of Nslice*1e12 
    + superSlabNum*1e9 + halo_position_superSlab
    '''
    
    # obtain slab number and index within slab
    id_factor = int(1e12)
    slab_factor = int(1e9)
    index = (halo_ids % slab_factor).astype(int)
    slab_number = ((halo_ids % id_factor - index) // slab_factor).astype(int)
    return slab_number, index

def pack_inds(halo_ids, slab_ids):
    '''
    Pack indices in Sownak's format of Nslice*1e12 
    + superSlabNum*1e9 + halo_position_superSlab
    '''
    # just as a place holder
    slice_ids = 0
    halo_ids = slab_ids*1e9 + halo_ids
    return halo_ids

def reorder_by_slab(fns):
    '''
    Reorder filenames in terms of their slab number
    '''
    return sorted(fns, key=extract_superslab)

def mark_ineligible_slow(nums, starts, main_progs, progs, halo_ind_prev, eligibility_prev, N_halos_slabs_prev, slabs_prev, inds_fn_prev):
    N_this_star_lc = len(nums)
    # loop around halos that were marked belonging to this redshift catalog
    for j in range(N_this_star_lc):
        # select all progenitors
        start = starts[j]
        num = nums[j]
        prog_inds = progs[start : start + num]

        # remove progenitors with no info
        prog_inds = prog_inds[prog_inds > 0]
        if len(prog_inds) == 0: continue

        # correct halo indices
        prog_inds = correct_inds(prog_inds, N_halos_slabs_prev, slabs_prev, inds_fn_prev)
        halo_inds = halo_ind_prev[prog_inds]

        # test output; remove in final version
        #if num > 1: print(halo_inds, Merger_prev['HaloIndex'][main_progs[j]])

        # mark ineligible
        eligibility_prev[halo_inds] = False
    return eligibility_prev

@jit(nopython = True)
def mark_ineligible(nums, starts, main_progs, progs, halo_ind_prev, eligibility_prev, offsets, slabs_prev_load):
    # constants used for unpacking
    id_factor = int(1e12)
    slab_factor = int(1e9)

    # number of objects marked belonging to this redshift catalog
    N_this_star_lc = len(nums)
    # loop around halos that were marked belonging to this redshift catalog
    for j in range(N_this_star_lc):
        # select all progenitors
        start = starts[j]
        num = nums[j]
        prog_inds = progs[start : start + num]

        # remove progenitors with no info
        prog_inds = prog_inds[prog_inds > 0]
        if len(prog_inds) == 0: continue

        # correct halo indices
        for k in range(len(prog_inds)):
            prog_ind = prog_inds[k]
            idx = (prog_ind % slab_factor)
            slab_id = ((prog_ind % id_factor - idx) // slab_factor)
            for i in range(len(slabs_prev_load)):
                slab_prev_load = slabs_prev_load[i]
                if slab_id == slab_prev_load:
                    idx += offsets[i]
            prog_inds[k] = idx
        # find halo indices in previous snapshot
        halo_inds = halo_ind_prev[prog_inds]

        # test output; remove in final version
        #if num > 1: print(halo_inds, halo_ind_prev[main_progs[j]])

        # mark ineligible
        eligibility_prev[halo_inds] = False
    return eligibility_prev

def mark_ineligible_extrap(nums, starts, main_progs, progs, halo_ind_prev, eligibility_prev, eligibility_extrap_prev, offsets, slabs_prev_load):
    # constants used for unpacking
    id_factor = int(1e12)
    slab_factor = int(1e9)

    # number of objects marked belonging to this redshift catalog
    N_this_star_lc = len(nums)
    # loop around halos that were marked belonging to this redshift catalog
    for j in range(N_this_star_lc):
        # select all progenitors
        start = starts[j]
        num = nums[j]
        prog_inds = progs[start : start + num]

        # remove progenitors with no info
        prog_inds = prog_inds[prog_inds > 0]
        if len(prog_inds) == 0: continue

        # correct halo indices
        for k in range(len(prog_inds)):
            prog_ind = prog_inds[k]
            idx = (prog_ind % slab_factor)
            slab_id = ((prog_ind % id_factor - idx) // slab_factor)
            for i in range(len(slabs_prev_load)):
                slab_prev_load = slabs_prev_load[i]
                if slab_id == slab_prev_load:
                    idx += offsets[i]
            prog_inds[k] = idx
        # find halo indices in previous snapshot
        halo_inds = halo_ind_prev[prog_inds]

        # mark ineligible
        eligibility_prev[halo_inds] = False
        eligibility_extrap_prev[halo_inds] = False
    return eligibility_prev, eligibility_extrap_prev


def simple_load(filenames, fields):
    if type(filenames) not in (list, tuple):
        filenames = [filenames]
    
    do_prog = 'Progenitors' in fields
    
    Ntot = 0
    dtypes = {}
    
    if do_prog:
        N_prog_tot = 0
        fields.remove('Progenitors')  # treat specially
    
    for fn in filenames:
        with asdf.open(fn) as af:
            # Peek at the first field to get the total length
            # If the lengths of fields don't match up, that will be an error later
            Ntot += len(af['data'][fields[0]])
            
            for field in fields:
                if field not in dtypes:
                    if 'Position' == field:
                        dtypes[field] = (af['data'][field].dtype,3)
                    else:
                        dtypes[field] = af['data'][field].dtype
            
            if do_prog:
                N_prog_tot += len(af['data']['Progenitors'])
                
    # Make the empty tables
    t = Table({f:np.empty(Ntot, dtype=dtypes[f]) for f in fields}, copy=False)
    if do_prog:
        p = Table({'Progenitors':np.empty(N_prog_tot, dtype=np.int64)}, copy=False)
    
    # Fill the data into the empty tables
    j = 0
    jp = 0
    for i, fn in enumerate(filenames):
        print(f"Assocations file number {i+1:d} of {len(filenames)}")
        with asdf.open(fn, lazy_load=True, memmap=False) as f:
            fdata = f['data']
            thisN = len(fdata[fields[0]])
            
            for field in fields:
                # Insert the data into the next slot in the table
                t[field][j:j+thisN] = fdata[field]
                
            if do_prog:
                thisNp = len(fdata['Progenitors'])
                p['Progenitors'][jp:jp+thisNp] = fdata['Progenitors']
                jp += thisNp
            
        j += thisN
    
    # Should have filled the whole table!
    assert j == Ntot
    
    ret = dict(merger=t)
    if do_prog:
        ret['progenitors'] = p
        assert jp == N_prog_tot
        fields.append('Progenitors')
        
    return ret

def simple_load_old(filenames, fields):
    
    for i in range(len(filenames)):
        fn = filenames[i]
        print("File number %i of %i" % (i, len(filenames) - 1))
        f = asdf.open(fn, lazy_load=True, copy_arrays=True)
        fdata = f['data']

        N_halos = fdata[fields[0]].shape[0]
        num_progs_cum = 0
        if i == 0:

            dtypes = []
            for field in fields:
                dtype = fdata[field].dtype
                try:
                    shape = fdata[field].shape[1]
                    dtype = (dtype, shape)
                except:
                    pass
                if field != "Progenitors":
                    dtypes.append((field, dtype))

                if field == "Progenitors":
                    final_progs = fdata["Progenitors"]
                    try:
                        num_progs = fdata["NumProgenitors"] + num_progs_cum
                    except:
                        print(
                            "You need to also request 'NumProgenitors' if requesting the 'Progenitors' field"
                        )
                        exit()
                    num_progs_cum += np.sum(num_progs)

            final = np.empty(N_halos, dtype=dtypes)
            for field in fields:
                if "Progenitors" != field:
                    final[field] = fdata[field]

            if "Progenitors" in fields:
                final["NumProgenitors"] = num_progs

        else:
            new = np.empty(N_halos, dtype=dtypes)
            for field in fields:

                if field != "Progenitors":
                    new[field] = fdata[field]

                if field == "Progenitors":
                    progs = fdata["Progenitors"]
                    final_progs = np.hstack((final_progs, progs))

                    num_progs = fdata["NumProgenitors"] + num_progs_cum
                    num_progs_cum += np.sum(num_progs)

            if "Progenitors" in fields:
                new["NumProgenitors"] = num_progs

            final = np.hstack((final, new))
        del fdata
        f.close()

    #cols = {col:np.empty(N_halos, dtype=user_dt[col]) for col in fields
    final = Table(final, copy=False)
    if "Progenitors" in fields:
        return final, final_progs

    return final


def get_halos_per_slab(filenames):
    # extract all slabs
    slabs = np.array([extract_superslab(fn) for fn in filenames])
    n_slabs = len(slabs)
    N_halos_slabs = np.zeros(n_slabs, dtype=int)

    # extract number of halos in each slab
    for i in range(len(filenames)):
        fn = filenames[i]
        print(f"Halo info file number {i + 1} of {len(filenames)}")
        with asdf.open(fn) as f:
            N_halos = f["data"]["HaloIndex"].shape[0]
        N_halos_slabs[i] = N_halos
        
    # sort in slab order
    i_sort = np.argsort(slabs)
    slabs = slabs[i_sort]
    N_halos_slabs = N_halos_slabs[i_sort]

    return N_halos_slabs, slabs

def extract_origin(fn):
    '''
    Extract index of the light cone origin
    example: 'Merger_lc1.02.asdf' should return 1
    '''
    origin = int(fn.split('lc')[-1].split('.')[0])
    return origin

def get_halos_per_slab_origin(filenames, minified):
    # number of halos in each file
    N_halos_slabs_origins = np.zeros(len(filenames), dtype=int)

    # extract all slabs
    slabs = np.array([extract_superslab(fn) for fn in filenames])

    # extract all origins
    origins = np.array([extract_origin(fn) for fn in filenames])
    
    # extract number of halos in each slab
    for i in range(len(filenames)):
        fn = filenames[i]
        print("File number %i of %i" % (i, len(filenames) - 1))
        f = asdf.open(fn)
        N_halos = f["data"]["HaloIndex"].shape[0]
        N_halos_slabs_origins[i] = N_halos
        f.close()
        
    # sort in slab order
    i_sort = np.argsort(slabs)
    slabs = slabs[i_sort]
    origins = origins[i_sort]
    N_halos_slabs_origins = N_halos_slabs_origins[i_sort]
    
    return N_halos_slabs_origins, slabs, origins
