import os
from pathlib import Path

import asdf
import numpy as np
from abacusnbody.data.bitpacked import unpack_pids, unpack_rvint
from numba import jit

from tools.compute_dist import dist

COMPRESSION_KWARGS = dict(typesize='auto', shuffle='bitshuffle', compression_block_size=12*1024**2, blosc_block_size=3*1024**2, nthreads=4)

# og
pos_key = 'pos_interp'
# TESTING
#pos_key = 'x_L2com'

def load_lc_pid_rv(lc_pid_fn, lc_rv_fn, Lbox, PPD):
    # load and unpack pids
    lc_pids = asdf.open(lc_pid_fn, lazy_load=True, memmap=False)
    lc_pid = lc_pids['data']['packedpid'][:]
    lc_pid = unpack_pids(lc_pid, box=Lbox, ppd=PPD, pid=True)['pid']
    lc_pids.close()

    # load positions and velocities
    lc_rvs = asdf.open(lc_rv_fn, lazy_load=True, memmap=False)
    lc_rv = lc_rvs['data']['rvint'][:]
    lc_rvs.close()
    return lc_pid, lc_rv

def load_mt_pid(mt_fn,Lbox,PPD):
    # load mtree catalog
    print("load mtree file = ", mt_fn)
    mt_pids = asdf.open(mt_fn, lazy_load=True, memmap=False)
    mt_pid = mt_pids['data']['pid'][:]
    mt_pid = unpack_pids(mt_pid, box=Lbox, ppd=PPD, pid=True)['pid']
    header = mt_pids['header']
    mt_pids.close()

    return mt_pid, header

def load_mt_pid_pos_vel(mt_fn,Lbox,PPD):
    # load mtree catalog
    print("load mtree file = ", mt_fn)
    mt_pids = asdf.open(mt_fn, lazy_load=True, memmap=False)
    mt_pid = mt_pids['data']['pid'][:]
    mt_pid = unpack_pids(mt_pid, box=Lbox, ppd=PPD, pid=True)['pid']
    mt_pos = mt_pids['data']['pos'][:]
    mt_vel = mt_pids['data']['vel'][:]
    header = mt_pids['header']
    mt_pids.close()

    return mt_pid, mt_pos, mt_vel, header

def load_mt_npout(halo_mt_fn):
    # load mtree catalog
    print("load halo mtree file = ",halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    mt_npout = f['data']['npoutA'][:]
    f.close()
    return mt_npout

def load_mt_npout_B(halo_mt_fn):
    # load mtree catalog
    print("load halo mtree file = ", halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    mt_npout = f['data']['npoutB'][:]
    f.close()
    return mt_npout

def load_mt_origin(halo_mt_fn):
    # load mtree catalog
    print("load halo mtree file = ", halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    mt_origin = (f['data']['origin'][:])%3
    f.close()
    return mt_origin

def load_mt_pos(halo_mt_fn):
    # load mtree catalog
    print("load halo mtree file = ", halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    mt_pos = f['data'][pos_key][:]
    f.close()
    return mt_pos

def load_mt_pos_yz(halo_mt_fn):
    # load mtree catalog
    print("load halo mtree file = ", halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    mt_pos_yz = f['data'][pos_key][:, 1:].astype(np.float16)
    f.close()
    return mt_pos_yz

def load_mt_cond_edge(halo_mt_fn, Lbox):
    # load mtree catalog
    print("load halo mtree file = ", halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    mt_pos_yz = f['data'][pos_key][:, 1:]
    mt_cond_edge = np.zeros(mt_pos_yz.shape[0], dtype=np.int8)
    mt_cond_edge[mt_pos_yz[:, 1] < Lbox/2.+10.] += 1
    mt_cond_edge[mt_pos_yz[:, 0] < Lbox/2.+10.] += 2
    mt_cond_edge[mt_pos_yz[:, 1] > Lbox/2.-10.] += 4
    mt_cond_edge[mt_pos_yz[:, 0] > Lbox/2.-10.] += 8
    f.close()
    return mt_cond_edge

def load_mt_origin_edge(halo_mt_fn, Lbox):
    # load mtree catalog
    print("load halo mtree file = ", halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    
    mt_pos_yz = f['data'][pos_key][:, 1:]
    mt_cond_edge = np.zeros(mt_pos_yz.shape[0], dtype=np.int8)
    mt_cond_edge[mt_pos_yz[:, 1] < Lbox/2.+10.] += 1
    mt_cond_edge[mt_pos_yz[:, 0] < Lbox/2.+10.] += 2
    mt_cond_edge[mt_pos_yz[:, 1] > Lbox/2.-10.] += 4
    mt_cond_edge[mt_pos_yz[:, 0] > Lbox/2.-10.] += 8
    del mt_pos_yz
    mt_cond_edge += 2**((f['data']['origin'][:])%3 + 4)
    f.close()
    return mt_cond_edge

def load_mt_dist(halo_mt_fn, origin):
    # load mtree catalog
    print("load halo mtree file = ", halo_mt_fn)
    f = asdf.open(halo_mt_fn, lazy_load=True, memmap=False)
    mt_pos = f['data'][pos_key]
    f.close()
    mt_dist = dist(mt_pos, origin)
    return mt_dist

@jit(nopython = True)
def reindex_pid(pid, npstart, npout):
    npstart_new = np.zeros(len(npout),dtype=np.int64)
    npstart_new[1:] = np.cumsum(npout)[:-1]
    npout_new = npout
    pid_new = np.zeros(np.sum(npout_new), dtype=pid.dtype)

    for j in range(len(npstart)):
        pid_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pid[npstart[j]:npstart[j]+npout[j]]

    return pid_new, npstart_new, npout_new

@jit(nopython = True)
def reindex_pid_AB(pid, npstartA, npoutA, npstartB, npoutB):
    # offsets for subsample A and B
    npstart_newAB = np.zeros(len(npoutA), dtype=np.int64)
    npstart_newAB[1:] = np.cumsum(npoutA+npoutB)[:-1]

    # those two are unchanged
    npout_newA = npoutA
    npout_newB = npoutB

    # create new array for the pid's containing A and B
    pid_new = np.zeros(np.sum(npout_newA+npout_newB), dtype=pid.dtype)

    # fill the pid array with corresponding values
    for j in range(len(npoutA)):
        st = npstart_newAB[j]
        pid_new[st:st+npout_newA[j]] = pid[npstartA[j]:npstartA[j]+npoutA[j]]
        st += npout_newA[j]
        pid_new[st:st+npout_newB[j]] = pid[npstartB[j]:npstartB[j]+npoutB[j]]
    return pid_new, npstart_newAB, npout_newA, npout_newB

@jit(nopython = True)
def reindex_pid_pos(pid,pos,npstart,npout):
    npstart_new = np.zeros(len(npout),dtype=np.int64)
    npstart_new[1:] = np.cumsum(npout)[:-1]
    npout_new = npout
    pid_new = np.zeros(np.sum(npout_new),dtype=pid.dtype)
    pos_new = np.zeros((np.sum(npout_new),3),dtype=pos.dtype)
    
    for j in range(len(npstart)):
        pid_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pid[npstart[j]:npstart[j]+npout[j]]
        pos_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pos[npstart[j]:npstart[j]+npout[j]]
            
    return pid_new, pos_new, npstart_new, npout_new

@jit(nopython = True)
def reindex_pid_pos_vel(pid,pos,vel,npstart,npout):
    npstart_new = np.zeros(len(npout),dtype=np.int64)
    npstart_new[1:] = np.cumsum(npout)[:-1]
    npout_new = npout
    pid_new = np.zeros(np.sum(npout_new),dtype=pid.dtype)
    pos_new = np.zeros((np.sum(npout_new),3),dtype=pos.dtype)
    vel_new = np.zeros((np.sum(npout_new),3),dtype=vel.dtype)
    
    for j in range(len(npstart)):
        pid_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pid[npstart[j]:npstart[j]+npout[j]]
        pos_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = pos[npstart[j]:npstart[j]+npout[j]]
        vel_new[npstart_new[j]:npstart_new[j]+npout_new[j]] = vel[npstart[j]:npstart[j]+npout[j]]
            
    return pid_new, pos_new, vel_new, npstart_new, npout_new

@jit(nopython = True)
def reindex_pid_pos_vel_AB(pid, pos, vel, npstartA, npoutA, npstartB, npoutB):
    # offsets for subsample A and B
    npstart_newAB = np.zeros(len(npoutA), dtype=np.int64)
    npstart_newAB[1:] = np.cumsum(npoutA+npoutB)[:-1]

    # those two are unchanged
    npout_newA = npoutA
    npout_newB = npoutB

    # create new array for the pid's containing A and B
    pid_new = np.zeros(np.sum(npout_newA+npout_newB), dtype=pid.dtype)
    pos_new = np.zeros(np.sum(npout_newA+npout_newB), dtype=pos.dtype)
    vel_new = np.zeros(np.sum(npout_newA+npout_newB), dtype=vel.dtype)

    # fill the pid array with corresponding values
    for j in range(len(npoutA)):
        st = npstart_newAB[j]
        pid_new[st:st+npout_newA[j]] = pid[npstartA[j]:npstartA[j]+npoutA[j]]
        pos_new[st:st+npout_newA[j]] = pos[npstartA[j]:npstartA[j]+npoutA[j]]
        vel_new[st:st+npout_newA[j]] = vel[npstartA[j]:npstartA[j]+npoutA[j]]
        st += npout_newA[j]
        pid_new[st:st+npout_newB[j]] = pid[npstartB[j]:npstartB[j]+npoutB[j]]
        pos_new[st:st+npout_newB[j]] = pos[npstartB[j]:npstartB[j]+npoutB[j]]
        vel_new[st:st+npout_newB[j]] = vel[npstartB[j]:npstartB[j]+npoutB[j]]

    return pid_new, pos_new, vel_new, npstart_newAB, npout_newA, npout_newB

# save light cone catalog
def save_asdf(table, filename, compress=False):
    filename = Path(filename)
    cols_as_arrays = {name: np.asarray(table[name]) for name in table.colnames}
    header = dict(table.meta)
    tree = {'data': cols_as_arrays, 'header': header}

    filename.parent.mkdir(parents=True, exist_ok=True)

    compression = 'blsc' if compress else None
    with asdf.AsdfFile(tree) as af:
        af.write_to(filename, all_array_compression=compression, compression_kwargs=COMPRESSION_KWARGS)
