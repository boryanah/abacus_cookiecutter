#!/usr/bin/env python3
"""
This is the first script in the "lightcone halo" pipeline.  The goal of this script is to use merger
tree information to flag halos that intersect the lightcone and make a unique determination of which
halo catalog epoch from which to draw the halo.

Prerequisites:
`all_headers/` directory containing copies of the headers of all light cone particle files (~1100 per
simulation). If missing can be generated with `tools/script_headers.sh`.

Usage
-----
$ ./build_mt.py --help
"""

import argparse
import gc
from pathlib import Path

import asdf
import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from astropy.table import Table
from scipy.interpolate import interp1d

from tools.aid_asdf import save_asdf
from tools.compute_dist import dist, wrapping
from tools.merger import (
    get_halos_per_slab,
    get_one_header,
    get_zs_from_headers,
    mark_ineligible,
    pack_inds,
    reorder_by_slab,
    simple_load,
    unpack_inds,
)

CONSTANTS = {'c': 299792.458}  # km/s, speed of light


def correct_inds(halo_ids, N_halos_slabs, slabs, inds_fn):
    """
    Reorder indices for given halo index array with
    corresponding n halos and slabs for its time epoch
    """

    # number of halos in the loaded superslabs
    N_halos_load = np.array([N_halos_slabs[i] for i in inds_fn])

    onefile = len(inds_fn) == 1
    # unpack slab and index for each halo
    slab_ids, ids = unpack_inds(halo_ids, unpack_slab_ids=not onefile)

    # total number of halos in the slabs that we have loaded
    # N_halos = np.sum(N_halos_load)
    offsets = np.zeros(len(inds_fn), dtype=int)
    offsets[1:] = np.cumsum(N_halos_load)[:-1]

    # determine if unpacking halos for only one file (Merger_this['HaloIndex']) -- no need to offset
    if onefile:
        return ids

    # select the halos belonging to given slab
    for i, ind_fn in enumerate(inds_fn):
        select = np.where(slab_ids == slabs[ind_fn])[0]
        ids[select] += offsets[i]

    return ids


def get_mt_info(fn_load, fn_halo_load, fields):
    """
    Load merger tree and progenitors information
    """

    # loading merger tree info
    mt_data = simple_load(fn_load, fields=fields)

    # turn data into astropy table
    Merger = mt_data['merger']
    Merger.add_column(
        np.empty(len(Merger['HaloIndex']), dtype=np.float32),
        copy=False,
        name='ComovingDistance',
    )

    # if loading all progenitors
    if 'Progenitors' in fields:
        num_progs = Merger['NumProgenitors']
        # get an array with the starting indices of the progenitors array
        start_progs = np.empty(len(num_progs), dtype=int)
        start_progs[0] = 0
        start_progs[1:] = num_progs.cumsum()[:-1]
        Merger.add_column(start_progs, name='StartProgenitors', copy=False)

    # add cleaned masses and halo velocities
    halos = CompaSOHaloCatalog(fn_halo_load, subsamples=False, fields=['N', 'v_L2com']).halos
    Merger.add_column(halos['N'], copy=False, name='N')
    Merger.add_column(halos['v_L2com'], copy=False, name='v_L2com')

    return mt_data


def solve_crossing(r1, r2, pos1, pos2, chi1, chi2, vel1, vel2, m1, m2, Lbox, origin, chs, complete=False, extra=4.0):
    """
    Solve when the crossing of the light cones occurs and the
    interpolated position and velocity. Merger trees loook for progenitors in a 4 Mpc/h radius
    """

    # periodic wrapping of the positions of the particles
    r1, r2, pos1, pos2 = wrapping(r1, r2, pos1, pos2, chs[1], chs[0], Lbox, origin, extra)
    assert np.all(((r2 <= chs[1]) & (r2 > chs[0])) | ((r1 <= chs[1]) & (r1 > chs[0]))), "Wrapping didn't work"

    # in a very very very very small number of cases (i.e. z = 0.8, corner halos), the current halo position
    # and the main progenitor would both be within chi1 and chi2, but will be on opposite ends. In that case,
    # we will just move things to the side of whoever's closer to the interpolated position (or just pick one)
    assert np.sum(np.abs(pos2 - pos1) > extra) == 0, 'There are halos on opposite ends after wrapping'

    # solve for chi_star, where chi(z) = eta(z=0)-eta(z)
    # equation is r1+(chi1-chi)/(chi1-chi2)*(r2-r1) = chi, with solution:
    chi_star = (r1 * (chi1 - chi2) + chi1 * (r2 - r1)) / ((chi1 - chi2) + (r2 - r1))

    # get interpolated positions of the halos
    v_avg = (pos2 - pos1) / (chi1 - chi2)
    pos_star = pos1 + v_avg * (chi1 - chi_star[:, None])

    # get interpolated masses of the halos
    # they're coming in as uint32, so avoid overflow by casting to float
    m_dot = (m2.astype(np.float64) - m1.astype(np.float64)) / (chi1 - chi2)
    m_star = m1 + m_dot * (chi1 - chi_star)

    # float32 precision is probably good enough,
    # and we'll round to make sure the floats compress well.
    # These could be ints, but float helps remind people that there was a lossy step.
    m_star = m_star.astype(np.float32).round()

    # mask halos with zero current or previous mass
    mask = np.isclose(m1, 0.0) | np.isclose(m2, 0.0)
    m_star[mask] = 0.0

    # enforce boundary conditions by periodic wrapping
    # pos_star[pos_star >= Lbox/2.] = pos_star[pos_star >= Lbox/2.] - Lbox
    # pos_star[pos_star < -Lbox/2.] = pos_star[pos_star < -Lbox/2.] + Lbox

    # interpolated velocity [km/s]
    a_avg = (vel2 - vel1) / (chi1 - chi2)
    vel_star = vel1 + a_avg * (chi1 - chi_star[:, None])

    # x is comoving position; r = x a; dr = a dx; r = a x; dr = da x + dx a; a/H
    # vel_star = dx/deta = dr/dt − H(t)r -> r is real space coord dr/dt = vel_star + a H(t) x
    # 'Htime', 'HubbleNow', 'HubbleTimeGyr', 'HubbleTimeHGyr'

    # mark True if closer to chi2 (this snapshot)
    bool_star = np.abs(chi1 - chi_star) > np.abs(chi2 - chi_star)

    # condition to check whether halo in this light cone band
    assert np.all((chi_star <= chs[1] + extra) & (chi_star > chs[0] - extra)), 'Solution is out of bounds'

    return chi_star, pos_star, vel_star, m_star, bool_star


def offset_pos(pos, ind_origin, all_origins):
    """
    Offset the interpolated positions to create continuous light cones
    """

    # location of initial observer
    first_observer = all_origins[0]
    current_observer = all_origins[ind_origin]
    offset = first_observer - current_observer
    pos += offset
    return pos


def main(
    sim_path,
    superslab_start=0,
    output_parent=None,
    merger_dir=None,
    tmpdir=None,
    z_start=None,
    z_stop=None,
    resume=False,
    plot=False,
    complete=False,
):
    """
    Main function.
    The algorithm: for each merger tree epoch, for
    each superslab, for each light cone origin,
    compute the intersection of the light cone with
    each halo, using the interpolated position
    to the previous merger epoch (and possibly a
    velocity correction).  If the intersection is
    between the current and previous merger epochs,
    then record the closer one as that halo's
    epoch and mark its progenitors as ineligible.
    Will need one padding superslab in the previous
    merger epoch.  Can process in a rolling fashion.
    """

    # turn directories into Paths
    sim_path = Path(sim_path)
    sim_name = sim_path.name

    if output_parent is None:
        output_parent = sim_path.parent / 'halo_light_cones'
    output_parent = Path(output_parent)

    if merger_dir is None:
        merger_dir = sim_path.parent / 'merger'
    merger_dir = Path(merger_dir)

    if tmpdir is None:
        tmpdir = output_parent / sim_name / 'tmp'
    tmpdir = Path(tmpdir) / sim_name

    header = get_one_header(merger_dir)

    # simulation parameters
    Lbox = header['BoxSize']
    # location of the LC origins in Mpc/h
    # FUTURE: each LCOrigin is repeated LCBoxRepeats times
    # origins = np.array(header['LightConeOrigins']).reshape(-1,3)
    assert len(np.array(header['LCOrigins']).reshape(-1, 3)) == 1
    rpd = 2 * header.get('LCBoxRepeats', 1) + 1

    # The rpd^3 box origins, in "Fourier" order, i.e.:
    # [0, 1, ..., N//2, -N//2, ..., -1] (in each dimension)
    origins = np.mgrid[:rpd, :rpd, :rpd].reshape(3, -1).T
    origins[origins > rpd // 2] -= rpd
    origins = origins * Lbox

    # directory where we save the final outputs
    cat_lc_dir = output_parent / sim_name
    cat_lc_dir.mkdir(exist_ok=True, parents=True)

    # directory where we save the current state if we want to resume
    tmpdir.mkdir(exist_ok=True, parents=True)
    print(f'Starting light cone catalog construction in simulation {sim_name:s}')

    # all redshifts, steps and comoving distances of light cones files; high z to low z
    states = Table.read(sim_path / 'state_log.asdf')
    zs_all = states['Redshift']
    chis_all = states['CoordinateDistanceHMpc']

    # get functions relating chi and z
    chi_of_z = interp1d(zs_all, chis_all)

    # if merger tree redshift information has been saved, load it (if not, save it)
    data_mt_fn = cat_lc_dir / 'zs_mt.asdf'
    if not data_mt_fn.exists():
        # all merger tree snapshots and corresponding redshifts
        mt_table = Table({'mt0_fns': list(merger_dir.glob('associations_z*.0.asdf'))})
        mt_table['zs_mt'] = get_zs_from_headers(mt_table['mt0_fns'])
        mt_table.sort('zs_mt')
        # "associations_z0.000", etc; low-z to high-z
        mt_table['mt_fn_stems'] = ['.'.join(fn.name.split('.')[:2]) for fn in mt_table['mt0_fns']]
        del mt_table['mt0_fns']

        data_mt_fn.parent.mkdir(exist_ok=True, parents=True)
        mt_table.write(data_mt_fn)
    else:
        mt_table = Table.read(data_mt_fn)

    # number of superslabs
    n_superslabs = len(list(merger_dir.glob(mt_table['mt_fn_stems'][0] + '.*.asdf')))
    print('number of superslabs = ', n_superslabs)

    zs_mt = mt_table['zs_mt']

    # starting and finishing redshift indices
    ind_start = np.argmin(np.abs(zs_mt - z_start)) if z_start is not None else 0
    ind_stop = np.argmin(np.abs(zs_mt - z_stop)) if z_stop is not None else len(mt_table)

    # initialize difference between the conformal time of the previous two catalogs
    delta_chi_old = 0.0

    build_log = tmpdir / 'build_log.asdf'
    if resume:
        # if user wants to resume from previous state, create padded array for marking whether superslab has been loaded
        resume_flags = np.ones((n_superslabs, origins.shape[1]), dtype=bool)

        # previous redshift, distance between shells
        build_state = load_build_state(build_log)
        z_this_tmp = build_state['z_prev']
        delta_chi_old = build_state['delta_chi']
        superslab = build_state['super_slab']

        assert np.abs(zs_mt[ind_start] - z_this_tmp) < 1.0e-6, (
            f"Your recorded state is not for the currently requested redshift, can't resume from old. Last recorded state is z = {z_this_tmp:.3f}"
        )
        assert np.abs((superslab_start - 1) % n_superslabs - superslab) < 1.0e-6, (
            f"Your recorded state is not for the currently requested superslab, can't resume from old. Last recorded state is superslab = {superslab:d}"
        )
        print(f'Resuming from redshift z = {z_this_tmp:4.3f}')
    else:
        # delete the exisiting temporary files
        for fn in tmpdir.glob('*'):
            fn.unlink()
        resume_flags = np.zeros((n_superslabs, origins.shape[0]), dtype=bool)

    # fields to extract from the merger trees
    fields_mt = [
        'HaloIndex',
        'Position',
        'MainProgenitor',
        'Progenitors',
        'NumProgenitors',
    ]
    # lighter version tuks could we add N here from the cleaned catalogs
    # fields_mt = ['HaloIndex', 'Position', 'MainProgenitor']

    for i in range(ind_start, ind_stop + 1):
        # this snapshot redshift and the previous
        z_this = zs_mt[i]
        z_prev = zs_mt[i + 1]
        # z_pprev = zs_mt[i + 2] # not currently used
        print('redshift of this and the previous snapshot = ', z_this, z_prev)

        # how to name folder
        zname_this = min(header['L1OutputRedshifts'], key=lambda z: abs(z - z_this))
        zname_prev = min(header['L1OutputRedshifts'], key=lambda z: abs(z - z_prev))

        # check that you are starting at a reasonable redshift
        assert z_this >= np.min(zs_all), 'You need to set starting redshift to the smallest value of the merger tree'

        # coordinate distance of the light cone at this redshift and the previous
        chi_this = chi_of_z(z_this)
        chi_prev = chi_of_z(z_prev)
        # chi_pprev = chi_of_z(z_pprev) # not currently used
        delta_chi = chi_prev - chi_this
        # delta_chi_new = chi_pprev - chi_prev # not currently used
        print('comoving distance between this and previous snapshot = ', delta_chi)

        # read merger trees file names at this and previous snapshot
        fns_this = list(merger_dir.glob(mt_table['mt_fn_stems'][i] + '.*.asdf'))
        fns_prev = list(merger_dir.glob(mt_table['mt_fn_stems'][i + 1] + '.*.asdf'))

        # number of merger tree files
        print('number of files = ', len(fns_this), len(fns_prev))
        assert n_superslabs == len(fns_this) and n_superslabs == len(fns_prev), 'Incomplete merger tree files'
        # reorder file names by super slab number
        fns_this = reorder_by_slab(fns_this)
        fns_prev = reorder_by_slab(fns_prev)

        # halo info files (used for masses and eligibility)
        fns_halo_this = [
            (sim_path / 'halos' / f'z{zname_this:4.3f}' / 'halo_info' / f'halo_info_{counter:03d}.asdf')
            for counter in range(n_superslabs)
        ]
        fns_halo_prev = [
            (sim_path / 'halos' / f'z{zname_prev:4.3f}' / 'halo_info' / f'halo_info_{counter:03d}.asdf')
            for counter in range(n_superslabs)
        ]

        # get number of halos in each slab and number of slabs
        N_halos_slabs_this, slabs_this = get_halos_per_slab(fns_this)
        N_halos_slabs_prev, slabs_prev = get_halos_per_slab(fns_prev)

        # We're going to be loading slabs in a rolling fashion:
        # reading the "high" slab at the leading edge, discarding the trailing "low" slab
        # and moving the mid to low. But first we need to read all three to prime the queue
        mt_prev = {}  # indexed by slab num
        mt_prev[(superslab_start - 1) % n_superslabs] = get_mt_info(
            fns_prev[(superslab_start - 1) % n_superslabs],
            fns_halo_prev[(superslab_start - 1) % n_superslabs],
            fields=fields_mt,
        )
        if superslab_start not in mt_prev:
            # In the case of 1 superslab, this may be the same as what we just loaded
            mt_prev[superslab_start] = get_mt_info(
                fns_prev[superslab_start], fns_halo_prev[superslab_start], fields=fields_mt
            )

        # loop over each superslab
        for k in range(superslab_start, n_superslabs):
            # starting and finishing superslab superslabs
            klow = (k - 1) % n_superslabs
            khigh = (k + 1) % n_superslabs

            # slide down by one
            klowm1 = (klow - 1) % n_superslabs
            if klowm1 in mt_prev and klowm1 not in (klow, k, khigh):
                del mt_prev[klowm1]
            if khigh not in mt_prev:
                mt_prev[khigh] = get_mt_info(fns_prev[khigh], fns_halo_prev[khigh], fields_mt)

            # starting and finishing superslab superslabs
            inds_fn_this = [k]
            if n_superslabs == 1:
                inds_fn_prev = np.array([k], dtype=int)
            else:  # there is also the case of only 2 superslabs
                inds_fn_prev = np.array([klow, k, khigh], dtype=int)
            print(
                'superslabs loaded in this and previous redshifts = ',
                inds_fn_this,
                inds_fn_prev,
            )

            # get merger tree data for this snapshot and for the previous one
            mt_data_this = get_mt_info(fns_this[k], fns_halo_this[k], fields_mt)

            # number of halos in this step and previous step; this depends on the number of files requested
            N_halos_this = np.sum(N_halos_slabs_this[inds_fn_this])
            N_halos_prev = np.sum(N_halos_slabs_prev[inds_fn_prev])
            print('N_halos_this = ', N_halos_this)
            print('N_halos_prev = ', N_halos_prev)

            # organize data from this redshift into astropy tables
            Merger_this = mt_data_this['merger']
            cols = {
                col: np.empty((N_halos_prev,) + Merger_this[col].shape[1:], dtype=Merger_this[col].dtype)
                for col in Merger_this.colnames
            }
            Merger_prev = Table(cols, copy=False)

            # make a boolean array with eligible halos
            clean_halos_this = Merger_this['N'] > 0

            # organize data from prev redshift into astropy tables
            offset = 0
            for ss in mt_prev:
                size_superslab = len(mt_prev[ss]['merger']['HaloIndex'])
                Merger_prev[offset : offset + size_superslab] = mt_prev[ss]['merger'][:]
                offset += size_superslab

            # mask where no merger tree info is available (because we don'to need to solve for eta star for those)
            noinfo_this = Merger_this['MainProgenitor'] <= 0
            info_this = Merger_this['MainProgenitor'] > 0

            # print percentage where no information is available or halo not eligible
            print('percentage no info = ', np.sum(noinfo_this) / len(noinfo_this) * 100.0)

            # no info is denoted by 0 or -999 (or regular if ineligible), but -999 messes with unpacking, so we set it to 0
            Merger_this['MainProgenitor'][noinfo_this] = 0

            # rework the main progenitor and halo indices to return in proper order
            Merger_this['HaloIndex'] = correct_inds(
                Merger_this['HaloIndex'],
                N_halos_slabs_this,
                slabs_this,
                inds_fn_this,
            )
            Merger_this['MainProgenitor'] = correct_inds(
                Merger_this['MainProgenitor'],
                N_halos_slabs_prev,
                slabs_prev,
                inds_fn_prev,
            )
            Merger_prev['HaloIndex'] = correct_inds(
                Merger_prev['HaloIndex'],
                N_halos_slabs_prev,
                slabs_prev,
                inds_fn_prev,
            )

            # loop over all origins
            for o in range(len(origins)):
                # location of the observer
                origin = origins[o]

                # https://math.stackexchange.com/questions/2133217/minimal-distance-to-a-cube-in-2d-and-3d-from-a-point-lying-outside
                d_min = np.sqrt(
                    np.max([0.0, np.abs(origin[0]) - Lbox / 2.0]) ** 2.0
                    + np.max([0.0, np.abs(origin[1]) - Lbox / 2.0]) ** 2.0
                    + np.max([0.0, np.abs(origin[2]) - Lbox / 2.0]) ** 2.0
                )
                d_max = (
                    d_min + np.sqrt(3.0) * Lbox
                )  # the max possible distance (larger than needed, but haven't done the proper calculation)

                # if interpolating to z = 0.1 (kinda ugly way to do this)
                if complete and np.abs(z_this - 0.1) < 1.0e-3:  # tuks
                    print(f'extending {z_this:4.3f} all the way to z = 0')
                    chi_low = 0.0
                else:
                    chi_low = chi_this

                # condition for selecting halos with merger tree information
                # chs = np.array([chi_low, chi_prev], dtype=np.float32) # doesn't take past halos
                chs = np.array(
                    [chi_low - delta_chi_old / 2.0, chi_prev], dtype=np.float32
                )  # taking some additional objects from before
                # chs = np.array([chi_low - delta_chi_old / 2.0, chi_prev + delta_chi_new / 2.0], dtype=np.float32) # TESTING
                if d_min > chs[1] or d_max < chs[0]:
                    continue

                # comoving distance to observer
                Merger_this['ComovingDistance'][:] = dist(Merger_this['Position'], origin)
                Merger_prev['ComovingDistance'][:] = dist(Merger_prev['Position'], origin)

                # merger tree data of main progenitor halos corresponding to the halos in current snapshot
                Merger_prev_main_this = Merger_prev[Merger_this['MainProgenitor']]

                # if eligible, can be selected for light cone redshift catalog
                if (i != ind_start) or resume_flags[k, o]:
                    # dealing with the fact that these files may not exist for all origins and all superslabs
                    if (tmpdir / ('eligibility_prev_z%4.3f_lc%d.%02d.npy' % (z_this, o, k))).exists():
                        eligibility_this = np.load(tmpdir / ('eligibility_prev_z%4.3f_lc%d.%02d.npy' % (z_this, o, k)))
                        eligibility_extrap_this = np.load(
                            tmpdir / ('eligibility_extrap_prev_z%4.3f_lc%d.%02d.npy' % (z_this, o, k))
                        )
                    else:
                        eligibility_this = np.ones(N_halos_this, dtype=bool) & clean_halos_this
                        eligibility_extrap_this = np.ones(N_halos_this, dtype=bool) & clean_halos_this
                else:
                    eligibility_this = np.ones(N_halos_this, dtype=bool) & clean_halos_this
                    eligibility_extrap_this = np.ones(N_halos_this, dtype=bool) & clean_halos_this

                # for a newly opened redshift, everyone is eligible to be part of the light cone catalog
                eligibility_prev = np.ones(N_halos_prev, dtype=bool)
                eligibility_extrap_prev = np.ones(N_halos_prev, dtype=bool)

                # only halos without merger tree info are allowed to use the extrap quantities; this is relevant if you're doing
                # mask for eligible halos for light cone origin with and without information
                mask_noinfo_this = noinfo_this & eligibility_this & eligibility_extrap_this
                mask_info_this = info_this & eligibility_this

                # halos that have merger tree information
                idx_info_this = mask_info_this.nonzero()
                Merger_this_info = Merger_this[idx_info_this]
                Merger_prev_main_this_info = Merger_prev_main_this[idx_info_this]

                # halos that don't have merger tree information
                idx_noinfo_this = mask_noinfo_this.nonzero()
                Merger_this_noinfo = Merger_this[idx_noinfo_this]

                # select objects that are crossing the light cones
                cond_1 = (Merger_this_info['ComovingDistance'] > chs[0]) & (
                    Merger_this_info['ComovingDistance'] <= chs[1]
                )
                cond_2 = (Merger_prev_main_this_info['ComovingDistance'] > chs[0]) & (
                    Merger_prev_main_this_info['ComovingDistance'] <= chs[1]
                )
                mask_lc_this_info = cond_1 | cond_2
                del cond_1, cond_2

                # for halos that have no merger tree information, we simply take their current position
                # og
                cond_1 = Merger_this_noinfo['ComovingDistance'] > chi_low - delta_chi_old / 2.0
                cond_2 = Merger_this_noinfo['ComovingDistance'] <= chi_low + delta_chi / 2.0

                # TESTING
                # cond_1 = (Merger_this_noinfo['ComovingDistance'] > chi_low)
                # cond_2 = (Merger_this_noinfo['ComovingDistance'] <= chi_low + delta_chi)

                mask_lc_this_noinfo = cond_1 & cond_2
                del cond_1, cond_2

                # spare the computer the effort and avert empty array errors
                # TODO: perhaps revise, as sometimes we might have no halos in
                # noinfo but some in info and vice versa
                if np.sum(mask_lc_this_info) == 0 or np.sum(mask_lc_this_noinfo) == 0:
                    print(
                        'either no halos with no info or no halos with info',
                        np.sum(mask_lc_this_info),
                        np.sum(mask_lc_this_noinfo),
                    )
                    continue

                # percentage of objects that are part of this or previous snapshot
                print(
                    f'percentage of halos in light cone {o:d} with and without progenitor info =',
                    np.sum(mask_lc_this_info) / len(mask_lc_this_info) * 100.0,
                    np.sum(mask_lc_this_noinfo) / len(mask_lc_this_noinfo) * 100.0,
                )

                # select halos with mt info that have had a light cone crossing
                idx_lc_this_info = mask_lc_this_info.nonzero()
                Merger_this_info_lc = Merger_this_info[idx_lc_this_info]
                Merger_prev_main_this_info_lc = Merger_prev_main_this_info[idx_lc_this_info]

                if plot:
                    import matplotlib

                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    x_min = -Lbox / 2.0 + k * (Lbox / n_superslabs)
                    x_max = x_min + (Lbox / n_superslabs)

                    x = Merger_this_info_lc['Position'][:, 0]
                    choice = (x > x_min) & (x < x_max)

                    y = Merger_this_info_lc['Position'][choice, 1]
                    z = Merger_this_info_lc['Position'][choice, 2]

                    plt.figure(1)
                    plt.scatter(y, z, color='dodgerblue', s=0.1, label='current objects')

                    plt.legend()
                    plt.axis('equal')
                    plt.savefig(f'this_{i:d}_{k:d}_{o:d}.png')
                    plt.close()

                    x = Merger_prev_main_this_info_lc['Position'][:, 0]

                    choice = (x > x_min) & (x < x_max)

                    y = Merger_prev_main_this_info_lc['Position'][choice, 1]
                    z = Merger_prev_main_this_info_lc['Position'][choice, 2]

                    plt.figure(2)
                    plt.scatter(y, z, color='orangered', s=0.1, label='main progenitor')

                    plt.legend()
                    plt.axis('equal')
                    plt.savefig(f'prev_{i:d}_{k:d}_{o:d}.png')
                    plt.close()

                # select halos without mt info that have had a light cone crossing
                idx_lc_this_noinfo = mask_lc_this_noinfo.nonzero()
                Merger_this_noinfo_lc = Merger_this_noinfo[idx_lc_this_noinfo]

                # add columns for new interpolated position, velocity and comoving distance tuks could add all in one line; could shorten names
                Merger_this_info_lc.add_column(
                    np.empty(len(Merger_this_info_lc), dtype=(np.float32, 3)),
                    copy=False,
                    name='InterpolatedPosition',
                )
                Merger_this_info_lc.add_column(
                    np.empty(len(Merger_this_info_lc), dtype=(np.float32, 3)),
                    copy=False,
                    name='InterpolatedVelocity',
                )
                Merger_this_info_lc.add_column(
                    np.empty(len(Merger_this_info_lc), dtype=np.float32),
                    copy=False,
                    name='InterpolatedComoving',
                )
                Merger_this_info_lc.add_column(
                    np.empty(len(Merger_this_info_lc), dtype=np.float32),
                    copy=False,
                    name='InterpolatedN',
                )

                # get chi star where lc crosses halo trajectory; bool is False where closer to previous
                (
                    Merger_this_info_lc['InterpolatedComoving'],
                    Merger_this_info_lc['InterpolatedPosition'],
                    Merger_this_info_lc['InterpolatedVelocity'],
                    Merger_this_info_lc['InterpolatedN'],
                    bool_star_this_info_lc,
                ) = solve_crossing(
                    Merger_prev_main_this_info_lc['ComovingDistance'],
                    Merger_this_info_lc['ComovingDistance'],
                    Merger_prev_main_this_info_lc['Position'],
                    Merger_this_info_lc['Position'],
                    chi_prev,
                    chi_this,
                    Merger_prev_main_this_info_lc['v_L2com'],
                    Merger_this_info_lc['v_L2com'],
                    Merger_prev_main_this_info_lc['N'],
                    Merger_this_info_lc['N'],
                    Lbox,
                    origin,
                    chs,
                    complete=(complete and np.abs(z_this - 0.1) < 1.0e-3),  # tuks
                )

                # number of objects in this light cone
                N_this_star_lc = np.sum(bool_star_this_info_lc)
                N_this_noinfo_lc = np.sum(mask_lc_this_noinfo)

                if i != ind_start or resume_flags[k, o]:
                    # check if we have information about this light cone origin, superslab and epoch
                    if (tmpdir / f'Merger_next_z{z_this:4.3f}_lc{o:d}.{k:02d}').exists():
                        # load leftover halos from previously loaded redshift
                        with asdf.open(
                            tmpdir / f'Merger_next_z{z_this:4.3f}_lc{o:d}.{k:02d}',
                            lazy_load=True,
                            memmap=False,
                        ) as f:
                            Merger_next = Table(f['data'])

                        # if you are a halo that appears here, we are gonna ignore you
                        N_next_lc = len(Merger_next['HaloIndex'])

                        # tmp1: to-append and extrapolated from before; tmp2: to-append and interpolated now; get rid of these; TODO: can be done less expensively
                        tmp1 = np.in1d(
                            Merger_next['HaloIndex'][:],
                            pack_inds(Merger_this['HaloIndex'][~eligibility_extrap_this], k),
                        )
                        tmp2 = np.in1d(
                            Merger_next['HaloIndex'][:],
                            pack_inds(Merger_this_info_lc['HaloIndex'][:], k),
                        )
                        tmp3 = ~(tmp1 & tmp2)

                        # if we found you in the interpolated halos in this redshift, you can't be allowed to be appended as part of Merger_next
                        Merger_next = Merger_next[tmp3]
                        del tmp1, tmp2, tmp3

                        # adding contributions from the previously loaded redshift
                        N_next_lc = len(Merger_next['HaloIndex'])

                    else:
                        N_next_lc = 0
                else:
                    N_next_lc = 0

                # total number of halos belonging to this light cone superslab and origin
                N_lc = N_this_star_lc + N_this_noinfo_lc + N_next_lc
                print(
                    'in this snapshot: interpolated, no info, next, total = ',
                    N_this_star_lc * 100.0 / N_lc,
                    N_this_noinfo_lc * 100.0 / N_lc,
                    N_next_lc * 100.0 / N_lc,
                    N_lc,
                )

                # save those arrays
                Merger_lc = Table(
                    {
                        'HaloIndex': np.empty(N_lc, dtype=Merger_this_info_lc['HaloIndex'].dtype),
                        'InterpolatedVelocity': np.empty(N_lc, dtype=(np.float32, 3)),
                        'InterpolatedPosition': np.empty(N_lc, dtype=(np.float32, 3)),
                        'InterpolatedComoving': np.empty(N_lc, dtype=np.float32),
                        'InterpolatedN': np.empty(N_lc, dtype=np.float32),
                    }
                )

                # record interpolated position and velocity for those with info belonging to current redshift
                Merger_lc['InterpolatedPosition'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedPosition'][
                    bool_star_this_info_lc
                ]
                Merger_lc['InterpolatedVelocity'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedVelocity'][
                    bool_star_this_info_lc
                ]
                Merger_lc['InterpolatedComoving'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedComoving'][
                    bool_star_this_info_lc
                ]
                Merger_lc['InterpolatedN'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedN'][
                    bool_star_this_info_lc
                ]
                Merger_lc['HaloIndex'][:N_this_star_lc] = Merger_this_info_lc['HaloIndex'][bool_star_this_info_lc]

                # record interpolated position and velocity of the halos in the light cone without progenitor information
                Merger_lc['InterpolatedPosition'][N_this_star_lc : N_this_star_lc + N_this_noinfo_lc] = (
                    Merger_this_noinfo_lc['Position']
                )
                Merger_lc['InterpolatedVelocity'][N_this_star_lc : N_this_star_lc + N_this_noinfo_lc] = (
                    Merger_this_noinfo_lc['v_L2com']
                )
                Merger_lc['InterpolatedComoving'][N_this_star_lc : N_this_star_lc + N_this_noinfo_lc] = (
                    Merger_this_noinfo_lc['ComovingDistance']
                )  # assign comoving distance based on position; used to be np.ones(Merger_this_noinfo_lc['Position'].shape[0])*chi_this
                Merger_lc['InterpolatedN'][N_this_star_lc : N_this_star_lc + N_this_noinfo_lc] = Merger_this_noinfo_lc[
                    'N'
                ]
                Merger_lc['HaloIndex'][N_this_star_lc : N_this_star_lc + N_this_noinfo_lc] = Merger_this_noinfo_lc[
                    'HaloIndex'
                ]
                del Merger_this_noinfo_lc

                # pack halo indices for all halos but those in Merger_next
                Merger_lc['HaloIndex'][: (N_this_star_lc + N_this_noinfo_lc)] = pack_inds(
                    Merger_lc['HaloIndex'][: (N_this_star_lc + N_this_noinfo_lc)], k
                )

                # record information from previously loaded redshift that was postponed
                if i != ind_start or resume_flags[k, o]:
                    if N_next_lc != 0:
                        Merger_lc['InterpolatedPosition'][-N_next_lc:] = Merger_next['InterpolatedPosition'][:]
                        Merger_lc['InterpolatedVelocity'][-N_next_lc:] = Merger_next['InterpolatedVelocity'][:]
                        Merger_lc['InterpolatedComoving'][-N_next_lc:] = Merger_next['InterpolatedComoving'][:]
                        Merger_lc['InterpolatedN'][-N_next_lc:] = Merger_next['InterpolatedN'][:]
                        Merger_lc['HaloIndex'][-N_next_lc:] = Merger_next['HaloIndex'][:]
                        del Merger_next
                    resume_flags[k, o] = False

                # offset position to make light cone continuous
                Merger_lc['InterpolatedPosition'] = offset_pos(
                    Merger_lc['InterpolatedPosition'], ind_origin=o, all_origins=origins
                )

                # halo indices for this origin
                mask_uni = np.zeros(len(Merger_lc), dtype=bool)

                # find unique halo indices (already for specific origins)
                _, inds = np.unique(Merger_lc['HaloIndex'], return_index=True)
                mask_uni[inds] = True
                Merger_lc = Merger_lc[mask_uni]
                del mask_uni, inds
                gc.collect()

                # Keep most of the position precision, since that's global over all repeats
                compress_lossy(Merger_lc['InterpolatedPosition'], keep_bits=20)
                compress_lossy(Merger_lc['InterpolatedComoving'], keep_bits=20)
                # Keep 12 bits of velocity precision, which is what we keep from rvint
                compress_lossy(Merger_lc['InterpolatedVelocity'], keep_bits=12)
                # Don't need to compress the masses, they were already rounded to integers

                # write table with interpolated information
                save_asdf(
                    Merger_lc,
                    header,
                    cat_lc_dir / f'z{zname_this:.3f}' / f'Merger_lc{o:d}.{k:02d}.asdf',
                    compress=True,
                )

                # mask of the extrapolated halos
                mask_extrap = (Merger_this_info_lc['InterpolatedComoving'] > chi_prev) | (
                    Merger_this_info_lc['InterpolatedComoving'] < chi_this
                )
                print(
                    'percentage extrapolated = ',
                    np.sum(mask_extrap) * 100.0 / len(mask_extrap),
                )

                # TODO: Need to make sure no bugs with eligibility
                # version 1: only the main progenitor is marked ineligible
                # if halo belongs to this redshift catalog or the previous redshift catalog
                eligibility_prev[Merger_prev_main_this_info_lc['HaloIndex'][~mask_extrap]] = False
                eligibility_extrap_prev[Merger_prev_main_this_info_lc['HaloIndex'][mask_extrap]] = False
                print(
                    'number eligible = ',
                    np.sum(eligibility_prev),
                    np.sum(eligibility_extrap_prev),
                )

                # version 2: all progenitors of halos belonging to this redshift catalog are marked ineligible
                # run version 1 AND 2 to mark ineligible Merger_next objects to avoid multiple entries
                # Note that some progenitor indices are zeros
                # For best result perhaps combine Progs with MainProgs
                if 'Progenitors' in fields_mt:
                    nums = Merger_this_info_lc['NumProgenitors'][bool_star_this_info_lc]
                    starts = Merger_this_info_lc['StartProgenitors'][bool_star_this_info_lc]
                    # for testing purposes (remove in final version)
                    main_progs = Merger_this_info_lc['MainProgenitor'][bool_star_this_info_lc]
                    progs = mt_data_this['progenitors']['Progenitors']
                    halo_ind_prev = Merger_prev['HaloIndex']

                    N_halos_load = np.array([N_halos_slabs_prev[i] for i in inds_fn_prev])
                    slabs_prev_load = np.array(
                        [slabs_prev[i] for i in slabs_prev[inds_fn_prev]],
                        dtype=np.int64,
                    )
                    offsets = np.zeros(len(inds_fn_prev), dtype=np.int64)
                    offsets[1:] = np.cumsum(N_halos_load)[:-1]

                    # mark ineligible the progenitors of the halos interpolated in this catalog
                    eligibility_prev = mark_ineligible(
                        nums,
                        starts,
                        main_progs,
                        progs,
                        halo_ind_prev,
                        eligibility_prev,
                        offsets,
                        slabs_prev_load,
                    )

                print(
                    'number eligible after progenitors removal = ',
                    np.sum(eligibility_prev),
                    np.sum(eligibility_extrap_prev),
                )

                # information to keep for next redshift considered
                N_next = np.sum(~bool_star_this_info_lc)
                Merger_next = Table(
                    {
                        'HaloIndex': np.empty(N_next, dtype=Merger_lc['HaloIndex'].dtype),
                        'InterpolatedVelocity': np.empty(N_next, dtype=(np.float32, 3)),
                        'InterpolatedPosition': np.empty(N_next, dtype=(np.float32, 3)),
                        'InterpolatedComoving': np.empty(N_next, dtype=np.float32),
                        'InterpolatedN': np.empty(N_next, dtype=np.float32),
                    }
                )
                Merger_next['HaloIndex'][:] = Merger_prev_main_this_info_lc['HaloIndex'][~bool_star_this_info_lc]
                Merger_next['InterpolatedVelocity'][:] = Merger_this_info_lc['InterpolatedVelocity'][
                    ~bool_star_this_info_lc
                ]
                Merger_next['InterpolatedPosition'][:] = Merger_this_info_lc['InterpolatedPosition'][
                    ~bool_star_this_info_lc
                ]
                Merger_next['InterpolatedComoving'][:] = Merger_this_info_lc['InterpolatedComoving'][
                    ~bool_star_this_info_lc
                ]
                Merger_next['InterpolatedN'][:] = Merger_this_info_lc['InterpolatedN'][~bool_star_this_info_lc]
                del Merger_this_info_lc, Merger_prev_main_this_info_lc

                if plot:
                    # select the halos in the light cones
                    pos_choice = Merger_lc['InterpolatedPosition']

                    # selecting thin slab
                    pos_x_min = -Lbox / 2.0 + k * (Lbox / n_superslabs)
                    pos_x_max = x_min + (Lbox / n_superslabs)

                    ijk = 0
                    choice = (pos_choice[:, ijk] >= pos_x_min) & (pos_choice[:, ijk] < pos_x_max)

                    circle_this = plt.Circle(
                        (origins[0][1], origins[0][2]),
                        radius=chi_this,
                        color='g',
                        fill=False,
                    )
                    circle_prev = plt.Circle(
                        (origins[0][1], origins[0][2]),
                        radius=chi_prev,
                        color='r',
                        fill=False,
                    )

                    # clear things for fresh plot
                    ax = plt.gca()
                    ax.cla()

                    # plot particles
                    ax.scatter(
                        pos_choice[choice, 1],
                        pos_choice[choice, 2],
                        s=0.1,
                        alpha=1.0,
                        color='dodgerblue',
                    )

                    # circles for in and prev
                    ax.add_artist(circle_this)
                    ax.add_artist(circle_prev)
                    plt.xlabel([-Lbox / 2.0, Lbox * 1.5])
                    plt.ylabel([-Lbox / 2.0, Lbox * 1.5])
                    plt.axis('equal')
                    plt.savefig('interp_%d_%d_%d.png' % (i, k, o))
                    # plt.show()
                    plt.close()

                # pack halo indices for the halos in Merger_next
                offset = 0
                for idx in inds_fn_prev:
                    print('k, idx = ', k, idx)
                    choice_idx = (offset <= Merger_next['HaloIndex'][:]) & (
                        Merger_next['HaloIndex'][:] < offset + N_halos_slabs_prev[idx]
                    )
                    Merger_next['HaloIndex'][choice_idx] = pack_inds(Merger_next['HaloIndex'][choice_idx] - offset, idx)
                    offset += N_halos_slabs_prev[idx]

                # split the eligibility array over three files for the three superslabs it's made up of
                offset = 0
                for idx in inds_fn_prev:
                    eligibility_prev_idx = eligibility_prev[offset : offset + N_halos_slabs_prev[idx]]
                    eligibility_extrap_prev_idx = eligibility_extrap_prev[offset : offset + N_halos_slabs_prev[idx]]
                    # combine current information with previously existing
                    if (tmpdir / ('eligibility_prev_z%4.3f_lc%d.%02d.npy' % (z_prev, o, idx))).exists():
                        eligibility_prev_old = np.load(
                            tmpdir / ('eligibility_prev_z%4.3f_lc%d.%02d.npy' % (z_prev, o, idx))
                        )
                        eligibility_prev_idx = eligibility_prev_old & eligibility_prev_idx
                        eligibility_extrap_prev_old = np.load(
                            tmpdir / ('eligibility_extrap_prev_z%4.3f_lc%d.%02d.npy' % (z_prev, o, idx))
                        )
                        eligibility_extrap_prev_idx = eligibility_extrap_prev_old & eligibility_extrap_prev_idx
                        print('Appending to existing eligibility file for %4.3f, %d, %02d!' % (z_prev, o, idx))
                    else:
                        print('First time seeing eligibility file for %4.3f, %d, %02d!' % (z_prev, o, idx))
                    np.save(
                        tmpdir / ('eligibility_prev_z%4.3f_lc%d.%02d.npy' % (z_prev, o, idx)),
                        eligibility_prev_idx,
                    )
                    np.save(
                        tmpdir / ('eligibility_extrap_prev_z%4.3f_lc%d.%02d.npy' % (z_prev, o, idx)),
                        eligibility_extrap_prev_idx,
                    )
                    offset += N_halos_slabs_prev[idx]
                gc.collect()

                # write as table the information about halos that are part of next loaded redshift
                save_asdf(
                    Merger_next,
                    header,
                    tmpdir / (f'Merger_next_z{z_prev:4.3f}_lc{o:d}.{k:02d}'),
                )

                # save redshift of catalog that is next to load and difference in comoving between this and prev
                save_build_state(
                    {'z_prev': z_prev, 'delta_chi': delta_chi, 'light_cone': o, 'super_slab': k}, build_log
                )

            del Merger_this, Merger_prev

        # update values for difference in comoving distance
        delta_chi_old = delta_chi


# dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])


def compress_lossy(arr, keep_bits):
    # Null out the (24 - keep_bits) least significant bits
    int_view = arr.view(np.uint32)
    mask = np.uint32(0xFFFFFFFF) << (24 - keep_bits)
    int_view[:] &= mask


def save_build_state(state, build_log):
    with asdf.AsdfFile({'state': state}) as af:
        af.write_to(build_log)


def load_build_state(build_log):
    with asdf.open(build_log, lazy_load=False, memmap=False) as af:
        return af['state']


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('sim_path', help='Simulation path')
    parser.add_argument(
        '--z-start',
        help='Initial redshift where we start building the trees',
        type=float,
    )
    parser.add_argument(
        '--z-stop',
        help='Final redshift (inclusive)',
        type=float,
    )
    parser.add_argument(
        '--merger-dir',
        help='Merger tree directory',
    )
    parser.add_argument(
        '--output-parent',
        help='The output light cone catalog directory',
    )
    parser.add_argument(
        '--tmpdir',
        help='Temporary working directory. Interrupted runs will be resumed from here.',
    )
    parser.add_argument(
        '--superslab-start',
        help='Initial superslab where we start building the trees',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--resume',
        help='Resume the calculation from the checkpoint on disk',
        action='store_true',
    )
    parser.add_argument('--plot', help='Want to show plots', action='store_true')
    parser.add_argument(
        '--complete',
        help='Interpolate the halos from  z = 0.1 to interpolate to z = 0',
        action='store_true',
    )

    args = vars(parser.parse_args())
    main(**args)
