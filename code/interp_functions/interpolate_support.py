#--------------------------------
# Name:         interpolate_support.py
# Purpose:      Interpolator support functions
#--------------------------------

from __future__ import division
import datetime as dt
# import gc
import logging
from multiprocessing import Process, Queue, cpu_count
import os
import sys
import warnings

import drigo
import numpy as np
from osgeo import gdal, ogr
from scipy import interpolate

# import et_common
import python_common as dripy

# np.seterr(invalid='ignore')
gdal.UseExceptions()


def landsat_dt_func(image_id):
    """"""
    # Assume image_id has been verified as a Landsat image ID
    # i.e. LC08_L1TP_043030_20150415_20170227_01_T1
    return dt.datetime.strptime(image_id.split('_')[3], '%Y%m%d').date()


def daterange_func(start_dt, end_dt, delta=1):
    """"""
    curr_dt = start_dt
    while curr_dt <= end_dt:
        yield curr_dt
        curr_dt += dt.timedelta(delta)


def tile_wkt_func(input_path, path_field='PATH', row_field='ROW',
                  tile_fmt='p{:03d}r{:03d}'):
    """Return a dictionary of path/rows and their geometries"""
    output_dict = dict()
    input_ds = ogr.Open(input_path, 0)
    input_lyr = input_ds.GetLayer()
    input_ftr = input_lyr.GetNextFeature()
    while input_ftr:
        path = input_ftr.GetFieldAsInteger(
            input_ftr.GetFieldIndex(path_field))
        row = input_ftr.GetFieldAsInteger(
            input_ftr.GetFieldIndex(row_field))
        input_wkt = input_ftr.GetGeometryRef().ExportToWkt()
        output_dict[tile_fmt.format(path, row)] = input_wkt
        input_ftr = input_lyr.GetNextFeature()
    input_ds = None
    return output_dict


# def clip_project_raster_worker(args, input_q, output_q):
#     """Worker function for multiprocessing with input and output queues
#
#     First input argument is an index that will be passed through to the output
#     Convert projection WKT parameters to OSR objects
#         4th and 7th?
#
#     """
#     while True:
#         args = input_q.get()
#         if args is None:
#             break
#         args_mod = args[:]
#         for i, arg in enumerate(args):
#             # DEADBEEF - Do all projection WKT's start with 'PROJCS'?
#             # Could try testing to see if the result of proj_osr is an OSR?
#             if type(arg) == str and arg.startswith('PROJCS'):
#                 args_mod[i] = drigo.proj_osr(arg)
#         output_q.put([args_mod[0], clip_project_raster_func(*args_mod[1:])])
#         # output_q.put(clip_project_raster_mp(args))
#
# def clip_project_raster_mp(args):
#     """MP wrapper for calling clip_project_raster_func with Pool
#
#     First input parameter is an index that will be passed through
#     Convert projection WKT parameters to OSR objects
#         4th and 7th?
#
#     """
#     args_mod = args[:]
#     for i, arg in enumerate(args):
#         # DEADBEEF - Do all projection WKT's start with 'PROJCS'?
#         # Could try testing to see if the result of proj_osr is an OSR?
#         if type(arg) == str and arg.startswith('PROJCS'):
#             args_mod[i] = drigo.proj_osr(arg)
#     return args_mod[0], clip_project_raster_func(*args_mod[1:])


def clip_project_raster_func(input_raster, resampling_type,
                             input_osr, input_cs, input_extent,
                             ouput_osr, output_cs, output_extent):
    """Clip and then project an input raster"""
    # Read array from input raster using input extent
    input_array = drigo.raster_to_array(
        input_raster, 1, input_extent, return_nodata=False)
    # Project and clip array to block
    output_array = drigo.project_array(
        input_array, resampling_type,
        input_osr, input_cs, input_extent,
        ouput_osr, output_cs, output_extent)
    return output_array


def mosaic_func(mosaic_array, input_array, mosaic_method):
    """"""
    input_mask = np.isfinite(input_array)
    if not np.any(input_mask):
        # Only mosaic if there is new data
        pass
    elif mosaic_method.lower() == 'first':
        # Fill cells that are currently empty
        input_mask &= np.isnan(mosaic_array)
        mosaic_array[input_mask] = input_array[input_mask]
    elif mosaic_method.lower() == 'last':
        # Overwrite any cells with new data
        mosaic_array[input_mask] = input_array[input_mask]
    elif mosaic_method.lower() == 'mean':
        # Fill cells that are currently empty
        temp_mask = input_mask & np.isnan(mosaic_array)
        mosaic_array[temp_mask] = input_array[temp_mask]
        # plt.imshow(mosaic_array)
        # plt.title('mosaic_array')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(input_array)
        # plt.title('input_array')
        # plt.colorbar()
        # plt.show()
        # plt.imshow((mosaic_array - input_array))
        # plt.title('mosaic_array - input_array')
        # plt.colorbar()
        # plt.show()
        # print((mosaic_array - input_array))
        # Mean with existing value (overlapping rows)
        temp_mask = input_mask & np.isfinite(mosaic_array)
        mosaic_array[temp_mask] += input_array[temp_mask]
        mosaic_array[temp_mask] *= 0.5
        del temp_mask
    return mosaic_array


def load_etrf_func(array_shape, date_list, year_ws, year,
                   etrf_raster, block_tile_list, block_extent,
                   tile_image_dict, mosaic_method, resampling_type,
                   output_osr, output_cs, output_extent, debug_flag):
    """Load ETrF from rasters to an array for all images/dates

    Parameters
    ----------
    array_shape : list
    date_list : list
        List of dates to be processed.
    year_ws : str
        File path of the workspace to the year folder from METRIC run.
    etrf_raster : str
        File path for the output ETrF.
    year : str
        Year that will be processed.
    block_tile_list : list
        List of the tiles to be processed in each block.
    block_extent(class:`gdal_common.env`):
        The gdal_common.extent of the block.
    tile_image_dict : dict
        A dictionary of the tiles/years to be processed.
    mosaic_method : str
        Mean, upper, or lower
    resampling_type : int
        GDAL resampling type used to reproject the daily ETrF.
    output_osr (class:`osr.SpatialReference):
        Desired spatial reference object.
    output_cs : int
        Desired cellsize of the output
    output_extent(class:`gdal_common.extent):
        Desired gdal_common.extent of the output.
    debug_flag : bool
        If True, NumPy RuntimeWarnings will be printed.

    """
    # Read in ETrF raster from each scene folder
    days, rows, cols = array_shape
    # days, x, y = etrf_array.shape

    tile_etrf_array = np.full(
        (days, len(block_tile_list), rows, cols), np.nan, np.float32)
    for tile_i, tile_name in enumerate(block_tile_list):
        if tile_name not in tile_image_dict[year].keys():
            continue

        for image_id in dripy.shuffle(tile_image_dict[year][tile_name]):
            tile_ws = os.path.join(year_ws, tile_name)
            image_ws = os.path.join(tile_ws, image_id)
            image_etrf_raster = os.path.join(image_ws, etrf_raster)
            if not os.path.isfile(image_etrf_raster):
                logging.debug('  ETrF raster does not exist')
                continue

            # Get projection and extent for each image
            block_tile_ds = gdal.Open(image_etrf_raster)
            block_tile_osr = drigo.raster_ds_osr(block_tile_ds)
            block_tile_cs = drigo.raster_ds_cellsize(block_tile_ds, x_only=True)
            block_tile_x, block_tile_y = drigo.raster_ds_origin(block_tile_ds)
            block_tile_extent = drigo.project_extent(
                block_extent, output_osr, block_tile_osr, output_cs)
            block_tile_extent.adjust_to_snap(
                'EXPAND', block_tile_x, block_tile_y, block_tile_cs)
            block_tile_ds = None

            # Use image_id to determine date
            date_i = date_list.index(landsat_dt_func(image_id))
            tile_etrf_array[date_i, tile_i, :, :] = clip_project_raster_func(
                image_etrf_raster, resampling_type,
                block_tile_osr, block_tile_cs, block_tile_extent,
                output_osr, output_cs, output_extent)

            # if low_etrf_limit is not None:
            #     temp_array[temp_array < low_etrf_limit] = low_etrf_limit
            # if high_etrf_limit is not None:
            #     temp_array[temp_array > high_etrf_limit] = high_etrf_limit

    # Suppress the numpy nan warning if the debug flag is off
    if not debug_flag:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            etrf_array = np.nanmean(tile_etrf_array, axis=1)
    else:
        etrf_array = np.nanmean(tile_etrf_array, axis=1)

    return etrf_array


# def load_etrf_swb_func(etrf_array, etrf_raster,
#                        low_etrf_limit, high_etrf_limit,
#                        date_list, year_ws, ndvi_raster, year,
#                        block_tile_list, block_extent,
#                        tile_image_dict, mosaic_method, resampling_type,
#                        output_osr, output_cs, output_extent, debug_flag,
#                        soil_water_balance_adjust_flag,
#                        year_tile_ndvi_paths, tile_ndvi_dict,
#                        awc_path, etr_input_ws, etr_input_re, ppt_input_ws,
#                        ppt_input_re, ndvi_threshold):
#     """
#
#     Parameters
#     ----------
#
#     Returns
#     -------
#         numpy.array: class:`numpy.array`
#     """
#     days, x, y = etrf_array.shape
#     tiles = len(block_tile_list)
#     temp_etrf_array = np.full((days, tiles, x, y), np.nan)
#     temp_ndvi_array = np.full((days, tiles, x, y), np.nan)

#     load_etrf_func(
#         etrf_array, date_list, year_ws, etrf_raster, year,
#         block_tile_list, block_extent,
#         tile_image_dict, mosaic_method, resampling_type,
#         output_osr, output_cs, output_extent, debug_flag,
#         low_etrf_limit, high_etrf_limit)
#     year = int(year)
#     for tile_i, tile_name in enumerate(block_tile_list):
#         if tile_name not in tile_image_dict[year].keys():
#             continue

#         for image_id in dripy.shuffle(tile_image_dict[year][tile_name]):
#             tile_ws = os.path.join(year_ws, tile_name)
#             image_ws = os.path.join(tile_ws, image_id)
#             image_ndvi_raster = os.path.join(image_ws, ndvi_raster)
#             if not os.path.isfile(image_ndvi_raster):
#                 continue

#             # Get projection and extent for each image
#             block_tile_ds = gdal.Open(image_ndvi_raster)
#             block_tile_osr = drigo.raster_ds_osr(block_tile_ds)
#             block_tile_cs = drigo.raster_ds_cellsize(block_tile_ds, x_only=True)
#             block_tile_x, block_tile_y = drigo.raster_ds_origin(block_tile_ds)
#             block_tile_extent = drigo.project_extent(
#                 block_extent, output_osr, block_tile_osr, output_cs)
#             # block_tile_extent.adjust_to_snap(
#             #     'EXPAND', block_tile_x, block_tile_y, block_tile_cs)
#             block_tile_ds = None

#             awc_ds = gdal.Open(awc_path)
#             awc_osr = drigo.raster_ds_osr(awc_ds)
#             awc_cs = drigo.raster_ds_cellsize(awc_ds, x_only=True)
#             awc_x, awc_y = drigo.raster_ds_origin(awc_ds)
#             awc_extent = drigo.project_extent(
#                 block_extent, output_osr, awc_osr, awc_cs)
#             awc_extent.adjust_to_snap(
#                 'EXPAND', awc_x, awc_y, awc_cs)
#             awc_ds = None

#             dt_object = landsat_dt_func(image_id)
#             date_i = date_list.index(dt_object)
#             etrf_array = daily_etrf_array[date_i,:,:,]

#             if np.all(np.isnan(etrf_array)):
#                 continue
#             etrf_background = et_common.array_swb_func(
#                 dt_object, awc_path, etr_input_ws, etr_input_re,
#                 ppt_input_ws, ppt_input_re, awc_osr, awc_cs, awc_extent,
#                 output_osr, output_cs, output_extent, 30)
#             ndvi_array = clip_project_raster_func(
#                 image_ndvi_raster, resampling_type,
#                 block_tile_osr, block_tile_cs, block_tile_extent,
#                 output_osr, output_cs, output_extent)
#             ndvi_mask = (ndvi_array > ndvi_threshold).astype(np.bool)
#             fc = calc_fc(
#                 # ndvi_array=temp_ndvi_array[date_i, tile_i,:,:,],
#                 ndvi_array=ndvi_array,
#                 ndvi_full_cover=tile_ndvi_dict[year][tile_name][image_id]['cold'],
#                 ndvi_bare_soil=tile_ndvi_dict[year][tile_name][image_id]['hot'])
#             etrf_transpiration = etrf_array - ((1 - fc) * etrf_background)
#             etrf_transpiration_adj = np.max(
#                 np.array([etrf_transpiration, etrf_background]),
#                 axis=0)
#             etrf_adjusted = (
#                 ((1 - fc) * etrf_background) + (fc * etrf_transpiration_adj))
#             etrf_adjusted[ndvi_mask] = etrf_array[ndvi_mask]
#             temp_etrf_array[date_i, tile_i,:,:,] = etrf_adjusted

#     # Suppress the numpy nan warning if the debug flag is off
#     if not debug_flag:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore', category=RuntimeWarning)
#             etrf_array[:] = np.nanmean(temp_etrf_array, axis=1)
#     elif debug_flag:
#         etrf_array[:] = np.nanmean(temp_etrf_array, axis=1)
#     else:
#         logging.error(
#             ('Could not calculate ETRF using ' +
#              'temp_etrf_array: {}, shape {}'.format(
#                 temp_etrf_array, temp_etrf_array.shape)))
#         sys.exit()


def spatial_fill_func(data_array, date_list, mp_flag, mp_procs):
    """"""
    return data_array


# def end_fill_func(data_array, block_mask, fill_method='linear'):
#     """"""
#
#     # Skip block if array is all nodata
#     if not np.any(block_mask):
#         return data_array
#     # Skip block if array is all nodata
#     # elif np.all(np.isnan(data_array)):
#     #     return data_array
#
#     # Fill first and last Landsat ETrF rasters
#     # Filling anchor rasters is independent of the fill method
#     # date_str_list = [d.strftime('%Y_%m_%d') for d in date_list]
#
#     data_shape = data_array.shape
#     data_index = np.tile(
#         np.arange(data_shape[0], dtype=np.float32)[:, np.newaxis, np.newaxis],
#         (data_shape[1], data_shape[2]))
#     data_index[np.isnan(data_array)] = np.nan
#
#     min_index = np.nanargmin(data_index, axis=0)
#     max_index = np.nanargmax(data_index, axis=0)
#     print min_index
#     print max_index
#     return data_array


def end_fill_func(data_array, block_mask, fill_method='linear'):
    """Fill start/end/anchor values using nearest value in time

    Parameters
    ----------
    data_array : ndarray
    block_mask : ndarray
    fill_method : {'linear' or 'cubicspline'}

    Returns
    -------
    ndarray

    Notes
    -----
    The actual spacing/timing of the images is not being considered.
    This approach would be inefficient if the full array was passed in.

    """
    # Skip block if array is all nodata
    if not np.any(block_mask):
        return data_array
    # Skip block if array is all nodata
    # elif np.all(np.isnan(data_array)):
    #     return data_array

    def fill_from_next(data_array, block_mask, data_i_list):
        """"""
        # First axis of block array is the date/doy
        fill_array = np.empty(data_array[0].shape, dtype=data_array.dtype)
        fill_array[:] = np.nan
        for data_i in data_i_list:
            next_array = data_array[data_i,:,:]
            next_mask = np.isfinite(next_array)
            # Only fill values that are nan
            next_mask &= np.isnan(fill_array)
            # Only fill values that are nan
            next_mask &= block_mask
            # Only fill pixels that have a usable number of scenes
            if np.any(next_mask):
                fill_array[next_mask] = next_array[next_mask]
            del next_array, next_mask
            # Stop once all usable scene pixels are filled
            if np.all(np.isfinite(fill_array[block_mask])):
                break
        return fill_array
    # The actual spacing/timing of the images is not being considered
    data_i_list = range(data_array.shape[0])
    # Calculate ETrF start raster
    if np.any(np.isnan(data_array[0, :, :])):
        data_array[0, :, :] = fill_from_next(
            data_array, block_mask, data_i_list)
    # Calculate ETrF end raster
    if np.any(np.isnan(data_array[-1, :, :])):
        data_array[-1, :, :] = fill_from_next(
            data_array, block_mask, sorted(data_i_list, reverse=True))
    # Calculate start/end anchor rasters
    if fill_method == 'cubicspline':
        if np.any(np.isnan(data_array[1, :, :])):
            data_array[1, :, :] = fill_from_next(
                data_array, block_mask, data_i_list)
        if np.any(np.isnan(data_array[-2, :, :])):
            data_array[-2, :, :] = fill_from_next(
                data_array, block_mask, sorted(data_i_list, reverse=True))
    return data_array


# DEADBEEF - Single core implementation
def temporal_fill_func(sub_array, sub_i_array, block_mask, fill_method='linear'):
    """Single core temporal fill function

    Fill Landsat scene dates so that interpolator only runs between known dates

    Parameters
    ----------
    sub_array : ndarray
    sub_i_array : ndarray
    block_mask : ndarray
    fill_method : {'linear' or 'cubicspline'}
        Interpolation method (the default is 'linear').

    Returns
    -------
    ndarray

    """
    # Skip block if array is all nodata
    if not np.any(block_mask):
        return sub_array
    # Skip block if array is all nodata
    # elif np.all(np.isnan(data_array)):
    #     return sub_array

    # Begin interpolating scene days with missing values
    # for interp_i, interp_doy in enumerate(sub_i_array):
    for interp_sub_i, interp_full_i in enumerate(sub_i_array):
        # Interp mask is False where pixels have data
        # (i.e. True for pixels that will be interpolated)
        interp_mask = np.isnan(sub_array[interp_sub_i, :, :])
        interp_mask &= block_mask
        if not np.any(interp_mask):
            continue
        # logging.info('    INTERP {} {}'.format(
        #     interp_sub_i, interp_full_i))

        # list of subsequent days
        for anchor_sub_i, anchor_full_i in enumerate(sub_i_array):
            if anchor_sub_i <= interp_sub_i:
                continue
            # Interpolate when next DOY has data
            anchor_mask = np.copy(interp_mask)
            anchor_mask &= np.isfinite(sub_array[anchor_sub_i, :, :])
            if not np.any(anchor_mask):
                continue
            # logging.info('      ANCHOR {} {}'.format(
            #     anchor_sub_i, anchor_full_i))
            if fill_method == 'cubicspline':
                for cubic_sub_i, cubic_full_i in enumerate(sub_i_array):
                    if cubic_sub_i <= anchor_sub_i:
                        continue
                    cubic_mask = np.copy(anchor_mask)
                    cubic_mask &= np.isfinite(sub_array[cubic_sub_i, :, :])
                    if not np.any(cubic_mask):
                        continue
                    # logging.info('      CUBIC {} {}'.format(
                    #     cubic_sub_i, cubic_full_i))
                    interp_i_array = np.array([
                        sub_i_array[interp_sub_i-2], sub_i_array[interp_sub_i-1],
                        sub_i_array[anchor_sub_i], sub_i_array[cubic_sub_i]])
                    interp_i_mask = np.in1d(sub_i_array, interp_i_array)
                    interp_array = sub_array[interp_i_mask, :, :][:, cubic_mask]
                    f = interpolate.interp1d(
                        interp_i_array, interp_array,
                        axis=0, kind=3)
                    sub_array[interp_sub_i, :, :][cubic_mask] = f(interp_full_i)
                    # sub_array[interp_sub_i,:,:][anchor_mask] = f(interp_full_i).astype(np.float32)
                    interp_mask[cubic_mask] = False
                    anchor_mask[cubic_mask] = False
                    del f, interp_i_array, interp_i_mask
                    del cubic_mask, interp_array
                    if not np.any(interp_mask):
                        break
            elif fill_method == 'linear':
                interp_i_array = np.array(
                    [sub_i_array[interp_sub_i-1], sub_i_array[anchor_sub_i]])
                interp_i_mask = np.in1d(sub_i_array, interp_i_array)
                interp_array = sub_array[interp_i_mask, :, :][:, anchor_mask]
                f = interpolate.interp1d(
                    interp_i_array, interp_array, axis=0, kind=fill_method)
                sub_array[interp_sub_i, :, :][anchor_mask] = f(interp_full_i)
                # sub_array[interp_sub_i,:,:][anchor_mask] = f(interp_full_i).astype(np.float32)
                interp_mask[anchor_mask] = False
                del f, interp_i_array, interp_i_mask, interp_array
                if not np.any(interp_mask):
                    break
            elif fill_method == 'nearest':
                pass
            # There is a memory leak with f/interp1d
            # gc.collect()
        del interp_mask
    return sub_array


def interpolate_func(full_array, sub_array, sub_i_array,
                     block_mask, interp_method):
    """Single core interpolator function

    This function should be used after scene dates have already been filled
    There is no error checking to see if the start/end/anchor have data

    Parameters
    ----------
        full_array : ndarray
        sub_array : ndarray
        sub_i_array : ndarray
        block_mask : ndarray
        interp_method : str

    Returns
    -------
    ndarray

    """
    # Skip block if array is all nodata
    if not np.any(block_mask):
        return full_array
    # Skip block if array is all nodata
    # elif np.all(np.isnan(data_array)):
    #     return full_array

    # Assume each step is a day
    full_i_array = np.arange(full_array.shape[0])

    # Copy start/end/anchor dates directly to output
    copy_i_list = [full_i_array[0], full_i_array[-1]]
    if interp_method in ['cubic', 'cubicspline']:
        copy_i_list.extend([full_i_array[1], full_i_array[-2]])
    copy_i_list.sort()

    # Begin interpolating scene days with missing values
    for interp_full_i in full_i_array:
        # Interp mask is False where pixels have data
        # (i.e. True for pixels that will be interpolated)
        interp_mask = np.isnan(full_array[interp_full_i, :, :])
        interp_mask &= block_mask
        if not np.any(interp_mask):
            continue
        # logging.info('    INTERP {}'.format(interp_full_i))

        # Copy start/end/anchor dates directly to output
        # if interp_full_i in list(sub_i_array):
        if interp_full_i in copy_i_list:
            full_array[interp_full_i, :, :][interp_mask] = sub_array[
                list(sub_i_array).index(interp_full_i), :, :][interp_mask]
            continue

        # Select anchor days (last day(s) before interp and first day(s) after)
        if interp_method in ['cubic', 'cubicspline']:
            interp_i_array = sub_i_array[np.concatenate(
                (np.where(sub_i_array <= interp_full_i)[0][-2:],
                 np.where(sub_i_array > interp_full_i)[0][:2]))]
        else:
            interp_i_array = sub_i_array[np.concatenate(
                (np.where(sub_i_array <= interp_full_i)[0][-1:],
                 np.where(sub_i_array > interp_full_i)[0][:1]))]
        interp_i_mask = np.in1d(sub_i_array, interp_i_array)
        interp_array = sub_array[interp_i_mask, :, :][:, interp_mask]
        f = interpolate.interp1d(
            interp_i_array, interp_array, axis=0, kind=interp_method)
        full_array[interp_full_i, :, :][interp_mask] = f(interp_full_i)
        # data_array[interp_full_i,:,:][:,interp_mask] = f(interp_full_i).astype(np.float32)
        del f, interp_array, interp_i_array
        # There is a memory leak with f/interp1d
        # gc.collect()
    return full_array


# def mp_interpolate_func(full_array, sub_array, sub_i_array,
#                         block_mask, interp_method,
#                         mp_flag=True, mp_procs=cpu_count()):
#     """"""
#     mp_procs = 1
#
#     # Skip block if array is all nodata
#     if not np.any(block_mask):
#         return data_array
#     # Skip block if array is all nodata
#     # elif np.all(np.isnan(data_array)):
#     #     return data_array
#
#     # Assume each step is a day
#     full_i_array = np.arange(full_array.shape[0])
#
#     # Create shared memory object of full_array
#     print sub_array[0,:,:]
#     print sub_array[:,0,0]
#     sub_ctypes = RawArray(ctypes.c_float, sub_array.size)
#     sub_shr_array = np.frombuffer(
#         sub_ctypes, dtype=np.float32, count=sub_array.size)
#     # Copy sub_array into the shared memory array
#     # sub_shr_array = sub_array
#     sub_shr_array = sub_array.flatten()
#
#     # Begin interpolating scene days with missing values
#     input_q = Queue()
#     output_q = Queue()
#     mp_tasks = 0
#     for interp_full_i in full_i_array:
#         # Interp mask is False where pixels have data
#         # (i.e. True for pixels that will be interpolated)
#         interp_mask = np.isnan(full_array[interp_full_i,:,:])
#         interp_mask &= block_mask
#         if not np.any(interp_mask):
#             continue
#         # Copy start/end/anchor dates directly to output
#         # if interp_i in list(sub_i_array):
#         if (interp_full_i == full_i_array[0] or
#             interp_full_i == full_i_array[-1] or
#             (interp_method in ['cubic', 'cubicspline'] and
#              (interp_full_i == full_i_array[1] or
#               interp_full_i == full_i_array[-2]))):
#             full_array[interp_full_i,:,:][interp_mask] = sub_array[
#                 list(sub_i_array).index(interp_full_i),:,:][interp_mask]
#             continue
#         # Select anchor days for each day being interpolated
#         if interp_method in ['cubic', 'cubicspline']:
#             interp_sub_i_array = np.concatenate(
#                 (np.where(sub_i_array <= interp_full_i)[0][-2:],
#                  np.where(sub_i_array > interp_full_i)[0][:2]))
#         else:
#             interp_sub_i_array = np.concatenate(
#                 (np.where(sub_i_array <= interp_full_i)[0][-1:],
#                  np.where(sub_i_array > interp_full_i)[0][:1]))
#         interp_full_i_array = sub_i_array[interp_sub_i_array]
#         # Put the items into the processing queue
#         input_q.put([
#             interp_full_i, interp_full_i_array,
#             interp_sub_i_array, interp_method])
#         mp_tasks += 1
#         del interp_full_i, interp_full_i_array, interp_sub_i_array
#
#     # Start the workers
#     for i in range(max(1, mp_procs - 1)):
#         p = Process(
#             target=interpolate_worker,
#             args=(sub_ctypes, sub_array.shape, input_q, output_q)).start()
#     # Start processing
#     for i in range(mp_tasks):
#     # for i in range(input_q.qsize()):
#         interp_i, interp_array = output_q.get()
#         full_array[interp_i,:,:][block_mask] = interp_array[block_mask]
#         del interp_i, interp_array
#     # Terminate the workers
#     for i in range(max(1, mp_procs - 1)):
#         input_q.put(None)
#     input_q.close()
#     output_q.close()
#     del input_q, output_q
#     del sub_ctypes, sub_shr_array
#     return full_array

# def interpolate_worker(sub_ctypes, sub_shape, input_q, output_q):
#     """Worker function for multiprocessing with input and output queues"""
#     # sub_array = np.ctypeslib.as_array(sub_ctypes)
#     # sub_array = sub_array.reshape(sub_shape)
#     # sub_array.shape = sub_shape
#     # sub_array = np.ctypeslib.as_array(sub_ctypes).reshape(sub_shape)
#     sub_array = np.asarray(np.frombuffer(sub_ctypes, dtype=np.float32))
#     sub_array = sub_array.reshape(sub_shape)
#     print sub_array
#     print sub_array.shape
#     print sub_array[:,0,0]
#     print sub_array.dtype
#     print input_q
#     print output_q
#     while True:
#         args = input_q.get()
#         if args is None:
#             break
#         interp_full_i = args[0]
#         interp_full_i_array = args[1]
#         interp_sub_i_array = args[2]
#         interp_method = args[3]
#         f = interpolate.interp1d(
#             interp_full_i_array, sub_array[interp_sub_i_array,:,:],
#             axis=0, kind=interp_method)
#         # f = interpolate.interp1d(
#         #     interp_i_array, sub_array[[0,2],:,:], axis=0, kind=interp_method)
#         output_q.put([interp_full_i, f(interp_full_i)])
#         # output_q.put(interpolate_mp(args))

# def interpolate_mp(args):
#     """MP wrapper for calling interpolate
#
#     First input parameter is the date index that will be passed through
#
#     """
#     f = interpolate.interp1d(args[1], args[2], axis=0, kind=args[3])
#     return args[0], f(args[0])

# def interpolate_mp(tup):
#     """MP wrapper for calling interpolate
#
#     First input parameter is the date index that will be passed through
#     Second input parameter is a mask that will be passed through
#
#     """
#     return tup[0], tup[1], interpolate_sp(*tup[2:])

# def interpolate_sp(x_array, y_array, interp_doy, interp_method):
#     """Wrapper function for clipping and then projecting an input raster"""
#     f = interpolate.interp1d(x_array, y_array, axis=0, kind=interp_method)
#     return f(interp_doy)


def block_interpolate_func(full_array, sub_array, sub_i_array,
                           block_mask, fill_method, interp_method,
                           mp_flag=True, mp_procs=cpu_count()):
    """Interpolate sub block using multiprocessing

    Parameters
    ----------
    full_array : ndarray
    sub_array : ndarray
    sub_i_array : ndarray
    block_mask : ndarray
    fill_method : str
    interp_method : str
    mp_flag : bool
    mp_procs : int

    Returns
    -------
    ndarray

    """
    logging.info('    Processing by sub block')
    block_rows, block_cols = block_mask.shape
    sub_bs = 64
    mp_list = []
    for s_i, s_j in drigo.block_gen(block_rows, block_cols, sub_bs):
        # logging.info('    Sub  y: {:5d}  x: {:5d}'.format(s_i, s_j))
        sub_rows, sub_cols = drigo.block_shape(
            block_rows, block_cols, s_i, s_j, sub_bs)
        # logging.info('    Sub rows: {}  cols: {}'.format(sub_rows, sub_cols))
        mp_list.append([s_i, s_j])
    if mp_list:
        input_q = Queue()
        output_q = Queue()

        # Load some inputs into the input queue
        mp_tasks = len(mp_list)
        for i in range(max(1, mp_procs - 1)):
            s_i, s_j = mp_list.pop()
            input_q.put([
                s_i, s_j, full_array[:, s_i:s_i+sub_rows, s_j:s_j+sub_cols],
                block_mask[s_i:s_i+sub_rows, s_j:s_j+sub_cols], interp_method])
        # Load all inputs into the input queue
        # for mp_args in mp_list:
        #     input_q.put(mp_args)

        # Start workers
        for i in range(max(1, mp_procs - 1)):
            p = Process(target=block_interpolate_worker,
                        args=(i, input_q, output_q)).start()
            del p

        # Get data from workers and add new items to queue
        for i in range(mp_tasks):
            s_i, s_j, interp_array = output_q.get()
            full_array[:, s_i:s_i+sub_rows, s_j:s_j+sub_cols] = sub_array
            del s_i, s_j, sub_array
            try:
                s_i, s_j = mp_list.pop()
                input_q.put([
                    s_i, s_j, full_array[:, s_i:s_i+sub_rows, s_j:s_j+sub_cols],
                    block_mask[s_i:s_i+sub_rows, s_j:s_j+sub_cols], interp_method])
                del s_i, s_j
            except IndexError:
                pass

        # Close workers
        for i in range(max(1, mp_procs - 1)):
            input_q.put(None)

        # Close queues
        input_q.close()
        output_q.close()
        del input_q, output_q

    return full_array

def block_interpolate_worker(args, input_q, output_q):
    """Worker function for multiprocessing with input and output queues"""
    while True:
        args = input_q.get()
        if args is None:
            break
        s_i, s_j, full_array, sub_array, sub_i_array, sub_mask, fill_method, interp_method = args
        sub_array = end_fill_func(sub_array, sub_mask, fill_method)
        sub_array = temporal_fill_func(
            sub_array, sub_i_array, sub_mask, fill_method)
        full_array = interpolate_func(
            full_array, sub_array, sub_i_array, sub_mask, interp_method)
        output_q.put([s_i, s_j, full_array])


def load_year_array_func(input_ws, input_re, date_list,
                         mask_osr, mask_cs, mask_extent,
                         name='ETr', return_geo_array=True):
    """Load

    Parameters
    ----------
    input_ws : str
    input_re
    date_list : list
    output_osr
    output_cs : float
    output_extent
    name : str
    return_geo_array : bool
        If True, return array geo-spatial properties (the default is True).

    Returns
    -------
    ndarray

    """
    logging.info('\n{}'.format(name))
    logging.debug('  {} workspace: {}'.format(name, input_ws))
    year_str_list = sorted(list(set([
        date.strftime('%Y') for date in date_list])))

    if not os.path.isdir(input_ws):
        logging.error(
            '\nERROR: The {} folder does not exist:\n  {}'.format(
                name, input_ws))
        sys.exit()
    input_dict = {
        input_match.group('YYYY'): os.path.join(input_ws, input_name)
        for input_name in os.listdir(os.path.join(input_ws))
        for input_match in [input_re.match(input_name)]
        if (input_match and input_match.group('YYYY') and
            input_match.group('YYYY') in year_str_list)}
    if not input_dict:
        logging.error(
            ('  No {0} files found in {1} for {2}\n'
             '  The {0} year folder may be empty or the regular '
             'expression is invalid\n  Exiting').format(
                name, input_ws, ', '.join(year_str_list)))
        sys.exit()

    # Assume all rasters have same projection, cellsize, and snap
    for date_obj in date_list:
        try:
            input_path = input_dict[date_obj.strftime('%Y')]
            break
        except KeyError:
            logging.debug(
                '    {} raster for date {} does not exist'.format(
                    name, date_obj.strftime('%Y%m%d')))
            sys.exit()
    input_ds = gdal.Open(input_path, 0)
    input_osr = drigo.raster_ds_osr(input_ds)
    # input_proj = drigo.osr_proj(input_osr)
    input_cs = drigo.raster_ds_cellsize(input_ds, x_only=True)
    input_x, input_y = drigo.raster_ds_origin(input_ds)
    input_ds = None

    # Get mask extent in the original spat. ref.
    output_extent = drigo.project_extent(
        mask_extent, mask_osr, input_osr, mask_cs)
    output_extent.adjust_to_snap('EXPAND', input_x, input_y, input_cs)
    output_rows, output_cols = output_extent.shape(cs=input_cs)

    # Initialize the common array
    output_array = np.full(
        (len(date_list), output_rows, output_cols), np.nan, np.float32)

    # Read in the raster for each date
    for date_i, date_obj in enumerate(date_list):
        try:
            input_path = input_dict[date_obj.strftime('%Y')]
        except KeyError:
            logging.debug(
                '  {} - {} raster does not exist'.format(
                    date_obj.strftime('%Y%m%d'), name))
            continue
        output_array[date_i, :, :] = drigo.raster_to_array(
            input_path, band=int(date_obj.strftime('%j')),
            mask_extent=output_extent, return_nodata=False,)

    if return_geo_array:
        return output_array, input_osr, input_cs, output_extent
    else:
        return output_array


def swb_adjust_fc(ndvi_array, ndvi_full_cover, ndvi_bare_soil):
    """"""
    return (1 - (ndvi_full_cover - ndvi_array) /
                (ndvi_full_cover - ndvi_bare_soil))


def unknown_proj_osr(input_proj):
    """Return the spatial reference object for a projection string"""
    try:
        output_osr = drigo.epsg_osr(input_proj)
        logging.debug('  OSR from EPSG string')
        return output_osr
    except:
        pass
    try:
        output_osr = drigo.epsg_osr(input_proj.replace('EPSG:'))
        logging.debug('  OSR from EPSG integer')
        return output_osr
    except:
        pass
    try:
        output_osr = drigo.proj_osr(input_proj)
        logging.debug('  OSR from WKT')
        return output_osr
    except:
        pass
    try:
        output_osr = drigo.proj4_osr(input_proj)
        logging.debug('  OSR from PROJ4')
        return output_osr
    except:
        pass
    try:
        output_osr = drigo.raster_path_osr(input_proj)
        logging.debug('  OSR from raster path')
        return output_osr
    except:
        pass
    try:
        output_osr = drigo.feature_path_osr(input_proj)
        logging.debug('  OSR from feature path')
        return output_osr
    except:
        pass

    return output_osr


# def feature_extents(input_path):
#     """Return a dictionary of zone FIDs and their extents"""
#     output_dict = dict()
#     # shp_driver = ogr.GetDriverByName('ESRI Shapefile')
#     input_ds = ogr.Open(input_path, 0)
#     input_lyr = input_ds.GetLayer()
#     input_lyr.ResetReading()
#     for input_ftr in input_lyr:
#         input_fid = input_ftr.GetFID()
#         input_extent = drigo.Extent(
#             input_ftr.GetGeometryRef().GetEnvelope()).ogrenv_swap()
#         output_dict[input_fid] = input_extent
#     input_ds = None
#     return output_dict

# def feature_geometries(input_path):
#     """Return a dictionary of zone FIDs and their geometries"""
#     output_dict = dict()
#     # shp_driver = ogr.GetDriverByName('ESRI Shapefile')
#     input_ds = ogr.Open(input_path, 0)
#     input_lyr = input_ds.GetLayer()
#     input_lyr.ResetReading()
#     for input_ftr in input_lyr:
#         input_fid = input_ftr.GetFID()
#         input_geom = input_ftr.GetGeometryRef().ExportToWkt()
#         output_dict[input_fid] = input_geom
#     input_ds = None
#     return output_dict

# def feature_field_values(input_path, field='FID'):
#     """Return a dictionary of zone FIDs and their field values"""
#     output_dict = dict()
#     # shp_driver = ogr.GetDriverByName('ESRI Shapefile')
#     input_ds = ogr.Open(input_path, 0)
#     input_lyr = input_ds.GetLayer()
#     input_lyr.ResetReading()
#     for input_ftr in input_lyr:
#         input_fid = input_ftr.GetFID()
#         output_dict[input_fid] = input_ftr.GetField(field)
#     input_ds = None
#     return output_dict
