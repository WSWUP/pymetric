#!/usr/bin/env python
#--------------------------------
# Name:         interpolate_rasters_func.py
# Purpose:      Interpolate ETrF rasters between Landsat scenes based on DOY
#--------------------------------

from __future__ import division
import argparse
from builtins import input
from collections import defaultdict
import ctypes
import datetime as dt
import logging
from multiprocessing import Pool, Process, Queue, cpu_count, sharedctypes
import os
import random
import re
import shutil
import sys
from time import clock, sleep
import warnings

# Python 2/3
try:
    import pickle
except:
    import cPickle as pickle

import drigo
import numpy as np
from numpy import ctypeslib
from osgeo import gdal, ogr, osr

import et_common
import interpolate_support as interp
from python_common import open_ini, read_param, remove_file, parse_int_set

np.seterr(invalid='ignore')
gdal.UseExceptions()


def metric_interpolate(year_ws, ini_path, mc_iter=None, bs=None,
                       pyramids_flag=None, stats_flag=None,
                       overwrite_flag=None, mp_procs=1, delay=0,
                       debug_flag=False, output_queue=1):
    """METRIC Raster Interpolator

    Parameters
    ----------
    year_ws : str
        Year folder path.
    ini_path : str
        The config file (path).
    mc_iter : int
        Iteration number for Monte Carlo processing (the default is None).
    bs : int
        Processing block size (the default is None).
    pyramids_flag : bool
        If True, compute raster pyramids (the default is None).
    stats_flag : bool
        If True, compute raster statistics (the default is None).
    overwrite_flag : bool
        If True, overwrite existing files (the default is None).
    mp_procs : int
        Number of cores to use (the default is 1).
    delay : int
        Max random delay starting workers in seconds (the default is 0).
    debug_flag : bool
        If True, enable debug level logging (the default is False).
    output_queue : int
        Size of output queue (the default is 1).

    Returns
    -------
    None

    """
    logging.info('\nInterpolating ET rasters')
    log_fmt = '  {:<22s} {}'
    main_clock = clock()

    env = drigo.env

    # Use iteration number to file iteration string
    if mc_iter is None:
        mc_str = ''
        iter_fmt = '.img'
    elif int(mc_iter) < 0:
        logging.error('\nERROR: Iteration number must be a positive integer')
        return False
    else:
        logging.info(log_fmt.format('Iteration:', mc_iter))
        mc_str = 'MC{:02d}_'.format(int(mc_iter))
        iter_fmt = '_{:02d}.img'.format(int(mc_iter))

    # Open config file
    config = open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    year = config.getint('INPUTS', 'year')
    output_folder_name = config.get('INPUTS', 'folder_name')
    study_area_path = config.get('INPUTS', 'study_area_path')
    study_area_mask_flag = read_param('study_area_mask', False, config)
    study_area_buffer = read_param('study_area_buffer', 0.0, config)
    output_snap = read_param('study_area_snap', (0, 0), config)
    output_cs = read_param('study_area_cellsize', 30.0, config)
    output_proj = read_param('study_area_proj', None, config)
    output_ws = read_param('output_folder', None, config)
    etrf_input_ws = read_param('etrf_input_folder', None, config)
    etr_input_ws = config.get('INPUTS', 'etr_input_folder')
    etr_input_re = re.compile(config.get('INPUTS', 'etr_input_re'))
    ppt_input_ws = config.get('INPUTS', 'ppt_input_folder')
    ppt_input_re = re.compile(config.get('INPUTS', 'ppt_input_re'))
    footprint_path = config.get('INPUTS', 'footprint_path')
    # Raster paths should be defined with a foward slash ("/") in INI file
    # On windows, normpath will convert "/" to "\\"
    etrf_raster = os.path.normpath(read_param(
        'etrf_raster', os.path.join('ETRF', 'et_rf.img'), config))
    ndvi_raster = os.path.normpath(read_param(
        'ndvi_raster', os.path.join('INDICES', 'ndvi_toa.img'), config))
    tile_list = read_param('tile_list', [], config)
    use_landsat4_flag = read_param('use_landsat4_flag', False, config)
    use_landsat5_flag = read_param('use_landsat5_flag', False, config)
    use_landsat7_flag = read_param('use_landsat7_flag', False, config)
    use_landsat8_flag = read_param('use_landsat8_flag', False, config)
    fill_method = read_param('fill_method', 'linear', config).lower()
    interp_method = read_param('interp_method', 'linear', config).lower()
    mosaic_method = read_param('mosaic_method', 'mean', config).lower()
    tile_gcs_buffer = read_param('tile_buffer', 0.25, config)
    # doy_remove_list = read_param('doy_remove_list', [], config)

    # Read Monte Carlo iteration ETrF raster
    if mc_iter is not None:
        etrf_raster = os.path.splitext(etrf_raster)[0] + iter_fmt
        # etrf_raster = etrf_raster.replac('.img', iter_fmt)

    # If command line arguments aren't set, trying reading from input file
    if bs is None:
        bs = read_param('blocksize', 256, config)
    if pyramids_flag is None:
        pyramids_flag = read_param('pyramids_flag', False, config)
    if stats_flag is None:
        stats_flag = read_param('statistics_flag', False, config)

    # Output file/folder names
    ndvi_name = read_param('ndvi_name', 'NDVI', config)
    etrf_name = read_param('etrf_name', 'ETrF', config)
    etr_name = read_param('etr_name', 'ETr', config)
    et_name = read_param('et_name', 'ET', config)
    ppt_name = read_param('ppt_name', 'PPT', config)

    # Clamp/limit extreme ETrF values
    try:
        low_etrf_limit = config.getfloat('INPUTS', 'low_etrf_limit')
    except:
        low_etrf_limit = None
    try:
        high_etrf_limit = config.getfloat('INPUTS', 'high_etrf_limit')
    except:
        high_etrf_limit = None

    # Adjust ETrF based on daily soil water balance
    swb_adjust_dict = dict()
    swb_adjust_dict['flag'] = read_param('swb_adjust_flag', False, config)
    if swb_adjust_dict['flag']:
        swb_adjust_dict['awc'] = read_param('awc_input_path', None, config)
        swb_adjust_dict['spinup'] = read_param('swb_spinup_days', 30, config)
        swb_adjust_dict['ndvi_bare'] = read_param(
            'swb_bare_soil_ndvi', 0.15, config)
        swb_adjust_dict['ndvi_full'] = read_param(
            'swb_full_cover_ndvi', 0.7, config)
        # hot_cold_pixels = read_param(
        #     'hot_cold_pixels', os.path.join('PIXELS', 'hot_cold.shp'), config)
        # ndvi_threshold = read_param('ndvi_threshold', 0.7, config)

    # NDVI as surrogate for ETrF parameters
    etrf_ndvi_dict = dict()
    etrf_ndvi_dict['flag'] = read_param('etrf_ndvi_flag', False, config)
    if etrf_ndvi_dict['flag']:
        etrf_ndvi_dict['doy'] = sorted(list(parse_int_set(
            read_param('etrf_ndvi_doy_list', '', config))))
        etrf_ndvi_dict['month'] = sorted(list(parse_int_set(
            read_param('etrf_ndvi_month_list', '', config))))
        etrf_ndvi_dict['slope'] = read_param('etrf_ndvi_slope', 1.25, config)
        etrf_ndvi_dict['offset'] = read_param('etrf_ndvi_offset', 0., config)

    # Process control flags
    calc_flags = dict()

    # NDVI
    calc_flags['daily_ndvi'] = read_param(
        'calc_daily_ndvi_rasters_flag', False, config)
    calc_flags['monthly_ndvi'] = read_param(
        'calc_monthly_ndvi_rasters_flag', False, config)
    calc_flags['annual_ndvi'] = read_param(
        'calc_annual_ndvi_rasters_flag', False, config)
    # calc_flags['seasonal_ndvi'] = read_param(
    #     'calc_seasonal_ndvi_rasters_flag', False, config)

    # ETrF
    calc_flags['daily_etrf'] = read_param(
        'calc_daily_etrf_rasters_flag', False, config)
    calc_flags['monthly_etrf'] = read_param(
        'calc_monthly_etrf_rasters_flag', False, config)
    calc_flags['annual_etrf'] = read_param(
        'calc_annual_etrf_rasters_flag', False, config)
    # calc_flags['seasonal_etrf'] = read_param(
    #     'calc_seasonal_etrf_rasters_flag', False, config)

    # ETr
    calc_flags['daily_etr'] = read_param(
        'calc_daily_etr_rasters_flag', False, config)
    calc_flags['monthly_etr'] = read_param(
        'calc_monthly_etr_rasters_flag', False, config)
    calc_flags['annual_etr'] = read_param(
        'calc_annual_etr_rasters_flag', False, config)
    # calc_flags['seasonal_etr'] = read_param(
    #     'calc_seasonal_etr_rasters_flag', False, config)

    # ET
    calc_flags['daily_et'] = read_param(
        'calc_daily_et_rasters_flag', False, config)
    calc_flags['monthly_et'] = read_param(
        'calc_monthly_et_rasters_flag', True, config)
    calc_flags['annual_et'] = read_param(
        'calc_annual_et_rasters_flag', True, config)
    # calc_flags['seasonal_et'] = read_param(
    #     'calc_seasonal_et_rasters_flag', False, config)

    # # Counts
    calc_flags['monthly_count'] = read_param(
        'calc_monthly_count_rasters_flag', True, config)
    calc_flags['annual_count'] = read_param(
        'calc_annual_count_rasters_flag', True, config)
    # calc_flags['seasonal_count'] = read_param(
    #     'calc_seasonal_count_rasters_flag', False, config)

    # PPT
    calc_flags['daily_ppt'] = read_param(
        'calc_daily_ppt_rasters_flag', False, config)
    calc_flags['monthly_ppt'] = read_param(
        'calc_monthly_ppt_rasters_flag', False, config)
    calc_flags['annual_ppt'] = read_param(
        'calc_annual_ppt_rasters_flag', False, config)
    # calc_flags['seasonal_ppt'] = read_param(
    #     'calc_seasonal_ppt_rasters_flag', False, config)

    if not any(calc_flags.values()):
        logging.error('\nERROR: All calc flags are false, exiting\n')
        sys.exit()

    # These flags control which products will be computed
    # regardless of the time steps requested
    calc_flags['ndvi'] = False
    calc_flags['etrf'] = False
    calc_flags['etr'] = False
    calc_flags['et'] = False
    calc_flags['ppt'] = False
    # ETrF must be computed to get counts
    if (calc_flags['monthly_count'] or
            # calc_flags['seasonal_count'] or
            calc_flags['annual_count']):
        calc_flags['etrf'] = True

    if (calc_flags['daily_ndvi'] or
            calc_flags['monthly_ndvi'] or
            # calc_flags['seasonal_ndvi'] or
            calc_flags['annual_ndvi']):
        calc_flags['ndvi'] = True
    if (calc_flags['daily_etrf'] or
            calc_flags['monthly_etrf'] or
            # calc_flags['seasonal_etrf'] or
            calc_flags['annual_etrf']):
        calc_flags['etrf'] = True
    if (calc_flags['daily_etr'] or
            calc_flags['monthly_etr'] or
            # calc_flags['seasonal_etr'] or
            calc_flags['annual_etr']):
        calc_flags['etr'] = True
    if (calc_flags['daily_et'] or
            calc_flags['monthly_et'] or
            # calc_flags['seasonal_et'] or
            calc_flags['annual_et']):
        calc_flags['etrf'] = True
        calc_flags['etr'] = True
        calc_flags['et'] = True
    if (calc_flags['daily_ppt'] or
            calc_flags['monthly_ppt'] or
            # calc_flags['seasonal_ppt'] or
            calc_flags['annual_ppt']):
        calc_flags['ppt'] = True

    if etrf_ndvi_dict['flag']:
        calc_flags['ndvi'] = True
    if swb_adjust_dict['flag']:
        calc_flags['etr'] = True
        calc_flags['ndvi'] = True
        calc_flags['ppt'] = True

    # These flags control the structure of the data
    #   that is returned from the block worker function
    # Only one of these can be True
    # Start with daily and work up to annual
    calc_flags['daily'] = False
    calc_flags['monthly'] = False
    calc_flags['annual'] = False
    if (calc_flags['daily_ndvi'] or
            calc_flags['daily_etrf'] or
            calc_flags['daily_etr'] or
            calc_flags['daily_et'] or
            calc_flags['daily_ppt']):
        calc_flags['daily'] = True
    if (calc_flags['monthly_ndvi'] or
            calc_flags['monthly_etrf'] or
            calc_flags['monthly_etr'] or
            calc_flags['monthly_et'] or
            calc_flags['monthly_ppt'] or
            calc_flags['monthly_count']):
        calc_flags['monthly'] = True
    if (calc_flags['annual_ndvi'] or
            calc_flags['annual_etrf'] or
            calc_flags['annual_etr'] or
            calc_flags['annual_et'] or
            calc_flags['annual_ppt'] or
            calc_flags['annual_count']):
        calc_flags['annual'] = True
    # Seasonal output will either need monthly or daily data
    # if (calc_flags['seasonal_ndvi'] or
    #     calc_flags['seasonal_etrf'] or
    #     calc_flags['seasonal_etr'] or
    #     calc_flags['seasonal_et'] or
    #     calc_flags['seasonal_ppt'] or
    #     calc_flags['seasonal_count']):
    #     calc_flags['daily'] = True
    #     calc_flags['monthly'] = True

    # Force lists to be integers/floats
    output_snap = list(map(float, output_snap))
    # doy_remove_list = map(int, doy_remove_list)

    if not os.path.isfile(footprint_path):
        logging.error('\n\n  File {} does not exist'.format(footprint_path))
        sys.exit()

    # ETrF rasters can be read from a different folder
    if etrf_input_ws is None:
        etrf_input_ws = year_ws
    elif not os.path.isdir(etrf_input_ws):
        logging.error(('\nERROR: The ETrF input workspace does not exist:'
                       '\n  {}').format(etrf_input_ws))
        sys.exit()
    # else:
    #     logging.info('  {}'.format(etrf_input_ws))

    # Set fill interpolation method
    fill_method_list = ['nearest', 'linear', 'cubicspline']
    # fill_method_list = ['nearest', 'linear', 'cubicspline', 'spatial']
    if fill_method not in fill_method_list:
        logging.error(
            ('\nERROR: The fill_method {} is not a valid option.'
             '\nERROR: Set fill_method to {}').format(
                fill_method, fill_method_list))
        sys.exit()

    # Set temporal interpolation method
    interp_method_list = ['nearest', 'linear', 'cubicspline']
    if interp_method not in interp_method_list:
        logging.error(
            ('\nERROR: The interp_method {} is not a valid option.'
             '\nERROR: Set interp_method to {}').format(
                interp_method, interp_method_list))
        sys.exit()

    # Set mosaic method
    mosaic_method_list = ['mean']
    if mosaic_method not in mosaic_method_list:
        logging.error(
            ('\nERROR: The mosaic_method {} is not a valid option.'
             '\nERROR: Set mosaic_method to {}').format(
                mosaic_method, mosaic_method_list))
        sys.exit()

    # If a blocksize isn't set by user, set to 1024
    try:
        bs = int(bs)
    except (NameError, ValueError):
        bs = 1024

    # Print run properties
    logging.info(log_fmt.format('ETrF Workspace:', etrf_input_ws))
    logging.info(log_fmt.format('Output Workspace:', output_ws))
    logging.info(log_fmt.format('Fill:', fill_method))
    logging.info(log_fmt.format('Interpolation:', interp_method))
    logging.info(log_fmt.format('Mosaic Method:', mosaic_method))
    if low_etrf_limit is not None:
        logging.info(log_fmt.format(
            'Low {} limit:'.format(etrf_name), low_etrf_limit))
    if high_etrf_limit is not None:
        logging.info(log_fmt.format(
            'High {} limit:'.format(etrf_name), high_etrf_limit))
    logging.info(log_fmt.format('Use Landsat4 scenes:', use_landsat4_flag))
    logging.info(log_fmt.format('Use Landsat5 scenes:', use_landsat5_flag))
    logging.info(log_fmt.format('Use Landsat7 scenes:', use_landsat7_flag))
    logging.info(log_fmt.format('Use Landsat8 scenes:', use_landsat8_flag))
    logging.info(log_fmt.format('SWB adjust:', swb_adjust_dict['flag']))
    if swb_adjust_dict['flag']:
        logging.info(log_fmt.format(
            '  Spinup days:', swb_adjust_dict['spinup']))
        logging.info(log_fmt.format(
            '  Bare soil NDVI:', swb_adjust_dict['ndvi_bare']))
        logging.info(log_fmt.format(
            '  Full cover NDVI:', swb_adjust_dict['ndvi_full']))
    logging.info(log_fmt.format('ETrF from NDVI:', etrf_ndvi_dict['flag']))
    if swb_adjust_dict['flag']:
        logging.info(log_fmt.format(
            '  Months:', ', '.join(map(str, etrf_ndvi_dict['month']))))
        logging.info(log_fmt.format('  Slope:', etrf_ndvi_dict['slope']))
        logging.info(log_fmt.format('  Offset:', etrf_ndvi_dict['offset']))
    logging.info(log_fmt.format('Blocksize:', bs))
    logging.info(log_fmt.format('Processors:', mp_procs))
    logging.info(log_fmt.format('Queue:', output_queue))
    logging.info(log_fmt.format('Pyramids:', pyramids_flag))
    logging.info(log_fmt.format('Statistics:', stats_flag))
    logging.info(log_fmt.format('Overwrite:', overwrite_flag))

    # Create folders for output
    if output_ws:
        output_ws = os.path.join(output_ws, output_folder_name)
    else:
        output_ws = os.path.join(year_ws, output_folder_name)
    daily_ndvi_ws = os.path.join(
        output_ws, '{}{}_DAILY_{}'.format(
            mc_str, interp_method.upper(), ndvi_name.upper()))
    daily_etrf_ws = os.path.join(
        output_ws, '{}{}_DAILY_{}'.format(
            mc_str, interp_method.upper(), etrf_name.upper()))
    daily_etr_ws = os.path.join(
        output_ws, '{}{}_DAILY_{}'.format(
            mc_str, interp_method.upper(), etr_name.upper()))
    daily_et_ws = os.path.join(
        output_ws, '{}{}_DAILY_{}'.format(
            mc_str, interp_method.upper(), et_name.upper()))
    daily_ppt_ws = os.path.join(
        output_ws, '{}{}_DAILY_{}'.format(
            mc_str, interp_method.upper(), ppt_name.upper()))
    monthly_ws = os.path.join(
        output_ws, '{}{}_MONTHLY'.format(mc_str, interp_method.upper()))
    annual_ws = os.path.join(
        output_ws, '{}{}_ANNUAL'.format(mc_str, interp_method.upper()))
    # seasonal_ws = os.path.join(
    #     output_ws, '{}{}_SEASONAL'.format(mc_str, interp_method.upper()))
    if not os.path.isdir(output_ws):
        os.makedirs(output_ws)
    if calc_flags['daily_ndvi'] and not os.path.isdir(daily_ndvi_ws):
        os.mkdir(daily_ndvi_ws)
    if calc_flags['daily_etrf'] and not os.path.isdir(daily_etrf_ws):
        os.mkdir(daily_etrf_ws)
    if calc_flags['daily_etr'] and not os.path.isdir(daily_etr_ws):
        os.mkdir(daily_etr_ws)
    if calc_flags['daily_et'] and not os.path.isdir(daily_et_ws):
        os.mkdir(daily_et_ws)
    if calc_flags['daily_ppt'] and not os.path.isdir(daily_ppt_ws):
        os.mkdir(daily_ppt_ws)
    if calc_flags['monthly'] and not os.path.isdir(monthly_ws):
        os.mkdir(monthly_ws)
    if calc_flags['annual'] and not os.path.isdir(annual_ws):
        os.mkdir(annual_ws)
    # if ((calc_flags['seasonal_etrf'] or
    #      calc_flags['seasonal_etr'] or
    #      calc_flags['seasonal_et'] or
    #      calc_flags['seasonal_count']) and
    #     not os.path.isdir(seasonal_ws)):
    #     os.mkdir(seasonal_ws)

    # Remove folders (and files) that aren't being calculated
    if not calc_flags['daily_etrf'] and os.path.isdir(daily_etrf_ws):
        shutil.rmtree(daily_etrf_ws)
    if not calc_flags['daily_etr'] and os.path.isdir(daily_etr_ws):
        shutil.rmtree(daily_etr_ws)
    if not calc_flags['daily_et'] and os.path.isdir(daily_et_ws):
        shutil.rmtree(daily_et_ws)
    if not calc_flags['daily_ppt'] and os.path.isdir(daily_ppt_ws):
        shutil.rmtree(daily_ppt_ws)
    if not calc_flags['monthly'] and os.path.isdir(monthly_ws):
        shutil.rmtree(monthly_ws)
    if not calc_flags['annual'] and os.path.isdir(annual_ws):
        shutil.rmtree(annual_ws)
    # if not calc_flags['seasonal'] and os.path.isdir(seasonal_ws):
    #     shutil.rmtree(seasonal_ws)

    # Regular expressions for the unmerged and merged directories
    tile_re = re.compile('p(\d{3})r(\d{3})')
    image_id_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(?:\w{4})_(\d{3})(\d{3})_'
        '(\d{4})(\d{2})(\d{2})_(?:\d{8})_(?:\d{2})_(?:\w{2})$')
    # hot_pixel_re = re.compile(
    #     'HOT_(?P<mc_iter>{0:02d})_(?P<cal_iter>\d{{2}})'.format(mc_iter))
    # cold_pixel_re = re.compile(
    #     'COLD_(?P<mc_iter>{0:02d})_(?P<cal_iter>\d{{2}})'.format(mc_iter))

    # Path/rows
    # First read in all years and path/rows in the input workspace
    # Currently year is an input and script will only process 1 year
    # Eventually this could support reading multiple years
    year_tile_list = [
        [year, tile_name]
        for tile_name in sorted(os.listdir(etrf_input_ws))
        if (os.path.isdir(os.path.join(etrf_input_ws, tile_name)) and
            tile_re.match(tile_name))]

    # Eventually this should be modified so that the interpolator can
    # be seamlessly run in the project, year, or path/row folder
    if not year_tile_list:
        logging.error(
            '\nERROR: No path/row tiles were found.\n  Check that the '
            '"workspace" positional command line argument is set to '
            'a year folder.')
        sys.exit()

    # If a path/row list was set, filter the path_row list
    if tile_list:
        year_tile_list = [
            [tile_year, tile_name]
            for tile_year, tile_name in year_tile_list
            if tile_name in tile_list]

    # Get scene lists for each year and path/row
    tile_image_dict = defaultdict(dict)
    for year, tile_name in year_tile_list:
        # Input workspace currently is only for 1 year
        # etrf_input_ws = os.path.join(etrf_input_ws, year)
        tile_ws = os.path.join(etrf_input_ws, tile_name)
        image_id_list = [
            image_id for image_id in sorted(os.listdir(tile_ws))
            if (image_id_re.match(image_id) and
                os.path.isdir(os.path.join(tile_ws, image_id)))]
        #         (image_keep_list and image_id in image_keep_list))]
        image_id_list = [
            image_id for image_id in image_id_list
            if ((use_landsat4_flag and image_id[:4] == 'LT04') or
                (use_landsat5_flag and image_id[:4] == 'LT05') or
                (use_landsat7_flag and image_id[:4] == 'LE07') or
                (use_landsat8_flag and image_id[:4] == 'LC08'))]
        if not image_id_list:
            continue
        tile_image_dict[year][tile_name] = image_id_list
        del image_id_list, tile_ws

    # For now interpolate entire year
    # Eventually let user control start and end date
    date_list = list(
        interp.daterange_func(dt.date(year, 1, 1), dt.date(year, 12, 31), 1))
    date_str_list = [date.strftime('%Y_%m_%d') for date in date_list]
    doy_str_list = [date.strftime('%Y_%j') for date in date_list]
    month_str_list = sorted(list(set([
        date.strftime('%Y_%m') for date in date_list])))
    year_str_list = sorted(list(set([
        date.strftime('%Y') for date in date_list])))

    # Add additional start/end dates for CUBICSPINE
    if fill_method == 'cubicspline' or interp_method == 'cubicspline':
        date_list.insert(
            0, (dt.datetime.strptime(date_list[0], '%Y%m%d') -
                dt.timedelta(days=1)).strftime('%Y%m%d'))
        date_list.append(
            (dt.datetime.strptime(date_list[-1], '%Y%m%d') +
                dt.timedelta(days=1)).strftime('%Y%m%d'))

    # Need extra ETr and PPT days to spinup SWB
    if swb_adjust_dict['flag'] and swb_adjust_dict['spinup'] > 0:
        etr_date_list = list(interp.daterange_func(
            date_list[0] - dt.timedelta(days=swb_adjust_dict['spinup']),
            date_list[-1]))
        ppt_date_list = etr_date_list[:]
    else:
        etr_date_list = date_list[:]
        ppt_date_list = date_list[:]

    # Study Area
    logging.info('\nStudy Area')
    if not os.path.isfile(study_area_path):
        logging.error(('\nERROR: The study area shapefile does not exist:'
                       '\n  {}').format(study_area_path))
        sys.exit()
    else:
        logging.info('  {}'.format(study_area_path))
    study_area_osr = drigo.feature_path_osr(study_area_path)

    # Get output projection from 1) study_area_proj, 2) study_area_path
    if output_proj:
        logging.debug('  Projection from study_area_proj')
        output_osr = interp.unknown_proj_osr(output_proj)
        if output_osr is None:
            logging.error(
                ('\nERROR: The study_area_proj string could not be '
                 'converted to a spatial reference \n  {}').format(
                    output_proj))
            sys.exit()
    else:
        logging.debug('  Projection from study_area_path')
        output_osr = study_area_osr

    # Zones spatial reference must be a projected coordinate system
    output_gcs_osr = output_osr.CloneGeogCS()
    if ((drigo.osr_proj4(output_osr) == drigo.osr_proj4(output_gcs_osr)) or
            (str(output_osr) == str(output_gcs_osr))):
        logging.warning('  OSR: {}'.format(output_osr))
        logging.warning('  GCS: {}'.format(output_gcs_osr))
        logging.warning('  Cellsize: {}'.format(output_cs))
        logging.warning(
            '\nWARNING:\n' +
            '  The study area shapefile appears to be in a geographic '
            'coordinate system\n' +
            '    (units in decimal degrees)\n' +
            '  It is recommended to use a shapefile with a projected '
            'coordinate system\n' +
            '    (units in meters or feet)\n' +
            '  Before continuing, please ensure the cellsize is in '
            'decimal degrees')
        input('Press ENTER to continue')
        if output_cs >= 1:
            logging.error('\nERROR: The output cellsize is too large, exiting')
            sys.exit()

    logging.info('\nSpatial Reference')
    env.snap_osr = output_osr
    env.snap_gcs_osr = output_gcs_osr
    env.snap_proj = env.snap_osr.ExportToWkt()
    env.snap_gcs_proj = env.snap_gcs_osr.ExportToWkt()
    env.cellsize = output_cs
    env.snap_x, env.snap_y = output_snap
    logging.debug('  Cellsize: {}'.format(env.cellsize))
    logging.debug('  Snap: {} {}'.format(env.snap_x, env.snap_y))
    logging.debug('  OSR: {}'.format(env.snap_osr))
    logging.debug('  GCS: {}'.format(env.snap_gcs_osr))

    # Use study area to set mask properties
    study_area_extent = drigo.path_extent(study_area_path)
    # DEADBEEF - Need a clean way to get the cellsize of study_area_path
    #   If it was GCS, the Landsat 30m cellsize doens't make sense.
    env.mask_extent = drigo.project_extent(
        study_area_extent, study_area_osr, env.snap_osr)
    # DEADBEEF - study area buffer is only being applied to the extent
    #   This is simple, but would a user expect this or would they expecct
    #   the study area polygon to be buffered
    #   The other approach would be to dilate/erode the study area mask
    env.mask_extent.buffer_extent(study_area_buffer)
    env.mask_extent.adjust_to_snap('EXPAND')
    env.mask_geo = env.mask_extent.geo(env.cellsize)
    env.mask_rows, env.mask_cols = env.mask_extent.shape()
    env.mask_shape = (env.mask_rows, env.mask_cols)
    logging.debug('  Mask rows: {}  cols: {}'.format(
        env.mask_rows, env.mask_cols))
    logging.debug('  Mask extent: {}'.format(env.mask_extent))
    logging.debug('  Mask geo: {}'.format(env.mask_geo))

    # Build a mask array from the study area shapefile
    if study_area_mask_flag:
        study_area_mask_ds = drigo.polygon_to_raster_ds(
            study_area_path, nodata_value=0, burn_value=1,
            output_osr=env.snap_osr, output_cs=env.cellsize,
            output_extent=env.mask_extent)
        study_area_mask = drigo.raster_ds_to_array(
            study_area_mask_ds, return_nodata=False)
        study_area_mask_ds = None
    else:
        study_area_mask = np.full(env.mask_shape, True, dtype=np.bool)

    # ETr
    if calc_flags['etr']:
        etr_array, etr_osr, etr_cs, etr_extent = interp.load_year_array_func(
            etr_input_ws, etr_input_re, etr_date_list,
            env.snap_osr, env.cellsize, env.mask_extent,
            etr_name, return_geo_array=True)
        if np.all(np.isnan(etr_array)):
            logging.error(
                '\nERROR: The Reference ET array is all nodata, exiting\n')
            sys.exit()
    else:
        logging.debug('ETr - empty array')
        etr_osr = drigo.epsg_osr(4269)
        etr_cs = 0.125
        etr_extent = drigo.project_extent(
            env.mask_extent, env.snap_osr, etr_osr, env.cellsize)
        etr_extent.adjust_to_snap('EXPAND', 0, 0, etr_cs)
        etr_rows, etr_cols = etr_extent.shape(cs=etr_cs)
        etr_array = np.full(
            (len(etr_date_list), etr_rows, etr_cols), np.nan, np.float32)

    # PPT
    if calc_flags['ppt']:
        ppt_array, ppt_osr, ppt_cs, ppt_extent = interp.load_year_array_func(
            ppt_input_ws, ppt_input_re, ppt_date_list,
            env.snap_osr, env.cellsize, env.mask_extent,
            ppt_name, return_geo_array=True)
        if np.all(np.isnan(ppt_array)):
            logging.error(
                '\nERROR: The precipitation array is all nodata, exiting\n')
            sys.exit()
    else:
        logging.debug('PPT - empty array')
        ppt_osr = drigo.epsg_osr(4269)
        ppt_cs = 0.125
        ppt_extent = drigo.project_extent(
            env.mask_extent, env.snap_osr, ppt_osr, env.cellsize)
        ppt_extent.adjust_to_snap('EXPAND', 0, 0, ppt_cs)
        ppt_rows, ppt_cols = ppt_extent.shape(cs=ppt_cs)
        ppt_array = np.full(
            (len(ppt_date_list), ppt_rows, ppt_cols), np.nan, np.float32)

    # AWC
    if swb_adjust_dict['flag']:
        awc_ds = gdal.Open(swb_adjust_dict['awc'], 0)
        awc_osr = drigo.raster_ds_osr(awc_ds)
        awc_cs = drigo.raster_ds_cellsize(awc_ds, x_only=True)
        awc_x, awc_y = drigo.raster_ds_origin(awc_ds)
        awc_extent = drigo.project_extent(
            env.mask_extent, env.snap_osr, awc_osr, env.cellsize)
        awc_extent.adjust_to_snap('EXPAND', awc_x, awc_y, awc_cs)
        awc_array = drigo.raster_ds_to_array(
            awc_ds, 1, awc_extent, return_nodata=False)
        awc_ds = None
    else:
        awc_osr = drigo.epsg_osr(4269)
        awc_cs = 0.125
        awc_extent = drigo.project_extent(
            env.mask_extent, env.snap_osr, awc_osr, env.cellsize)
        awc_extent.adjust_to_snap('EXPAND', 0, 0, awc_cs)
        awc_rows, awc_cols = awc_extent.shape(cs=awc_cs)
        awc_array = np.full((awc_rows, awc_cols), np.nan, np.float32)

    # DEADBEEF - Working implementation of a shared memory ETr/PPT arrays
    # Set nan to the nodata value to avoid needing to make a copy while projecting
    # etr_array[np.isnan(etr_array)] = drigo.numpy_type_nodata(etr_array.dtype)
    # ppt_array[np.isnan(ppt_array)] = drigo.numpy_type_nodata(ppt_array.dtype)
    # awc_array[np.isnan(awc_array)] = drigo.numpy_type_nodata(awc_array.dtype)
    # Replace ETr array with a shared memory version
    etr_shape = etr_array.shape
    etr_ctypes = sharedctypes.RawArray(ctypes.c_float, etr_array.flat)
    etr_shmem = np.frombuffer(
        etr_ctypes, dtype=np.float32, count=etr_array.size)
    etr_shmem = etr_array

    # Replace PPT array with a shared memory version
    ppt_shape = ppt_array.shape
    ppt_ctypes = sharedctypes.RawArray(ctypes.c_float, ppt_array.flat)
    ppt_shmem = np.frombuffer(
        ppt_ctypes, dtype=np.float32, count=ppt_array.size)
    ppt_shmem = ppt_array

    # Replace AWX array with a shared memory version
    awc_shape = awc_array.shape
    awc_ctypes = sharedctypes.RawArray(ctypes.c_float, awc_array.flat)
    awc_shmem = np.frombuffer(
        awc_ctypes, dtype=np.float32, count=awc_array.size)
    awc_shmem = awc_array

    # Footprint (WRS2 Descending Polygons)
    logging.info('\nFootprints')
    logging.debug('\nFootprint (WRS2 descending should be GCS84):')
    if not os.path.isfile(footprint_path):
        logging.error(('\nERROR: The footprint shapefile does not exist:'
                       '\n  {}').format(footprint_path))
        sys.exit()
    tile_gcs_osr = drigo.feature_path_osr(footprint_path)
    logging.debug('  {}'.format(tile_gcs_osr))

    # Doublecheck that WRS2 descending shapefile is GCS84
    # if tile_gcs_osr != epsg_osr(4326):
    #     logging.error('  WRS2 is not GCS84')
    #     sys.exit()

    # Get geometry for each path/row
    tile_gcs_wkt_dict = interp.tile_wkt_func(
        footprint_path, path_field='PATH', row_field='ROW')

    # Get list of all intersecting Landsat path/rows
    tile_proj_wkt_dict = dict()
    for tile_name, tile_gcs_wkt in tile_gcs_wkt_dict.items():
        tile_gcs_geom = ogr.CreateGeometryFromWkt(tile_gcs_wkt)
        # Transform path/row from GCS to study area projected CS
        tile_proj_tx = osr.CoordinateTransformation(
            tile_gcs_osr, env.snap_osr)
        tile_gcs_geom = ogr.CreateGeometryFromWkt(
            tile_gcs_wkt_dict[tile_name])
        # Apply a small buffer (in decimal degrees)
        # DEADBEEF - Buffer fails if GDAL is not built with GEOS support
        try:
            # Project the path/row geometry
            tile_proj_geom = tile_gcs_geom.Clone()
            tile_proj_geom = tile_proj_geom.Buffer(tile_gcs_buffer)
        except:
            logging.error(
                '  GDAL does not appear to be built with GEOS support\n'
                '  Using tile extents instead')
        if tile_proj_geom is None:
            # Project the path/row extent
            tile_gcs_extent = drigo.Extent(tile_gcs_geom.GetEnvelope())
            tile_gcs_extent = tile_gcs_extent.ogrenv_swap()
            tile_gcs_extent.buffer_extent(tile_gcs_buffer)
            tile_proj_geom = tile_gcs_extent.geometry()
        tile_proj_geom.Transform(tile_proj_tx)
        tile_proj_wkt_dict[tile_name] = tile_proj_geom.ExportToWkt()

    # Raster path format
    daily_ndvi_fmt = os.path.join(daily_et_ws, 'ndvi_{}.img')
    daily_etrf_fmt = os.path.join(daily_etrf_ws, 'etrf_{}.img')
    daily_etr_fmt = os.path.join(daily_etr_ws, 'etr_{}.img')
    daily_et_fmt = os.path.join(daily_et_ws, 'et_{}.img')
    daily_ppt_fmt = os.path.join(daily_ppt_ws, 'ppt_{}.img')

    monthly_ndvi_fmt = os.path.join(monthly_ws, 'ndvi_{}.img')
    monthly_etrf_fmt = os.path.join(monthly_ws, 'etrf_{}.img')
    monthly_etr_fmt = os.path.join(monthly_ws, 'etr_{}.img')
    monthly_et_fmt = os.path.join(monthly_ws, 'et_{}.img')
    monthly_ppt_fmt = os.path.join(monthly_ws, 'ppt_{}.img')
    monthly_count_fmt = os.path.join(monthly_ws, 'count_{}.img')

    # seasonal_ndvi_fmt = os.path.join(seasonal_ws, 'ndvi_season_{}.img')
    # seasonal_etrf_fmt = os.path.join(seasonal_ws, 'etrf_season_{}.img')
    # seasonal_etr_fmt = os.path.join(seasonal_ws, 'etr_season_{}.img')
    # seasonal_et_fmt = os.path.join(seasonal_ws, 'et_season_{}.img')
    # seasonal_ppt_fmt = os.path.join(seasonal_ws, 'ppt_season_{}.img')
    # seasonal_count_fmt = os.path.join(seasonal_ws, 'count_season_{}.img')

    annual_ndvi_fmt = os.path.join(annual_ws, 'ndvi_annual_{}.img')
    annual_etrf_fmt = os.path.join(annual_ws, 'etrf_annual_{}.img')
    annual_etr_fmt = os.path.join(annual_ws, 'etr_annual_{}.img')
    annual_et_fmt = os.path.join(annual_ws, 'et_annual_{}.img')
    annual_ppt_fmt = os.path.join(annual_ws, 'ppt_annual_{}.img')
    annual_count_fmt = os.path.join(annual_ws, 'count_annual_{}.img')

    # Build empty output rasters
    def build_raster_list_func(ws, raster_fmt, date_str_list, overwrite_flag):
        existing_list = [os.path.join(ws, i) for i in os.listdir(ws)]
        return [
            raster_fmt.format(date_str) for date_str in date_str_list
            if (raster_fmt.format(date_str) not in existing_list or
                overwrite_flag)]

    logging.info('\nBuild empty output rasters')
    # Daily
    build_raster_list = []
    if calc_flags['daily_ndvi']:
        build_raster_list.extend(build_raster_list_func(
            daily_ndvi_ws, daily_ndvi_fmt, doy_str_list, overwrite_flag))
    if calc_flags['daily_etrf']:
        build_raster_list.extend(build_raster_list_func(
            daily_etrf_ws, daily_etrf_fmt, doy_str_list, overwrite_flag))
    if calc_flags['daily_etr']:
        build_raster_list.extend(build_raster_list_func(
            daily_etr_ws, daily_etr_fmt, doy_str_list, overwrite_flag))
    if calc_flags['daily_et']:
        build_raster_list.extend(build_raster_list_func(
            daily_et_ws, daily_et_fmt, doy_str_list, overwrite_flag))
    if calc_flags['daily_ppt']:
        build_raster_list.extend(build_raster_list_func(
            daily_ppt_ws, daily_ppt_fmt, doy_str_list, overwrite_flag))
    # Monthly
    if calc_flags['monthly_ndvi']:
        build_raster_list.extend(build_raster_list_func(
            monthly_ws, monthly_ndvi_fmt, month_str_list, overwrite_flag))
    if calc_flags['monthly_etrf']:
        build_raster_list.extend(build_raster_list_func(
            monthly_ws, monthly_etrf_fmt, month_str_list, overwrite_flag))
    if calc_flags['monthly_etr']:
        build_raster_list.extend(build_raster_list_func(
            monthly_ws, monthly_etr_fmt, month_str_list, overwrite_flag))
    if calc_flags['monthly_et']:
        build_raster_list.extend(build_raster_list_func(
            monthly_ws, monthly_et_fmt, month_str_list, overwrite_flag))
    if calc_flags['monthly_ppt']:
        build_raster_list.extend(build_raster_list_func(
            monthly_ws, monthly_ppt_fmt, month_str_list, overwrite_flag))
    # Annual
    if calc_flags['annual_ndvi']:
        build_raster_list.extend(build_raster_list_func(
            annual_ws, annual_ndvi_fmt, year_str_list, overwrite_flag))
    if calc_flags['annual_etrf']:
        build_raster_list.extend(build_raster_list_func(
            annual_ws, annual_etrf_fmt, year_str_list, overwrite_flag))
    if calc_flags['annual_etr']:
        build_raster_list.extend(build_raster_list_func(
            annual_ws, annual_etr_fmt, year_str_list, overwrite_flag))
    if calc_flags['annual_et']:
        build_raster_list.extend(build_raster_list_func(
            annual_ws, annual_et_fmt, year_str_list, overwrite_flag))
    if calc_flags['annual_ppt']:
        build_raster_list.extend(build_raster_list_func(
            annual_ws, annual_ppt_fmt, year_str_list, overwrite_flag))
    # # Seasonal
    # if calc_flags['seasonal_ndvi']:
    #     build_raster_list.extend(build_raster_list_func(
    #         seasonal_ws, seasonal_ndvi_fmt, year_str_list, overwrite_flag))
    # if calc_flags['seasonal_etrf']:
    #     build_raster_list.extend(build_raster_list_func(
    #         seasonal_ws, seasonal_etrf_fmt, year_str_list, overwrite_flag))
    # if calc_flags['seasonal_etr']:
    #     build_raster_list.extend(build_raster_list_func(
    #         seasonal_ws, seasonal_etr_fmt, year_str_list, overwrite_flag))
    # if calc_flags['seasonal_et']:
    #     build_raster_list.extend(build_raster_list_func(
    #         seasonal_ws, seasonal_et_fmt, year_str_list, overwrite_flag))
    # if calc_flags['seasonal_ppt']:
    #     build_raster_list.extend(build_raster_list_func(
    #         seasonal_ws, seasonal_ppt_fmt, year_str_list, overwrite_flag))

    # NDVI, ETrF, ETr, and ET rasters are float32 type
    mp_list = []
    for build_raster_path in build_raster_list:
        if mp_procs > 1:
            mp_list.append([
                build_raster_path, 1, np.float32,
                float(np.finfo(np.float32).min), env.snap_proj, env.cellsize,
                env.mask_extent, True])
        else:
            logging.debug('  {}'.format(build_raster_path))
            drigo.build_empty_raster(
                build_raster_path, band_cnt=1, output_dtype=np.float32,
                output_nodata=float(np.finfo(np.float32).min),
                output_proj=env.snap_proj, output_cs=env.cellsize,
                output_extent=env.mask_extent,
                output_fill_flag=True)
        # drigo.build_empty_raster(
        #     build_raster_path, band_cnt=1, output_dtype=np.float32,
        #     output_nodata=float(np.finfo(np.float32).min),
        #     output_proj=env.snap_proj, output_cs=env.cellsize,
        #     output_extent=env.mask_extent,
        #     output_fill_flag=True)

    # Count rasters are integer type
    build_raster_list, remove_raster_list = [], []
    if calc_flags['monthly_count']:
        build_raster_list.extend(build_raster_list_func(
            monthly_ws, monthly_count_fmt, month_str_list, overwrite_flag))
    if calc_flags['annual_count']:
        build_raster_list.extend(build_raster_list_func(
            annual_ws, annual_count_fmt, year_str_list, overwrite_flag))
    # if calc_flags['seasonal_count']:
    #     build_raster_list.extend(build_raster_list_func(
    #         seasonal_ws, seasonal_count_fmt, year_str_list, overwrite_flag))
    for build_raster_path in build_raster_list:
        if mp_procs > 1:
            mp_list.append([
                build_raster_path, 1, np.uint8, 0, env.snap_proj,
                env.cellsize, env.mask_extent, True])
        else:
            logging.debug('  {}'.format(build_raster_path))
            drigo.build_empty_raster(
                build_raster_path, band_cnt=1, output_dtype=np.uint8,
                output_nodata=0, output_proj=env.snap_proj,
                output_cs=env.cellsize, output_extent=env.mask_extent,
                output_fill_flag=True)
        # drigo.build_empty_raster(
        #     build_raster_path, band_cnt=1, output_dtype=np.uint8,
        #     output_nodata=0, output_proj=env.snap_proj,
        #     output_cs=env.cellsize, output_extent=env.mask_extent,
        #     output_fill_flag=True)

    # Build rasters using multiprocessing
    if mp_list:
        pool = Pool()
        results = pool.map(drigo.build_empty_raster_mp, mp_list, chunksize=1)
        pool.close()
        pool.join()
        del results, pool, mp_list

    # Remove existing rasters
    def remove_raster_list_func(ws, raster_fmt, date_str_list):
        """"""
        existing_list = [os.path.join(ws, i) for i in os.listdir(ws)]
        return [raster_fmt.format(date_str) for date_str in date_str_list
                if (raster_fmt.format(date_str) in existing_list)]

    remove_raster_list = []
    if not calc_flags['monthly_ndvi'] and os.path.isdir(monthly_ws):
        remove_raster_list.extend(remove_raster_list_func(
            monthly_ws, monthly_ndvi_fmt, doy_str_list))
    if not calc_flags['monthly_etrf'] and os.path.isdir(monthly_ws):
        remove_raster_list.extend(remove_raster_list_func(
            monthly_ws, monthly_etr_fmt, doy_str_list))
    if not calc_flags['monthly_etr'] and os.path.isdir(monthly_ws):
        remove_raster_list.extend(remove_raster_list_func(
            monthly_ws, monthly_etr_fmt, doy_str_list))
    if not calc_flags['monthly_et'] and os.path.isdir(monthly_ws):
        remove_raster_list.extend(remove_raster_list_func(
            monthly_ws, monthly_et_fmt, doy_str_list))
    if not calc_flags['monthly_count'] and os.path.isdir(monthly_ws):
        remove_raster_list.extend(remove_raster_list_func(
            monthly_ws, monthly_count_fmt, month_str_list))
    if not calc_flags['annual_ndvi'] and os.path.isdir(annual_ws):
        remove_raster_list.extend(remove_raster_list_func(
            annual_ws, annual_ndvi_fmt, doy_str_list))
    if not calc_flags['annual_etrf'] and os.path.isdir(annual_ws):
        remove_raster_list.extend(remove_raster_list_func(
            annual_ws, annual_etr_fmt, doy_str_list))
    if not calc_flags['annual_etr'] and os.path.isdir(annual_ws):
        remove_raster_list.extend(remove_raster_list_func(
            annual_ws, annual_etr_fmt, doy_str_list))
    if not calc_flags['annual_et'] and os.path.isdir(annual_ws):
        remove_raster_list.extend(remove_raster_list_func(
            annual_ws, annual_et_fmt, doy_str_list))
    if not calc_flags['annual_count'] and os.path.isdir(annual_ws):
        remove_raster_list.extend(remove_raster_list_func(
            annual_ws, annual_count_fmt, year_str_list))
    # if not calc_flags['seasonal_ndvi'] and os.path.isdir(seasonal_ws):
    #     remove_raster_list.extend(remove_raster_list_func(
    #         seasonal_ws, seasonal_ndvi_fmt, doy_str_list))
    # if not calc_flags['seasonal_etrf'] and os.path.isdir(seasonal_ws):
    #     remove_raster_list.extend(remove_raster_list_func(
    #         seasonal_ws, seasonal_etrf_fmt, doy_str_list))
    # if not calc_flags['seasonal_etr'] and os.path.isdir(seasonal_ws):
    #     remove_raster_list.extend(remove_raster_list_func(
    #         seasonal_ws, seasonal_etr_fmt, doy_str_list))
    # if not calc_flags['seasonal_et'] and os.path.isdir(seasonal_ws):
    #     remove_raster_list.extend(remove_raster_list_func(
    #         seasonal_ws, seasonal_et_fmt, doy_str_list))
    # if not calc_flags['seasonal_count'] and os.path.isdir(seasonal_ws):
    #     remove_raster_list.extend(remove_raster_list_func(
    #         seasonal_ws, seasonal_count_fmt, year_str_list))
    for remove_raster_path in remove_raster_list:
        remove_file(remove_raster_path)

    # Initialize queues
    input_q = Queue()
    # output_q = Queue(mp_procs + 1)
    output_q = Queue(output_queue)
    queue_cnt = 0

    # Load each block into queue
    logging.debug('\nGenerating block tasks')
    logging.debug('  Mask cols/rows: {}/{}'.format(
        env.mask_cols, env.mask_rows))
    for b_i, b_j in drigo.block_gen(env.mask_rows, env.mask_cols, bs):
        # logging.debug('Block  y: {:5d}  x: {:5d}'.format(b_i, b_j))
        block_geo = drigo.array_offset_geo(env.mask_geo, b_j, b_i)
        block_extent = drigo.geo_extent(block_geo, bs, bs)
        block_extent = drigo.intersect_extents(
            [block_extent, env.mask_extent])
        block_x, block_y = block_extent.origin()
        block_rows, block_cols = block_extent.shape(env.cellsize)
        # logging.debug('  Block rows: {}  cols: {}'.format(
        #     block_rows, block_cols))
        # logging.debug('  Block geo: {}'.format(block_geo))
        # logging.debug('  Block extent: {}'.format(block_extent))

        # Determine which path/rows to read
        block_tile_list = []
        for tile_name in sorted(tile_list):
            tile_proj_geom = ogr.CreateGeometryFromWkt(
                tile_proj_wkt_dict[tile_name])
            if tile_proj_geom.Intersects(block_extent.geometry()):
                block_tile_list.append(tile_name)
        if not block_tile_list:
            continue

        if study_area_mask_flag:
            block_data_mask = drigo.array_to_block(
                study_area_mask, b_i, b_j, bs).astype(np.bool)
            if not np.any(block_data_mask):
                del block_data_mask
                continue
            else:
                del block_data_mask

        # Place inputs into queue by block
        usable_scene_cnt = 2
        input_q.put([
            b_i, b_j, block_rows, block_cols, block_extent, block_tile_list,
            date_list, etr_date_list, ppt_date_list,
            year, etrf_input_ws, tile_image_dict, env.cellsize, env.snap_proj,
            # etr_array, drigo.osr_proj(etr_osr), etr_cs, etr_extent,
            # ppt_array, drigo.osr_proj(ppt_osr), ppt_cs, ppt_extent,
            # awc_array, drigo.osr_proj(awc_osr), awc_cs, awc_extent,
            etr_shmem, etr_shape, drigo.osr_proj(etr_osr), etr_cs, etr_extent,
            ppt_shmem, ppt_shape, drigo.osr_proj(ppt_osr), ppt_cs, ppt_extent,
            awc_shmem, awc_shape, drigo.osr_proj(awc_osr), awc_cs, awc_extent,
            etrf_raster, ndvi_raster, swb_adjust_dict, etrf_ndvi_dict,
            study_area_mask_flag, study_area_path,
            usable_scene_cnt, mosaic_method, fill_method, interp_method,
            calc_flags, low_etrf_limit, high_etrf_limit, debug_flag])
        queue_cnt += 1
        #break
        # if queue_cnt >= 16:
        #     break
    if queue_cnt == 0:
        logging.error(
            '\nERROR: No blocks were loaded into the queue, exiting')
        return False
    else:
        logging.debug('  {} blocks'.format(queue_cnt))
    if study_area_mask_flag:
        del study_area_mask

    # Start processing
    logging.info('\nProcessing by block')
    # Leave one processer for writing
    for mp_i in range(max(1, mp_procs - 1)):
        Process(target=block_worker, args=(mp_i, input_q, output_q)).start()
        sleep(random.uniform(0, max([0, delay])))
        # sleep(1)

    # Don't start timer until output has something in it
    logging.info('  waiting for output queue to intiailize')
    while output_q.empty():
        pass

    proc_clock = clock()
    for queue_i in range(queue_cnt):
        block_clock = clock()

        b_i, b_j, count, ndvi, et, etr, ppt = output_q.get()
        logging.info('Block  y: {:5d}  x: {:5d}  ({}/{})'.format(
            b_i, b_j, queue_i + 1, queue_cnt))
        count_array = pickle.loads(count)
        ndvi_array = pickle.loads(ndvi)
        et_array = pickle.loads(et)
        etr_array = pickle.loads(etr)
        ppt_array = pickle.loads(ppt)
        del count, et, etr, ndvi, ppt

        # Block mask could come from ETrF/image counts
        # block_ndvi_mask = np.any(count_array, axis=0)
        block_et_mask = np.any(count_array, axis=0)
        # block_et_mask = np.sum(count_array, axis=0) > 2
        block_etr_mask = np.any(etr_array, axis=0)
        block_ppt_mask = np.any(ppt_array, axis=0)

        # Write DOY, monthly, and annual, ETrF, ETr, and ET arrays to raster
        if calc_flags['daily']:
            # Write daily directly
            if calc_flags['daily_ndvi']:
                for doy_i, doy_str in enumerate(doy_str_list):
                    drigo.block_to_raster(
                        ndvi_array[doy_i, :, :],
                        daily_ndvi_fmt.format(doy_str),
                        b_i, b_j, bs)
                del etrf_array
            if calc_flags['daily_etrf']:
                etrf_array = et_array / etr_array
                for doy_i, doy_str in enumerate(doy_str_list):
                    drigo.block_to_raster(
                        etrf_array[doy_i, :, :],
                        daily_etrf_fmt.format(doy_str),
                        b_i, b_j, bs)
                del etrf_array
            if calc_flags['daily_etr']:
                for doy_i, doy_str in enumerate(doy_str_list):
                    drigo.block_to_raster(
                        etr_array[doy_i, :, :],
                        daily_etr_fmt.format(doy_str),
                        b_i, b_j, bs)
            if calc_flags['daily_et']:
                for doy_i, doy_str in enumerate(doy_str_list):
                    drigo.block_to_raster(
                        et_array[doy_i, :, :],
                        daily_et_fmt.format(doy_str),
                        b_i, b_j, bs)
            if calc_flags['daily_ndvi']:
                for doy_i, doy_str in enumerate(doy_str_list):
                    drigo.block_to_raster(
                        ndvi_array[doy_i, :, :],
                        daily_ndvi_fmt.format(doy_str),
                        b_i, b_j, bs)
            if calc_flags['daily_ppt']:
                for doy_i, doy_str in enumerate(doy_str_list):
                    drigo.block_to_raster(
                        ppt_array[doy_i, :, :],
                        daily_ppt_fmt.format(doy_str),
                        b_i, b_j, bs)
            # Write monthly from daily
            for month_i, month_str in enumerate(month_str_list):
                date_i_list = [
                    date_i for date_i, date_str in enumerate(date_str_list)
                    if month_str in date_str]
                start_i, end_i = date_i_list[0], date_i_list[-1] + 1
                if calc_flags['monthly_count']:
                    month_count_array = np.nansum(
                        count_array[start_i:end_i, :, :], axis=0)
                    month_count_array[~block_et_mask] = 0
                    drigo.block_to_raster(
                        month_count_array,
                        monthly_count_fmt.format(month_str),
                        b_i, b_j, bs)
                    del month_count_array
                if calc_flags['monthly_etrf']:
                    month_etrf_array = (
                        np.nansum(et_array[start_i:end_i, :, :], axis=0) /
                        np.nansum(etr_array[start_i:end_i, :, :], axis=0))
                    # month_etrf_array = np.nanmean(
                    #     etrf_array[start_i:end_i, :, :], axis=0)
                    month_etrf_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        month_etrf_array,
                        monthly_etrf_fmt.format(month_str),
                        b_i, b_j, bs)
                    del month_etrf_array
                if calc_flags['monthly_etr']:
                    month_etr_array = np.nansum(
                        etr_array[start_i:end_i, :, :], axis=0)
                    month_etr_array[~block_etr_mask] = np.nan
                    drigo.block_to_raster(
                        month_etr_array,
                        monthly_etr_fmt.format(month_str),
                        b_i, b_j, bs)
                    del month_etr_array
                if calc_flags['monthly_et']:
                    month_et_array = np.nansum(
                        et_array[start_i:end_i, :, :], axis=0)
                    month_et_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        month_et_array,
                        monthly_et_fmt.format(month_str),
                        b_i, b_j, bs)
                    del month_et_array
                if calc_flags['monthly_ndvi']:
                    month_ndvi_array = np.nansum(
                        ndvi_array[start_i:end_i, :, :], axis=0)
                    month_ndvi_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        month_ndvi_array,
                        monthly_ndvi_fmt.format(month_str),
                        b_i, b_j, bs)
                    del month_ndvi_array
                if calc_flags['monthly_ppt']:
                    month_ppt_array = np.nansum(
                        ppt_array[start_i:end_i, :, :], axis=0)
                    month_ppt_array[~block_ppt_mask] = np.nan
                    drigo.block_to_raster(
                        month_ppt_array,
                        monthly_ppt_fmt.format(month_str),
                        b_i, b_j, bs)
                    del month_ppt_array
            # Write annual from daily
            for year_i, year_str in enumerate(year_str_list):
                date_i_list = sorted([
                    date_i for date_i, date_str in enumerate(date_str_list)
                    if year_str in date_str])
                start_i = date_i_list[0]
                end_i = date_i_list[-1] + 1
                if calc_flags['annual_count']:
                    annual_count_array = np.nansum(
                        count_array[start_i:end_i, :, :], axis=0)
                    annual_count_array[~block_et_mask] = 0
                    drigo.block_to_raster(
                        annual_count_array,
                        annual_count_fmt.format(year_str), b_i, b_j, bs)
                    del annual_count_array
                if calc_flags['annual_etrf']:
                    annual_etrf_array = (
                        np.nansum(et_array[start_i:end_i, :, :], axis=0) /
                        np.nansum(etr_array[start_i:end_i, :, :], axis=0))
                    annual_etrf_array[~block_et_mask] = np.nan
                    # annual_etrf_array = np.nanmean(
                    #     etrf_array[start_i:end_i, :, :], axis=0)
                    drigo.block_to_raster(
                        annual_etrf_array,
                        annual_etrf_fmt.format(year_str), b_i, b_j, bs)
                    del annual_etrf_array
                if calc_flags['annual_etr']:
                    annual_etr_array = np.nansum(
                        etr_array[start_i:end_i, :, :], axis=0)
                    annual_etr_array[~block_etr_mask] = np.nan
                    drigo.block_to_raster(
                        annual_etr_array,
                        annual_etr_fmt.format(year_str), b_i, b_j, bs)
                    del annual_etr_array
                if calc_flags['annual_et']:
                    annual_et_array = np.nansum(
                        et_array[start_i:end_i, :, :], axis=0)
                    annual_et_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        annual_et_array,
                        annual_et_fmt.format(year_str), b_i, b_j, bs)
                    del annual_et_array
                if calc_flags['annual_ndvi']:
                    annual_ndvi_array = np.nansum(
                        ndvi_array[start_i:end_i, :, :], axis=0)
                    annual_ndvi_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        annual_ndvi_array,
                        annual_ndvi_fmt.format(year_str), b_i, b_j, bs)
                    del annual_ndvi_array
                if calc_flags['annual_ppt']:
                    annual_ppt_array = np.nansum(
                        ppt_array[start_i:end_i, :, :], axis=0)
                    annual_ppt_array[~block_etr_mask] = np.nan
                    drigo.block_to_raster(
                        annual_ppt_array,
                        annual_ppt_fmt.format(year_str), b_i, b_j, bs)
                    del annual_ppt_array
        elif calc_flags['monthly']:
            # Write monthlies direcly
            for month_i, month_str in enumerate(month_str_list):
                if calc_flags['monthly_count']:
                    drigo.block_to_raster(
                        count_array[month_i, :, :],
                        monthly_count_fmt.format(month_str),
                        b_i, b_j, bs)
                if calc_flags['monthly_etrf']:
                    drigo.block_to_raster(
                        et_array[month_i, :, :] / etr_array[month_i, :, :],
                        # etrf_array[month_i, :, :],
                        monthly_etrf_fmt.format(month_str),
                        b_i, b_j, bs)
                if calc_flags['monthly_etr']:
                    drigo.block_to_raster(
                        etr_array[month_i, :, :],
                        monthly_etr_fmt.format(month_str),
                        b_i, b_j, bs)
                if calc_flags['monthly_et']:
                    drigo.block_to_raster(
                        et_array[month_i, :, :],
                        monthly_et_fmt.format(month_str),
                        b_i, b_j, bs)
                if calc_flags['monthly_ndvi']:
                    drigo.block_to_raster(
                        ndvi_array[month_i, :, :],
                        monthly_ndvi_fmt.format(month_str),
                        b_i, b_j, bs)
                if calc_flags['monthly_ppt']:
                    drigo.block_to_raster(
                        ppt_array[month_i, :, :],
                        monthly_ppt_fmt.format(month_str),
                        b_i, b_j, bs)
            # Write annual from monthly
            for year_i, year_str in enumerate(year_str_list):
                month_i_list = sorted([
                    month_i for month_i, month_str in enumerate(month_str_list)
                    if year_str in month_str])
                start_i = month_i_list[0]
                end_i = month_i_list[-1] + 1
                if calc_flags['annual_count']:
                    annual_count_array = np.nansum(
                        count_array[start_i:end_i, :, :], axis=0)
                    annual_count_array[~block_et_mask] = 0
                    drigo.block_to_raster(
                        annual_count_array,
                        annual_count_fmt.format(year_str), b_i, b_j, bs)
                    del annual_count_array
                if calc_flags['annual_etrf']:
                    # Recompute ETrF from ET and ETr
                    annual_etrf_array = (
                        np.nansum(et_array[start_i:end_i, :, :], axis=0) /
                        np.nansum(etr_array[start_i:end_i, :, :], axis=0))
                    # annual_etrf_array = np.nanmean(
                    #     etrf_array[start_i:end_i, :, :], axis=0)
                    annual_etrf_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        annual_etrf_array,
                        annual_etrf_fmt.format(year_str), b_i, b_j, bs)
                    del annual_etrf_array
                if calc_flags['annual_etr']:
                    annual_etr_array = np.nansum(
                        etr_array[start_i:end_i, :, :], axis=0)
                    annual_etr_array[~block_etr_mask] = np.nan
                    drigo.block_to_raster(
                        annual_etr_array,
                        annual_etr_fmt.format(year_str), b_i, b_j, bs)
                    del annual_etr_array
                if calc_flags['annual_et']:
                    annual_et_array = np.nansum(
                        et_array[start_i:end_i, :, :], axis=0)
                    annual_et_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        annual_et_array,
                        annual_et_fmt.format(year_str), b_i, b_j, bs)
                    del annual_et_array
                if calc_flags['annual_ndvi']:
                    annual_ndvi_array = np.nanmean(
                        ndvi_array[start_i:end_i, :, :], axis=0)
                    annual_ndvi_array[~block_et_mask] = np.nan
                    drigo.block_to_raster(
                        annual_ndvi_array,
                        annual_ndvi_fmt.format(year_str), b_i, b_j, bs)
                    del annual_ndvi_array
                if calc_flags['annual_ppt']:
                    annual_ppt_array = np.nansum(
                        ppt_array[start_i:end_i, :, :], axis=0)
                    annual_ppt_array[~block_etr_mask] = np.nan
                    drigo.block_to_raster(
                        annual_ppt_array,
                        annual_ppt_fmt.format(year_str), b_i, b_j, bs)
                    del annual_ppt_array
        elif calc_flags['annual']:
            # Write annual directly
            for year_i, year_str in enumerate(year_str_list):
                if calc_flags['annual_count']:
                    drigo.block_to_raster(
                        count_array[year_i, :, :],
                        annual_count_fmt.format(year_str), b_i, b_j, bs)
                if calc_flags['annual_etrf']:
                    # Recompute ETrF from ET and ETr
                    drigo.block_to_raster(
                        et_array[year_i, :, :] / etr_array[year_i, :, :],
                        # etrf_array[year_i, :, :],
                        annual_etrf_fmt.format(year_str), b_i, b_j, bs)
                if calc_flags['annual_etr']:
                    drigo.block_to_raster(
                        etr_array[year_i, :, :],
                        annual_etr_fmt.format(year_str), b_i, b_j, bs)
                if calc_flags['annual_et']:
                    drigo.block_to_raster(
                        et_array[year_i, :, :],
                        annual_et_fmt.format(year_str), b_i, b_j, bs)
                if calc_flags['annual_ppt']:
                    drigo.block_to_raster(
                        ppt_array[year_i, :, :],
                        annual_ppt_fmt.format(year_str), b_i, b_j, bs)

        # # DEADBEEF - Need to figure out how to set season
        # # For now, flags are hardcoded to false, but set to annual start/end
        # for year_i, year_str in enumerate(year_str_list):
        #     seasonal_date_i_list = sorted([
        #         date_i for date_i, date_str in enumerate(date_str_list)
        #         if year_str in date_str])
        #     start_i = seasonal_date_i_list[0]
        #     end_i = seasonal_date_i_list[-1] + 1
        #     if calc_flags['seasonal_count']:
        #         seasonal_count_array = np.sum(
        #             etrf_mask[start_i:end_i, :, :], axis=0)
        #         drigo.block_to_raster(
        #             seasonal_count_array,
        #             seasonal_count_fmt.format(year_str),
        #             b_i, b_j, bs)
        #         del seasonal_count_array
        #     if calc_flags['seasonal_etrf']:
        #         seasonal_etrf_array = np.mean(
        #             etrf_array[start_i:end_i, :, :], axis=0)
        #         drigo.block_to_raster(
        #             seasonal_etrf_array, seasonal_etrf_fmt.format(year_str),
        #             b_i, b_j, bs)
        #         del seasonal_etrf_array
        #     if calc_flags['seasonal_etr']:
        #         seasonal_etr_array = np.nansum(
        #             etr_array[start_i:end_i, :, :], axis=0)
        #         seasonal_etr_array[~block_mask] = np.nan
        #         drigo.block_to_raster(
        #             seasonal_etr_array, seasonal_etr_fmt.format(year_str),
        #             b_i, b_j, bs)
        #         del seasonal_etr_array
        #     if calc_flags['seasonal_et']:
        #         seasonal_et_array = np.nansum(
        #             et_array[start_i:end_i, :, :], axis=0)
        #         seasonal_et_array[~block_mask] = np.nan
        #         drigo.block_to_raster(
        #             seasonal_et_array, seasonal_et_fmt.format(year_str),
        #             b_i, b_j, bs)
        #         del seasonal_et_array

        del et_array, etr_array, ppt_array, count_array
        del block_et_mask, block_etr_mask, block_ppt_mask
        logging.info(
            ('  Block Time: {:.1f}s  (mean {:.1f}s, ' +
             '{:.2f} hours remaining)').format(
                clock() - block_clock,
                (clock() - proc_clock) / (queue_i + 1),
                (queue_cnt - queue_i + 1) *
                (clock() - proc_clock) / (queue_i + 1) / 3600))

        # DEADBEEF
        # input('ENTER')

    # Close the queueus
    for i in range(max(1, mp_procs - 1)):
        input_q.put(None)
    input_q.close()
    output_q.close()
    del input_q, output_q
    logging.info('Time: {:.1f}'.format(clock() - main_clock))


    mp_list = []
    if stats_flag:
        logging.info('Calculating Statistics')
        stats_clock = clock()
        mp_list.extend([
            monthly_count_fmt.format(month_str)
            for month_str in month_str_list if calc_flags['monthly_count']])
        mp_list.extend([
            annual_count_fmt.format(year_str)
            for year_str in year_str_list if calc_flags['annual_count']])
        # mp_list.extend([
        #     seasonal_count_fmt.format(year_str)
        #     for year_str in year_str_list if calc_flags['seasonal_count']])

        mp_list.extend([
            daily_etrf_fmt.format(doy_str)
            for doy_str in doy_str_list if calc_flags['daily_etrf']])
        mp_list.extend([
            monthly_etrf_fmt.format(month_str)
            for month_str in month_str_list if calc_flags['monthly_etrf']])
        mp_list.extend([
            annual_etrf_fmt.format(year_str)
            for year_str in year_str_list if calc_flags['annual_etrf']])
        # mp_list.extend([
        #     seasonal_etrf_fmt.format(year_str)
        #     for year_str in year_str_list if calc_flags['seasonal_etrf']])

        mp_list.extend([
            daily_etr_fmt.format(doy_str)
            for doy_str in doy_str_list if calc_flags['daily_etr']])
        mp_list.extend([
            monthly_etr_fmt.format(month_str)
            for month_str in month_str_list if calc_flags['monthly_etr']])
        mp_list.extend([
            annual_etr_fmt.format(year_str)
            for year_str in year_str_list if calc_flags['annual_etr']])
        # mp_list.extend([
        #     seasonal_etr_fmt.format(year_str)
        #     for year_str in year_str_list if calc_flags['seasonal_etr']])

        mp_list.extend([
            daily_et_fmt.format(doy_str)
            for doy_str in doy_str_list if calc_flags['daily_et']])
        mp_list.extend([
            monthly_et_fmt.format(month_str)
            for month_str in month_str_list if calc_flags['monthly_et']])
        mp_list.extend([
            annual_et_fmt.format(year_str)
            for year_str in year_str_list if calc_flags['annual_et']])
        # mp_list.extend([
        #     seasonal_et_fmt.format(year_str)
        #     for year_str in year_str_list if calc_flags['seasonal_et']])

        mp_list.extend([
            daily_ndvi_fmt.format(doy_str)
            for doy_str in doy_str_list if calc_flags['daily_ndvi']])
        mp_list.extend([
            monthly_ndvi_fmt.format(month_str)
            for month_str in month_str_list if calc_flags['monthly_ndvi']])
        mp_list.extend([
            annual_ndvi_fmt.format(year_str)
            for year_str in year_str_list if calc_flags['annual_ndvi']])
        # mp_list.extend([
        #     seasonal_ndvi_fmt.format(year_str)
        #     for year_str in year_str_list if calc_flags['seasonal_ndvi']])

        mp_list.extend([
            daily_ppt_fmt.format(doy_str)
            for year_str in year_str_list if calc_flags['daily_ppt']])
        mp_list.extend([
            monthly_ppt_fmt.format(month_str)
            for year_str in year_str_list if calc_flags['monthly_ppt']])
        mp_list.extend([
            annual_ppt_fmt.format(year_str)
            for year_str in year_str_list if calc_flags['annual_ppt']])
        # mp_list.extend([
        #     seasonal_ppt_fmt.format(year_str)
        #     for year_str in year_str_list if calc_flags['seasonal_ppt']])

    if stats_flag and mp_list:
        if mp_procs == 1:
            for raster_path in mp_list:
                drigo.raster_statistics(raster_path)
        elif mp_procs > 1:
            pool = Pool()
            results = pool.map(drigo.raster_statistics, mp_list, chunksize=1)
            pool.close()
            pool.join()
            del results, pool, mp_list
        logging.debug('  Time: {:.1f}'.format(clock() - stats_clock))
    logging.info('Time: {:.1f}'.format(clock() - main_clock))


def block_worker(args, input_q, output_q):
    """Worker function for multiprocessing with input and output queues
    Pass block indices through to the output
    """
    # signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        args = input_q.get(block=True)
        if args is None:
            break
        output_q.put([args[0], args[1]] + block_func(*args[2:]), block=True)


def block_func(block_rows, block_cols, block_extent, block_tile_list,
               interp_date_list, etr_date_list, ppt_date_list,
               year, etrf_input_ws, tile_image_dict, cellsize, snap_proj,
               etr_shmem, etr_shape, etr_proj, etr_cs, etr_extent,
               ppt_shmem, ppt_shape, ppt_proj, ppt_cs, ppt_extent,
               awc_shmem, awc_shape, awc_proj, awc_cs, awc_extent,
               etrf_raster, ndvi_raster, swb_adjust_dict, etrf_ndvi_dict,
               study_area_mask_flag=False, study_area_path=None,
               usable_image_count=2, mosaic_method='mean',
               fill_method='linear', interp_method='linear',
               calc_flags={}, low_etrf_limit=None, high_etrf_limit=None,
               debug_flag=False):
    """

    # Input variables for array copies
    etr_input_array, etr_proj, etr_cs, etr_extent,
    ppt_input_array, ppt_proj, ppt_cs, ppt_extent,
    awc_input_array, awc_proj, awc_cs, awc_extent,
    # Input variables for shared memory array
    etr_shmem, etr_shape, etr_proj, etr_cs, etr_extent,
    ppt_shmem, ppt_shape, ppt_proj, ppt_cs, ppt_extent,
    awc_shmem, awc_shape, awc_proj, awc_cs, awc_extent,

    """
    # If time step flags are not set, or block doesn't intersect tiles
    #   immediately return empty arrays
    if ((not calc_flags['daily'] and
         not calc_flags['monthly'] and
         not calc_flags['annual']) or
        not block_tile_list):
        return [
            pickle.dumps(np.array([]), protocol=-1),
            pickle.dumps(np.array([]), protocol=-1),
            pickle.dumps(np.array([]), protocol=-1),
            pickle.dumps(np.array([]), protocol=-1)]

    # Initially assume block has data unless study area indicates false
    block_data_flag = True

    # Build arrays for storing data
    # ET array will initially be loaded with ETrF
    array_shape = len(interp_date_list), block_rows, block_cols

    # Check study area
    # block_data_mask is True for pixels in the study area
    if study_area_mask_flag and study_area_path and block_data_flag:
        study_area_mask_ds = drigo.polygon_to_raster_ds(
            study_area_path, nodata_value=0, burn_value=1,
            output_osr=drigo.proj_osr(snap_proj), output_cs=cellsize,
            output_extent=block_extent)
        block_data_mask = drigo.raster_ds_to_array(
            study_area_mask_ds, return_nodata=False).astype(np.bool)
        study_area_mask_ds = None
        # Don't calculate ETrF/ETr/ET for blocks that are outside study area
        if not np.any(block_data_mask):
            block_data_flag = False
    else:
        block_data_mask = np.full((block_rows, block_cols), True, np.bool)

    if calc_flags['etrf'] and block_data_flag:
        etrf_array = interp.load_etrf_func(
            array_shape, interp_date_list, etrf_input_ws, year,
            etrf_raster, block_tile_list, block_extent,
            tile_image_dict, mosaic_method, gdal.GRA_Bilinear,
            drigo.proj_osr(snap_proj), cellsize, block_extent, debug_flag)
        if np.any(block_data_mask):
            etrf_array[:, ~block_data_mask] = np.nan
        etrf_mask = np.isfinite(etrf_array)

        # Clear pixels that don't have a suitable number of scenes
        # I could also check the distribution of scenes (i.e. early and late)
        count_array = etrf_mask.sum(dtype=np.uint8, axis=0)
        count_mask = count_array >= usable_image_count
        # I only need to clear/reset pixels > 0 and < count
        clear_mask = (count_array > 0) & (~count_mask)
        if np.any(clear_mask):
            etrf_array[:, clear_mask] = np.nan
            etrf_mask[:, clear_mask] = False
        del clear_mask, count_array

    if calc_flags['ndvi'] and block_data_flag:
        ndvi_array = interp.load_etrf_func(
            array_shape, interp_date_list, etrf_input_ws, year,
            ndvi_raster, block_tile_list, block_extent,
            tile_image_dict, mosaic_method, gdal.GRA_Bilinear,
            drigo.proj_osr(snap_proj), cellsize, block_extent, debug_flag)
        if np.any(block_data_mask):
            ndvi_array[:, ~block_data_mask] = np.nan
        # ndvi_mask = np.isfinite(ndvi_array)

    # Compute ETrF from NDVI
    # DEADBEEF - Should this be before or after the soil water balance?
    if etrf_ndvi_dict['flag']:
        # First identify days that are missing ETrF but have NDVI
        ndvi_date_mask = np.isfinite(
            etrf_array).sum(axis=2).sum(axis=1).astype(np.bool)
        np.logical_not(ndvi_date_mask, out=ndvi_date_mask)
        ndvi_date_mask &= np.isfinite(
            ndvi_array).sum(axis=2).sum(axis=1).astype(np.bool)
        # Then add additional override dates
        ndvi_date_mask |= np.array([
            (d.month in etrf_ndvi_dict['month'] or
             int(d.strftime('%j')) in etrf_ndvi_dict['doy'])
            for d in interp_date_list])
        if ndvi_date_mask.any():
            etrf_array[ndvi_date_mask, :, :] = ndvi_array[ndvi_date_mask, :, :]
            etrf_array[ndvi_date_mask, :, :] *= etrf_ndvi_dict['slope']
            etrf_array[ndvi_date_mask, :, :] += etrf_ndvi_dict['offset']
        del ndvi_date_mask

    if calc_flags['etr'] and block_data_flag:
        # # DEADBEEF - Working implementation of a shared memory array
        # User must pass in etr_shmem and etr_shape
        etr_input_array = ctypeslib.as_array(etr_shmem)
        etr_input_array.shape = etr_shape

        # Project all ETr bands/days at once
        etr_array = drigo.project_array(
            etr_input_array, gdal.GRA_Bilinear,
            drigo.proj_osr(etr_proj), etr_cs, etr_extent,
            drigo.proj_osr(snap_proj), cellsize, block_extent)
        # Project each ETr band/day separately
        # etr_array = np.full(array_shape, np.nan, np.float32)
        # for date_i, date_obj in enumerate(date_list):
        #     etr_array[date_i, :, :] = drigo.project_array(
        #         etr_input_array[date_i, :, :], gdal.GRA_Bilinear,
        #         drigo.proj_osr(etr_proj), etr_cs, etr_extent,
        #         drigo.proj_osr(snap_proj), cellsize, block_extent)

        # Build ETr mask for reseting nodata after nansum/nanmean
        etr_mask = block_data_mask & np.any(np.isfinite(etr_array), axis=0)
    else:
        # If the user wants ETrF (and not ETr or ET),
        #   ETr needs to have a value since monthly and annual ETrF
        #   are computed from the sums of ET and ETr
        etr_array = np.ones(array_shape, dtype=np.float32)
        etr_mask = np.ones((block_rows, block_cols), dtype=np.bool)
        # etr_array = np.full(array_shape, np.nan, np.float32)
        # etr_mask = np.full((block_rows, block_cols), False, np.bool)

    if calc_flags['ppt'] and block_data_flag:
        # # DEADBEEF - Working implementation of a shared memory array
        # User must pass in ppt_shmem and ppt_shape
        ppt_input_array = ctypeslib.as_array(ppt_shmem)
        ppt_input_array.shape = ppt_shape

        # Project all PPT bands/days at once
        ppt_array = drigo.project_array(
            ppt_input_array, gdal.GRA_Bilinear,
            drigo.proj_osr(ppt_proj), ppt_cs, ppt_extent,
            drigo.proj_osr(snap_proj), cellsize, block_extent)
        # Project each PPT band/day separately
        # ppt_array = np.full(array_shape, np.nan, np.float32)
        # for date_i, date_obj in enumerate(date_list):
        #     ppt_array[date_i, :, :] = drigo.project_array(
        #         ppt_input_array[date_i, :, :], gdal.GRA_Bilinear,
        #         drigo.proj_osr(ppt_proj), ppt_cs, ppt_extent,
        #         drigo.proj_osr(snap_proj), cellsize, block_extent)

        # Build PPT mask for reseting nodata after nansum/nanmean
        ppt_mask = block_data_mask & np.any(np.isfinite(ppt_array), axis=0)
    else:
        ppt_array = np.full(array_shape, np.nan, np.float32)
        ppt_mask = np.full((block_rows, block_cols), False, np.bool)

    # Adjust ETrF based on SWB and NDVI
    if swb_adjust_dict['flag']:
        # DEADBEEF - Working implementation of a shared memory array
        # User must pass in awc_shmem and awc_shape
        awc_input_array = ctypeslib.as_array(awc_shmem)
        awc_input_array.shape = awc_shape

        # Clip/extract awc_block_array
        awc_x, awc_y = awc_extent.origin()
        awc_swb_extent = drigo.project_extent(
            block_extent, drigo.proj_osr(snap_proj),
            drigo.proj_osr(awc_proj), cellsize=cellsize)
        # This will cause errors for very small study areas...
        # awc_swb_extent.buffer_extent(awc_cs * 4)
        awc_swb_extent.adjust_to_snap('EXPAND', awc_x, awc_y, awc_cs)
        awc_swb_xi, awc_swb_yi = drigo.array_geo_offsets(
            awc_extent.geo(awc_cs), awc_swb_extent.geo(awc_cs), awc_cs)
        awc_swb_rows, awc_swb_cols = awc_swb_extent.shape(awc_cs)
        awc_swb_array = awc_input_array[
            awc_swb_yi:awc_swb_yi + awc_swb_rows,
            awc_swb_xi:awc_swb_xi + awc_swb_cols]

        # Clip and project ETr/PPT to AWC spat. ref., cellsize, and extent
        etr_swb_array = drigo.project_array(
            etr_input_array, gdal.GRA_Bilinear,
            drigo.proj_osr(etr_proj), etr_cs, etr_extent,
            drigo.proj_osr(awc_proj), awc_cs, awc_swb_extent)
        ppt_swb_array = drigo.project_array(
            ppt_input_array, gdal.GRA_Bilinear,
            drigo.proj_osr(ppt_proj), ppt_cs, ppt_extent,
            drigo.proj_osr(awc_proj), awc_cs, awc_swb_extent)

        # Compute the daily soil water balance
        ke_swb_array = et_common.array_swb_func(
            etr=etr_swb_array, ppt=ppt_swb_array, awc=awc_swb_array)

        # Project Ke to Landsat spat. ref., cellsize, and extent
        ke_array = drigo.project_array(
            ke_swb_array, gdal.GRA_Bilinear,
            drigo.proj_osr(awc_proj), awc_cs, awc_swb_extent,
            drigo.proj_osr(snap_proj), cellsize, block_extent)
        # print('Ke')
        # print(ke_array[:,1,1])
        # print(ke_array.shape)
        del ke_array
        # input('ENTER')

        # # Build AWC mask for reseting nodata after nansum/nanmean
        # # ke_mask = block_data_mask & np.any(np.isfinite(ke_array), axis=0)
        #
        # # How should background mean get computed?
        # # Mean of the previous 30 days maybe?
        # def moving_average(a, n=30) :
        #     ret = np.cumsum(a, dtype=float)
        #     ret[n:] = ret[n:] - ret[:-n]
        #     return ret[n - 1:] / n
        # ke_mean = moving_average(ke_array, n=30)
        # print('Ke Mean')
        # print(ke_swb_array)
        # print(ke_swb_array.shape)
        # sleep(5)
        #
        # # fc = interp.swb_adjust_fc(
        # #     ndvi_array, ndvi_full_cover=swb_adjust_dict['full'],
        # #     ndvi_bare_soil=swb_adjust_dict['bare'])
        # #     # ndvi_full_cover=tile_ndvi_dict[year][tile_name][image_id]['cold'],
        # #     # ndvi_bare_soil=tile_ndvi_dict[year][tile_name][image_id]['hot'])
        # # etrf_transpiration = etrf_array - ((1 - fc) * etrf_background_mean)
        # # np.maximum(
        # #     etrf_transpiration, etrf_background, out=etrf_transpiration)
        # # etrf_adjusted = (
        # #     ((1 - fc) * etrf_background) + (fc * etrf_transpiration))
        # # etrf_array[etrf_mask] = etrf_adjusted[etrf_mask]
        # del ke_array

    # Interpolate ETrF after SWB adjust
    if calc_flags['etrf'] and np.any(count_mask):
        # Clamp/limit ETrF values
        if low_etrf_limit is not None:
            etrf_array[etrf_array < low_etrf_limit] = low_etrf_limit + 0.0000001
        if high_etrf_limit is not None:
            etrf_array[etrf_array > high_etrf_limit] = high_etrf_limit

        # Calculate dates where ETrF has data (scene dates)
        sub_i_mask = np.isfinite(
            etrf_array).sum(axis=2).sum(axis=1).astype(np.bool)
        # sub_i_mask = np.isfinite(
        #     etrf_array).sum(dtype=np.bool, axis=(1,2))
        # Also include start/end/anchor dates
        sub_i_mask[0], sub_i_mask[-1] = True, True
        if (fill_method == 'cubicspline' or interp_method == 'cubicspline'):
            sub_i_mask[1], sub_i_mask[-2] = True, True

        #
        sub_i_array = np.arange(len(interp_date_list))[sub_i_mask]
        sub_etrf_array = np.copy(etrf_array[sub_i_mask, :, :])
        del sub_i_mask

        # If continues happens here, then rasters aren't updated/overwritten
        # if not np.any(np.isfinite(sub_etrf_array)):
        #     continue

        # Fill missing ETrF on image dates (SLC-off, clouds, etc.)
        if fill_method == 'spatial':
            # print('  Filling ETrF Spatially')
            sys.exit()
            # image_et_array = interp.spatial_fill_func(
            #     image_et_array, block_mask)
        elif fill_method in ['nearest', 'linear', 'cubicspline']:
            # Fill the end/anchor values if they are missing
            # print('  Filling end/anchor scenes')
            sub_etrf_array = interp.end_fill_func(
                sub_etrf_array, count_mask, fill_method)
            # Temporally fill image dates functions
            logging.debug('  Filling ETrF Temporally')
            sub_etrf_array = interp.temporal_fill_func(
                sub_etrf_array, sub_i_array,
                count_mask, fill_method)
        else:
            sys.exit()

        # Interpolate between image dates
        # print('  Interpolating ETrF Temporally')
        etrf_array = interp.interpolate_func(
            etrf_array, sub_etrf_array, sub_i_array,
            count_mask, interp_method)

        # # Interpolate between image dates
        # logging.debug('  Interpolating ETrF Temporally')
        # etrf_array = interp.mp_interpolate_func(
        #     etrf_array, sub_etrf_array, sub_i_array,
        #     block_mask, interp_method)

        # # Interpolate between image dates
        # logging.debug('  Interpolating ETrF Temporally')
        # interp_clock = clock()
        # etrf_array = interp.block_interpolate_func(
        #     etrf_array, sub_etrf_array, sub_i_array,
        #     block_mask, fill_method, interp_method)
        del sub_i_array, sub_etrf_array
    else:
        etrf_array = np.full(array_shape, np.nan, np.float32)
        etrf_mask = np.full(array_shape, False, np.bool)
        count_mask = np.full((block_rows, block_cols), True, np.bool)

    # Interpolate NDVI
    if calc_flags['ndvi'] and np.any(count_mask):
        # Calculate dates where NDVI has data (scene dates)
        sub_i_mask = np.isfinite(ndvi_array).sum(axis=2).sum(axis=1) \
            .astype(np.bool)
        # Also include start/end/anchor dates
        sub_i_mask[0], sub_i_mask[-1] = True, True
        if (fill_method == 'cubicspline' or interp_method == 'cubicspline'):
            sub_i_mask[1], sub_i_mask[-2] = True, True

        sub_i_array = np.arange(len(interp_date_list))[sub_i_mask]
        sub_ndvi_array = np.copy(ndvi_array[sub_i_mask, :, :])
        del sub_i_mask

        # Fill missing NDVI on image dates (SLC-off, clouds, etc.)
        if fill_method == 'spatial':
            sys.exit()
            # sub_ndvi_array = interp.spatial_fill_func(
            #     sub_ndvi_array, block_mask)
        elif fill_method in ['nearest', 'linear', 'cubicspline']:
            # Fill the end/anchor values if they are missing
            sub_ndvi_array = interp.end_fill_func(
                sub_ndvi_array, count_mask, fill_method)
            # Temporally fill image dates functions
            logging.debug('  Filling NDVI Temporally')
            sub_ndvi_array = interp.temporal_fill_func(
                sub_ndvi_array, sub_i_array,
                count_mask, fill_method)
        else:
            sys.exit()

        # Interpolate between image dates
        ndvi_array = interp.interpolate_func(
            ndvi_array, sub_ndvi_array, sub_i_array,
            count_mask, interp_method)

        del sub_i_array, sub_ndvi_array
    else:
        ndvi_array = np.full(array_shape, np.nan, np.float32)

    # Remove SWB spinup dates from ETr and PPT
    if interp_date_list != etr_date_list:
        etr_date_mask = [d in interp_date_list for d in etr_date_list]
        etr_array = etr_array[np.array(etr_date_mask), :, :]
    if interp_date_list != ppt_date_list:
        ppt_date_mask = [d in interp_date_list for d in ppt_date_list]
        ppt_array = ppt_array[np.array(ppt_date_mask), :, :]

    # Always compuate ET from ETrF and ETr
    et_array = etrf_array * etr_array

    # Only one of these flags should be True
    # Start with daily and work up to annual
    if calc_flags['daily']:
        # If daily, return the full datasets
        # Apply count/data mask before returning
        etrf_mask[:, ~count_mask] = False
        ndvi_array[:, ~count_mask] = np.nan
        # etrf_array[:, ~count_mask] = np.nan
        et_array[:, ~count_mask] = np.nan
        etr_array[:, ~etr_mask] = np.nan
        ppt_array[:, ~ppt_mask] = np.nan
        return [
            pickle.dumps(etrf_mask, protocol=-1),
            pickle.dumps(ndvi_array, protocol=-1),
            # pickle.dumps(etrf_array, protocol=-1),
            pickle.dumps(et_array, protocol=-1),
            pickle.dumps(etr_array, protocol=-1),
            pickle.dumps(ppt_array, protocol=-1)]

    elif calc_flags['monthly']:
        # If monthly, collapse by month
        date_str_list = [
            date.strftime('%Y_%m_%d') for date in interp_date_list]
        month_str_list = sorted(list(set([
            date.strftime('%Y_%m') for date in interp_date_list])))
        month_count_array = np.full(
            (len(month_str_list), block_rows, block_cols), 0, np.int8)
        month_ndvi_array = np.full(
            (len(month_str_list), block_rows, block_cols), np.nan, np.float32)
        # month_etrf_array = np.full(
        #     (len(month_str_list), block_rows, block_cols), np.nan, np.float32)
        month_et_array = np.full(
            (len(month_str_list), block_rows, block_cols), np.nan, np.float32)
        month_etr_array = np.full(
            (len(month_str_list), block_rows, block_cols), np.nan, np.float32)
        month_ppt_array = np.full(
            (len(month_str_list), block_rows, block_cols), np.nan, np.float32)
        for month_i, month_str in enumerate(month_str_list):
            date_i_list = [
                date_i for date_i, date_str in enumerate(date_str_list)
                if month_str in date_str]
            start_i, end_i = date_i_list[0], date_i_list[-1] + 1
            month_count_array[month_i] = np.sum(
                etrf_mask[start_i:end_i, :, :], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                month_ndvi_array[month_i] = np.nanmean(
                    ndvi_array[start_i:end_i, :, :], axis=0)
                # month_etrf_array[month_i] = np.nanmean(
                #     etrf_array[start_i:end_i, :, :], axis=0)
                month_et_array[month_i] = np.nansum(
                    et_array[start_i:end_i, :, :], axis=0)
                month_etr_array[month_i] = np.nansum(
                    etr_array[start_i:end_i, :, :], axis=0)
                month_ppt_array[month_i] = np.nansum(
                    ppt_array[start_i:end_i, :, :], axis=0)

        # Apply count/data mask before returning
        month_count_array[:, ~count_mask] = False
        month_ndvi_array[:, ~count_mask] = np.nan
        # month_etrf_array[:, ~count_mask] = np.nan
        month_et_array[:, ~count_mask] = np.nan
        month_etr_array[:, ~etr_mask] = np.nan
        month_ppt_array[:, ~ppt_mask] = np.nan
        return [
            pickle.dumps(month_count_array, protocol=-1),
            pickle.dumps(month_ndvi_array, protocol=-1),
            # pickle.dumps(month_etrf_array, protocol=-1),
            pickle.dumps(month_et_array, protocol=-1),
            pickle.dumps(month_etr_array, protocol=-1),
            pickle.dumps(month_ppt_array, protocol=-1)]

    elif calc_flags['annual']:
        # If annual, collapse by year
        date_str_list = [
            date.strftime('%Y_%m_%d') for date in interp_date_list]
        year_str_list = sorted(list(set([
            date.strftime('%Y') for date in interp_date_list])))
        annual_count_array = np.full(
            (len(year_str_list), block_rows, block_cols), 0, dtype=np.int8)
        annual_ndvi_array = np.full(
            (len(year_str_list), block_rows, block_cols), np.nan, np.float32)
        # annual_etrf_array = np.full(
        #     (len(year_str_list), block_rows, block_cols), np.nan, np.float32)
        annual_et_array = np.full(
            (len(year_str_list), block_rows, block_cols), np.nan, np.float32)
        annual_etr_array = np.full(
            (len(year_str_list), block_rows, block_cols), np.nan, np.float32)
        annual_ppt_array = np.full(
            (len(year_str_list), block_rows, block_cols), np.nan, np.float32)

        for year_i, year_str in enumerate(year_str_list):
            date_i_list = [
                date_i for date_i, date_str in enumerate(date_str_list)
                if year_str in date_str]
            start_i, end_i = date_i_list[0], date_i_list[-1] + 1
            annual_count_array[year_i] = np.sum(
                etrf_mask[start_i:end_i, :, :], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                annual_ndvi_array[year_i] = np.nanmean(
                    ndvi_array[start_i:end_i, :, :], axis=0)
                # annual_etrf_array[year_i] = np.nanmean(
                #     etrf_array[start_i:end_i, :, :], axis=0)
                annual_et_array[year_i] = np.nansum(
                    et_array[start_i:end_i, :, :], axis=0)
                annual_etr_array[year_i] = np.nansum(
                    etr_array[start_i:end_i, :, :], axis=0)
                annual_ppt_array[year_i] = np.nansum(
                    ppt_array[start_i:end_i, :, :], axis=0)

        # Apply count/data mask before returning
        annual_count_array[:, ~count_mask] = False
        annual_ndvi_array[:, ~count_mask] = np.nan
        # annual_etrf_array[:, ~count_mask] = np.nan
        annual_et_array[:, ~count_mask] = np.nan
        annual_etr_array[:, ~etr_mask] = np.nan
        annual_ppt_array[:, ~ppt_mask] = np.nan
        return [
            pickle.dumps(annual_count_array, protocol=-1),
            pickle.dumps(annual_ndvi_array, protocol=-1),
            # pickle.dumps(annual_etrf_array, protocol=-1),
            pickle.dumps(annual_et_array, protocol=-1),
            pickle.dumps(annual_etr_array, protocol=-1),
            pickle.dumps(annual_ppt_array, protocol=-1)]


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Interpolator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', default=None, help='Year folder', metavar='FOLDER')
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Interpolate input file', metavar='FILE')
    parser.add_argument(
        '-bs', '--blocksize', default=None, type=int, metavar='N',
        help='Block size')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-mc', '--montecarlo', default=None, type=int, metavar='N',
        help='Monte Carlo iteration number')
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int, nargs='?',
        metavar="[1-{}]".format(cpu_count()), const=cpu_count(),
        choices=range(1, cpu_count() + 1),
        help='Number of processers to use')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    parser.add_argument(
        '--pyramids', default=False, action="store_true",
        help='Compute raster pyramids')
    parser.add_argument(
        '-q', '--queue', default=1, type=int, metavar='N',
        help='Size of output queue')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.workspace and os.path.isdir(os.path.abspath(args.workspace)):
        args.workspace = os.path.abspath(args.workspace)
    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_console = logging.StreamHandler()
    log_console.setLevel(args.loglevel)
    formatter = logging.Formatter('%(message)s')
    log_console.setFormatter(formatter)
    logger.addHandler(log_console)

    if not args.no_file_logging:
        if args.montecarlo is None:
            log_file_name = 'interpolate_log.txt'
        else:
            log_file_name = 'mc{0:02d}_interpolate_rasters_log.txt'.format(
                args.montecarlo)
        log_file = logging.FileHandler(
            os.path.join(args.workspace, log_file_name), "w")
        log_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

    logging.info('\n{}'.format('#' * 80))
    log_fmt = '{:<20s} {}'
    logging.info(log_fmt.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_fmt.format('Current Directory:', args.workspace))
    logging.info(log_fmt.format('Script:', os.path.basename(sys.argv[0])))

    # Delay
    sleep(random.uniform(0, max([0, args.delay])))

    # Run ET interpolator
    metric_interpolate(year_ws=args.workspace, ini_path=args.ini,
                       mc_iter=args.montecarlo, bs=args.blocksize,
                       pyramids_flag=args.stats, stats_flag=args.stats,
                       overwrite_flag=args.overwrite,
                       mp_procs=args.multiprocessing, delay=args.delay,
                       debug_flag=args.loglevel==logging.DEBUG,
                       output_queue=args.queue)
