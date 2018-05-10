#!/usr/bin/env python
#--------------------------------
# Name:         interpolate_tables_func.py
# Purpose:      Interpolate ETrF rasters between Landsat scenes based on DOY
#--------------------------------

from __future__ import division
import argparse
from builtins import input
from collections import defaultdict
import ctypes
import datetime as dt
import logging
from multiprocessing import Process, Queue, cpu_count, sharedctypes
import os
import random
import re
import shutil
import sys
from time import clock, sleep

# Python 2/3
try:
    import pickle
except:
    import cPickle as pickle

import drigo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from numpy import ctypeslib
from osgeo import gdal, ogr, osr
import pandas as pd

import et_common
import interpolate_support as interp
from python_common import open_ini, read_param, parse_int_set

np.seterr(invalid='ignore')
# gdal.UseExceptions()


def metric_interpolate(year_ws, ini_path, mc_iter=None, bs=None,
                       overwrite_flag=None, mp_procs=None,
                       delay=0, debug_flag=False, output_queue=1,
                       dpi=150):
    """METRIC Table Interpolator

    Parameters
    ----------
    year_ws : str
        Year folder path
    ini_path : str
        The config file (path)
    mc_iter : int, optional
        Iteration number for Monte Carlo processing (the default is None).
    bs : int, optional
        Processing block size (the default is None).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is None).
    mp_procs : int, optional
        Number of cores to use (the default is None).
    delay : int, optional
        Max random delay starting workers in seconds (the default is 0).
    debug_flag : bool, optional
        If True, enable debug level logging (the default is False).
    output_queue : int, optional
        Size of output queue (the default is 1).
    dpi : int, optional
        Output image resolution (dots per inch) (the default is 150).

    Returns
    -------
    None

    """
    logging.info('\nInterpolating ET tables')
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
    zones_path = config.get('INPUTS', 'zones_path')
    zones_mask = read_param('zones_mask', None, config)
    zones_name_field = read_param('zones_name_field', 'FID', config)
    # zones_buffer = read_param('zones_buffer', 0.0, config)
    zones_buffer = 0
    output_snap = read_param('zones_snap', (0, 0), config)
    output_cs = read_param('zones_cellsize', 30.0, config)
    # output_proj = read_param('zones_proj', None, config)
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

    # Plot parameters
    plots_zone_area_flag = read_param('plots_zone_area_flag', False, config)
    plots_ndvi_ylim = read_param('plots_ndvi_ylim', [0.0, 1.2], config)
    plots_etrf_ylim = read_param('plots_etrf_ylim', [0.0, 1.2], config)
    plots_etr_ylim = read_param('plots_etr_ylim', [], config)
    plots_et_ylim = read_param('plots_et_ylim', [], config)
    plots_ppt_ylim = read_param('plots_ppt_ylim', [], config)
    if plots_ndvi_ylim:
        plots_ndvi_ylim = list(map(float, plots_ndvi_ylim))
    if plots_etrf_ylim:
        plots_etrf_ylim = list(map(float, plots_etrf_ylim))
    if plots_etr_ylim:
        plots_etr_ylim = list(map(float, plots_etr_ylim))
    if plots_et_ylim:
        plots_et_ylim = list(map(float, plots_et_ylim))
    if plots_ppt_ylim:
        plots_ppt_ylim = list(map(float, plots_ppt_ylim))

    # Table parameters
    ndvi_field = 'NDVI'
    etrf_field = 'ETRF'
    etr_field = 'ETR_MM'
    et_field = 'ET_MM'
    ppt_field = 'PPT_MM'

    # DEADBEEF - Still need to apply zones_buffer in worker function
    if zones_buffer != 0:
        logging.warning('\nWARNING: zones_buffer is not implemented\n')
        sys.exit()

    # Read Monte Carlo iteration ETrF raster
    if mc_iter is not None:
        etrf_raster = os.path.splitext(etrf_raster)[0] + iter_fmt
        # etrf_raster = etrf_raster.replac('.img', iter_fmt)

    # If command line arguments aren't set, trying reading from input file
    if bs is None:
        bs = read_param('blocksize', 256, config)

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
    swb_adjust_dict['flag'] = read_param(
        'swb_adjust_flag', False, config)
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

    # Zones
    calc_flags['daily_zones_table'] = read_param(
        'calc_daily_zones_table_flag', False, config)
    calc_flags['monthly_zones_table'] = read_param(
        'calc_monthly_zones_table_flag', False, config)
    calc_flags['annual_zones_table'] = read_param(
        'calc_annual_zones_table_flag', False, config)

    calc_flags['daily_ndvi_plots'] = read_param(
        'calc_daily_ndvi_plots_flag', False, config)
    calc_flags['daily_etrf_plots'] = read_param(
        'calc_daily_etrf_plots_flag', False, config)
    calc_flags['daily_etr_plots'] = read_param(
        'calc_daily_etr_plots_flag', False, config)
    calc_flags['daily_et_plots'] = read_param(
        'calc_daily_et_plots_flag', False, config)
    calc_flags['daily_ppt_plots'] = read_param(
        'calc_daily_ppt_plots_flag', False, config)

    if not any(calc_flags.values()):
        logging.error('\nERROR: All calc flags are false, exiting\n')
        sys.exit()

    # Collarpse zones and plots flags
    calc_flags['zones'] = False
    calc_flags['plots'] = False
    if (calc_flags['daily_zones_table'] or
            calc_flags['monthly_zones_table'] or
            calc_flags['annual_zones_table']):
        calc_flags['zones'] = True
    if (calc_flags['daily_ndvi_plots'] or
            calc_flags['daily_etrf_plots'] or
            calc_flags['daily_etr_plots'] or
            calc_flags['daily_et_plots'] or
            calc_flags['daily_ppt_plots']):
        calc_flags['plots'] = True

    # For now, compute all datasets if any zones_table flags are True
    calc_flags['ndvi'] = False
    calc_flags['etrf'] = False
    calc_flags['etr'] = False
    calc_flags['et'] = False
    calc_flags['ppt'] = False
    if (calc_flags['daily_ndvi_plots'] or
            calc_flags['zones']):
            calc_flags['ndvi'] = True
    if (calc_flags['daily_etrf_plots'] or
            calc_flags['zones']):
            calc_flags['etrf'] = True
    if (calc_flags['daily_etr_plots'] or
            calc_flags['zones']):
            calc_flags['etr'] = True
    if (calc_flags['daily_et_plots'] or
            calc_flags['zones']):
            calc_flags['et'] = True
    if (calc_flags['daily_ppt_plots'] or
            calc_flags['zones']):
            calc_flags['ppt'] = True

    if etrf_ndvi_dict['flag']:
        calc_flags['ndvi'] = True
    if swb_adjust_dict['flag']:
        calc_flags['etr'] = True
        calc_flags['ndvi'] = True
        calc_flags['ppt'] = True

    # Daily data is always returned by worker
    # calc_flags['daily'] = True
    # calc_flags['monthly'] = True
    # calc_flags['annual'] = True

    # Force lists to be integers/floats
    output_snap = map(float, output_snap)
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
    logging.info(log_fmt.format('Overwrite:', overwrite_flag))

    # Create folders for output
    if output_ws is not None:
        output_ws = os.path.join(output_ws, output_folder_name)
    else:
        output_ws = os.path.join(year_ws, output_folder_name)
    zones_table_ws = os.path.join(
        output_ws, '{}{}_ZONES'.format(mc_str, interp_method.upper()))
    zones_ndvi_plots_ws = os.path.join(
        output_ws, '{}{}_PLOTS_{}'.format(
            mc_str, interp_method.upper(), ndvi_name.upper()))
    zones_etrf_plots_ws = os.path.join(
        output_ws, '{}{}_PLOTS_{}'.format(
            mc_str, interp_method.upper(), etrf_name.upper()))
    zones_etr_plots_ws = os.path.join(
        output_ws, '{}{}_PLOTS_{}'.format(
            mc_str, interp_method.upper(), etr_name.upper()))
    zones_et_plots_ws = os.path.join(
        output_ws, '{}{}_PLOTS_{}'.format(
            mc_str, interp_method.upper(), et_name.upper()))
    zones_ppt_plots_ws = os.path.join(
        output_ws, '{}{}_PLOTS_{}'.format(
            mc_str, interp_method.upper(), ppt_name.upper()))
    if not os.path.isdir(output_ws):
        os.makedirs(output_ws)
    if calc_flags['zones'] and not os.path.isdir(zones_table_ws):
        os.mkdir(zones_table_ws)
    if calc_flags['daily_ndvi_plots'] and not os.path.isdir(zones_ndvi_plots_ws):
        os.mkdir(zones_ndvi_plots_ws)
    if calc_flags['daily_etrf_plots'] and not os.path.isdir(zones_etrf_plots_ws):
        os.mkdir(zones_etrf_plots_ws)
    if calc_flags['daily_etr_plots'] and not os.path.isdir(zones_etr_plots_ws):
        os.mkdir(zones_etr_plots_ws)
    if calc_flags['daily_et_plots'] and not os.path.isdir(zones_et_plots_ws):
        os.mkdir(zones_et_plots_ws)
    if calc_flags['daily_ppt_plots'] and not os.path.isdir(zones_ppt_plots_ws):
        os.mkdir(zones_ppt_plots_ws)

    # Remove folders (and files) that aren't being calculated
    if not calc_flags['zones'] and os.path.isdir(zones_table_ws):
        shutil.rmtree(zones_table_ws)
    if not calc_flags['daily_ndvi_plots'] and os.path.isdir(zones_ndvi_plots_ws):
        shutil.rmtree(zones_ndvi_plots_ws)
    if not calc_flags['daily_etrf_plots'] and os.path.isdir(zones_etrf_plots_ws):
        shutil.rmtree(zones_etrf_plots_ws)
    if not calc_flags['daily_etr_plots'] and os.path.isdir(zones_etr_plots_ws):
        shutil.rmtree(zones_etr_plots_ws)
    if not calc_flags['daily_et_plots'] and os.path.isdir(zones_et_plots_ws):
        shutil.rmtree(zones_et_plots_ws)
    if not calc_flags['daily_ppt_plots'] and os.path.isdir(zones_ppt_plots_ws):
        shutil.rmtree(zones_ppt_plots_ws)

    # Regular expressions for the unmerged and merged directories
    tile_re = re.compile('[p](\d{3})[r](\d{3})')
    image_re = re.compile(
        '^(?P<prefix>LT04|LT05|LE07|LC08)_(?P<path>\d{3})(?P<row>\d{3})_'
        '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})')
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
            if (os.path.isdir(os.path.join(tile_ws, image_id)) or
                image_re.match(image_id))]
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
    # date_str_list = [date.strftime('%Y_%m_%d') for date in date_list]
    # doy_str_list = [date.strftime('%Y_%j') for date in date_list]
    # month_str_list = sorted(list(set([
    #     date.strftime('%Y_%m') for date in date_list])))
    # year_str_list = sorted(list(set([
    #     date.strftime('%Y') for date in date_list])))

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

    # Zones
    logging.info('\nZones')
    if not os.path.isfile(zones_path):
        logging.error(
            ('\nERROR: The zones shapefile does not exist:'
             '\n  {}').format(zones_path))
        sys.exit()
    else:
        logging.info('  {}'.format(zones_path))
    if zones_mask and not os.path.isfile(zones_mask):
        logging.error(
            ('\nERROR: The zones mask does not exist:'
             '\n  {}').format(zones_mask))
        sys.exit()
    else:
        logging.info('  {}'.format(zones_mask))

    # For now, get output projection from zones_path
    logging.debug('  Projection from zones_path')
    zones_osr = drigo.feature_path_osr(zones_path)
    output_osr = drigo.feature_path_osr(zones_path)
    logging.debug('  OSR: {}'.format(zones_osr))
    logging.debug('  PROJ4: {}'.format(drigo.osr_proj4(zones_osr)))

    # Get output projection from 1) zones_mask, 2) zones_proj, 3) zones_path
    # zones_osr = drigo.feature_path_osr(zones_path)
    # if zones_mask:
    #     logging.debug('  Projection from zones_mask')
    #     output_osr = drigo.raster_path_osr(zones_mask)
    # elif output_proj:
    #     logging.debug('  Projection from zones_proj')
    #     output_osr = interp.unknown_proj_osr(output_proj)
    #     if output_osr is None:
    #         logging.error(
    #             ('\nERROR: The zones_proj string could not be ' +
    #              'converted to a spatial reference \n  {}'.format(
    #                 output_proj)))
    #         sys.exit()
    # else:
    #     logging.debug('  Projection from zones_path')
    #     output_osr = drigo.feature_path_osr(zones_path)

    # Output spatial reference must be a projected coordinate system
    output_gcs_osr = output_osr.CloneGeogCS()
    if ((drigo.osr_proj4(output_osr) == drigo.osr_proj4(output_gcs_osr)) or
            (str(output_osr) == str(output_gcs_osr))):
        logging.warning('  OSR: {}'.format(output_osr))
        logging.warning('  GCS: {}'.format(output_gcs_osr))
        logging.warning('  Cellsize: {}'.format(output_cs))
        logging.warning(
            '\nWARNING:\n'
            '  The zones shapefile appears to be in a geographic '
            'coordinate system\n'
            '    (units in decimal degrees)\n'
            '  It is recommended to use a shapefile with a projected '
            'coordinate system\n'
            '    (units in meters or feet)\n'
            '  Before continuing, please ensure the cellsize is in '
            'decimal degrees')
        input('Press ENTER to continue')
        if output_cs >= 1:
            logging.error('\nERROR: The output cellsize is too large, exiting')
            sys.exit()

    # Check if zones mask has the same projection
    # For now just warn the user
    if zones_mask:
        logging.debug('\nZones mask')
        zones_mask_osr = drigo.raster_path_osr(zones_mask)
        logging.debug('  OSR: {}'.format(zones_mask_osr))
        logging.debug('  PROJ4: {}'.format(drigo.osr_proj4(zones_mask_osr)))
        if drigo.osr_proj4(zones_mask_osr) != drigo.osr_proj4(output_osr):
            logging.warning(
                '\nWARNING: The zone features and mask '
                'may have different projections\n')

    logging.info('\nSpatial Reference')
    env.snap_osr = output_osr
    env.snap_gcs_osr = output_gcs_osr
    env.snap_proj = env.snap_osr.ExportToWkt()
    env.snap_gcs_proj = env.snap_gcs_osr.ExportToWkt()
    if zones_mask:
        env.cellsize = drigo.raster_path_cellsize(zones_mask, x_only=True)
        env.snap_x, env.snap_y = drigo.raster_path_origin(zones_mask)
    else:
        env.cellsize = output_cs
        env.snap_x, env.snap_y = output_snap
    logging.debug('  Cellsize: {}'.format(env.cellsize))
    logging.debug('  Snap: {} {}'.format(env.snap_x, env.snap_y))
    logging.debug('  OSR: {}'.format(env.snap_osr))
    logging.debug('  PROJ4: {}'.format(drigo.osr_proj4(env.snap_osr)))
    logging.debug('  GCS: {}'.format(env.snap_gcs_osr))

    # Footprint (WRS2 Descending Polygons)
    logging.info('\nFootprints')
    logging.debug('\nFootprint (WRS2 descending should be GCS84):')
    if not os.path.isfile(footprint_path):
        logging.error(('\nERROR: The footprint shapefile does not exist:'
                       '\n  {}').format(footprint_path))
        sys.exit()
    tile_gcs_osr = drigo.feature_path_osr(footprint_path)
    logging.debug('  OSR: {}'.format(tile_gcs_osr))

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
                '  GDAL does not appear to be built with GEOS support\n' +
                '  Using tile extents instead')
        if tile_proj_geom is None:
            # Project the path/row extent
            tile_gcs_extent = drigo.Extent(tile_gcs_geom.GetEnvelope())
            tile_gcs_extent = tile_gcs_extent.ogrenv_swap()
            tile_gcs_extent.buffer_extent(tile_gcs_buffer)
            tile_proj_geom = tile_gcs_extent.geometry()
        tile_proj_geom.Transform(tile_proj_tx)
        tile_proj_wkt_dict[tile_name] = tile_proj_geom.ExportToWkt()

    # Get the extent and name of each zone feature
    logging.debug('\nZones')
    zones_name_dict = dict()
    zones_extent_dict = dict()
    zones_geom_hull_dict = dict()
    logging.debug('  Zone name field: {}'.format(zones_name_field))
    # shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    zones_ds = ogr.Open(zones_path, 0)
    zones_lyr = zones_ds.GetLayer()
    zones_lyr.ResetReading()
    for zones_ftr in zones_lyr:
        zones_fid = zones_ftr.GetFID()
        zones_geom = zones_ftr.GetGeometryRef()
        zones_geom_hull_dict[zones_fid] = zones_geom.ConvexHull()
        zones_extent_dict[zones_fid] = drigo.Extent(
            zones_geom.GetEnvelope()).ogrenv_swap()
        if zones_name_field.upper() != 'FID':
            zone_name = str(zones_ftr.GetField(zones_name_field))
            zone_name = zone_name.replace('/', ' ')
            zones_name_dict[zones_fid] = zone_name
        else:
            zones_name_dict[zones_fid] = zones_fid
    zones_ds = None

    # # DEADBEEF - Don't project extents since output OSR is from zones_path
    # # Project the zone extents to the snap spatial reference
    # logging.debug('  {}'.format(zones_osr))
    # for fid, extent in sorted(zones_extent_dict.items()):
    #     proj_extent = drigo.project_extent(
    #         extent, zones_osr, env.snap_osr, env.cellsize)
    #     proj_extent.adjust_to_snap(
    #         'EXPAND', env.snap_x, env.snap_y, env.cellsize)
    #     zones_extent_dict[fid] = proj_extent

    # # DEADBEEF - Transform wasn't working on linux
    # # Project the zone convex hull geometries to the snap spatial reference
    # logging.debug('  {}'.format(zones_osr))
    # transform = osr.CoordinateTransformation(zones_osr, env.snap_osr)
    # for fid, geom in sorted(zones_geom_hull_dict.items()):
    #     geom.Transform(transform)
    #     zones_geom_hull_dict[fid] = geom

    # Log the zone names and extents
    logging.debug('  FID: Zone name')
    for fid, name in sorted(zones_name_dict.items()):
        logging.debug('    {}: {}'.format(fid, name))
    # logging.debug('  FID: Zone extent')
    # for fid, extent in sorted(zones_extent_dict.items()):
    #     logging.debug('    {}: {}'.format(fid, extent))

    # Union zones extent and tiles extent to get mask extent
    zones_extent = drigo.union_extents(zones_extent_dict.values())
    logging.debug('  Zones extent: {}'.format(zones_extent))
    tiles_extent = drigo.union_extents([
        drigo.Extent(ogr.CreateGeometryFromWkt(
            tile_proj_wkt_dict[tile]).GetEnvelope()).ogrenv_swap()
        for tile in tile_list])
    logging.debug('  Tiles extent: {}'.format(tiles_extent))
    env.mask_extent = drigo.intersect_extents([zones_extent, tiles_extent])
    env.mask_extent.buffer_extent(env.cellsize)
    env.mask_extent.adjust_to_snap('EXPAND')
    env.mask_geo = env.mask_extent.geo(env.cellsize)
    env.mask_rows, env.mask_cols = env.mask_extent.shape()
    env.mask_shape = (env.mask_rows, env.mask_cols)
    logging.debug('  Mask rows: {}  cols: {}'.format(
        env.mask_rows, env.mask_cols))
    logging.debug('  Mask extent: {}'.format(env.mask_extent))
    logging.debug('  Mask geo: {}'.format(env.mask_geo))

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
            env.mask_extent, env.snap_osr, ppt_osr, env.cellsize)
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

    # Table path format
    logging.info('\nOutput')
    block_table_path = os.path.join(
        zones_table_ws,
        'daily_block_stats_{}{}.csv'.format(mc_str.lower(), year))
    daily_zones_table_path = os.path.join(
        zones_table_ws,
        'daily_zonal_stats_{}{}.csv'.format(mc_str.lower(), year))
    monthly_zones_table_path = os.path.join(
        zones_table_ws,
        'monthly_zonal_stats_{}{}.csv'.format(mc_str.lower(), year))
    annual_zones_table_path = os.path.join(
        zones_table_ws,
        'annual_zonal_stats_{}{}.csv'.format(mc_str.lower(), year))
    # Fow now, always overwrite
    if os.path.isfile(block_table_path):
        os.remove(block_table_path)
    if os.path.isfile(daily_zones_table_path):
        os.remove(daily_zones_table_path)
    if os.path.isfile(monthly_zones_table_path):
        os.remove(monthly_zones_table_path)
    if os.path.isfile(annual_zones_table_path):
        os.remove(annual_zones_table_path)
    block_header_list = [
        'FID', 'BLOCK_I', 'BLOCK_J', 'DATE',
        'PIXELS', ndvi_field, etrf_field, etr_field, et_field, ppt_field]
    daily_zone_header_list = [
        'FID', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'PIXELS', ndvi_field, etrf_field, etr_field, et_field, ppt_field]
    monthly_zone_header_list = [
        'FID', 'YEAR', 'MONTH',
        'PIXELS', ndvi_field, etrf_field, etr_field, et_field, ppt_field]
    annual_zone_header_list = [
        'FID', 'YEAR',
        'PIXELS', ndvi_field, etrf_field, etr_field, et_field, ppt_field]
    if zones_name_field.upper() != 'FID':
        block_header_list.insert(1, zones_name_field)
        daily_zone_header_list.insert(1, zones_name_field)
        monthly_zone_header_list.insert(1, zones_name_field)
        annual_zone_header_list.insert(1, zones_name_field)
    float_fields = [etrf_field, etr_field, et_field, ppt_field]
    int_fields = ['FID', 'PIXELS']
    block_zones_df = pd.DataFrame(columns=block_header_list)
    block_zones_df[int_fields] = block_zones_df[int_fields].astype(np.int64)
    block_zones_df[float_fields] = block_zones_df[float_fields].astype(np.float32)

    # Initialize queues
    input_q = Queue()
    # output_q = Queue(mp_procs + 1)
    output_q = Queue(output_queue)
    queue_cnt = 0

    # Load each block into queue
    logging.info('\nGenerating block tasks')
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

        # Only queue blocks that intersect zone extents
        zones_fid_list = sorted([
            zone_fid
            for zone_fid, zone_extent in zones_extent_dict.items()
            if drigo.extents_overlap(zone_extent, block_extent)])
        if not zones_fid_list:
            continue
        # logging.debug(zones_fid_list)

        # DEADBEEF - I can't Intersect on NEX due to GDAL/osgeo/proj4 issues
        # # Only queue blocks that intersect zone geometries
        # zones_fid_list = sorted([
        #     zone_fid for zone_fid in zones_fid_list
        #     if zones_geom_hull_dict[zone_fid].Intersects(
        #         block_extent.geometry())])
        # if not zones_fid_list:
        #     continue
        # # logging.debug(zones_fid_list)

        # Place inputs into queue by block
        usable_scene_cnt = 2
        input_q.put([
            b_i, b_j, block_rows, block_cols, block_extent, block_tile_list,
            date_list, etr_date_list, ppt_date_list,
            year, etrf_input_ws, tile_image_dict, env.cellsize, env.snap_proj,
            # etr_array, drigo.osr_proj(etr_osr), etr_cs, etr_extent, etr_date_list,
            # ppt_array, drigo.osr_proj(ppt_osr), ppt_cs, ppt_extent, ppt_date_list,
            # awc_array, drigo.osr_proj(awc_osr), awc_cs, awc_extent,
            etr_shmem, etr_shape, drigo.osr_proj(etr_osr), etr_cs, etr_extent,
            ppt_shmem, ppt_shape, drigo.osr_proj(ppt_osr), ppt_cs, ppt_extent,
            awc_shmem, awc_shape, drigo.osr_proj(awc_osr), awc_cs, awc_extent,
            etrf_raster, ndvi_raster, swb_adjust_dict, etrf_ndvi_dict,
            zones_path, zones_fid_list, zones_mask, zones_buffer,
            usable_scene_cnt, mosaic_method, fill_method, interp_method,
            calc_flags, low_etrf_limit, high_etrf_limit, debug_flag])
        queue_cnt += 1
    if queue_cnt == 0:
        logging.error(
            '\nERROR: No blocks were loaded into the queue, exiting')
        return False
    else:
        logging.debug('  {} blocks'.format(queue_cnt))

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

        b_i, b_j, output_list = output_q.get()
        logging.info('Block  y: {:5d}  x: {:5d}  ({}/{})'.format(
            b_i, b_j, queue_i + 1, queue_cnt))
        output_list = pickle.loads(output_list)
        if not output_list:
            continue

        # Write daily ETrF, ETr, and ET means to a table
        row_list = []
        for zone_fid, zone_count, ndvi_a, etrf_a, etr_a, ppt_a in output_list:
            zone_dict = {
                'FID': zone_fid,
                'BLOCK_I': b_i,
                'BLOCK_J': b_j,
                'PIXELS': zone_count}
            if zones_name_field.upper() != 'FID':
                zone_dict[zones_name_field] = zones_name_dict[zone_fid]
            for date_i, (ndvi, etrf, etr, ppt) in enumerate(
                    zip(ndvi_a, etrf_a, etr_a, ppt_a)):
                row_dict = zone_dict.copy()
                row_dict.update({
                    'DATE': date_list[date_i].isoformat(),
                    ndvi_field: float(ndvi),
                    etrf_field: float(etrf),
                    etr_field: float(etr),
                    et_field: float(etrf) * float(etr),
                    ppt_field: float(ppt)})
                row_list.append(row_dict)
        block_zones_df = block_zones_df.append(row_list, ignore_index=True)
        # block_zones_df = block_zones_df.reindex(header_list, axis=1)
        # block_zones_df[int_fields] = block_zones_df[int_fields].astype(np.int64)
        # block_zones_df[float_fields] = block_zones_df[float_fields].astype(np.float32)
        # block_zones_df.sort_values(
        #     ['FID', 'DATE'], ascending=[True, True], inplace=True)
        # block_zones_df.to_csv(block_table_path, index=False)

        del output_list
        logging.info(
            ('  Block Time: {:.1f}s  (mean {:.1f}s, '
             '{:.2f} hours remaining)').format(
                clock() - block_clock,
                (clock() - proc_clock) / (queue_i + 1),
                (queue_cnt - queue_i + 1) *
                (clock() - proc_clock) / (queue_i + 1) / 3600))

    # Close the queueus
    for i in range(max(1, mp_procs - 1)):
        input_q.put(None)
    input_q.close()
    output_q.close()
    del input_q, output_q
    logging.info('Time: {:.1f}'.format(clock() - main_clock))

    # Write the block zonal stats
    if calc_flags['zones']:
        logging.info('\nWriting block zonal stats')
        block_zones_df.sort_values(
            ['FID', 'DATE'], ascending=[True, True], inplace=True)
        block_zones_df = block_zones_df.reindex(block_header_list, axis=1)
        block_zones_df.to_csv(block_table_path, index=False)

    # Combine block means
    if calc_flags['zones'] or calc_flags['plots']:
        logging.info('\nAggregating zonal stats')

        # Manually compute the weighted average, account for nodata
        scale_fields = [ndvi_field, etrf_field, etr_field, et_field, ppt_field]
        block_zones_df[scale_fields] = block_zones_df[scale_fields].multiply(
            block_zones_df['PIXELS'], axis="index")
        daily_f = {
            'PIXELS': np.nansum,
            ndvi_field: np.nansum,
            etrf_field: np.nansum,
            etr_field: np.nansum,
            et_field: np.nansum,
            ppt_field: np.nansum}
        if zones_name_field.upper() != 'FID':
            daily_zones_df = block_zones_df.groupby(
                ['FID', zones_name_field, 'DATE'], as_index=False).agg(daily_f)
        else:
            daily_zones_df = block_zones_df.groupby(
                ['FID', 'DATE'], as_index=False).agg(daily_f)
        daily_zones_df[scale_fields] = daily_zones_df[scale_fields].divide(
            daily_zones_df['PIXELS'], axis="index")

        # This approach doesn't work if there are NaN
        # This was causing a problem when trying to interpolate folders that
        #   only had ETrF and not NDVI
        # block_wm = lambda x: np.average(
        #     x, weights=block_zones_df.loc[x.index, "PIXELS"])
        # daily_f = {
        #     'PIXELS': {'PIXELS': np.sum},
        #     # ndvi_field: {ndvi_field: block_wm},
        #     ndvi_field: {ndvi_field: np.nanmean},
        #     # ndvi_field: {ndvi_field: np.mean},
        #     etrf_field: {etrf_field: block_wm},
        #     etr_field: {etr_field: block_wm},
        #     et_field: {et_field: block_wm},
        #     ppt_field: {ppt_field: block_wm}}
        # if zones_name_field.upper() != 'FID':
        #     daily_zones_df = block_zones_df.groupby(
        #         ['FID', zones_name_field, 'DATE'],
        #         as_index=False).agg(daily_f)
        # else:
        #     daily_zones_df = block_zones_df.groupby(
        #         ['FID', 'DATE'], as_index=False).agg(daily_f)
        # daily_zones_df.columns = daily_zones_df.columns.droplevel(1)

        daily_zones_df.sort_values(
            ['FID', 'DATE'], ascending=[True, True], inplace=True)
        # Add extra date columns
        daily_zones_df['DATE'] = pd.to_datetime(daily_zones_df['DATE'])
        daily_zones_df['YEAR'] = daily_zones_df['DATE'].dt.year
        daily_zones_df['MONTH'] = daily_zones_df['DATE'].dt.month
        daily_zones_df['DAY'] = daily_zones_df['DATE'].dt.day
        daily_zones_df['DOY'] = daily_zones_df['DATE'].dt.strftime('%j')
        daily_zones_df = daily_zones_df.reindex(daily_zone_header_list, axis=1)

    # Write tables
    if calc_flags['daily_zones_table']:
        daily_zones_df.to_csv(
            daily_zones_table_path, index=False, float_format='%.6f')

    if calc_flags['monthly_zones_table']:
        monthly_f = {
            'PIXELS': np.max,
            ndvi_field: np.nanmean,
            etrf_field: np.nanmean,
            etr_field: np.nansum,
            et_field: np.nansum,
            ppt_field: np.nansum}
        if zones_name_field.upper() != 'FID':
            monthly_zones_df = daily_zones_df.groupby(
                ['FID', zones_name_field, 'MONTH'],
                as_index=False).agg(monthly_f)
        else:
            monthly_zones_df = daily_zones_df.groupby(
                ['FID', 'MONTH'], as_index=False).agg(monthly_f)
        # monthly_zones_df.columns = monthly_zones_df.columns.droplevel(1)
        # Recompute ETrF from ET and ETr
        monthly_zones_df[etrf_field] = (
            monthly_zones_df[et_field] / monthly_zones_df[etr_field])
        monthly_zones_df.sort_values(
            ['FID', 'MONTH'], ascending=[True, True], inplace=True)
        monthly_zones_df = monthly_zones_df.reindex(
            monthly_zone_header_list, axis=1)
        monthly_zones_df.to_csv(
            monthly_zones_table_path, index=False, float_format='%.6f')

    if calc_flags['annual_zones_table']:
        print('ANNUAL')
        annual_f = {
            'PIXELS': np.max,
            ndvi_field: np.nanmean,
            etrf_field: np.nanmean,
            etr_field: np.nansum,
            et_field: np.nansum,
            ppt_field: np.nansum}
        if zones_name_field.upper() != 'FID':
            annual_zones_df = daily_zones_df.groupby(
                ['FID', zones_name_field, 'YEAR'],
                as_index=False).agg(annual_f)
        else:
            annual_zones_df = daily_zones_df.groupby(
                ['FID', 'YEAR'], as_index=False).agg(annual_f)
        # annual_zones_df.columns = annual_zones_df.columns.droplevel(1)
        # Recompute ETrF from ET and ETr
        annual_zones_df[etrf_field] = (
            annual_zones_df[et_field] / annual_zones_df[etr_field])
        annual_zones_df.sort_values(
            ['FID', 'YEAR'], ascending=[True, True], inplace=True)
        annual_zones_df = annual_zones_df.reindex(
            annual_zone_header_list, axis=1)
        annual_zones_df.to_csv(
            annual_zones_table_path, index=False, float_format='%.6f')

    # Build plots for each FID
    if calc_flags['plots']:
        logging.info('\nPlots')
        for zone_fid, zone_name in zones_name_dict.items():
            count_array = daily_zones_df.loc[
                daily_zones_df['FID'] == zone_fid, 'PIXELS'].values
            if not np.any(count_array):
                logging.debug('  FID {}: {}'.format(zone_fid, zone_name))
                logging.debug('    Zonal stats not computed, skipping')
                continue
            else:
                logging.info('  FID {}: {}'.format(zone_fid, zone_name))

            # Square meters to acres
            zone_count = max(count_array)
            zone_area = 0.000247105 * zone_count * float(env.cellsize) ** 2

            doy_array = daily_zones_df.loc[
                daily_zones_df['FID'] == zone_fid, 'DOY'].values
            zone_ndvi_array = daily_zones_df.loc[
                daily_zones_df['FID'] == zone_fid, ndvi_field].values
            zone_etrf_array = daily_zones_df.loc[
                daily_zones_df['FID'] == zone_fid, etrf_field].values
            zone_etr_array = daily_zones_df.loc[
                daily_zones_df['FID'] == zone_fid, etr_field].values
            zone_et_array = daily_zones_df.loc[
                daily_zones_df['FID'] == zone_fid, et_field].values
            zone_ppt_array = daily_zones_df.loc[
                daily_zones_df['FID'] == zone_fid, ppt_field].values

            if zones_name_field.upper() != 'FID':
                zone_title = ' - {}'.format(zones_name_dict[zone_fid])
                zone_path = '_{}'.format(
                    str(zones_name_dict[zone_fid]).lower().replace(' ', '_'))
            else:
                zone_title = ''
                zone_path = ''

            if zones_buffer != 0:
                buffer_title = ' ({}m BUFFER)'.format()
            else:
                buffer_title = ''

            zone_ndvi_plot_path = os.path.join(
                zones_ndvi_plots_ws,
                'fid_{}{}.png'.format(zone_fid, zone_path))
            zone_etrf_plot_path = os.path.join(
                zones_etrf_plots_ws,
                'fid_{}{}.png'.format(zone_fid, zone_path))
            zone_etr_plot_path = os.path.join(
                zones_etr_plots_ws,
                'fid_{}{}.png'.format(zone_fid, zone_path))
            zone_et_plot_path = os.path.join(
                zones_et_plots_ws,
                'fid_{}{}.png'.format(zone_fid, zone_path))
            zone_ppt_plot_path = os.path.join(
                zones_ppt_plots_ws,
                'fid_{}{}.png'.format(zone_fid, zone_path))

            if calc_flags['daily_ndvi_plots'] and np.any(zone_ndvi_array):
                fig, ax = plt.subplots()
                ax.set_xlabel('Day of Year')
                plt.plot(doy_array, zone_ndvi_array, '.b')
                ax.set_xlim([0, 365])
                if any(plots_ndvi_ylim):
                    ax.set_ylim(plots_ndvi_ylim)
                ax.set_ylabel('{}'.format(ndvi_name))
                ax.set_title(
                    '{}{} - FID {}{}'.format(
                        ndvi_name, buffer_title, zone_fid, zone_title),
                    y=1.05)
                ax.xaxis.set_major_locator(MultipleLocator(30))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                ax.tick_params(axis='x', which='both', top='off')
                ax.tick_params(axis='y', which='both', right='off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if plots_zone_area_flag:
                    ax.annotate(
                        '{:0.2f} acres\n{:d} cells'.format(
                            zone_area, zone_count),
                        xy=(0.80, 0.89), xycoords='axes fraction')
                plt.savefig(zone_ndvi_plot_path, dpi=dpi)
                plt.close(fig)
                plt.clf()

            if calc_flags['daily_etrf_plots'] and np.any(zone_etrf_array):
                fig, ax = plt.subplots()
                ax.set_xlabel('Day of Year')
                plt.plot(doy_array, zone_etrf_array, '.b')
                ax.set_xlim([0, 365])
                if any(plots_etrf_ylim):
                    ax.set_ylim(plots_etrf_ylim)
                ax.set_ylabel('{}'.format(etrf_name))
                ax.set_title(
                    'METRIC {}{} - FID {}{}'.format(
                        etrf_name, buffer_title, zone_fid, zone_title),
                    y=1.05)
                ax.xaxis.set_major_locator(MultipleLocator(30))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                ax.tick_params(axis='x', which='both', top='off')
                ax.tick_params(axis='y', which='both', right='off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if plots_zone_area_flag:
                    ax.annotate(
                        '{:0.2f} acres\n{:d} cells'.format(
                            zone_area, zone_count),
                        xy=(0.80, 0.89), xycoords='axes fraction')
                plt.savefig(zone_etrf_plot_path, dpi=dpi)
                plt.close(fig)
                plt.clf()

            if calc_flags['daily_etr_plots'] and np.any(zone_etr_array):
                fig, ax = plt.subplots()
                ax.set_xlabel('Day of Year')
                plt.plot(doy_array, zone_etr_array, '.b')
                ax.set_xlim([0, 365])
                if any(plots_etr_ylim):
                    ax.set_ylim(plots_etr_ylim)
                ax.set_ylabel('{} [mm]'.format(etr_name))
                ax.set_title(
                    '{}{} - FID {}{}'.format(
                        etr_name, buffer_title, zone_fid, zone_title),
                    y=1.05)
                ax.xaxis.set_major_locator(MultipleLocator(30))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.tick_params(axis='x', which='both', top='off')
                ax.tick_params(axis='y', which='both', right='off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if plots_zone_area_flag:
                    ax.annotate(
                        '{:0.2f} acres\n{:d} cells'.format(
                            zone_area, zone_count),
                        xy=(0.80, 0.89), xycoords='axes fraction')
                plt.savefig(zone_etr_plot_path, dpi=dpi)
                plt.close(fig)
                plt.clf()

            if calc_flags['daily_et_plots'] and np.any(zone_et_array):
                fig, ax = plt.subplots()
                ax.set_xlabel('Day of Year')
                plt.plot(doy_array, zone_et_array, '.b')
                ax.set_xlim([0, 365])
                if any(plots_et_ylim):
                    ax.set_ylim(plots_et_ylim)
                ax.set_ylabel('{} [mm]'.format(et_name))
                ax.set_title(
                    'METRIC {}{} - FID {}{}'.format(
                        et_name, buffer_title, zone_fid, zone_title),
                    y=1.05)
                ax.xaxis.set_major_locator(MultipleLocator(30))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.tick_params(axis='x', which='both', top='off')
                ax.tick_params(axis='y', which='both', right='off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if plots_zone_area_flag:
                    ax.annotate(
                        '{:0.2f} acres\n{:d} cells'.format(
                            zone_area, zone_count),
                        xy=(0.80, 0.89), xycoords='axes fraction')
                plt.savefig(zone_et_plot_path, dpi=dpi)
                plt.close(fig)
                plt.clf()

            if calc_flags['daily_ppt_plots'] and np.any(zone_ppt_array):
                fig, ax = plt.subplots()
                ax.set_xlabel('Day of Year')
                plt.bar(doy_array, zone_ppt_array, width=1, color='0.5',
                        linewidth=0)
                ax.set_xlim([0, 365])
                if any(plots_ppt_ylim):
                    ax.set_ylim(plots_ppt_ylim)
                ax.set_ylabel('{} [mm]'.format(ppt_name))
                ax.set_title(
                    '{}{} - FID {}{}'.format(
                        ppt_name, buffer_title, zone_fid, zone_title),
                    y=1.05)
                ax.xaxis.set_major_locator(MultipleLocator(30))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.yaxis.set_minor_locator(MultipleLocator(10))
                ax.tick_params(axis='x', which='both', top='off')
                ax.tick_params(axis='y', which='both', right='off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if plots_zone_area_flag:
                    ax.annotate(
                        '{:0.2f} acres\n{:d} cells'.format(
                            zone_area, zone_count),
                        xy=(0.80, 0.89), xycoords='axes fraction')
                plt.savefig(zone_ppt_plot_path, dpi=dpi)
                plt.close(fig)
                plt.clf()


def block_worker(args, input_q, output_q):
    """Worker function for multiprocessing with input and output queues
    Pass block indices through to the output
    """
    # signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        args = input_q.get(block=True)
        if args is None:
            break
        output_q.put([args[0], args[1], block_func(*args[2:])], block=True)
        # output_q.put([args[0], args[1]] + block_func(*args[2:]), block=True)


def block_func(block_rows, block_cols, block_extent, block_tile_list,
               interp_date_list, etr_date_list, ppt_date_list,
               year, etrf_input_ws, tile_image_dict, cellsize, snap_proj,
               etr_shmem, etr_shape, etr_proj, etr_cs, etr_extent,
               ppt_shmem, ppt_shape, ppt_proj, ppt_cs, ppt_extent,
               awc_shmem, awc_shape, awc_proj, awc_cs, awc_extent,
               etrf_raster, ndvi_raster, swb_adjust_dict, etrf_ndvi_dict,
               zones_path, zones_fid_list, zones_mask, zones_buffer,
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
    if not block_tile_list:
        return pickle.dumps([[]], protocol=-1)
    # if ((not calc_flags['daily'] and
    #      not calc_flags['monthly'] and
    #      not calc_flags['annual']) or
    #     not block_tile_list):
    #     return pickle.dumps([], protocol=-1)

    # Build arrays for storing data
    # ET array will initially be loaded with ETrF
    array_shape = len(interp_date_list), block_rows, block_cols

    # Get Count mask from ETrF
    if calc_flags['etrf']:
        etrf_array = interp.load_etrf_func(
            array_shape, interp_date_list, etrf_input_ws, year,
            etrf_raster, block_tile_list, block_extent,
            tile_image_dict, mosaic_method, gdal.GRA_Bilinear,
            drigo.proj_osr(snap_proj), cellsize, block_extent, debug_flag)
        etrf_mask = np.isfinite(etrf_array)

        # Clear pixels that don't have a suitable number of scenes
        # I could also check the distribution of scenes (i.e. early and late)
        count_array = etrf_mask.sum(dtype=np.uint8, axis=0)
        count_mask = count_array >= usable_image_count
        if not np.any(count_mask):
            return pickle.dumps([], protocol=-1)

        # I only need to clear/reset pixels > 0 and < count
        clear_mask = (count_array > 0) & (~count_mask)
        if np.any(clear_mask):
            etrf_array[:, clear_mask] = np.nan
            etrf_mask[:, clear_mask] = False
        del clear_mask, count_array

    if calc_flags['ndvi']:
        ndvi_array = interp.load_etrf_func(
            array_shape, interp_date_list, etrf_input_ws, year,
            ndvi_raster, block_tile_list, block_extent,
            tile_image_dict, mosaic_method, gdal.GRA_Bilinear,
            drigo.proj_osr(snap_proj), cellsize, block_extent, debug_flag)
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

    if calc_flags['etr']:
        # # DEADBEEF - Working implementation of a shared memory array
        # User must pass in etr_shmem and etr_shape
        etr_input_array = ctypeslib.as_array(etr_shmem)
        etr_input_array.shape = etr_shape

        # Project all ETr bands/days at once
        etr_array = drigo.project_array(
            etr_input_array, gdal.GRA_Bilinear,
            drigo.proj_osr(etr_proj), etr_cs, etr_extent,
            drigo.proj_osr(snap_proj), cellsize, block_extent)
        # Project each PPT band/day separately
        # etr_array = np.full(array_shape, np.nan, np.float32)
        # for date_i, date_obj in enumerate(date_list):
        #     etr_array[date_i, :, :] = drigo.project_array(
        #         etr_input_array[date_i, :, :], gdal.GRA_Bilinear,
        #         drigo.proj_osr(etr_proj), etr_cs, etr_extent,
        #         drigo.proj_osr(snap_proj), cellsize, block_extent)

        # In table interpolator, only include ETr for pixels with ETrF data
        etr_array[:, ~count_mask] = np.nan
    else:
        etr_array = np.full(array_shape, np.nan, np.float32)

    if calc_flags['ppt']:
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

        # In table interpolator, only include PPT for pixels with ETrF data
        ppt_array[:, ~count_mask] = np.nan
    else:
        ppt_array = np.full(array_shape, np.nan, np.float32)

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
        # awc_swb_extent.buffer_extent(awc_cs * 3)
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
        del ke_swb_array

    #     # etrf_background = et_common.array_swb_func(
    #     #     etr=etr_array, ppt=ppt_array, awc=awc_array)
    #     # # How should background mean get computed?
    #     # # Mean of the previous 30 days maybe?
    #     # etrf_background_mean =
    #     # sleep(5)
    #     # fc = interp.swb_adjust_fc(
    #     #     ndvi_array, ndvi_full_cover=swb_adjust_dict['full'],
    #     #     ndvi_bare_soil=swb_adjust_dict['bare'])
    #     #     # ndvi_full_cover=tile_ndvi_dict[year][tile_name][image_id]['cold'],
    #     #     # ndvi_bare_soil=tile_ndvi_dict[year][tile_name][image_id]['hot'])
    #     # etrf_transpiration = etrf_array - ((1 - fc) * etrf_background_mean)
    #     # np.maximum(
    #     #     etrf_transpiration, etrf_background, out=etrf_transpiration)
    #     # etrf_adjusted = (
    #     #     ((1 - fc) * etrf_background) + (fc * etrf_transpiration))
    #     # etrf_array[etrf_mask] = etrf_adjusted[etrf_mask]
    #     del ke_array

    # Interpolate ETrF after SWB adjust
    if calc_flags['etrf']:
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
    if calc_flags['ndvi']:
        # Calculate dates where NDVI has data (scene dates)
        sub_i_mask = np.isfinite(
            ndvi_array).sum(axis=2).sum(axis=1).astype(np.bool)
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
            # image_et_array = interp.spatial_fill_func(
            #     image_et_array, block_mask)
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

    if zones_mask:
        # For now assume zones_mask was used to define the spatial reference
        zones_array, zones_nodata = drigo.raster_to_array(
            zones_mask, mask_extent=block_extent, return_nodata=True)
        # Assume nodata and 0 should be masked in statistics
        count_mask &= (zones_array != 0)
        if zones_nodata != 0:
            count_mask &= (zones_array != zones_nodata)
        del zones_array, zones_nodata

    block_zone_stats = []
    if zones_fid_list:
        # Initialize the zones in memory raster
        mem_driver = gdal.GetDriverByName('MEM')
        zones_raster_ds = mem_driver.Create(
            '', block_cols, block_rows, 1, gdal.GDT_Byte)
        zones_raster_ds.SetProjection(snap_proj)
        zones_raster_ds.SetGeoTransform(
            drigo.extent_geo(block_extent, cs=cellsize))
        zones_band = zones_raster_ds.GetRasterBand(1)
        zones_band.SetNoDataValue(0)

        # Open the zones shapefile once
        # zones_ftr_ds = ogr.Open(zones_path, 0)
        # zones_layer = zones_ftr_ds.GetLayer()
        # zones_layer.ResetReading()

        # Compute zonal stats for each zone separately
        for zone_fid in zones_fid_list:
            zones_ftr_ds = ogr.Open(zones_path, 0)
            zones_layer = zones_ftr_ds.GetLayer()
            zones_layer.SetAttributeFilter("{} = {}".format('FID', zone_fid))
            # Clear the raster before rasterizing
            zones_band.Fill(0)
            gdal.RasterizeLayer(zones_raster_ds, [1], zones_layer)
            zones_ftr_ds = None
            zone_array = drigo.raster_ds_to_array(
                zones_raster_ds, return_nodata=False)
            zone_mask = (zone_array > 0) & count_mask
            if np.any(zone_mask):
                block_zone_stats.append(
                    [zone_fid, np.sum(zone_mask),
                     np.nanmean(ndvi_array[:, zone_mask], axis=1),
                     np.nanmean(etrf_array[:, zone_mask], axis=1),
                     np.nanmean(etr_array[:, zone_mask], axis=1),
                     np.nanmean(ppt_array[:, zone_mask], axis=1)])
            del zone_array, zone_mask
        zones_ftr_ds = None
        zones_raster_ds = None
    return pickle.dumps(block_zone_stats, protocol=-1)


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
        '-dpi', '--dpi', default=150, type=int,
        help='Desired resolution of output PNG files in DPI (dots per inch)')
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
        '-q', '--queue', default=1, type=int, metavar='N',
        help='Size of output queue')
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
            log_file_name = 'mc{:02d}_interpolate_tables_log.txt'.format(
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
                       overwrite_flag=args.overwrite,
                       mp_procs=args.multiprocessing, delay=args.delay,
                       debug_flag=args.loglevel==logging.DEBUG,
                       output_queue=args.queue, dpi=args.dpi)
