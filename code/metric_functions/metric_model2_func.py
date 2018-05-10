#!/usr/bin/env python
#--------------------------------
# Name:         metric_model2_func.py
# Purpose:      Calculate METRIC Model 2
# Notes:        GDAL Block Version
#--------------------------------

import argparse
import datetime as dt
import json
import logging
import os
import random
import sys
from time import sleep

import drigo
import numpy as np
from osgeo import gdal

import et_common
import et_image
import et_numpy
from python_common import open_ini, read_param, remove_file


def metric_model2(image_ws, ini_path,
                  mc_iter=None, kc_cold=None, kc_hot=None,
                  cold_xy=None, hot_xy=None, stats_flag=None,
                  overwrite_flag=None, ts_diff_threshold=4):
    """METRIC Model 2 Version

    Parameters
    ----------
    image_ws : str
        Image folder path.
    ini_path : str
        METRIC config file path.
    mc_iter : int
        Iteration number for Monte Carlo processing.
    kc_cold : float
        Kc value at the cold calibration point.
    kc_hot : float
        Kc value at the hot calibration point.
    cold_xy : tuple
        Location of the cold calibration point.
    hot_xy : tuple
        Location of the hot calibration point.
    ovewrite_flag : bool, optional
        If True, overwrite existing files (the default is None).

    Returns
    -------
    True if successful

    """

    logging.info('METRIC Model 2')
    log_fmt = '  {:<18s} {}'
    pixel_str_fmt = '    {:<14s}  {:>14s}  {:>14s}'
    pixel_flt_fmt = '    {:<14s}  {:>14.2f}  {:>14.2f}'

    env = drigo.env
    image = et_image.Image(image_ws, env)
    logging.info(log_fmt.format('Image:', image.folder_id))
    np.seterr(invalid='ignore')

    # # Check that image_ws is valid
    # image_re = re.compile(
    #     '^(LT04|LT05|LE07|LC08)_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})')
    # if not os.path.isdir(image_ws) or not image_re.match(scene_id):
    #     logging.error('\nERROR: Image folder is invalid or does not exist\n')
    #     return False

    # Folder Paths
    zom_ws = os.path.join(image_ws, 'ROUGHNESS')
    rn_ws = os.path.join(image_ws, 'RN')
    g_ws = os.path.join(image_ws, 'G')
    h_ws = os.path.join(image_ws, 'H')
    le_ws = os.path.join(image_ws, 'LE')
    etrf_ws = os.path.join(image_ws, 'ETRF')
    et24_ws = os.path.join(image_ws, 'ET24')

    # Open config file
    config = open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    # Recently added to read in variables for Ts correction
    ts_correction_flag  = read_param(
        'Ts_correction_flag', True, config, 'INPUTS')
    k_value = read_param('K_value', 2, config, 'INPUTS')
    dense_veg_min_albedo = read_param(
        'dense_veg_min_albedo', 0.18, config, 'INPUTS')

    # Arrays are processed by block
    bs = read_param('block_size', 1024, config, 'INPUTS')
    logging.info(log_fmt.format('Block Size:', bs))

    # Raster pyramids/statistics
    pyramids_flag = read_param('pyramids_flag', False, config, 'INPUTS')
    if pyramids_flag:
        gdal.SetConfigOption('HFA_USE_RRD', 'YES')
    if stats_flag is None:
        stats_flag = read_param('statistics_flag', False, config, 'INPUTS')

    # Overwrite
    if overwrite_flag is None:
        overwrite_flag = read_param('overwrite_flag', True, config, 'INPUTS')

    # Use iteration number to file iteration string
    if mc_iter is None:
        iter_fmt = '.img'
    elif int(mc_iter) < 0:
        logging.error('\nERROR: Iteration number must be a positive integer')
        return False
    else:
        iter_fmt = '_{:02d}.img'.format(int(mc_iter))
        logging.info('  {:<18s} {}'.format('Iteration:', mc_iter))

    # Check that common_area raster exists
    if not os.path.isfile(image.common_area_raster):
        logging.error(
            '\nERROR: A common area raster was not found.' +
            '\nERROR: Please rerun prep tool to build these files.\n')
        return False

    # Use common_area to set mask parameters
    common_ds = gdal.Open(image.common_area_raster)
    env.mask_geo = drigo.raster_ds_geo(common_ds)
    env.mask_rows, env.mask_cols = drigo.raster_ds_shape(common_ds)
    env.mask_extent = drigo.geo_extent(
        env.mask_geo, env.mask_rows, env.mask_cols)
    env.mask_array = drigo.raster_ds_to_array(common_ds, return_nodata=False)
    env.mask_path = image.common_area_raster
    common_ds = None
    logging.debug(log_fmt.format('Mask Extent:', env.mask_extent))

    # Set raster names
    r_fmt = '.img'
    raster_dict = dict()
    raster_dict['dem'] = os.path.join(image.support_ws, 'dem' + r_fmt)
    raster_dict['landuse'] = os.path.join(image.support_ws, 'landuse' + r_fmt)
    raster_dict['slp'] = os.path.join(image.support_ws, 'slope' + r_fmt)
    raster_dict['asp'] = os.path.join(image.support_ws, 'aspect' + r_fmt)
    raster_dict['lat'] = os.path.join(image.support_ws, 'latitude' + r_fmt)
    raster_dict['lon'] = os.path.join(image.support_ws, 'longitude' + r_fmt)
    raster_dict['cos_theta'] = os.path.join(image.support_ws, 'cos_theta' + r_fmt)
    raster_dict['albedo_sur'] = image.albedo_sur_raster
    raster_dict['tau'] = os.path.join(image_ws, 'transmittance' + r_fmt)

    raster_dict['ea'] = image.metric_ea_raster
    raster_dict['wind'] = image.metric_wind_raster
    raster_dict['etr'] = image.metric_etr_raster
    raster_dict['etr_24hr'] = image.metric_etr_24hr_raster
    raster_dict['tair'] = image.metric_tair_raster

    raster_dict['ndvi_toa'] = image.ndvi_toa_raster
    raster_dict['ndwi_toa'] = image.ndwi_toa_raster
    raster_dict['lai_toa'] = image.lai_toa_raster
    raster_dict['ndvi_sur'] = image.ndvi_sur_raster
    raster_dict['ndwi_sur'] = image.ndwi_sur_raster
    raster_dict['lai_sur'] = image.lai_sur_raster
    raster_dict['em_0'] = os.path.join(image_ws, 'broadband_em' + r_fmt)
    raster_dict['ts'] = image.ts_raster
    raster_dict['ts_dem'] = os.path.join(image_ws, 'ts_dem' + r_fmt)

    raster_dict['rn'] = os.path.join(rn_ws, 'rn' + iter_fmt)
    raster_dict['rn_24'] = os.path.join(rn_ws, 'rn' + iter_fmt)
    raster_dict['g_water'] = os.path.join(g_ws, 'g_water' + iter_fmt)
    raster_dict['g_snow'] = os.path.join(g_ws, 'g_snow' + iter_fmt)
    raster_dict['g_wetland']= os.path.join(g_ws, 'g_wetland' + iter_fmt)
    raster_dict['g'] = os.path.join(g_ws, 'g' + iter_fmt)

    raster_dict['zom'] = os.path.join(zom_ws, 'zom' + iter_fmt)

    raster_dict['dt'] = os.path.join(h_ws, 'dt' + iter_fmt)
    raster_dict['h'] = os.path.join(h_ws, 'h' + iter_fmt)
    raster_dict['psi_z3'] = os.path.join(h_ws, 'psi_z3' + iter_fmt)
    raster_dict['psi_z2'] = os.path.join(h_ws, 'psi_z2' + iter_fmt)
    raster_dict['psi_z1'] = os.path.join(h_ws, 'psi_z1' + iter_fmt)
    raster_dict['l_stabil'] = os.path.join(h_ws, 'l_stabil' + iter_fmt)
    raster_dict['rah'] = os.path.join(h_ws, 'rah' + iter_fmt)
    raster_dict['u_star'] = os.path.join(h_ws, 'u_star' + iter_fmt)

    raster_dict['le'] = os.path.join(le_ws, 'le' + iter_fmt)
    raster_dict['et_inst'] = os.path.join(le_ws, 'et_inst' + iter_fmt)
    raster_dict['etrf'] = os.path.join(etrf_ws, 'et_rf' + iter_fmt)
    raster_dict['et_24'] = os.path.join(et24_ws, 'et_24' + iter_fmt)
    # raster_dict['ef'] = os.path.join(le_ws, 'ef' + iter_fmt)

    # Read MODEL 2 raster flags
    save_dict = dict()
    save_dict['rn'] = read_param(
        'save_rn_raster_flag', False, config, 'INPUTS')
    save_dict['rn_24'] = read_param(
        'save_rn_24_raster_flag', False, config, 'INPUTS')
    save_dict['g'] = read_param(
        'save_g_raster_flag', False, config, 'INPUTS')
    save_dict['g_water'] = read_param(
        'save_g_landuse_rasters_flag', False, config, 'INPUTS')
    save_dict['g_snow'] = read_param(
        'save_g_landuse_rasters_flag', False, config, 'INPUTS')
    save_dict['g_wetland'] = read_param(
        'save_g_landuse_rasters_flag', False, config, 'INPUTS')
    save_dict['zom'] = read_param(
        'save_zom_raster_flag', False, config, 'INPUTS')

    save_dict['dt'] = read_param(
        'save_dt_raster_flag', False, config, 'INPUTS')
    save_dict['h'] = read_param(
        'save_h_raster_flag', False, config, 'INPUTS')
    save_dict['psi_z1'] = read_param(
        'save_psi_raster_flag', False, config, 'INPUTS')
    save_dict['psi_z2'] = read_param(
        'save_psi_raster_flag', False, config, 'INPUTS')
    save_dict['psi_z3'] = read_param(
        'save_psi_raster_flag', False, config, 'INPUTS')
    save_dict['l_stabil'] = read_param(
        'save_l_stabil_raster_flag', False, config, 'INPUTS')
    save_dict['rah'] = read_param(
        'save_rah_raster_flag', False, config, 'INPUTS')
    save_dict['u_star'] = read_param(
        'save_u_star_raster_flag', False, config, 'INPUTS')

    save_dict['le'] = read_param(
        'save_le_raster_flag', False, config, 'INPUTS')
    save_dict['et_inst'] = read_param(
        'save_et_inst_raster_flag', False, config, 'INPUTS')
    save_dict['etrf'] = read_param(
        'save_etrf_raster_flag', True, config, 'INPUTS')
    save_dict['et_24'] = read_param(
        'save_et_24_raster_flag', False, config, 'INPUTS')
    # save_dict['ef'] = read_param(
    #     'save_ef_raster_flag', False, config, 'INPUTS')

    # If overwrite, remove all existing rasters that can be saved
    logging.debug('\nRemoving existing rasters')
    for name, save_flag in sorted(save_dict.items()):
        if ((overwrite_flag or save_flag) and
                os.path.isfile(raster_dict[name])):
            remove_file(raster_dict[name])

    # If raster flag is true, than calc flag has to be true
    calc_dict = save_dict.copy()

    # Initialize model1 rasters to False
    calc_dict['dem'] = False
    calc_dict['elev'] = False
    calc_dict['cos_theta'] = False
    calc_dict['tau'] = False
    calc_dict['albedo_sur'] = False
    calc_dict['ndvi_toa'] = False
    calc_dict['ndwi_toa'] = False
    calc_dict['lai_toa'] = False
    calc_dict['ndvi_sur'] = False
    calc_dict['ndwi_sur'] = False
    calc_dict['lai_sur'] = False
    calc_dict['em_0'] = False
    calc_dict['ts'] = False
    calc_dict['ts_dem'] = False

    # NDLAS gridded weather data flags
    calc_dict['ea'] = False
    calc_dict['wind'] = False
    calc_dict['etr'] = False
    calc_dict['etr_24'] = False
    calc_dict['tair'] = False


    # Working backwords,
    #   Adjust calc flags based on function dependencies
    #   Read in additional parameters based on calc flags

    # # Compute evaporative fraction based ET 24hr estimate
    # #   for target landuses
    # calc_dict['ef'] = read_param(
    #     'use_ef_flag', False, config, 'INPUTS')
    # if calc_dict['ef']:
    #     calc_dict['et_24'] = True
    #     calc_dict['et_inst'] = True
    #     calc_dict['landuse'] = True
    #     calc_dict['rn_24'] = True
    #     ef_landuse_list = read_param(
    #         'ef_landuses', [21, 52, 71], config, 'INPUTS')
    #     ef_landuse_list = list(map(int, ef_landuse_list))

    if calc_dict['et_24']:
        calc_dict['etrf'] = True
    if calc_dict['etrf']:
        calc_dict['et_inst'] = True
    if calc_dict['et_inst']:
        calc_dict['le'] = True
        calc_dict['ts'] = True
    if calc_dict['le']:
        calc_dict['rn'] = True
        calc_dict['g'] = True
        calc_dict['h'] = True

    # Sensible heat flux
    if calc_dict['h']:
        # Read the Kc values from the input file if they were not set
        if kc_cold is None:
            kc_cold = read_param('kc_cold_pixel', 1.05, config, 'INPUTS')
        if kc_hot is None:
            kc_hot = read_param('kc_hot_pixel', 0.1, config, 'INPUTS')
        if kc_cold <= kc_hot:
            logging.error(
                '\nERROR: Kc cold ({}) is less than Kc hot ({})'.format(
                    kc_cold, kc_hot))
            return False
        kc_array = np.array([kc_cold, kc_hot])
        calc_dict['dt'] = True
        calc_dict['air_density'] = True
        calc_dict['l_stabil'] = True
        calc_dict['u_star'] = True
        calc_dict['rah'] = True
        calc_dict['x'] = True
        calc_dict['psi'] = True

        # Controls for pixel stability calculation iterations
        stabil_pixel_mode_str = read_param(
            'stability_pixel_mode', 'MANUAL', config, 'INPUTS').upper()
        stabil_pixel_a_max = 100
        stabil_pixel_b_max = 10000
        if 'MANUAL' in stabil_pixel_mode_str:
            stabil_pixel_iter_max = read_param(
                'stability_pixel_iters', 20, config, 'INPUTS')
        # AUTO and AUTO2 mode
        elif 'AUTO' in stabil_pixel_mode_str:
            stabil_pixel_tolerance = 0.001
            stabil_pixel_iter_max = 100
        # Controls for raster stability calculation iterations
        stabil_raster_mode_str = read_param(
            'stability_raster_mode', 'MANUAL', config, 'INPUTS').upper()
        if 'MANUAL' in stabil_raster_mode_str:
            stabil_raster_iter_max = read_param(
                'stability_raster_iters', 6, config, 'INPUTS')
        # AUTO and AUTO2 mode
        elif 'AUTO' in stabil_raster_mode_str:
            stabil_raster_iter_max = 10

    # Then recheck H sub components for non-H inputs
    if calc_dict['dt'] or calc_dict['l_stabil'] or calc_dict['air_density']:
        calc_dict['ts'] = True
    if calc_dict['dt']:
        calc_dict['ts_dem'] = True
    if calc_dict['u_star']:
        calc_dict['zom'] = True
    if calc_dict['air_density']:
        calc_dict['dem'] = True

    # Zom
    if calc_dict['zom']:
        calc_dict['landuse'] = True

        zom_lai_refl_type = read_param(
            'zom_lai_refl_type', 'TOA', config, 'INPUTS').upper()
        if zom_lai_refl_type == 'TOA':
            calc_dict['lai_toa'] = True
        elif zom_lai_refl_type == 'SUR':
            calc_dict['lai_sur'] = True
        else:
            logging.error(
                ('\nERROR: The LAI reflectance type {} is invalid.' +
                 '\nERROR: Set zom_lai_refl_type to TOA or SUR').format(
                    zom_lai_refl_type))
            return False

        zom_remap_path = read_param('zom_remap_path', None, config, 'INPUTS')
        if zom_remap_path is None:
            logging.error(
                 '\nERROR: The zom_remap_path parameter was not set in '
                 'the INI\n'.format(zom_remap_path))
            return False
        elif not os.path.isfile(zom_remap_path):
            logging.error(
                 '\nERROR: The Zom remap file does not exist\n  {}\n'.format(
                     zom_remap_path))
            return False

        # For now assuming the dict is all string values
        # Conversions will be handled inside et_numpy.zom_func()
        try:
            with open(zom_remap_path) as json_f:
                zom_remap_dict = json.load(json_f)
        except Exception as e:
            logging.error(
                '\nERROR: The Zom remap file could not be read\n  {}\n'.format(
                    zom_remap_path))
            logging.debug('Exception: {}\n'.format(str(e)))
            return False

    # Soil heat flux
    if calc_dict['g']:
        g_model_type = read_param(
            'g_model_type', 'METRIC', config, 'INPUTS').upper()
        if g_model_type not in ['METRIC', 'WIM']:
            logging.error(
                 '\nERROR: The G model type {} is invalid.' +
                 '\nERROR: Set g_model_type to METRIC or WIM'.format(
                    g_model_type))
            return False
        g_refl_type = read_param(
            'g_refl_type', 'TOA', config, 'INPUTS').upper()
        if g_refl_type not in ['TOA', 'SUR']:
            logging.error(
                 '\nERROR: The G reflectance type {} is invalid.' +
                 '\nERROR: Set g_refl_type to TOA or SUR'.format(
                    g_refl_type))
            return False

        if g_model_type == 'METRIC':
            calc_dict['ts'] = True
            calc_dict['rn'] = True
            if g_refl_type == 'TOA':
                calc_dict['lai_toa'] = True
            elif g_refl_type == 'SUR':
                calc_dict['lai_sur'] = True
        elif g_model_type == 'WIM':
            calc_dict['ts'] = True
            calc_dict['albedo_sur'] = True
            if g_refl_type == 'TOA':
                calc_dict['ndvi_toa'] = True
            elif g_refl_type == 'SUR':
                calc_dict['ndvi_sur'] = True

        # Check landuse specific G flags
        # Save flags were set generally from "save_g_landuse_rasters_flag"
        # Clear save flags if "use" flag is false
        calc_dict['g_water'] = read_param(
            'use_g_water_flag', False, config, 'INPUTS')
        calc_dict['g_snow'] = read_param(
            'use_g_snow_flag', False, config, 'INPUTS')
        calc_dict['g_wetland'] = read_param(
            'use_g_wetland_flag', False, config, 'INPUTS')
        if calc_dict['g_water']:
            # calc_dict['slope'] = True
            calc_dict['ts'] = True
            # calc_dict['ts_avg_delap'] = True
            calc_dict['rn'] = True
        else:
            save_dict['g_water'], save_dict['g_water'] = False, False
        if calc_dict['g_snow']:
            calc_dict['ts'] = True
            calc_dict['albedo_sur'] = True
            calc_dict['rn'] = True
        else:
            save_dict['g_snow'], save_dict['g_snow'] = False, False
        if calc_dict['g_wetland']:
            calc_dict['landuse'] = True
            calc_dict['rn'] = True
        else:
            save_dict['g_wetland'], save_dict['g_wetland'] = False, False
        # if calc_dict['ts_avg_delap']:
        #     calc_dict['dem'] = True

        # Get refl. type after checking water G flags
        if calc_dict['g_wetland'] or calc_dict['g_water']:
            if g_refl_type == 'TOA':
                calc_dict['ndvi_toa'] = True
            elif g_refl_type == 'SUR':
                calc_dict['ndvi_sur'] = True
        # if calc_dict['g_water']:
        #     if g_refl_type == 'TOA':
        #         calc_dict['ndwi_toa'] = True
        #     elif g_refl_type == 'SUR':
        #         calc_dict['ndwi_sur'] = True

    # Daily net radiation
    if calc_dict['rn_24']:
        # calc_dict['ts'] = True
        calc_dict['rs_in'] = True
        calc_dict['albedo_sur'] = True
        calc_dict['lat'] = True

    # Net radiation
    if calc_dict['rn']:
        calc_dict['rl_in'] = True
        calc_dict['rl_out'] = True
        calc_dict['rs_in'] = True
        calc_dict['rs_out'] = True
    if calc_dict['rl_in']:
        calc_dict['tau'] = True
        calc_dict['ts_cold_lap'] = True
        rl_in_coef1_flt = read_param('rl_in_coef1', 0.85, config, 'INPUTS')
        rl_in_coef2_flt = read_param('rl_in_coef2', 0.09, config, 'INPUTS')
    if calc_dict['rl_out']:
        calc_dict['ts'] = True
        calc_dict['em_0'] = True
    if calc_dict['rs_in']:
        calc_dict['cos_theta'] = True
        calc_dict['tau'] = True
    if calc_dict['rs_out']:
        calc_dict['rs_in'] = True
        calc_dict['albedo_sur'] = True
    if calc_dict['ts_cold_lap']:
        calc_dict['dem'] = True
        datum_flt = float(config.get('INPUTS', 'datum'))

    # Re-calculate emissivity if it doesn't exist from model 1
    if calc_dict['em_0'] and not os.path.isfile(raster_dict['em_0']):
        # Emissivity is a function of TOA LAI or at-surface LAI
        em_refl_type = read_param(
            'em_refl_type', 'TOA', config, 'INPUTS').upper()
        if em_refl_type == 'TOA':
            calc_dict['lai_toa'] = True
        elif em_refl_type == 'SUR':
            calc_dict['lai_sur'] = True
        else:
            logging.error(
                ('\nERROR: The emissivity reflectance type {} is invalid.' +
                 '\nERROR: Set em_refl_type to TOA or SUR').format(
                    em_refl_type))
            return False
        # Emissivity of water can be set using either NDVI or NDWI
        em_water_index_type = 'NDVI'
        # em_water_index_type = read_param(
        #    'em_water_index_type', 'NDVI', config, 'INPUTS').upper()
        if em_water_index_type == 'NDVI' and em_refl_type == 'TOA':
            calc_dict['ndvi_toa'] = True
        elif em_water_index_type == 'NDVI' and em_refl_type == 'SUR':
            calc_dict['ndvi_sur'] = True
        # elif em_water_index_type == 'NDWI' and em_refl_type == 'TOA':
        #     calc_dict['ndwi_toa'] = True
        # elif em_water_index_type == 'NDWI' and em_refl_type == 'SUR':
        #     calc_dict['ndwi_sur'] = True
        else:
            logging.error(
                ('\nERROR: The emissivity water type {} is invalid.' +
                 '\nERROR: Set em_water_index_type to NDVI').format(
                    em_water_index_type))
            # '\nERROR: Set em_water_index_type to NDVI or NDWI').format(
            return False

    # Re-calculate transmittance if it doesn't exist from Model 1
    if calc_dict['tau'] and not os.path.isfile(raster_dict['tau']):
        calc_dict['cos_theta'] = True
        # Air pressure model dependent parameters
        if calc_dict['tau']:
            pair_model_list = ['DATUM', 'DEM']
            pair_model = read_param(
                'pair_model', 'DEM', config, 'INPUTS').upper()
            if pair_model not in pair_model_list:
                logging.error(
                    ('\nERROR: The Pair model {} is not a valid option.' +
                     '\nERROR: Set pair_model to DATUM or DEM').format(
                        pair_model))
                return False
            if pair_model == 'DATUM':
                datum_flt = float(config.get('INPUTS', 'datum'))
            # Get DEM elevation
            elif pair_model == 'DEM':
                calc_dict['dem'] = True

    # Re-calculate ts_dem if it doesn't exist from Model 1
    if calc_dict['ts_dem'] and not os.path.isfile(raster_dict['ts_dem']):
        calc_dict['ts'] = True
        calc_dict['dem'] = True
        lapse_rate_flt = read_param('lapse_rate', 6.5, config, 'INPUTS')
        datum_flt = float(config.get('INPUTS', 'datum'))

    # Check that rasters from model 1 exist
    # Check DEM and NLCD separately below
    for raster_name in ['cos_theta', 'albedo_sur', 'ts',
                        'ndvi_toa', 'lai_toa', 'ndvi_sur', 'lai_sur']:
        try:
            if (calc_dict[raster_name] and
                    not os.path.isfile(raster_dict[raster_name])):
                logging.error(
                    ('\nERROR: The {} raster was not found.\n' +
                     'ERROR: Please rerun model1 \n').format(
                         raster_name))
                return False
        except KeyError:
            calc_dict[raster_name] = False
    # Read in lapse rates and elevations
    # if (calc_dict['ts_cold_lap'] or calc_dict['ts_avg_delap']):
    if calc_dict['ts_cold_lap']:
        lapse_rate_flt = read_param('lapse_rate', 6.5, config, 'INPUTS')
        datum_flt = float(config.get('INPUTS', 'datum'))
    # Read in calibration parameters/paths
    # if (calc_dict['ts_cold_lap'] or calc_dict['ts_avg_delap']):
    if calc_dict['ts_cold_lap'] or calc_dict['h']:
        # use_pixel_database_flag = read_param(
        #    'use_pixels_database_flag', False, config, 'INPUTS')
        # if use_pixel_database_flag:
        #     pixel_database_file = config.get(
        #         'INPUTS','pixels_database_file')
        #     pd_user_pref_str = read_param(
        #         'pd_user_preference', None, config, 'INPUTS')
        #     pd_set_pref_int = read_param(
        #         'pd_set_preference', None, config, 'INPUTS')
        #     update_config_file_flag = read_param(
        #         'update_config_file_flag', False, config, 'INPUTS')
        pixel_folder_str = read_param(
            'pixels_folder', 'PIXELS', config, 'INPUTS')
        pixel_ws = os.path.join(image_ws, pixel_folder_str)
        cold_str = read_param('cold_pixel', 'cold.shp', config, 'INPUTS')
        cold_path = os.path.join(pixel_ws, cold_str)
        hot_str = read_param('hot_pixel', 'hot.shp', config, 'INPUTS')
        hot_path = os.path.join(pixel_ws, hot_str)
    # Calibration parameter overrides
    # if (calc_dict['ts_cold_lap'] or calc_dict['ts_avg_delap']):
    if calc_dict['ts_cold_lap']:
        ts_cold_override_flag = read_param(
            'ts_cold_override_flag', False, config, 'INPUTS')
        ts_hot_override_flag = read_param(
            'ts_hot_override_flag', False, config, 'INPUTS')
        if ts_cold_override_flag:
            ts_cold_override_flt = read_param(
                'ts_cold_override', 0., config, 'INPUTS')
            if not ts_cold_override_flt:
                ts_cold_override_flag = False
        if ts_hot_override_flag:
            ts_hot_override_flt = read_param(
                'ts_hot_override', 0., config, 'INPUTS')
            if not ts_hot_override_flt:
                ts_hot_override_flag = False

    # DEM
    if calc_dict['dem']:
        # Get the input file DEM raster path if needed
        if (not os.path.isfile(raster_dict['dem']) or
                drigo.raster_path_extent(raster_dict['dem']) != env.mask_extent):
            raster_dict['dem_full'] = config.get(
                'INPUTS', 'dem_raster')
            if not os.path.isfile(raster_dict['dem_full']):
                logging.error(
                    '\nERROR: The dem_raster path {} is not valid'.format(
                        raster_dict['dem_full']))
                return False
        # Otherwise script reads DEM from "full" path,
        #   so set full path to local path
        else:
            raster_dict['dem_full'] = raster_dict['dem']

    # Landuse
    if calc_dict['landuse']:
        # Get the input file DEM raster path if needed
        if (not os.path.isfile(raster_dict['landuse']) or
                drigo.raster_path_extent(raster_dict['landuse']) != env.mask_extent):
            raster_dict['landuse_full'] = config.get(
                'INPUTS', 'landuse_raster')
            if not os.path.isfile(raster_dict['landuse_full']):
                logging.error(
                    '\nERROR: The landuse raster {} does not exist'.format(
                        raster_dict['landuse_full']))
                return False
            landuse_type = read_param(
                'landuse_type', 'NLCD', config, 'INPUTS').upper()
            landuse_type_list = ['NLCD']
            # landuse_type_list = ['NLCD', 'CDL', 'MOD12']
            if landuse_type not in landuse_type_list:
                logging.error(
                     '\nERROR: The landuse type {} is invalid.'
                     '\nERROR: Set landuse_type to {}'.format(
                        landuse_type, ', '.join(landuse_type_list)))
                return False
        # Otherwise script reads landuse from "full" path,
        #   so set full path to local path
        else:
            raster_dict['landuse_full'] = raster_dict['landuse']

    # Cold/Hot Pixels preperation and check
    # Check pixel locations before loading weather data
    if calc_dict['h'] or calc_dict['ts_cold_lap']:
        # Read the calibration point locations from the shapefiles
        #   if they were not passed to the function
        if cold_xy is None and os.path.isfile(cold_path):
            cold_xy = drigo.point_path_xy(cold_path)
        elif cold_xy is None and not os.path.isfile(cold_path):
            logging.error(
                 '\nERROR: The cold calibration shapefile '
                 'does not exist\n  {}\n'.format(cold_path))
            return False
        if hot_xy is None and os.path.isfile(hot_path):
            hot_xy = drigo.point_path_xy(hot_path)
        elif hot_xy is None and not os.path.isfile(hot_path):
            logging.error(
                 '\nERROR: The hot calibration shapefile '
                 'does not exist\n  {}\n'.format(hot_path))
            return False

        # Check that hot/cold pixel are within study area extent
        cold_mask = drigo.raster_value_at_xy(env.mask_path, cold_xy)
        hot_mask = drigo.raster_value_at_xy(env.mask_path, hot_xy)
        if np.isnan(cold_mask) or np.isnan(hot_mask):
            logging.error('\nERROR: The hot or cold pixel are '
                          'located outside the study area mask')
            logging.error('    {:<14s}  {:14.8f}  {:14.8f}\n'.format(
                'MASK' + ':', cold_mask, hot_mask))
            return False
        del cold_mask, hot_mask

    # Weather data parameters
    if calc_dict['h']:
        wind_speed_height_flt = read_param(
            'wind_speed_height', 2.0, config, 'INPUTS')
        station_roughness_flt = read_param(
            'station_roughness', 0.015, config, 'INPUTS')
        add_wind_speed_flt = read_param(
            'additional_wind_speed', 0.0, config, 'INPUTS')

    # Weather Data
    if (calc_dict['tau'] or
            calc_dict['h'] or
            calc_dict['etrf'] or
            calc_dict['et_24']):
        weather_data_source = config.get(
            'INPUTS', 'weather_data_source').upper()
        log_fmt = '    {:<22s} {}'
        if weather_data_source not in ['NLDAS', 'REFET', 'MANUAL']:
            logging.error(
                 '\nERROR: The weather data source {} is invalid.'
                 '\nERROR: Set weather_data_source to REFET or MANUAL'.format(
                    weather_data_source))
            return False
        elif weather_data_source == 'NLDAS':
            logging.info('\n  Weather parameters from NDLAS rasters')
            # DEADBEEF - Testing where to store Landsat scene NLDAS rasters
            # Assuming Ea/Wind/ETr rasters were clipped/projected into SUPPORT_RASTERS
            # After establishing Wind/ETr rasters
            #   Extract values at hot and cold calibration points
            # For now, only use cold calibration point value
            # Project point to raster spatial reference
            if (calc_dict['h'] or
                    calc_dict['etrf'] or
                    calc_dict['et_24']):
                logging.info(pixel_str_fmt.format(
                    '', 'Cold Pixel', 'Hot Pixel'))
            if calc_dict['tau']:
                if not os.path.isfile(raster_dict['ea']):
                    logging.error(
                         '\nERROR: NLDAS Ea raster does not exist\n'
                         '  {}'.format(raster_dict['ea']))
                    return False
                calc_dict['ea'] = True
                ea_flt = float(np.array(et_common.cell_value_set(
                    raster_dict['ea'], 'Ea [kPa]', cold_xy, hot_xy))[0])
            if calc_dict['h']:
                if not os.path.isfile(raster_dict['wind']):
                    logging.error(
                         '\nERROR: NLDAS wind raster does not exist\n'
                         '  {}'.format(raster_dict['wind']))
                    return False
                raster_dict['wind_full'] = raster_dict['wind']
                # calc_dict['wind'] = True
                wind_speed_flt = float(np.array(et_common.cell_value_set(
                    raster_dict['wind_full'], 'Wind [m/s]',
                    cold_xy, hot_xy))[0])
                wind_speed_mod_flt = wind_speed_flt + add_wind_speed_flt
            if calc_dict['etrf'] or calc_dict['et_24']:
                if not os.path.isfile(raster_dict['etr']):
                    logging.error(
                         '\nERROR: NLDAS ETr raster does not exist\n'
                         '  {}'.format(raster_dict['etr']))
                    return False
                elif not os.path.isfile(raster_dict['etr_24hr']):
                    logging.error(
                        ('\nERROR: NLDAS 24hr ETr raster does not exist\n' +
                         '  {}').format(raster_dict['etr_24hr']))
                    return False
                raster_dict['etr_full'] = raster_dict['etr']
                raster_dict['etr_24hr_full'] = raster_dict['etr_24hr']
                # calc_dict['etr'] = True
                # calc_dict['etr_24hr'] = True
                etr_flt = float(np.array(et_common.cell_value_set(
                    raster_dict['etr_full'], 'ETr [mm/hr]',
                    cold_xy, hot_xy))[0])
                etr_24hr_flt = float(np.array(et_common.cell_value_set(
                    raster_dict['etr_24hr_full'], 'ETr [mm/day]',
                    cold_xy, hot_xy))[0])

            # Add an option to METRIC INI for reading in and using Tair
            # Something like "use_tair_h_flag" or "use_tair_adjust_flag"
            # if tair_flag:
            #     calc_dict['tair'] = True
            if calc_dict['tair']:
                if not os.path.isfile(raster_dict['tair']):
                    logging.error(
                        ('\nERROR: NLDAS Tair raster does not exist\n' +
                         '  {}').format(raster_dict['tair']))
                    calc_dict['tair'] = False
                    # return False
        elif weather_data_source == 'REFET':
            gmt_offset = float(config.get('INPUTS', 'gmt_offset'))
            logging.debug('\n  Weather parameters from RefET file')
            refet_file = config.get('INPUTS', 'refet_file')
            logging.debug('  {}'.format(refet_file))
            if not os.path.isfile(refet_file):
                logging.error('\nERROR: The refet_file path is not valid')
                return False
            # The RefET data is in localtime, scene acquisition is GMT
            acq_localtime = image.acq_time + gmt_offset
            # Get RefET Data
            (dew_point_flt, wind_speed_flt, ea_flt,
             etr_flt, etr_24hr_flt) = et_common.read_refet_instantaneous_func(
                 refet_file, image.acq_year, image.acq_doy, acq_localtime)
            wind_speed_mod_flt = wind_speed_flt + add_wind_speed_flt
            # Output RefET Data
            logging.debug('\n  Interpolated Values:')
            logging.debug(log_fmt.format('Scene Time:', acq_localtime))
            logging.debug(log_fmt.format('Dew Point [C]:', dew_point_flt))
            logging.debug(log_fmt.format('Wind Speed [m/s]:', wind_speed_flt))
            logging.debug(log_fmt.format(
                'Wind Speed Mod. [m/s]:', wind_speed_mod_flt))
            logging.debug(log_fmt.format('Ea [kPa]:', ea_flt))
            logging.debug(log_fmt.format('ETr [mm/hr]:', etr_flt))
            logging.debug(log_fmt.format('ETr 24hr [mm/day]:', etr_24hr_flt))
        elif weather_data_source == 'MANUAL':
            logging.debug('\n  Weather parameters from INI file')
            if calc_dict['tau']:
                ea_flt = float(config.get('INPUTS', 'ea'))
                logging.debug(log_fmt.format('Ea [kPa]:', ea_flt))
            if calc_dict['h']:
                wind_speed_flt = float(
                    config.get('INPUTS', 'wind_speed'))
                wind_speed_mod_flt = wind_speed_flt
                logging.debug(log_fmt.format(
                    'Wind Speed [m/s]:', wind_speed_mod_flt))
            if calc_dict['etrf'] or calc_dict['et_24']:
                etr_flt = float(config.get('INPUTS', 'etr'))
                etr_24hr_flt = float(config.get('INPUTS', 'etr_24hr'))
                logging.debug(log_fmt.format('ETr [mm/hr]:', etr_flt))
                logging.debug(log_fmt.format(
                    'ETr 24hr [mm/day]:', etr_24hr_flt))

    # Initialize stability parameters
    if  calc_dict['h']:
        logging.debug('\n  Initialize Stability Calculation Parameters')
        # Near surface height dictionary
        z_flt_dict = {1: 0.1, 2: 2.0, 3: 200.0}
        logging.debug('\n    Blending Layers:')
        for z_index, z_flt in z_flt_dict.items():
            z_log_str = 'Z{:d} [m]:'.format(z_index)
            logging.debug('    {:<8s} {:f}'.format(z_log_str, z_flt))

        # Initialize wind speed parameters
        if calc_dict['h']:
            logging.debug('\n  Initial Wind Speed Parameters')
            # U* at the station [m/s]
            log_fmt = '    {:<32s} {:f}'
            logging.debug('\n' + log_fmt.format(
                'Wind Speed Measured height [m]:', wind_speed_height_flt))
            logging.debug(log_fmt.format(
                'Station Roughness:', station_roughness_flt))
            u_star_station_flt = et_common.u_star_station_func(
                wind_speed_height_flt, station_roughness_flt,
                wind_speed_mod_flt)
            logging.debug(log_fmt.format(
                'U* (Friction Velocity) [m/s]:', u_star_station_flt))
            # U at blending height (200m) [m/s]
            u3_flt = et_common.u3_func(
                u_star_station_flt, z_flt_dict[3], station_roughness_flt)
            logging.debug(log_fmt.format(
                'U3 (Wind Velocity @ {:3.0f}m) [m/s]:'.format(z_flt_dict[3]),
                u3_flt))
            del u_star_station_flt

    # Get hot and cold pixel values once
    # if (calc_dict['h'] or calc_dict['ts_cold_lap'] or
    #     calc_dict['ts_avg_delap']):
    if calc_dict['h'] or calc_dict['ts_cold_lap']:
        # DEADBEEF - Include test to see if the shapefiles has one point
        logging.info('\n  Calibration pixel values')
        logging.info(pixel_str_fmt.format('', 'Cold Pixel', 'Hot Pixel'))
        logging.info(pixel_flt_fmt.format('Kc:', *kc_array))
        logging.debug(pixel_flt_fmt.format('X:', cold_xy[0], hot_xy[0]))
        logging.debug(pixel_flt_fmt.format('Y:', cold_xy[1], hot_xy[1]))
        elev_array = np.array(et_common.cell_value_set(
            raster_dict['dem_full'], 'Elevation', cold_xy, hot_xy))
        landuse_array = np.array(et_common.cell_value_set(
            raster_dict['landuse_full'], 'Landuse', cold_xy, hot_xy))
        cos_theta_array = np.array(et_common.cell_value_set(
            raster_dict['cos_theta'], 'Cos Theta',
            cold_xy, hot_xy, 'DEBUG'))
        if calc_dict['tau'] and not os.path.isfile(raster_dict['tau']):
            # Air Pressure
            if pair_model in ['DEM']:
                pair_array = et_common.air_pressure_func(elev_array)
            elif pair_model in ['DATUM']:
                pair_array = et_common.air_pressure_func(
                    np.array([datum_flt, datum_flt]))
            # Vapor pressure
            if calc_dict['ea']:
                ea_array = np.array(et_common.cell_value_set(
                    raster_dict['ea'], 'Ea', cold_xy, hot_xy))
            else:
                ea_array = np.array([ea_flt, ea_flt])
            #
            tau_array = et_numpy.tau_broadband_func(
                pair_array,
                et_common.precipitable_water_func(pair_array, ea_array),
                cos_theta_array)
            del pair_array, ea_array
        else:
            tau_array = np.array(et_common.cell_value_set(
                raster_dict['tau'], 'Tau', cold_xy, hot_xy, 'DEBUG'))
        albedo_sur_array = np.array(et_common.cell_value_set(
            raster_dict['albedo_sur'], 'Albedo', cold_xy, hot_xy))
        ts_array = np.array(et_common.cell_value_set(
            raster_dict['ts'], 'Ts', cold_xy, hot_xy))

        # This creates a variable from the temp of hot and cold pixels
        # (for Ts correction)
        cold_px_temp = ts_array.item(0)
        hot_px_temp = ts_array.item(1)

        if calc_dict['ts_dem'] and not os.path.isfile(raster_dict['ts_dem']):

            ts_dem_array = et_numpy.ts_delapsed_func(
                ts_array, elev_array, datum_flt, lapse_rate_flt)
        else:
            ts_dem_array = np.array(et_common.cell_value_set(
                raster_dict['ts_dem'], 'Ts_dem', cold_xy, hot_xy, 'DEBUG'))
        # Get index values
        if calc_dict['ndvi_toa']:
            ndvi_toa_array = np.array(et_common.cell_value_set(
                raster_dict['ndvi_toa'], 'NDVI_TOA', cold_xy, hot_xy))
        # if calc_dict['ndwi_toa']:
        #     ndwi_toa_array = np.array(et_common.cell_value_set(
        #         raster_dict['ndwi_toa'], 'NDWI_TOA', cold_xy, hot_xy))
        if calc_dict['lai_toa']:
            lai_toa_array = np.array(et_common.cell_value_set(
                raster_dict['lai_toa'], 'LAI_TOA', cold_xy, hot_xy))
        if calc_dict['ndvi_sur']:
            ndvi_sur_array = np.array(et_common.cell_value_set(
                raster_dict['ndvi_sur'], 'NDVI', cold_xy, hot_xy))
        # if calc_dict['ndwi_sur']:
        #     ndwi_sur_array = np.array(et_common.cell_value_set(
        #         raster_dict['ndwi_sur'], 'NDWI', cold_xy, hot_xy))
        if calc_dict['lai_sur']:
            lai_sur_array = np.array(et_common.cell_value_set(
                raster_dict['lai_sur'], 'LAI', cold_xy, hot_xy))
        if calc_dict['em_0'] and not os.path.isfile(raster_dict['em_0']):
            if em_refl_type == 'TOA' and em_water_index_type == 'NDVI':
                em_0_array = et_numpy.em_0_func(
                    lai_toa_array, ndvi_toa_array, 0)
            elif em_refl_type == 'SUR' and em_water_index_type == 'NDVI':
                em_0_array = et_numpy.em_0_func(
                    lai_sur_array, ndvi_sur_array, 0)
            # elif em_refl_type == 'TOA' and em_water_index_type == 'NDWI':
            #     em_0_array = em_0_func(lai_toa_array, ndwi_toa_array, -0.4)
            # elif em_refl_type == 'SUR' and em_water_index_type == 'NDWI':
            #     em_0_array = em_0_func(lai_array, ndwi_array, -0.4)
        else:
            em_0_array = np.array(et_common.cell_value_set(
                raster_dict['em_0'], 'Em_0', cold_xy, hot_xy, 'DEBUG'))
        if abs(ts_array[0] - ts_array[1]) < ts_diff_threshold:
            logging.error(
                '\nERROR: The hot or cold pixel temperatures ' +
                'are within {}K of each other'.format(ts_diff_threshold))
            logging.error('ERROR: The scene cannot be accurately calibrated\n')
            return False
            # raise et_common.TemperatureError
        elif ts_array[0] > ts_array[1]:
            logging.error(
                '\nERROR: The cold pixel temperature is greater ' +
                'than the hot pixel temperature\n' +
                'ERROR: The scene cannot be accurately calibrated\n')
            return False
            # raise et_common.TemperatureError

    # Override hot and cold pixel Ts and TsDEM
    # if (calc_dict['h'] or calc_dict['ts_cold_lap'] or
    #     calc_dict['ts_avg_delap']):
    if calc_dict['h'] or calc_dict['ts_cold_lap']:
        # Get Ts override values
        if ts_cold_override_flag:
            # ts_array[0] = ts_cold_override_flt
            # ts_dem_array[0] = ts_delapsed_func(
            #     ts_hot_flt, elev_cold_flt, datum_flt, lapse_rate_flt)
            ts_dem_array[0] = ts_cold_override_flt
            ts_array[0] = float(et_numpy.ts_lapsed_func(
                ts_dem_array[0], elev_array[0], datum_flt,
                lapse_rate_flt))
        if ts_hot_override_flag:
            # ts_array[1] = ts_hot_override_flt
            # ts_dem_array[1] = ts_delapsed_func(
            #     ts_array[1], elev_hot_flt, datum, lapse_rate)
            ts_dem_array[1] = ts_hot_override_flt
            ts_array[1] = float(et_numpy.ts_lapsed_func(
                ts_dem_array[1], elev_array[1], datum_flt, lapse_rate_flt))
        if ts_cold_override_flag or ts_hot_override_flag:
            logging.info('\nOverride Hot and Cold pixel temperatures')
            logging.info(pixel_str_fmt.format('', 'Cold Pixel', 'Hot Pixel'))
            logging.info(pixel_flt_fmt.format('Ts', *ts_array))
            logging.info(pixel_flt_fmt.format('TsDEM', *ts_dem_array))

    # Calculate Rn/G at hot and cold point
    if calc_dict['h']:
        # Make a copy of ts and ts_dem at calibration points
        #   for use in block calculations below
        ts_point_array = np.copy(ts_array)
        ts_dem_point_array = np.copy(ts_dem_array)

        # Net Radiation
        ts_cold_lapsed_array = et_numpy.ts_lapsed_func(
            ts_array[0], elev_array, datum_flt, lapse_rate_flt)
        rs_in_array = et_numpy.rs_in_func(cos_theta_array, tau_array, image.dr)
        rs_out_array = et_numpy.rs_out_func(rs_in_array, albedo_sur_array)
        rl_in_array = et_numpy.rl_in_func(
            tau_array, ts_cold_lapsed_array, rl_in_coef1_flt, rl_in_coef2_flt)
        rl_out_array = et_numpy.rl_out_func(rl_in_array, ts_array, em_0_array)
        rn_array = et_numpy.rn_func(
            rs_in_array, rs_out_array, rl_in_array, rl_out_array)
        # Averaged of delapsed ts_hot & ts_cold (for water filter)
        # ts_avg_delap_array = ts_delapsed_func(
        #     float(np.mean(ts_array)), elev_array,
        #     datum_flt, lapse_rate_flt)

        # Soil Heat Flux
        if g_model_type == 'METRIC' and g_refl_type == 'TOA':
            g_array = et_numpy.g_ag_func(
                lai_toa_array, ts_array, rn_array, 1.8, 0.084)
        elif g_model_type == 'METRIC' and g_refl_type == 'SUR':
            g_array = et_numpy.g_ag_func(
                lai_sur_array, ts_array, rn_array, 1.8, 0.084)
        elif g_model_type == 'WIM' and g_refl_type == 'TOA':
            g_array = et_numpy.g_wim_func(
                ts_array, albedo_sur_array, ndvi_toa_array)
        elif g_model_type == 'WIM' and g_refl_type == 'SUR':
            g_array = et_numpy.g_wim_func(
                ts_array, albedo_sur_array, ndvi_sur_array)

        # Calculate Zom
        if zom_lai_refl_type == 'TOA':
            zom_array = et_numpy.zom_func(
                lai_toa_array, landuse_array, zom_remap_dict)
        elif zom_lai_refl_type == 'SUR':
            zom_array = et_numpy.zom_func(
                lai_sur_array, landuse_array, zom_remap_dict)

        # Cleanup
        if calc_dict['ndvi_toa']:
            del ndvi_toa_array
        # if calc_dict['ndwi_toa']:
        #     del ndwi_toa_array
        if calc_dict['lai_toa']:
            del lai_toa_array
        if calc_dict['ndvi_sur']:
            del ndvi_sur_array
        # if calc_dict['ndwi_sur']:
        #     del ndwi_sur_array
        if calc_dict['lai_sur']:
            del lai_sur_array
        del rs_in_array, rs_out_array, rl_in_array, rl_out_array
        del ts_cold_lapsed_array

        # Calculate surface energy balance at calibration points
        le_array = et_numpy.le_calibration_func(etr_flt, kc_array, ts_array)
        h_array = rn_array - g_array - le_array
        logging.info('    {:<14s}  {:14.8f}  {:14.8f}'.format(
            'Rn:', *rn_array))
        logging.info('    {:<14s}  {:14.8f}  {:14.8f}'.format(
            'G:', *g_array))
        logging.info('    {:<14s}  {:14.8f}  {:14.8f}'.format(
            'LE:', *le_array))
        logging.info('    {:<14s}  {:14.8f}  {:14.8f}'.format(
            'H:', *h_array))
        del rn_array, g_array, le_array

        # Calculate A & B values for atmospheric stability
        auto_wind_speed_list = [0]
        if stabil_pixel_mode_str == 'AUTO2':
            auto_wind_speed_list = list(np.arange(0, 2.5, 0.5))
        converge_flag = False
        for i, auto_wind_speed_flt in enumerate(auto_wind_speed_list):
            # Initialize parameters
            a, b = 1, -1000
            psi_dict = {
                1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0]),
                3: np.array([0.0, 0.0])}
            # First time through, set dt so that Ts-dt=293 (dt = Ts-293)
            dt_array = ts_array - 293

            # Iteration can only be greater than 0 if convergence failed
            #   on first iteration
            # Recalculate u* and u3 with additional wind speed
            if i > 0:
                logging.info(
                    '  Updated additonal wind speed - {:4.2f} [m/s]'.format(
                        auto_wind_speed_flt))
                logging.info(
                    '  U* and U3 will be recalculated')
                log_fmt = '    {:<32s} {:f}'
                # Re-calculate wind speed at blending height with additional wind
                u3_flt = et_common.u3_func(
                    et_common.u_star_station_func(
                        wind_speed_height_flt, station_roughness_flt,
                        wind_speed_mod_flt + auto_wind_speed_flt),
                    z_flt_dict[3], station_roughness_flt)
                logging.debug('    {:<32s} {}'.format(
                    'U3 (Wind Velocity @ {:3.0f}m) [m/s]:'.format(
                        z_flt_dict[3]),
                    u3_flt))

            logging.debug('\n  Calculate Stability')
            logging.debug(
                ('          {:>6s} {:>8s} {:>6s} {:>7s} {:>7s} ' +
                 '{:>8s} {:>10s} {:>8}').format(
                    'U*', 'Rah', 'den.', 'dT', 'L',
                    'Psi1', 'Psi2', 'Psi3'))
            # logging.debug(
            #     ('    {:4s}  {:>10s}  {:>10s}  {:>8s}' +
            #      '  {:>8s}  {:>8s}  {:>8s}').format(
            #         'Iter', 'A Coef', 'B Coef', 'Rah Cold',
            #         'dT Cold', 'Rah Hot', 'dT Hot'))
            # log_format = '    {:<14s}  {:14.9f}  {:14.9f}'

            for stabil_pixel_iter in range(0, stabil_pixel_iter_max):
                logging.debug('  ITERATION: {}'.format(stabil_pixel_iter + 1))
                a_old, b_old = a, b
                # U*
                u_star_array = et_numpy.u_star_func(
                    u3_flt, z_flt_dict[3], zom_array, psi_dict[3])
                # Rah
                rah_array = et_numpy.rah_func(
                    z_flt_dict, psi_dict[2], psi_dict[1], u_star_array)
                # Density
                density_array = et_numpy.density_func(
                    elev_array, ts_array, dt_array)
                # dT
                dt_array = et_numpy.dt_calibration_func(
                    h_array, rah_array, density_array)
                # A/B
                a = ((dt_array[1] - dt_array[0]) /
                     (ts_dem_array[1] - ts_dem_array[0]))
                b = -(a * ts_dem_array[0]) + dt_array[0]
                # L
                l_array = et_numpy.l_calibration_func(
                    h_array, density_array, u_star_array, ts_array)

                # Psi - Z3 (200m), Z2 (2m), Z1 (0.1m)
                for z_index, z_flt in z_flt_dict.items():
                    psi_dict[z_index] = et_numpy.psi_func(
                        l_array, z_index, z_flt)

                logging.debug(
                    ('    Cold: {:6.3f} {:8.3f} {:6.3f} {:7.3f} ' +
                     '{:7.3f} {:10.3f} {:10.3f} {:8.3f}').format(
                        u_star_array[0], rah_array[0], density_array[0],
                        dt_array[0], a, b, l_array[0], psi_dict[1][0],
                        psi_dict[2][0], psi_dict[3][0]))
                logging.debug(
                    ('    Hot:  {:6.3f} {:8.3f} {:6.3f} {:7.3f} ' +
                     '{:7.3f} {:10.3f} {:10.3f} {:8.3f}').format(
                        u_star_array[1], rah_array[1], density_array[1],
                        dt_array[1], a, b, l_array[1], psi_dict[1][1],
                        psi_dict[2][1], psi_dict[3][1]))
                logging.debug('    A/B:  {:10.4f} {:10.4f}'.format(a, b))

                # logging.debug(
                #     ('    {:4d}  {:>10.6f}  {:>10.4f}  {:>8.2f}  ' +
                #      '{:>8.2f}  {:>8.2f}  {:>8.2f}').format(
                #         (stabil_pixel_iter+1), a, b,
                #          rah_array[0], dt_array[0],
                #          rah_array[1], dt_array[1]))

                # logging.debug(log_format.format('U*', *u_star_array))
                # logging.debug(log_format.format('Rah:', *rah_array))
                # logging.debug(log_format.format('p:', *density_array))
                # logging.debug(log_format.format('dT:', *dt_array))
                # logging.debug(log_format.format('A/B:', a, b))
                # logging.debug(log_format.format('L:', *l_array))
                # logging.debug(
                #     '    Psi{} ({:5.1f}m):  {:14.9f}  {:14.9f}'.format(
                #         z_index, z_flt, *psi_dict[z_index]))

                # Check for stop conditions
                # Solutions have converged
                # MANUAL mode doesn't check convergence, only iterations
                if ('AUTO' in stabil_pixel_mode_str and
                        abs(a - a_old) < stabil_pixel_tolerance and
                        abs(b - b_old) < stabil_pixel_tolerance):
                    logging.info('\n    A & B values have stabilized')
                    converge_flag = True
                    break
                # An input value is NaN
                elif np.isnan(a) or np.isnan(b):
                    logging.error('\n  ERROR: A & B values are NaN\n')
                    return False
                # Solutions are getting too big
                elif (abs(a) > stabil_pixel_a_max or
                      abs(b) > stabil_pixel_b_max):
                    logging.warning(
                        '\n  WARNING: A & B values are too large and ' +
                        'do not appear to be converging')
                    if (stabil_pixel_mode_str == 'MANUAL' or
                            stabil_pixel_mode_str == 'AUTO'):
                        logging.error(
                            '  ERROR: Try adding additional wind speed\n')
                        return False
                    elif (stabil_pixel_mode_str == 'AUTO2' and
                          auto_wind_speed_flt < auto_wind_speed_list[-1]):
                        logging.warning(
                            '  WARNING: Adding Additional wind speed \n')
                        break
                    elif (stabil_pixel_mode_str == 'AUTO2' and
                          auto_wind_speed_flt >= auto_wind_speed_list[-1]):
                        logging.error(
                            '\n  ERROR: Max additional wind speed applied' +
                            '\n  ERROR: Check the image and inputs\n')
                        return False
                # Too many iterations (limit is set to a large #, ~100)
                elif stabil_pixel_iter + 1 >= stabil_pixel_iter_max:
                    logging.info(
                        '\n  ERROR: Number of iterations exceeds maximum')
                    if (stabil_pixel_mode_str == 'MANUAL' or
                            stabil_pixel_mode_str == 'AUTO'):
                        logging.error(
                            '  ERROR: Try adding additional wind speed\n')
                        return False
                    elif (stabil_pixel_mode_str == 'AUTO2' and
                          auto_wind_speed_flt < auto_wind_speed_list[-1]):
                        logging.warning(
                            '  WARNING: Adding additional wind speed \n')
                        break
                    elif (stabil_pixel_mode_str == 'AUTO2' and
                          auto_wind_speed_flt >= auto_wind_speed_list[-1]):
                        logging.error(
                            '\n  ERROR: Max additional wind speed applied' +
                            '\n  ERROR: Check the image and inputs\n')

                        return False
                    # logging.info(
                    #     '  ERROR: Continuing with last value of A & B')
                    # converge_flag = True
                    # break

            # Break if solution converged, otherwise
            #   additional wind speed will be added in subsequent iterations
            if converge_flag:
                logging.info('      A: {}'.format(a))
                logging.info('      B: {}'.format(b))
                break
            del auto_wind_speed_flt
        del auto_wind_speed_list

        # Cleanup
        del a_old, b_old
        del u_star_array, rah_array, h_array,
        del zom_array
        del density_array, dt_array, l_array, psi_dict

    # Cleanup
    # if (calc_dict['h'] or calc_dict['ts_cold_lap'] or
    #     calc_dict['ts_avg_delap']):
    if calc_dict['h'] or calc_dict['ts_cold_lap']:
        del elev_array, landuse_array, cos_theta_array
        del tau_array, albedo_sur_array, em_0_array
        del ts_array, ts_dem_array


    # Build necessary output folders
    logging.debug('\nBuilding output folders')
    if any([v for k, v in save_dict.items() if rn_ws in raster_dict[k]]):
        if not os.path.isdir(rn_ws):
            os.makedirs(rn_ws)
    if any([v for k, v in save_dict.items() if g_ws in raster_dict[k]]):
        if not os.path.isdir(g_ws):
            os.makedirs(g_ws)
    if any([v for k, v in save_dict.items() if zom_ws in raster_dict[k]]):
        if not os.path.isdir(zom_ws):
            os.makedirs(zom_ws)
    if any([v for k, v in save_dict.items() if h_ws in raster_dict[k]]):
        if not os.path.isdir(h_ws):
            os.makedirs(h_ws)
    if any([v for k, v in save_dict.items() if le_ws in raster_dict[k]]):
        if not os.path.isdir(le_ws):
            os.makedirs(le_ws)
    if any([v for k, v in save_dict.items() if etrf_ws in raster_dict[k]]):
        if not os.path.isdir(etrf_ws):
            os.makedirs(etrf_ws)
    if any([v for k, v in save_dict.items() if et24_ws in raster_dict[k]]):
        if not os.path.isdir(et24_ws):
            os.makedirs(et24_ws)

    # Remove existing and build new empty rasters if necessary
    logging.debug('\nBuilding empty rasters')
    for name, save_flag in sorted(save_dict.items()):
        if save_flag:
            drigo.build_empty_raster(raster_dict[name], 1, np.float32)

    # Process by block
    logging.info('\nProcessing by block')
    logging.debug('  Mask  cols/rows: {}/{}'.format(
        env.mask_cols, env.mask_rows))
    for b_i, b_j in drigo.block_gen(env.mask_rows, env.mask_cols, bs):
        logging.debug('  Block  y: {:5d}  x: {:5d}'.format(b_i, b_j))
        block_data_mask = drigo.array_to_block(
            env.mask_array, b_i, b_j, bs).astype(np.bool)
        block_nodata_mask = ~block_data_mask
        block_rows, block_cols = block_nodata_mask.shape
        block_geo = drigo.array_offset_geo(env.mask_geo, b_j, b_i)
        block_extent = drigo.geo_extent(block_geo, block_rows, block_cols)
        logging.debug('    Block rows: {}  cols: {}'.format(
            block_rows, block_cols))
        logging.debug('    Block extent: {}'.format(block_extent))
        logging.debug('    Block geo: {}'.format(block_geo))

        # Skips blocks that are entirely nodata
        if not np.any(block_data_mask):
            continue

        # Input rasters from Model 1
        if calc_dict['dem']:
            elev_array = drigo.raster_to_array(
                raster_dict['dem_full'], 1, block_extent, -9999.0,
                return_nodata=False)
        if calc_dict['ts']:
            ts_array = drigo.raster_to_block(
                raster_dict['ts'], b_i, b_j, bs, return_nodata=False)
        if calc_dict['cos_theta']:
            cos_theta_array = drigo.raster_to_block(
                raster_dict['cos_theta'], b_i, b_j, bs, return_nodata=False)
        if calc_dict['albedo_sur']:
            albedo_sur_array = drigo.raster_to_block(
                raster_dict['albedo_sur'], b_i, b_j, bs, return_nodata=False)

        # LAI and NDVI primaily for G calculation
        if calc_dict['ndvi_toa']:
            ndvi_toa_array = drigo.raster_to_block(
                raster_dict['ndvi_toa'], b_i, b_j, bs, return_nodata=False)
        if calc_dict['lai_toa']:
            lai_toa_array = drigo.raster_to_block(
                raster_dict['lai_toa'], b_i, b_j, bs, return_nodata=False)
        # if calc_dict['ndwi_toa']:
        #     ndwi_toa_array = drigo.raster_to_block(
        #         raster_dict['ndwi_toa'], b_i, b_j, bs, return_nodata=False)
        if calc_dict['ndvi_sur']:
            ndvi_sur_array = drigo.raster_to_block(
                raster_dict['ndvi_sur'], b_i, b_j, bs, return_nodata=False)
        if calc_dict['lai_sur']:
            lai_sur_array = drigo.raster_to_block(
                raster_dict['lai_sur'], b_i, b_j, bs, return_nodata=False)
        # if calc_dict['ndwi_sur']:
        #     ndwi_sur_array = drigo.raster_to_block(
        #         raster_dict['ndwi_sur'], b_i, b_j, bs, return_nodata=False)

        # Ts DEM can be read in or re-calculated
        if calc_dict['ts_dem'] and not os.path.isfile(raster_dict['ts_dem']):
            ts_dem_array = et_numpy.ts_delapsed_func(
                ts_array, elev_array, datum_flt, lapse_rate_flt)
        else:
            ts_dem_array = drigo.raster_to_block(
                raster_dict['ts_dem'], b_i, b_j, bs, return_nodata=False)

        # Transmittance can be read in or re-calculated
        if calc_dict['tau'] and not os.path.isfile(raster_dict['tau']):
            if pair_model in ['DEM']:
                pair_array = et_common.air_pressure_func(elev_array)
            elif pair_model in ['DATUM']:
                pair_array = np.empty(
                    block_data_mask.shape, dtype=np.float32)
                pair_array[block_data_mask] = et_common.air_pressure_func(
                    datum_flt)
                pair_array[block_nodata_mask] = np.nan
            # Vapor pressure (Ea)
            if calc_dict['ea']:
                ea_array = drigo.raster_to_block(
                    raster_dict['ea'], b_i, b_j, bs, return_nodata=False)
            else:
                ea_array = np.array([ea_flt])
            #
            tau_array = et_numpy.tau_broadband_func(
                pair_array,
                et_common.precipitable_water_func(pair_array, ea_array),
                cos_theta_array)
            del pair_array, ea_array
        else:
            tau_array = drigo.raster_to_block(
                raster_dict['tau'], b_i, b_j, bs, return_nodata=False)
        # Emissivity can be read in or re-calculated
        if calc_dict['em_0'] and not os.path.isfile(raster_dict['em_0']):
            if em_refl_type == 'TOA' and em_water_index_type == 'NDVI':
                em_0_array = et_numpy.em_0_func(lai_toa_array, ndvi_toa_array)
            elif em_refl_type == 'SUR' and em_water_index_type == 'NDVI':
                em_0_array = et_numpy.em_0_func(lai_sur_array, ndvi_sur_array)
            # elif em_refl_type == 'TOA' and em_water_index_type == 'NDWI':
            #     em_0_array = em_0_func(lai_toa_array, ndwi_toa_array)
            # elif em_refl_type == 'SUR' and em_water_index_type == 'NDWI':
            #     em_0_array = em_0_func(lai_array, ndwi_array)
        elif calc_dict['em_0']:
            em_0_array = drigo.raster_to_block(
                raster_dict['em_0'], b_i, b_j, bs, return_nodata=False)

        # Lapse Adjusted ts_cold
        if calc_dict['ts_cold_lap']:
            ts_cold_lapsed_array = et_numpy.ts_lapsed_func(
                ts_point_array[0], elev_array, datum_flt, lapse_rate_flt)
        # Incoming Longwave Radiation
        if calc_dict['rl_in']:
            rl_in_array = et_numpy.rl_in_func(
                tau_array, ts_cold_lapsed_array,
                rl_in_coef1_flt, rl_in_coef2_flt)
        if calc_dict['ts_cold_lap']:
            del ts_cold_lapsed_array
        # Emitted Longwave Radiation
        if calc_dict['rl_out']:
            rl_out_array = et_numpy.rl_out_func(
                rl_in_array, ts_array, em_0_array)
        # Incoming Shortwave Radiation
        # possible rounding error in rs_in_array
        if calc_dict['rs_in']:
            rs_in_array = et_numpy.rs_in_func(
                cos_theta_array, tau_array, image.dr)
        # Outgoing Shortwave Radiation
        if calc_dict['rs_out']:
            rs_out_array = et_numpy.rs_out_func(
                rs_in_array, albedo_sur_array)
        # Net Radiation
        if calc_dict['rn']:
            rn_array = et_numpy.rn_func(
                rs_in_array, rs_out_array, rl_in_array, rl_out_array)
        if save_dict['rn']:
            drigo.block_to_raster(rn_array, raster_dict['rn'], b_i, b_j, bs)

        # Daily Net Radiation (for evaporative fraction)
        if calc_dict['rn_24']:
            lat_array, lon_array = drigo.array_lat_lon_func(
                env.snap_osr, env.cellsize, block_extent,
                gcs_cs=0.005, radians_flag=True)
            rn_24_array = et_numpy.rn_24_func(
                albedo_sur=albedo_sur_array, rs_in=rs_in_array,
                lat=lat_array, doy=image.acq_doy)
            del lat_array, lon_array
        if save_dict['rn_24']:
            drigo.block_to_raster(
                rn_24_array, raster_dict['rn_24'], b_i, b_j, bs)

        # Cleanup
        if calc_dict['rs_in']:
            del rs_in_array
        if calc_dict['rs_out']:
            del rs_out_array
        if calc_dict['rl_in']:
            del rl_in_array
        if calc_dict['rl_out']:
            del rl_out_array
        if calc_dict['em_0']:
            del em_0_array
        if calc_dict['tau']:
            del tau_array
        if calc_dict['cos_theta']:
            del cos_theta_array

        # Load landuse
        if calc_dict['landuse']:
            landuse_array = drigo.raster_to_array(
                raster_dict['landuse_full'], 1, block_extent, fill_value=0,
                return_nodata=False)

        # Soil Heat Flux
        # (Uncommenting would apply Ag function to all pixels)
        if calc_dict['g'] and g_model_type == 'METRIC':
            if g_refl_type == 'TOA':
                g_array = et_numpy.g_ag_func(
                    lai_toa_array, ts_array, rn_array, 1.8, 0.084)
            elif g_refl_type == 'SUR':
                g_array = et_numpy.g_ag_func(
                    lai_sur_array, ts_array, rn_array, 1.8, 0.084)
        elif calc_dict['g'] and g_model_type == 'WIM':
            if g_refl_type == 'TOA':
                g_array = et_numpy.g_wim_func(
                    ts_array, albedo_sur_array, ndvi_toa_array)
            elif g_refl_type == 'SUR':
                g_array = et_numpy.g_wim_func(
                    ts_array, albedo_sur_array, ndvi_sur_array)
        # Averaged of delapsed ts_hot & ts_cold (for water filter)
        # if calc_dict['ts_avg_delap']:
        #     ts_avg_delap_array = ts_delapsed_func(
        #         float(np.mean(ts_point_array)), elev_array,
        #         datum_flt, lapse_rate_flt)
        # if save_dict['ts_avg_delap']:
        #     drigo.block_to_raster(
        #         ts_avg_delap_array, raster_dict['ts_avg_delap'],
        #         b_i, b_j, bs)

        # Water Body Filter
        # Basalt (and maybe playa?) can have a negative NDVI but not be water
        if calc_dict['g_water']:
            g_water_mask = np.copy(block_data_mask)
            if g_refl_type == 'TOA':
                g_water_mask &= (ndvi_toa_array <= 0)
                # g_water_mask &= (ndwi_toa_array <= 0)
            elif g_refl_type == 'SUR':
                g_water_mask &= (ndvi_sur_array <= 0)
                # g_water_mask &= (ndwi_array <= 0)
            # g_water_mask &= (slope_array < 1)
            ts_avg_delap_flt = np.mean(ts_dem_point_array)
            g_water_mask &= (ts_array < ts_avg_delap_flt)
            if np.any(g_water_mask):
                g_array[g_water_mask] = (
                    0.5 * rn_array[g_water_mask])
            if save_dict['g_water']:
                g_water_array = np.copy(g_array)
                g_water_array[~g_water_mask] = np.nan
                drigo.block_to_raster(
                    g_water_array, raster_dict['g_water'], b_i, b_j, bs)
            del g_water_mask
        # if calc_dict['ts_avg_delap']:
        #     del ts_avg_delap_array

        # Snow Filter
        if calc_dict['g_snow']:
            snow_temp_threshold_flt = 277.0
            snow_albedo_threshold_flt = 0.35
            g_snow_mask = np.copy(block_data_mask)
            g_snow_mask &= (albedo_sur_array > snow_albedo_threshold_flt)
            g_snow_mask &= (ts_array < snow_temp_threshold_flt)
            if np.any(g_snow_mask):
                g_array[g_snow_mask] = (
                    0.5 * rn_array[g_snow_mask])
            if save_dict['g_snow']:
                g_snow_array = np.copy(g_array)
                g_snow_array[~g_snow_mask] = np.nan
                drigo.block_to_raster(
                    g_snow_array, raster_dict['g_snow'], b_i, b_j, bs)
            del g_snow_mask
        # Wetland Filter
        # Ag G function is applied to wetlands with NDVI > 0.5
        if calc_dict['g_wetland']:
            g_wetland_mask = np.copy(block_data_mask)
            if landuse_type in ['NLCD', 'CDL']:
                g_wetland_mask &= (
                    (landuse_array == 90) | (landuse_array == 95))
            else:
                pass
            if g_refl_type == 'TOA':
                g_wetland_mask &= (ndvi_toa_array <= 0.5)
            elif g_refl_type == 'SUR':
                g_wetland_mask &= (ndvi_sur_array <= 0.5)
            if np.any(g_wetland_mask):
                g_array[g_wetland_mask] = (
                    -51 + (0.41 * rn_array[g_wetland_mask]))
            if save_dict['g_wetland']:
                g_wetland_array = np.copy(g_array)
                g_wetland_array[~g_wetland_mask] = np.nan
                drigo.block_to_raster(
                    g_wetland_array, raster_dict['g_wetland'], b_i, b_j, bs)
            del g_wetland_mask
        # Save G
        if save_dict['g']:
            drigo.block_to_raster(g_array, raster_dict['g'], b_i, b_j, bs)

        # Cleanup
        if calc_dict['ndvi_toa']:
            del ndvi_toa_array
        if calc_dict['ndvi_sur']:
            del ndvi_sur_array
        # if calc_dict['ndwi_toa']:
        #     del ndwi_toa_array
        # if calc_dict['ndwi_sur']:
        #     del ndwi_sur_array
        if calc_dict['albedo_sur']:
            del albedo_sur_array

        # Calculate Zom
        if calc_dict['zom'] and zom_lai_refl_type == 'TOA':
            zom_array = et_numpy.zom_func(
                lai_toa_array, landuse_array, zom_remap_dict)
        elif calc_dict['zom'] and zom_lai_refl_type == 'SUR':
            zom_array = et_numpy.zom_func(
                lai_sur_array, landuse_array, zom_remap_dict)
        if save_dict['zom']:
            drigo.block_to_raster(zom_array, raster_dict['zom'], b_i, b_j, bs)
        if calc_dict['lai_toa']:
            del lai_toa_array
        if calc_dict['lai_sur']:
            del lai_sur_array

        # dT (Eqn 45)
        if calc_dict['dt']:
            dt_array = et_numpy.dt_func(ts_dem_array, a, b)
        if save_dict['dt']:
            drigo.block_to_raster(dt_array, raster_dict['dt'], b_i, b_j, bs)

        # Sensible Heat Flux
        if calc_dict['h']:
            logging.debug('\n  Calculate L (stability)')
            # Set these to 0 for first iteration of stability loop calculation
            # They will be modified/updated within the loop
            psi_z3_array = np.zeros(block_data_mask.shape, dtype=np.float32)
            psi_z2_array = np.zeros(block_data_mask.shape, dtype=np.float32)
            psi_z1_array = np.zeros(block_data_mask.shape, dtype=np.float32)
            l_stabil_array = np.zeros(block_data_mask.shape, dtype=np.float32)

            for stabil_raster_iter in range(0, stabil_raster_iter_max):
                logging.debug(
                    '  ITERATION: {}'.format(stabil_raster_iter + 1))
                # Use the corrected u_star and rah after the first iteration
                u_star_array = et_numpy.u_star_func(
                    u3_flt, z_flt_dict[3], zom_array, psi_z3_array)
                rah_array = et_numpy.rah_func(
                    z_flt_dict, psi_z2_array, psi_z1_array,
                    u_star_array)
                del psi_z3_array, psi_z2_array, psi_z1_array, l_stabil_array

                l_stabil_array = et_numpy.l_func(
                    dt_array, u_star_array, ts_array, rah_array)
                psi_z3_array = et_numpy.psi_func(
                    l_stabil_array, 3, z_flt_dict[3])
                psi_z2_array = et_numpy.psi_func(
                    l_stabil_array, 2, z_flt_dict[2])
                psi_z1_array = et_numpy.psi_func(
                    l_stabil_array, 1, z_flt_dict[1])
                # Cleanup
                del u_star_array, rah_array

            if save_dict['l_stabil']:
                drigo.block_to_raster(
                    l_stabil_array, raster_dict['l_stabil'], b_i, b_j, bs)
            if save_dict['psi_z3']:
                drigo.block_to_raster(
                    psi_z3_array, raster_dict['psi_z3'], b_i, b_j, bs)
            if save_dict['psi_z2']:
                drigo.block_to_raster(
                    psi_z2_array, raster_dict['psi_z2'], b_i, b_j, bs)
            if save_dict['psi_z1']:
                drigo.block_to_raster(
                    psi_z1_array, raster_dict['psi_z1'], b_i, b_j, bs)
            del l_stabil_array

            # Final calculation of H sub components
            u_star_array = et_numpy.u_star_func(
                u3_flt, z_flt_dict[3], zom_array, psi_z3_array)
            if save_dict['u_star']:
                drigo.block_to_raster(
                    u_star_array, raster_dict['u_star'], b_i, b_j, bs)
            del psi_z3_array
            rah_array = et_numpy.rah_func(
                z_flt_dict, psi_z2_array, psi_z1_array,
                u_star_array)
            if save_dict['rah']:
                drigo.block_to_raster(
                    rah_array, raster_dict['rah'], b_i, b_j, bs)
            del u_star_array, psi_z2_array, psi_z1_array

            air_density_array = et_numpy.density_func(
                elev_array, ts_array, dt_array)
            # H [W/m^2]
            h_array = et_numpy.h_func(
                air_density_array, dt_array, rah_array)
            del air_density_array, dt_array, rah_array
        # Save H
        if save_dict['h']:
            drigo.block_to_raster(h_array, raster_dict['h'], b_i, b_j, bs)

        # Cleanup
        if calc_dict['zom']:
            del zom_array
        if calc_dict['dem']:
            del elev_array

        # Latent Heat Flux [W/m^2]
        if calc_dict['le']:
            le_array = et_numpy.le_func(rn_array, g_array, h_array)
        if save_dict['le']:
            drigo.block_to_raster(le_array, raster_dict['le'], b_i, b_j, bs)

        # ET instantaneous [mm/hr]
        if calc_dict['et_inst']:
            et_inst_array = et_numpy.et_inst_func(le_array, ts_array)
        if save_dict['et_inst']:
            drigo.block_to_raster(
                et_inst_array, raster_dict['et_inst'], b_i, b_j, bs)

        # ET Reference Fraction - ETrF
        if calc_dict['etrf']:
            etrf_array = et_numpy.etrf_func(et_inst_array, etr_flt)
        if save_dict['etrf']:
            drigo.block_to_raster(
                etrf_array, raster_dict['etrf'], b_i, b_j, bs)

        # ET 24hr [mm/day]
        # This is computed before E.F. based estimate is applied
        #   but is saved after E.F.
        if calc_dict['et_24']:
            et_24_array = etr_24hr_flt * etrf_array

        # # Calc ET 24hr [mm/day] from evaporative fraction`
        # if calc_dict['ef']:
        #     # ef_array = et_numpy.ef_func(le_array, rn_array, g_array)
        #     ef_24_array = et_inst_array * (rn_24_array - 0) / (rn_array - g_array)
        #     # for lu in ef_landuse_list:
        #     #     et_24_array[landuse_array == lu] = ef_24_array[landuse_array == lu]
        #     del ef_24_array
        # # Save evaporative fraction raster (not ET raster)
        # if save_dict['ef']:
        #     ef_array = et_numpy.ef_func(le_array, rn_array, g_array)
        #     drigo.block_to_raster(ef_array, raster_dict['ef'], b_i, b_j, bs)
        if calc_dict['et_inst']:
            del et_inst_array

        # Save ET 24hr after updating with EF based values
        if save_dict['et_24']:
            drigo.block_to_raster(
                et_24_array, raster_dict['et_24'], b_i, b_j, bs)
        if calc_dict['et_24']:
            del et_24_array
        if calc_dict['etrf']:
            del etrf_array

        # Cleanup
        if calc_dict['ts']:
            del ts_array
        if calc_dict['h']:
            del h_array
        if calc_dict['g']:
            del g_array
        if calc_dict['rn']:
            del rn_array
        del block_nodata_mask, block_data_mask, block_rows, block_cols

    # Raster Statistics
    if stats_flag:
        logging.info('\nCalculating Statistics')
        for name, save_flag in save_dict.items():
            if save_flag:
                logging.debug('  {}'.format(raster_dict[name]))
                drigo.raster_statistics(raster_dict[name])

    # Raster Pyramids
    if pyramids_flag:
        logging.info('\nBuilding Pyramids')
        for name, save_flag in save_dict.items():
            if save_flag:
                logging.debug('  {}'.format(raster_dict[name]))
                drigo.raster_pyramids(raster_dict[name])

    # Cleanup
    del save_dict, calc_dict, image

    return True


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Model 2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-i', '--ini', required=True,
        help='METRIC input file', metavar='PATH')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-kc', '--kc', type=float,
        default=[None, None], nargs=2, metavar=('COLD', 'HOT'),
        help='Kc at the cold and hot calibration point')
    parser.add_argument(
        '-mc', '--iter', default=None, type=int, metavar='N',
        help='Monte Carlo iteration number')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--stats', default=None, action="store_true",
        help='Compute raster statistics')
    parser.add_argument(
        '-xyc', '--xy_cold', default=None, type=float, nargs=2,
        help='Location of the the cold calibration point', metavar=('X', 'Y'))
    parser.add_argument(
        '-xyh', '--xy_hot', default=None, type=float, nargs=2,
        help='Location of the hot calibration point', metavar=('X', 'Y'))
    args = parser.parse_args()

    # Convert relative paths to absolute paths
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
        log_file_name = 'metric_model2_log.txt'
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
    logging.info('')

    # Delay
    sleep(random.uniform(0, max([0, args.delay])))

    # METRIC Model 2
    metric_model2(image_ws=args.workspace, ini_path=args.ini,
                  mc_iter=args.iter, kc_cold=args.kc[0], kc_hot=args.kc[1],
                  cold_xy=args.xy_cold, hot_xy=args.xy_hot,
                  stats_flag=args.stats, overwrite_flag=args.overwrite)
