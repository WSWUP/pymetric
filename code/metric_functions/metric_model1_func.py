#!/usr/bin/env python
#--------------------------------
# Name:         metric_model1_func.py
# Purpose:      Calculate METRIC Model 1
# Notes:        GDAL Block Version
#--------------------------------

import argparse
import datetime as dt
import logging
import math
import os
import random
import shutil
import sys
from time import sleep

import drigo
import numpy as np
from osgeo import gdal

import et_common
import et_image
import et_numpy
from python_common import open_ini, read_param, remove_file


def metric_model1(image_ws, ini_path, bs=None, stats_flag=None,
                  overwrite_flag=None):
    """METRIC Model 1 Version

    Parameters
    ----------
    image_ws : str
        Image folder path.
    ini_path : str
        METRIC config file path.
    bs : int, optional
        Processing block size (the default is None).  If set, this blocksize
        parameter will be used instead of the value in the INI file.
    stats_flag : bool, optional
        If True, compute raster statistics (the default is None).
    ovewrite_flag : bool, optional
        If True, overwrite existing files (the default is None).

    Returns
    -------
    True if successful

    """

    logging.info('METRIC Model 1')
    log_fmt = '  {:<18s} {}'

    env = drigo.env
    image = et_image.Image(image_ws, env)
    np.seterr(invalid='ignore')
    # env.cellsize = 463.313
    # env.snap_xmin, env.snap_ymin = 231.6565, 231.6565

    # # Check that image_ws is valid
    # image_re = re.compile(
    #     '^(LT04|LT05|LE07|LC08)_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})')
    # if not os.path.isdir(image_ws) or not image_re.match(scene_id):
    #     logging.error('\nERROR: Image folder is invalid or does not exist\n')
    #     return False

    # Open config file
    config = open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')

    # Arrays are processed by block
    if bs is None:
        bs = read_param('block_size', 1024, config)
    logging.info(log_fmt.format('Block Size:', bs))

    # Raster pyramids/statistics
    pyramids_flag = read_param('pyramids_flag', False, config)
    if pyramids_flag:
        gdal.SetConfigOption('HFA_USE_RRD', 'YES')
    if stats_flag is None:
        stats_flag = read_param('statistics_flag', False, config)

    # Remove reflectance rasters after calculating Model 1
    remove_refl_toa_flag = read_param('remove_refl_toa_flag', False, config)
    remove_refl_sur_flag = read_param('remove_refl_sur_flag', False, config)
    remove_ts_bt_flag = read_param('remove_ts_bt_flag', False, config)

    # Check that common_area raster exists
    if not os.path.isfile(image.common_area_raster):
        logging.error('\nERROR: A common area raster was not found.')
        logging.error('ERROR: Please rerun prep tool to build these files.\n')
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

    raster_dict['ndvi_toa'] = image.ndvi_toa_raster
    raster_dict['ndwi_toa'] = image.ndwi_toa_raster
    raster_dict['lai_toa'] = image.lai_toa_raster
    raster_dict['ndvi_sur'] = image.ndvi_sur_raster
    raster_dict['lai_sur'] = image.lai_sur_raster
    raster_dict['ndwi_sur'] = image.ndwi_sur_raster
    raster_dict['savi_toa'] = os.path.join(image.indices_ws, 'savi_toa' + r_fmt)
    raster_dict['savi_sur'] = os.path.join(image.indices_ws, 'savi' + r_fmt)

    raster_dict['em_nb'] = os.path.join(image_ws, 'narrowband_em' + r_fmt)
    raster_dict['em_0'] = os.path.join(image_ws, 'broadband_em' + r_fmt)
    raster_dict['rc'] = os.path.join(image_ws, 'corrected_rad' + r_fmt)
    raster_dict['ts_dem'] = os.path.join(image_ws, 'ts_dem' + r_fmt)

    raster_dict['ts'] = image.ts_raster
    raster_dict['ts_bt'] = image.ts_bt_raster
    raster_dict['refl_toa'] = image.refl_toa_raster
    raster_dict['refl_sur_ledaps'] = image.refl_sur_ledaps_raster
    raster_dict['refl_sur_tasumi'] = image.refl_sur_tasumi_raster
    raster_dict['refl_sur'] = ''  # DEADBEEF - this is a sloppy work around
    # to a KeyError that was being thrown under the comment
    # 'Calculate refl_toa if any TOA indices flags are True'

    # Read MODEL 1 raster flags
    save_dict = dict()
    save_dict['dem'] = read_param('save_dem_raster_flag', False, config)
    save_dict['landuse'] = read_param('save_landuse_raster_flag', False, config)
    save_dict['slp'] = read_param('save_mountain_rasters_flag', False, config)
    save_dict['asp'] = read_param('save_mountain_rasters_flag', False, config)
    save_dict['lat'] = read_param('save_mountain_rasters_flag', False, config)
    save_dict['lon'] = read_param('save_mountain_rasters_flag', False, config)
    save_dict['cos_theta'] = read_param('save_cos_theta_raster_flag', True, config)

    # You can only save Tasumi, not LEDAPS, at-surface reflectance
    save_dict['refl_sur_tasumi'] = read_param('save_refl_sur_raster_flag', True, config)
    save_dict['tau'] = read_param('save_tau_raster_flag', True, config)
    save_dict['albedo_sur'] = read_param('save_albedo_sur_raster_flag', True, config)
    # Default for all TOA reflectance indices is True except SAVI
    save_dict['ndvi_toa'] = read_param('save_ndvi_toa_raster_flag', True, config)
    save_dict['ndwi_toa'] = read_param('save_ndwi_toa_raster_flag', True, config)
    save_dict['savi_toa'] = read_param('save_savi_toa_raster_flag', False, config)
    save_dict['lai_toa'] = read_param('save_lai_toa_raster_flag', True, config)
    # Default for all at-surface reflectance indices is False
    save_dict['ndvi_sur'] = read_param('save_ndvi_raster_flag', False, config)
    save_dict['ndwi_sur'] = read_param('save_ndwi_raster_flag', False, config)
    save_dict['savi_sur'] = read_param('save_savi_raster_flag', False, config)
    save_dict['lai_sur'] = read_param('save_lai_raster_flag', False, config)
    # Surface temperature and emissivity
    save_dict['em_nb'] = read_param('save_em_nb_raster_flag', False, config)
    save_dict['em_0'] = read_param('save_em_0_raster_flag', True, config)
    save_dict['rc'] = read_param('save_rc_raster_flag', False, config)
    save_dict['ts'] = read_param('save_ts_raster_flag', True, config)
    save_dict['ts_dem'] = read_param('save_ts_dem_raster_flag', True, config)

    # Clear SUR save flags if input rasters from prep_scene are not present
    em_refl_type = read_param('em_refl_type', 'TOA', config).upper()
    refl_sur_model_type = read_param(
        'refl_sur_model_type', 'TASUMI', config).upper()
    refl_sur_model_type_list = ['TASUMI', 'LEDAPS']
    if refl_sur_model_type.upper() not in refl_sur_model_type_list:
        logging.error(
            '\nERROR: Surface reflectance model type {} is invalid.'
            '\nERROR: Set refl_sur_model_type to {}'.format(
                refl_sur_model_type, ','.join(refl_sur_model_type_list)))
        return False
    elif (refl_sur_model_type == 'LEDAPS' and
        not os.path.isfile(raster_dict['refl_sur_ledaps'])):
        logging.warning(
            '\nLEDAPS at-surface refl. composite raster does not exist'
            '\nLEDAPS at-surface refl. products will not be calculated')
        save_dict['refl_sur_ledaps'] = False
        clear_refl_sur_flag = True
    elif (refl_sur_model_type == 'TASUMI' and
          not os.path.isfile(raster_dict['refl_toa'])):
        logging.warning(
            '\nTOA reflectance composite raster does not exist'
            '\nTasumi at-surface refl. products will not be calculated')
        save_dict['refl_sur_tasumi'] = False
        clear_refl_sur_flag = True
    else:
        clear_refl_sur_flag = False

    if clear_refl_sur_flag:
        save_dict['refl_sur'] = False
        save_dict['ndvi_sur'] = False
        save_dict['ndwi_sur'] = False
        save_dict['savi_sur'] = False
        save_dict['lai_sur'] = False
        save_dict['albedo_sur'] = False
        if em_refl_type == 'SUR':
            save_dict['em_nb'] = False
            save_dict['em_0'] = False
            save_dict['rc'] = False
            save_dict['ts'] = False

    # Clear TOA save flags if input TOA raster is not present
    if not os.path.isfile(raster_dict['refl_toa']):
        logging.warning(
            '\nTOA reflectance composite raster does not exist'
            '\nTOA reflectance products will not be calculated')
        save_dict['ndvi_toa'] = False
        save_dict['ndwi_toa'] = False
        save_dict['savi_toa'] = False
        save_dict['lai_toa'] = False
        if em_refl_type == 'TOA':
            save_dict['em_nb'] = False
            save_dict['em_0'] = False
            save_dict['rc'] = False
            save_dict['ts'] = False
            save_dict['ts_dem'] = False

    # Clear Ts save flags if input Ts brightness raster is not present
    if not os.path.isfile(raster_dict['ts_bt']):
        logging.warning('\nTs brightness raster does not exist')
        save_dict['rc'] = False
        save_dict['ts'] = False
        save_dict['ts_dem'] = False

    # If overwrite, remove all existing rasters that can be saved
    # DEADBEEF - changed the overwrite_flag or save_flag line to and. Not sure
    #  what else this will affect.
    logging.debug('\nRemoving existing rasters')
    for name, save_flag in sorted(save_dict.items()):
        if ((overwrite_flag and save_flag) and
                os.path.isfile(raster_dict[name])):
            remove_file(raster_dict[name])

    # If save flag is true, than calc flag has to be true
    calc_dict = save_dict.copy()

    # Initialize prep_scene rasters to False
    calc_dict['refl_toa'] = False
    calc_dict['refl_sur'] = False
    calc_dict['refl_sur_ledaps'] = False
    calc_dict['ts_bt'] = False
    calc_dict['ea'] = False

    # Working backwards,
    #   Adjust calc flags based on function dependencies
    #   Read in additional parameters based on calc flags
    # Surface temperature
    if calc_dict['ts_dem']:
        calc_dict['ts'] = True
        calc_dict['dem'] = True
        lapse_rate_flt = read_param('lapse_rate', 6.5, config)
    if calc_dict['ts']:
        calc_dict['rc'] = True
    if calc_dict['rc']:
        calc_dict['ts_bt'] = True
        calc_dict['em_nb'] = True
        rsky_flt = read_param('rsky', 1.32, config)
        rp_flt = read_param('rp', 0.91, config)
        tnb_flt = read_param('tnb', 0.866, config)

    # Emissivity
    if calc_dict['em_nb'] or calc_dict['em_0']:
        # Emissivity is a function of TOA LAI or at-surface LAI
        em_refl_type = read_param('em_refl_type', 'TOA', config).upper()
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
        #    'em_water_index_type', 'NDVI', config).upper()
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
            return False

    # Vegetation indices
    if calc_dict['lai_sur']:
        lai_veg_index_type = read_param(
            'lai_veg_index_type', 'SAVI', config).upper()
        if lai_veg_index_type == 'SAVI':
            calc_dict['savi_sur'] = True
        elif lai_veg_index_type == 'NDVI':
            calc_dict['ndvi_sur'] = True
        else:
            logging.error(
                ('\nERROR: The LAI veg. index type {} is invalid.' +
                 '\nERROR: Set lai_veg_index_type to SAVI or NDVI').format(
                    lai_veg_index_type))
            return False
    if calc_dict['lai_toa']:
        lai_toa_veg_index_type = read_param(
            'lai_toa_veg_index_type', 'SAVI', config).upper()
        if lai_toa_veg_index_type == 'SAVI':
            calc_dict['savi_toa'] = True
        elif lai_toa_veg_index_type == 'NDVI':
            calc_dict['ndvi_toa'] = True
        else:
            logging.error(
                ('\nERROR: The LAI TOA veg. index type {} is invalid.' +
                 '\nERROR: Set lai_toa_veg_index_type to SAVI or NDVI').format(
                    lai_toa_veg_index_type))
            return False
    if calc_dict['savi_toa'] or calc_dict['savi_sur']:
        savi_l_flt = read_param('savi_l', 0.1, config)

    # Calculate refl_toa if any TOA indices flags are True
    if any([v for k, v in calc_dict.items()
            if image.indices_ws in raster_dict[k] and '_toa' in k]):
        calc_dict['refl_toa'] = True
    # Calculate refl_sur if any non-TOA indices flags are True
    refl_toa_index_flag = False
    if any([v for k, v in calc_dict.items()
            if image.indices_ws in raster_dict[k] and
            ('_toa' not in k)]):
        refl_toa_index_flag = True
        calc_dict['refl_sur'] = True

    # At-surface albedo
    if calc_dict['albedo_sur']:
        calc_dict['refl_sur'] = True

    # At-surface reflectance
    if calc_dict['refl_sur']:
        # Remove refl_sur key/value then set LEDAPS or Tasumi
        del calc_dict['refl_sur']
        refl_sur_model_type_list = ['LEDAPS', 'TASUMI']
        refl_sur_model_type = read_param(
            'refl_sur_model_type', 'TASUMI', config).upper()
        if refl_sur_model_type.upper() not in refl_sur_model_type_list:
            logging.error(
                ('\nERROR: Surface reflectance model type {} is invalid.' +
                 '\nERROR: Set refl_sur_model_type to {}').format(
                    refl_sur_model_type, ','.join(refl_sur_model_type_list)))
            return False
        elif refl_sur_model_type.upper() == 'LEDAPS':
            calc_dict['refl_sur_ledaps'] = True
            calc_dict['refl_sur_tasumi'] = False
        elif refl_sur_model_type.upper() == 'TASUMI':
            calc_dict['refl_toa'] = True
            calc_dict['refl_sur_tasumi'] = True
            calc_dict['refl_sur_ledaps'] = False
    kt_flt = read_param('kt', 1.0, config)
    # Tasumi at-surface reflectance and transmittance
    if ((calc_dict['refl_sur_tasumi'] or calc_dict['tau']) and not
            os.path.isfile(raster_dict['cos_theta'])):
        calc_dict['cos_theta'] = True
        kt_flt = read_param('kt', 1.0, config)
    # Air pressure model dependent parameters
    if calc_dict['refl_sur_tasumi'] or calc_dict['tau']:
        pair_model_list = ['DATUM', 'DEM']
        pair_model = read_param('pair_model', 'DEM', config).upper()
        if pair_model not in pair_model_list:
            logging.error(
                ('\nERROR: The Pair model {} is not a valid option.' +
                 '\nERROR: Set pair_model to DATUM or DEM').format(
                    pair_model))
            return False
        # Get Datum elevation
        if pair_model == 'DATUM' or calc_dict['ts_dem']:
            datum_flt = float(config.get('INPUTS', 'datum'))
        # Get DEM elevation
        if pair_model == 'DEM':
            calc_dict['dem'] = True
    else:
        pair_model = None

    # Calculate a centroid based cos_theta value
    # DEADBEEF - Move this to image class?
    if calc_dict['cos_theta']:
        logging.debug('\nCos(theta)')
        # Get mask extent center in decimal degrees
        lon_center, lat_center = drigo.project_point(
            env.mask_extent.center(), env.snap_osr, env.snap_gcs_osr)
        cos_theta_centroid_flt = et_common.cos_theta_centroid_func(
            image.acq_time, image.acq_doy, image.dr,
            lon_center * math.pi / 180, lat_center * math.pi / 180)
        del lon_center, lat_center
        logging.debug('  Centroid: {}'.format(cos_theta_centroid_flt))

    # Spatial/Mountain model input rasters
    if calc_dict['cos_theta']:
        cos_theta_model_list = ['SOLAR', 'CENTROID', 'SPATIAL', 'MOUNTAIN']
        cos_theta_model = read_param(
            'cos_theta_model', 'CENTROID', config).upper()
        if cos_theta_model not in cos_theta_model_list:
            logging.error(
                ('\nERROR: The Cos(theta) model {} is not a valid option.' +
                 '\nERROR: Set cos_theta_model to {}').format(
                    cos_theta_model, ', '.join(cos_theta_model_list)))
            return False
        # I can't move these up since I have to read cos_theta_model first
        if cos_theta_model == 'MOUNTAIN':
            calc_dict['lon'] = True
            calc_dict['lat'] = True
            calc_dict['slp'] = True
            calc_dict['asp'] = True
        elif cos_theta_model == 'SPATIAL':
            calc_dict['lon'] = True
            calc_dict['lat'] = True
            calc_dict['slp'] = False
            calc_dict['asp'] = False
        else:
            calc_dict['lon'] = False
            calc_dict['lat'] = False
            calc_dict['slp'] = False
            calc_dict['asp'] = False

    # Rasters can be read from local copy or clipped from remote copy
    for key, raster_name in [
            ['dem', 'dem_raster'],
            ['landuse', 'landuse_raster'],
            ['slp', 'slope_raster'],
            ['asp', 'aspect_raster'],
            ['lat', 'latitude_raster'],
            ['lon', 'longitude_raster']]:
        # Skip if raster is not needed and reset save flag
        if not calc_dict[key]:
            save_dict[key] = False
        # Read local raster if possible
        elif (os.path.isfile(raster_dict[key]) and
                drigo.raster_path_extent(raster_dict[key]) == env.mask_extent):
            raster_dict[key + '_full'] = raster_dict[key]
            save_dict[key] = False
        # Otherwise try to read read full/external path
        else:
            raster_dict[key + '_full'] = config.get('INPUTS', raster_name)
            if not os.path.isfile(raster_dict[key + '_full']):
                logging.error(
                    '\nERROR: The raster path {} is not valid'.format(
                        raster_dict[key + '_full']))
                return False

    # Landuse type
    if calc_dict['landuse']:
        # For now only read NLCD landuse rasters
        landuse_type = read_param(
            'landuse_type', 'NLCD', config).upper()
        landuse_type_list = ['NLCD']
        # landuse_type_list = ['NLCD', 'CDL']
        if landuse_type not in landuse_type_list:
            logging.error(
                ('\nERROR: The landuse type {} is invalid.' +
                 '\nERROR: Set landuse_type to {}').format(
                    landuse_type, ', '.join(landuse_type_list)))
            return False

    # # Spatial/Mountain model input rasters
    # if calc_dict['cos_theta']:
    #     cos_theta_model_list = ['SOLAR', 'CENTROID', 'SPATIAL', 'MOUNTAIN']
    #     cos_theta_model = read_param('cos_theta_model', 'CENTROID', config).upper()
    #     if cos_theta_model not in cos_theta_model_list:
    #          logging.error(
    #              ('\nERROR: The Cos(theta) model {} is not a valid option.' +
    #               '\nERROR: Set cos_theta_model to {}').format(
    #              cos_theta_model, ', '.join(cos_theta_model_list)))
    #          return False
    #     # I can't move these up since I have to read cos_theta_model first
    #     if cos_theta_model in ['SPATIAL', 'MOUNTAIN']:
    #          calc_dict['lon'] = True
    #          calc_dict['lat'] = True
    #     if cos_theta_model == 'MOUNTAIN':
    #          calc_dict['slp'] = True
    #          calc_dict['asp'] = True
    #     for local_key, full_key, raster_name in [
    #          ['slp', 'slp_full', 'slope_raster'],
    #          ['asp', 'asp_full', 'aspect_raster'],
    #          ['lat', 'lat_full', 'latitude_raster'],
    #          ['lon', 'lon_full', 'longitude_raster']]:
    #          # Check that the output/sub rasters exist
    #          # Check that they have the correct shape
    #          if calc_dict[local_key]:
    #              if (save_dict[local_key] or
    #                  not os.path.isfile(raster_dict[local_key]) or
    #                  drigo.raster_path_extent(raster_dict[local_key]) != env.mask_extent):
    #                  save_dict[local_key] = True
    #                  raster_dict[full_key] = config.get('INPUTS', raster_name)
    #                  # Check that the input rasters exist
    #                  if not os.path.isfile(raster_dict[full_key]):
    #                      logging.error(
    #                          '\nERROR: The raster path {} is not valid'.format(
    #                              raster_dict[full_key]))
    #                      return False
    #              # Otherwise script reads from "full" path,
    #              #   so set full path to local path
    #              else:
    #                  raster_dict[full_key] = raster_dict[local_key]
    # # Terrain model dependent parameters
    # # if True:
    # #     terrain_model_list = ['FLAT', 'MOUNTAIN']
    # #     terrain_model = read_param('terrain_model', 'FLAT', config).upper()
    # #     if terrain_model not in terrain_model_list:
    # #         logging.error(
    # #             ('\nERROR: The terrain model {} is not a valid option.' +
    # #              '\nERROR: Set terrain_model to FLAT or MOUNTAIN').format(
    # #             terrain_model))
    # #         return False
    # # For elevation rasters, calc means it will be read locally
    # #   save means it will be extracted from remote location first

    # # DEM
    # if calc_dict['dem']:
    #     # Get the input file DEM raster path if needed
    #     if (save_dict['dem'] or
    #         not os.path.isfile(raster_dict['dem']) or
    #         drigo.raster_path_extent(raster_dict['dem']) != env.mask_extent):
    #         raster_dict['dem_full'] = config.get('INPUTS','dem_raster')
    #         if not os.path.isfile(raster_dict['dem_full']):
    #             logging.error(
    #                 '\nERROR: The dem_raster path {} is not valid'.format(
    #                     raster_dict['dem_full']))
    #             return False
    #     # Otherwise script reads from "full" path,
    #     #   so set full path to local path
    #     else:
    #         raster_dict['dem_full'] = raster_dict['dem']
    #
    # # Landuse
    # if calc_dict['landuse']:
    #     # Get the input file NLCD raster path if needed
    #     if (save_dict['nlcd'] or
    #         not os.path.isfile(raster_dict['nlcd']) or
    #         drigo.raster_path_extent(raster_dict['nlcd']) != env.mask_extent):
    #         raster_dict['landuse_full'] = config.get('INPUTS', 'landuse_raster')
    #         if not os.path.isfile(raster_dict['landuse_full']):
    #             logging.error(
    #                 '\nERROR: The landuse raster {} does not exist'.format(
    #                     raster_dict['landuse_full']))
    #             return False
    #         landuse_type = read_param('landuse_type', 'NLCD', config).upper()
    #         if landuse_type not in ['NLCD', 'CDL']:
    #             logging.error(
    #                 ('\nERROR: The landuse type {} is invalid.' +
    #                  '\nERROR: Set landuse_type to NLCD or CDL').format(
    #                 landuse_type))
    #             return False
    #     # Otherwise script reads from "full" path,
    #     #   so set full path to local path
    #     else:
    #         raster_dict['landuse_full'] = raster_dict['nlcd']
    # Weather Data
    if calc_dict['refl_sur_tasumi'] or calc_dict['tau']:
        weather_data_source = config.get(
            'INPUTS', 'weather_data_source').upper()
        log_fmt = '    {:<18s} {}'
        if weather_data_source not in ['NLDAS', 'REFET', 'MANUAL']:
            logging.error(
                ('\nERROR: The weather data source {} is invalid.' +
                 '\nERROR: Set weather_data_source to REFET or MANUAL').format(
                    weather_data_source))
            return False
        elif weather_data_source == 'NLDAS':
            logging.info('\nWeather parameters from NDLAS rasters')
            # DEADBEEF - Testing where to store Landsat scene NLDAS Ea rasters
            # Assuming Ea raster was clipped/projected into SUPPORT_RASTERS
            if not os.path.isfile(raster_dict['ea']):
                logging.error(
                    ('\nERROR: NLDAS Ea raster does not exist\n' +
                     '  {}').format(raster_dict['ea']))
                return False
            calc_dict['ea'] = True
        elif weather_data_source == 'REFET':
            gmt_offset_flt = float(config.get('INPUTS', 'gmt_offset'))
            logging.debug('\n  Weather parameters from RefET file')
            refet_file = config.get('INPUTS', 'refet_file')
            logging.debug('  {}'.format(refet_file))
            if not os.path.isfile(refet_file):
                logging.error('\nERROR: The refet_file path is not valid')
                return False
            # The RefET data is localtime, scene acquisition time is GMT
            acq_localtime = image.acq_time + gmt_offset_flt
            # Get RefET Data
            (dew_point_flt, wind_speed_flt, ea_flt,
             etr_flt, etr_24hr_flt) = et_common.read_refet_instantaneous_func(
                 refet_file, image.acq_year, image.acq_doy, acq_localtime)
            ea_array = np.array([ea_flt])
            # Output RefET Data
            logging.debug('\n  Interpolated Values:')
            log_fmt = '    {:<22s} {}'
            logging.debug(log_fmt.format('Scene Time:', acq_localtime))
            logging.debug(log_fmt.format('Dew Point [C]:', dew_point_flt))
            logging.debug(log_fmt.format('Wind Speed [m/s]:', wind_speed_flt))
            logging.debug(log_fmt.format('Ea [kPa]:', ea_flt))
            logging.debug(log_fmt.format('ETr [mm/hr]:', etr_flt))
            logging.debug(log_fmt.format('ETr 24hr [mm/day]:', etr_24hr_flt))
        elif weather_data_source == 'MANUAL':
            logging.info('\n  Weather parameters from INI file')
            ea_flt = float(config.get('INPUTS', 'ea'))
            ea_array = np.array([ea_flt])
            logging.debug(log_fmt.format('Ea [kPa]:', ea_flt))

    # Build necessary output folders
    logging.debug('\nBuilding output folders')
    if save_dict['refl_sur_tasumi']:
        if not os.path.isdir(image.refl_sur_ws):
            os.makedirs(image.refl_sur_ws)
    if any([v for k, v in save_dict.items()
            if image.indices_ws in raster_dict[k]]):
        if not os.path.isdir(image.indices_ws):
            os.makedirs(image.indices_ws)

    # Remove existing and build new empty rasters if necessary
    logging.debug('\nBuilding empty rasters')
    for name, save_flag in sorted(save_dict.items()):
        # logging.debug('{} {}'.format(name, save_flag))
        if save_flag:
            band_cnt, raster_type = 1, np.float32
            if name == 'refl_sur_tasumi':
                band_cnt = image.band_sur_cnt
            elif name == 'landuse_sub':
                raster_type = np.uint8
            logging.debug(raster_dict[name])
            drigo.build_empty_raster(raster_dict[name], band_cnt, raster_type)
            del band_cnt

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

        # Prebuild Landuse array even though it isn't used in Model 1
        if calc_dict['landuse']:
            landuse_array, landuse_nodata = drigo.raster_to_array(
                raster_dict['landuse_full'], 1, block_extent,
                return_nodata=True)
            landuse_array[block_nodata_mask] = landuse_nodata
        if save_dict['landuse']:
            drigo.block_to_raster(
                landuse_array, raster_dict['landuse'], b_i, b_j, bs)
        if calc_dict['landuse']:
            del landuse_array, landuse_nodata

        # Mountain rasters, and landuse by block
        if calc_dict['slp']:
            slope_array, slope_nodata = drigo.raster_to_array(
                raster_dict['slp'], 1, block_extent, return_nodata=True)
            slope_array[block_nodata_mask] = slope_nodata
        if calc_dict['asp']:
            aspect_array, aspect_nodata = drigo.raster_to_array(
                raster_dict['asp'], 1, block_extent, return_nodata=True)
            aspect_array[block_nodata_mask] = aspect_nodata
        if calc_dict['lat']:
            lat_array, lat_nodata = drigo.raster_to_array(
                raster_dict['lat'], 1, block_extent, return_nodata=True)
            lat_array[block_nodata_mask] = lat_nodata
        if calc_dict['lon']:
            lon_array, lon_nodata = drigo.raster_to_array(
                raster_dict['lon'], 1, block_extent, return_nodata=True)
            lon_array[block_nodata_mask] = lon_nodata
        if save_dict['slp']:
            drigo.block_to_raster(slope_array, raster_dict['slp'], b_i, b_j, bs)
        if save_dict['asp']:
            drigo.block_to_raster(aspect_array, raster_dict['asp'], b_i, b_j, bs)
        if save_dict['lat']:
            drigo.block_to_raster(lat_array, raster_dict['lat'], b_i, b_j, bs)
        if save_dict['lon']:
            drigo.block_to_raster(lon_array, raster_dict['lon'], b_i, b_j, bs)
        # logging.info('Build Latitude/Longitude Rasters for Common Area')
        # lat_lon_array_func(lat_sub_raster, lon_sub_raster)

        # Cos(theta) by block
        if calc_dict['cos_theta']:
            if cos_theta_model == 'MOUNTAIN':
                # lon_array = drigo.raster_to_block(
                #     raster_dict['lon_sub'],
                #     b_i, b_j, bs, return_nodata=False)
                # lat_array = drigo.raster_to_block(
                #     raster_dict['lat_sub'],
                #     b_i, b_j, bs, return_nodata=False)
                # slope_array = drigo.raster_to_block(
                #     raster_dict['slope_sub'],
                #     b_i, b_j, bs, return_nodata=False)
                # aspect_array = drigo.raster_to_block(
                #     raster_dict['aspect_sub'],
                #     b_i, b_j, bs, return_nodata=False)
                cos_theta_array = et_numpy.cos_theta_mountain_func(
                    image.acq_time, image.acq_doy, image.dr,
                    lon_array, lat_array, slope_array, aspect_array)
                del lon_array, lat_array, slope_array, aspect_array
                # Also build a simple cos(theta) array for refl_toa
                cos_theta_toa_array = np.empty(
                    block_data_mask.shape).astype(np.float32)
                cos_theta_toa_array[block_data_mask] = image.cos_theta_solar
                cos_theta_toa_array[block_nodata_mask] = np.nan
            elif cos_theta_model == 'SPATIAL':
                # lon_array = drigo.raster_to_block(
                #     raster_dict['lon_sub'],
                #     b_i, b_j, bs, return_nodata=False)
                # lat_array = drigo.raster_to_block(
                #     raster_dict['lat_sub'],
                #     b_i, b_j, bs, return_nodata=False)
                cos_theta_array = et_numpy.cos_theta_spatial_func(
                    image.acq_time, image.acq_doy, image.dr,
                    lon_array, lat_array)
                del lon_array, lat_array
            elif cos_theta_model == 'CENTROID':
                cos_theta_array = np.empty(
                    block_data_mask.shape).astype(np.float32)
                cos_theta_array[block_data_mask] = cos_theta_centroid_flt
                cos_theta_array[block_nodata_mask] = np.nan
            elif cos_theta_model == 'SOLAR':
                cos_theta_array = np.empty(
                    block_data_mask.shape).astype(np.float32)
                cos_theta_array[block_data_mask] = image.cos_theta_solar
                cos_theta_array[block_nodata_mask] = np.nan
        if save_dict['cos_theta']:
            drigo.block_to_raster(
                cos_theta_array, raster_dict['cos_theta'],
                b_i, b_j, bs)
        if calc_dict['slp']:
            del slope_array
        if calc_dict['asp']:
            del aspect_array
        if calc_dict['lat']:
            del lat_array
        if calc_dict['lon']:
            del lon_array

        # Read in TOA Reflectance
        if calc_dict['refl_toa']:
            refl_toa_array = np.zeros(
                (block_rows, block_cols, image.band_toa_cnt),
                dtype=np.float32)
            for band, band_i in sorted(image.band_toa_dict.items()):
                refl_toa_array[:, :, band_i - 1] = drigo.raster_to_block(
                    raster_dict['refl_toa'], b_i, b_j, bs, band_i,
                    return_nodata=False)
            refl_toa_array[block_nodata_mask, :] = np.nan

        # METRIC default indices using TOA reflectance
        # All other indices will use surface reflectance instead
        # Don't remove NDVI or LAI
        # NDVI
        if calc_dict['ndvi_toa']:
            ndvi_toa_array = et_numpy.ndi_func(
                refl_toa_array[:, :, 4 - 1], refl_toa_array[:, :, 3 - 1])
        if save_dict['ndvi_toa']:
            drigo.block_to_raster(
                ndvi_toa_array, raster_dict['ndvi_toa'], b_i, b_j, bs)
        # NDVI
        if save_dict['ndwi_toa']:
            ndwi_toa_array = et_numpy.ndi_func(
                refl_toa_array[:, :, 5 - 1], refl_toa_array[:, :, 2 - 1])
        if calc_dict['ndwi_toa']:
            drigo.block_to_raster(
                ndwi_toa_array, raster_dict['ndwi_toa'], b_i, b_j, bs)
        # SAVI
        if calc_dict['savi_toa']:
            savi_toa_array = et_numpy.ndi_func(
                refl_toa_array[:, :, 4 - 1], refl_toa_array[:, :, 3 - 1],
                savi_l_flt)
        if save_dict['savi_toa']:
            drigo.block_to_raster(
                savi_toa_array, raster_dict['savi_toa'], b_i, b_j, bs)

        # LAI (from SAVI or NDVI)
        if calc_dict['lai_toa'] and lai_toa_veg_index_type == 'SAVI':
            lai_toa_array = et_numpy.savi_lai_func(savi_toa_array)
        elif calc_dict['lai_toa'] and lai_toa_veg_index_type == 'NDVI':
            lai_toa_array = et_numpy.ndvi_lai_func(ndvi_toa_array)
        if save_dict['lai_toa']:
            drigo.block_to_raster(
                lai_toa_array, raster_dict['lai_toa'], b_i, b_j, bs)
        if calc_dict['savi_toa']:
            del savi_toa_array

        # DEM
        if calc_dict['dem']:
            elev_array = drigo.raster_to_array(
                raster_dict['dem_full'], 1, block_extent, -9999.0,
                return_nodata=False)
            elev_array = elev_array.astype(np.float32)
            elev_array[block_nodata_mask] = np.nan
        if save_dict['dem']:
            drigo.block_to_raster(
                elev_array, raster_dict['dem'], b_i, b_j, bs)

        # At surface reflectance, transmittance, & albedo
        # Pre calculate air pressure and precipitable water
        if calc_dict['refl_sur_tasumi'] or calc_dict['tau']:
            # Air Pressure
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
                ea_array = drigo.raster_to_array(
                    raster_dict['ea'], 1, block_extent, return_nodata=False)
                ea_array = ea_array.astype(np.float32)
                ea_array[block_nodata_mask] = np.nan
            else:
                ea_array = np.array([ea_flt])
            # Precipitable water
            w_array = et_common.precipitable_water_func(pair_array, ea_array)
            del ea_array

        # Transmittance can be pre-calculated for Model2 Rn calculation
        if calc_dict['tau'] or save_dict['tau']:
            if (not calc_dict['cos_theta'] and
                    os.path.isfile(raster_dict['cos_theta'])):
                # read in cos_theta
                cos_theta_array = drigo.raster_to_block(
                    raster_dict['cos_theta'], b_i, b_j, bs,
                    return_nodata=False)
            tau_array = et_numpy.tau_broadband_func(
                pair_array, w_array, cos_theta_array)
            drigo.block_to_raster(
                tau_array, raster_dict['tau'], b_i, b_j, bs)
            del tau_array

        # Read in LEDAPS at-surface reflectance
        if calc_dict['refl_sur_ledaps']:
            refl_sur_array = np.zeros(
                (block_rows, block_cols, image.band_sur_cnt),
                dtype=np.float32)
            for band, band_i in sorted(image.band_sur_dict.items()):
                refl_sur_array[:, :, band_i - 1] = drigo.raster_to_block(
                    raster_dict['refl_sur_ledaps'], b_i, b_j, bs, band_i,
                    return_nodata=False)
            refl_sur_array[block_nodata_mask, :] = np.nan
        # Calculate Tasumi at-surface reflectance
        elif calc_dict['refl_sur_tasumi']:
            refl_sur_array = et_numpy.refl_sur_tasumi_func(
                refl_toa_array[:, :, image.band_toa_sur_mask],
                pair_array, w_array, cos_theta_array, kt_flt,
                image.c1, image.c2, image.c3, image.c4, image.c5,
                image.cb, image.band_sur_cnt)
        if save_dict['refl_sur_tasumi']:
            for band, band_i in sorted(image.band_sur_dict.items()):
                drigo.block_to_raster(
                    refl_sur_array[:, :, band_i - 1],
                    raster_dict['refl_sur_tasumi'],
                    b_i, b_j, bs, band_i)

        # Cleanup
        if calc_dict['refl_sur_tasumi'] or calc_dict['tau']:
            del pair_array, w_array
        if calc_dict['refl_toa']:
            del refl_toa_array
        if calc_dict['cos_theta']:
            del cos_theta_array

        # Calculate at surface albedo
        if calc_dict['albedo_sur']:
            albedo_sur_array = et_numpy.albedo_sur_func(
                refl_sur_array, image.wb)
        if save_dict['albedo_sur']:
            drigo.block_to_raster(
                albedo_sur_array, raster_dict['albedo_sur'], b_i, b_j, bs)
            del albedo_sur_array

        # Non METRIC Indices (using surface reflectance)
        if calc_dict['ndvi_sur']:
            ndvi_sur_array = et_numpy.ndi_func(
                refl_sur_array[:, :, 4 - 1], refl_sur_array[:, :, 3 - 1])
        if save_dict['ndvi_sur']:
            drigo.block_to_raster(
                ndvi_sur_array, raster_dict['ndvi_sur'], b_i, b_j, bs)
        if calc_dict['ndwi_sur'] or save_dict['ndwi_sur']:
            # This is the NDWI Rick Allen uses in the METRIC model,
            #   but it is identical to MNDWI below
            ndwi_sur_array = et_numpy.ndi_func(
                refl_sur_array[:, :, 5 - 1], refl_sur_array[:, :, 2 - 1])
            drigo.block_to_raster(
                ndwi_sur_array, raster_dict['ndwi_sur'], b_i, b_j, bs)
        if calc_dict['savi_sur']:
            savi_sur_array = et_numpy.ndi_func(
                refl_sur_array[:, :, 4 - 1], refl_sur_array[:, :, 3 - 1],
                savi_l_flt)
        if save_dict['savi_sur']:
            drigo.block_to_raster(
                savi_sur_array, raster_dict['savi_sur'], b_i, b_j, bs)
        if calc_dict['lai_sur']:
            if lai_veg_index_type == 'SAVI':
                lai_sur_array = et_numpy.savi_lai_func(savi_sur_array)
            else:
                lai_sur_array = et_numpy.ndvi_lai_func(ndvi_sur_array)
        if save_dict['lai_sur']:
            drigo.block_to_raster(
                lai_sur_array, raster_dict['lai_sur'], b_i, b_j, bs)
        if calc_dict['savi_sur']:
            del savi_sur_array

        # Narrowband emissivity
        if calc_dict['em_nb']:
            if em_refl_type == 'TOA' and em_water_index_type == 'NDVI':
                em_nb_array = et_numpy.em_nb_func(
                    lai_toa_array, ndvi_toa_array)
            elif em_refl_type == 'SUR' and em_water_index_type == 'NDVI':
                em_nb_array = et_numpy.em_nb_func(
                    lai_sur_array, ndvi_sur_array)
            elif em_refl_type == 'TOA' and em_water_index_type == 'NDWI':
                em_nb_array = et_numpy.em_nb_func(
                    lai_toa_array, ndwi_toa_array)
            elif em_refl_type == 'SUR' and em_water_index_type == 'NDWI':
                em_nb_array = et_numpy.em_nb_func(
                    lai_sur_array, ndwi_sur_array)
        if save_dict['em_nb']:
            drigo.block_to_raster(
                em_nb_array, raster_dict['em_nb'], b_i, b_j, bs)

        # Broadband emissivity
        if calc_dict['em_0']:
            if em_refl_type == 'TOA' and em_water_index_type == 'NDVI':
                em_0_array = et_numpy.em_0_func(lai_toa_array, ndvi_toa_array)
            elif em_refl_type == 'SUR' and em_water_index_type == 'NDVI':
                em_0_array = et_numpy.em_0_func(lai_sur_array, ndvi_sur_array)
            # elif em_refl_type == 'TOA' and em_water_index_type == 'NDWI':
            #     em_0_array = em_0_func(lai_toa_array, ndwi_toa_array)
            # elif em_refl_type == 'SUR' and em_water_index_type == 'NDWI':
            #     em_0_array = em_0_func(lai_array, ndwi_array)
        if save_dict['em_0']:
            drigo.block_to_raster(
                em_0_array, raster_dict['em_0'], b_i, b_j, bs)
        if calc_dict['em_0']:
            del em_0_array

        # Cleanup
        if calc_dict['ndvi_sur']:
            del ndvi_sur_array
        if calc_dict['ndwi_sur']:
            del ndwi_sur_array
        if calc_dict['lai_sur']:
            del lai_sur_array
        if calc_dict['ndvi_toa']:
            del ndvi_toa_array
        if calc_dict['ndwi_toa']:
            del ndwi_toa_array
        if calc_dict['lai_toa']:
            del lai_toa_array

        # Corrected radiance
        if calc_dict['ts_bt']:
            ts_bt_array = drigo.raster_to_block(
                raster_dict['ts_bt'], b_i, b_j, bs, return_nodata=False)
            ts_bt_array[block_nodata_mask] = np.nan
        if calc_dict['rc']:
            thermal_rad_array = et_numpy.thermal_rad_func(
                ts_bt_array, image.k1_dict[image.thermal_band],
                image.k2_dict[image.thermal_band])
            rc_array = et_numpy.rc_func(
                thermal_rad_array, em_nb_array,
                rp_flt, tnb_flt, rsky_flt)
            del thermal_rad_array
        if save_dict['rc']:
            drigo.block_to_raster(rc_array, raster_dict['rc'], b_i, b_j, bs)
        if calc_dict['ts_bt']:
            del ts_bt_array

        # Surface temperature
        if calc_dict['ts']:
            ts_array = et_numpy.ts_func(
                em_nb_array, rc_array, image.k1_dict[image.thermal_band],
                image.k2_dict[image.thermal_band])
        if save_dict['ts']:
            drigo.block_to_raster(ts_array, raster_dict['ts'], b_i, b_j, bs)
        if calc_dict['rc']:
            del rc_array
        if calc_dict['em_nb']:
            del em_nb_array

        # Delapsed Surface temperature
        # if calc_dict['ts_dem'] and calc_dict['ts']:
        #     ts_dem_array = et_numpy.ts_delapsed_func(
        #          ts_array, elev_array, datum_flt, lapse_rate_flt)
        # if calc_dict['ts_dem'] and not calc_dict['ts']:
        #     ts_array = drigo.raster_to_block(
        #         raster_dict['ts'], b_i, b_j, bs, return_nodata=False)
        if calc_dict['ts_dem']:
            ts_dem_array = et_numpy.ts_delapsed_func(
                ts_array, elev_array, datum_flt, lapse_rate_flt)
        if save_dict['ts_dem']:
            drigo.block_to_raster(
                ts_dem_array, raster_dict['ts_dem'], b_i, b_j, bs)
            del ts_dem_array
        if calc_dict['ts']:
            del ts_array

        # DEADBEEF - Brightness temp is provided by LEDAPS/ESPA
        # Brightness temperature
        # if calc_dict['ts_bt']:
        #     rc_bt_array = et_numpy.rc_func(
        #         thermal_rad_toa_array, 1., 0, 1., 1.)
        #     # em_nb is 1, but needs to be an array of
        #     ts_bt_array = et_numpy.ts_func(
        #         block_data_mask.astype(np.float32),
        #         rc_bt_array, image.k_dict)
        # if save_dict['ts_bt']:
        #     drigo.block_to_raster(
        #         ts_bt_array, raster_dict['ts_bt'], b_i, b_j, bs)
        # if calc_dict['ts_bt']:
        #     del ts_bt_array, rc_bt_array
        # if calc_dict['rc'] or calc_dict['ts_bt']:
        #     del thermal_rad_toa_array

        # Cleanup
        if calc_dict['dem']:
            del elev_array
        del block_nodata_mask, block_data_mask, block_rows, block_cols

    # Raster Pyramids
    if pyramids_flag:
        logging.info('\nBuild Pyramids')
        for name, save_flag in sorted(save_dict.items()):
            if save_flag:
                logging.debug('  {}'.format(raster_dict[name]))
                drigo.raster_pyramids(raster_dict[name])

    # Raster Statistics
    if stats_flag:
        logging.info('\nCalculate Statistics')
        for name, save_flag in sorted(save_dict.items()):
            if save_flag:
                logging.debug('  {}'.format(raster_dict[name]))
                drigo.raster_statistics(raster_dict[name])

    # Cleanup
    if remove_refl_toa_flag and os.path.isdir(image.refl_toa_ws):
        shutil.rmtree(image.refl_toa_ws)
    if remove_refl_sur_flag and os.path.isdir(image.refl_sur_ws):
        shutil.rmtree(image.refl_sur_ws)
    if remove_ts_bt_flag and os.path.isfile(image.ts_bt_raster):
        remove_file(image.ts_bt_raster)
    del save_dict, calc_dict, image

    return True


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Model 1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-i', '--ini', required=True,
        help='METRIC input file', metavar='PATH')
    parser.add_argument(
        '-bs', '--blocksize', default=None, type=int,
        help='Processing block size (overwrite INI blocksize parameter)')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--stats', default=None, action="store_true",
        help='Compute raster statistics')
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
        log_file_name = 'metric_model1_log.txt'
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

    metric_model1(image_ws=args.workspace, ini_path=args.ini,
                  bs=args.blocksize, stats_flag=args.stats,
                  overwrite_flag=args.overwrite)
