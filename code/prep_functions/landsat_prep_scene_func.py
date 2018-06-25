#!/usr/bin/env python
#--------------------------------
# Name:         landsat_prep_scene_func.py
# Purpose:      Prepare a single Landsat scene
#--------------------------------

import argparse
from datetime import datetime, timedelta
import logging
import os
import random
import re
import shutil
from time import sleep

import drigo
import numpy as np
from scipy import ndimage
from osgeo import gdal

import et_common
import et_image
import et_numpy
import python_common


def main(image_ws, ini_path, bs=2048, smooth_flag=False,
         stats_flag=False, overwrite_flag=False):
    """Prep a Landsat scene for METRIC

    Parameters
    ----------
    image_ws : str
        Landsat scene folder that will be prepped.
    ini_path : str
        File path of the input parameters file.
    bs : int, optional
        Processing block size (the default is 2048).
    smooth_flag : bool, optional
        If True, dilate/erode image to remove fringe/edge pixels
        (the default is False).
    stats_flag : bool, optional
        If True, compute raster statistics (the default is True).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    True is successful

    """

    # Open config file
    config = python_common.open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    calc_refl_toa_flag = python_common.read_param(
        'calc_refl_toa_flag', True, config, 'INPUTS')
    calc_refl_toa_qa_flag = python_common.read_param(
        'calc_refl_toa_qa_flag', True, config, 'INPUTS')
    # calc_refl_sur_ledaps_flag = python_common.read_param(
    #     'calc_refl_sur_ledaps_flag', False, config, 'INPUTS')
    # calc_refl_sur_qa_flag = python_common.read_param(
    #     'calc_refl_sur_qa_flag', False, config, 'INPUTS')
    calc_ts_bt_flag = python_common.read_param(
        'calc_ts_bt_flag', True, config, 'INPUTS')

    # Use QA band to set common area
    # Fmask cloud, shadow, & snow pixels will be removed from common area
    calc_fmask_common_flag = python_common.read_param(
        'calc_fmask_common_flag', True, config, 'INPUTS')
    fmask_buffer_flag = python_common.read_param(
        'fmask_buffer_flag', False, config, 'INPUTS')
    fmask_erode_flag = python_common.read_param(
        'fmask_erode_flag', False, config, 'INPUTS')
    if fmask_erode_flag:
        fmask_erode_cells = int(python_common.read_param(
            'fmask_erode_cells', 10, config, 'INPUTS'))
        if fmask_erode_cells == 0 and fmask_erode_flag:
            fmask_erode_flag = False
    # Number of cells to buffer Fmask clouds
    # For now use the same buffer radius and apply to
    if fmask_buffer_flag:
        fmask_buffer_cells = int(python_common.read_param(
            'fmask_buffer_cells', 25, config, 'INPUTS'))
        if fmask_buffer_cells == 0 and fmask_buffer_flag:
            fmask_buffer_flag = False
    # Include hand made cloud masks
    cloud_mask_flag = python_common.read_param(
        'cloud_mask_flag', False, config, 'INPUTS')
    cloud_mask_ws = ""
    if cloud_mask_flag:
        cloud_mask_ws = config.get('INPUTS', 'cloud_mask_ws')

    # Extract separate Fmask rasters
    calc_fmask_flag = python_common.read_param(
        'calc_fmask_flag', True, config, 'INPUTS')
    calc_fmask_cloud_flag = python_common.read_param(
        'calc_fmask_cloud_flag', True, config, 'INPUTS')
    calc_fmask_snow_flag = python_common.read_param(
        'calc_fmask_snow_flag', True, config, 'INPUTS')
    calc_fmask_water_flag = python_common.read_param(
        'calc_fmask_water_flag', True, config, 'INPUTS')

    # Keep Landsat DN, LEDAPS, and Fmask rasters
    keep_dn_flag = python_common.read_param(
        'keep_dn_flag', True, config, 'INPUTS')
    # keep_sr_flag = python_common.read_param(
    #     'keep_sr_flag', True, config, 'INPUTS')

    # For this to work I would need to pass in the metric input file
    # calc_elev_flag = python_common.read_param(
    #     'calc_elev_flag', False, config, 'INPUTS')
    # calc_landuse_flag = python_common.read_param(
    #     'calc_landuse_flag', False, config, 'INPUTS')

    # calc_acca_cloud_flag = python_common.read_param(
    #     'calc_acca_cloud_flag', True, config, 'INPUTS')
    # calc_acca_snow_flag = python_common.read_param(
    #     'calc_acca_snow_flag', True, config, 'INPUTS')
    # calc_ledaps_dem_land_flag = python_common.read_param(
    #     'calc_ledaps_dem_land_flag', False, config, 'INPUTS')
    # calc_ledaps_veg_flag = python_common.read_param(
    #     'calc_ledaps_veg_flag', False, config, 'INPUTS')
    # calc_ledaps_snow_flag = python_common.read_param(
    #     'calc_ledaps_snow_flag', False, config, 'INPUTS')
    # calc_ledaps_land_flag = python_common.read_param(
    #     'calc_ledaps_land_flag', False, config, 'INPUTS')
    # calc_ledaps_cloud_flag = python_common.read_param(
    #     'calc_ledaps_cloud_flag', False, config, 'INPUTS')

    # Interpolate/clip/project hourly rasters for each Landsat scene
    # calc_metric_flag = python_common.read_param(
    #     'calc_metric_flag', False, config, 'INPUTS')
    calc_metric_ea_flag = python_common.read_param(
        'calc_metric_ea_flag', False, config, 'INPUTS')
    calc_metric_wind_flag = python_common.read_param(
        'calc_metric_wind_flag', False, config, 'INPUTS')
    calc_metric_etr_flag = python_common.read_param(
        'calc_metric_etr_flag', False, config, 'INPUTS')
    calc_metric_tair_flag = python_common.read_param(
        'calc_metric_tair_flag', False, config, 'INPUTS')

    # Interpolate/clip/project AWC and daily ETr/PPT rasters
    # to compute SWB Ke for each Landsat scene
    calc_swb_ke_flag = python_common.read_param(
        'calc_swb_ke_flag', False, config, 'INPUTS')
    if cloud_mask_flag:
        spinup_days = python_common.read_param(
            'swb_spinup_days', 30, config, 'INPUTS')
        min_spinup_days = python_common.read_param(
            'swb_min_spinup_days', 5, config, 'INPUTS')

    # Round ea raster to N digits to save space
    rounding_digits = python_common.read_param(
        'rounding_digits', 3, config, 'INPUTS')

    env = drigo.env
    image = et_image.Image(image_ws, env)
    np.seterr(invalid='ignore', divide='ignore')
    gdal.UseExceptions()

    # Input file paths
    dn_image_dict = et_common.landsat_band_image_dict(
        image.orig_data_ws, image.image_re)

    # # Open METRIC config file
    # if config_file:
    #    logging.info(
    #        log_f.format('METRIC INI File:', os.path.basename(config_file)))
    #    config = configparser.ConfigParser()
    #    try:
    #        config.read(config_file)
    #    except:
    #        logging.error('\nERROR: Config file could not be read, ' +
    #                      'is not an input file, or does not exist\n' +
    #                      'ERROR: config_file = {}\n').format(config_file)
    #        sys.exit()
    #    #  Overwrite
    #    overwrite_flag = read_param('overwrite_flag', True, config)
    #
    #    #  Elevation and landuse parameters/flags from METRIC input file
    #    calc_elev_flag = read_param('save_dem_raster_flag', True, config)
    #    calc_landuse_flag = read_param(
    #        'save_landuse_raster_flag', True, config)
    #    if calc_elev_flag:
    #        elev_pr_path = config.get('INPUTS','dem_raster')
    #    if calc_landuse_flag:
    #        landuse_pr_path = config.get('INPUTS', 'landuse_raster')
    # else:
    #    overwrite_flag = False
    #    calc_elev_flag = False
    #    calc_landuse_flag = False
    #
    # Elev raster must exist
    # if calc_elev_flag and not os.path.isfile(elev_pr_path):
    #    logging.error('\nERROR: Elevation raster {} does not exist\n'.format(
    #        elev_pr_path))
    #    return False
    # Landuse raster must exist
    # if calc_landuse_flag and not os.path.isfile(landuse_pr_path):
    #    logging.error('\nERROR: Landuse raster {} does not exist\n'.format(
    #        landuse_pr_path))
    #    return False

    # Removing ancillary files before checking for inputs
    if os.path.isdir(os.path.join(image.orig_data_ws, 'gap_mask')):
        shutil.rmtree(os.path.join(image.orig_data_ws, 'gap_mask'))
    for item in os.listdir(image.orig_data_ws):
        if (image.type == 'Landsat7' and
            (item.endswith('_B8.TIF') or
             item.endswith('_B6_VCID_2.TIF'))):
            os.remove(os.path.join(image.orig_data_ws, item))
        elif (image.type == 'Landsat8' and
              (item.endswith('_B1.TIF') or
               item.endswith('_B8.TIF') or
               item.endswith('_B9.TIF') or
               item.endswith('_B11.TIF'))):
            os.remove(os.path.join(image.orig_data_ws, item))
        elif (item.endswith('_VER.jpg') or
              item.endswith('_VER.txt') or
              item.endswith('_GCP.txt') or
              item == 'README.GTF'):
            os.remove(os.path.join(image.orig_data_ws, item))

    # Check correction level (image must be L1T to process)
    if image.correction != 'L1TP':
        logging.debug('  Image is not L1TP corrected, skipping')
        return False
        # calc_fmask_common_flag = False
        # calc_refl_toa_flag = False
        # calc_ts_bt_flag = False
        # calc_metric_ea_flag = False
        # calc_metric_wind_flag = False
        # calc_metric_etr_flag = False
        # overwrite_flag = False

    # QA band must exist
    if (calc_fmask_common_flag and image.qa_band not in dn_image_dict.keys()):
        logging.warning(
             '\nQA band does not exist but calc_fmask_common_flag=True'
             '\n  Setting calc_fmask_common_flag=False\n  {}'.format(
                 image.qa_band))
        calc_fmask_common_flag = False
    if cloud_mask_flag and not os.path.isdir(cloud_mask_ws):
        logging.warning(
             '\ncloud_mask_ws is not a directory but cloud_mask_flag=True.'
             '\n  Setting cloud_mask_flag=False\n   {}'.format(cloud_mask_ws))
        cloud_mask_flag = False

    # Check for Landsat TOA images
    if (calc_refl_toa_flag and
        (set(list(image.band_toa_dict.keys()) +
                 [image.thermal_band, image.qa_band]) !=
            set(dn_image_dict.keys()))):
        logging.warning(
            '\nMissing Landsat images but calc_refl_toa_flag=True'
            '\n  Setting calc_refl_toa_flag=False')
        calc_refl_toa_flag = False

    # Check for Landsat brightness temperature image
    if calc_ts_bt_flag and image.thermal_band not in dn_image_dict.keys():
        logging.warning(
            '\nThermal band image does not exist but calc_ts_bt_flag=True'
            '\n  Setting calc_ts_bt_flag=False')
        calc_ts_bt_flag = False
        # DEADBEEF - Should the function return False if Ts doesn't exist?
        # return False

    # Check for METRIC hourly/daily input folders
    if calc_metric_ea_flag:
        metric_ea_input_ws = config.get('INPUTS', 'metric_ea_input_folder')
        if not os.path.isdir(metric_ea_input_ws):
            logging.warning(
                 '\nHourly Ea folder does not exist but calc_metric_ea_flag=True'
                 '\n  Setting calc_metric_ea_flag=False\n  {}'.format(
                     metric_ea_input_ws))
            calc_metric_ea_flag = False
    if calc_metric_wind_flag:
        metric_wind_input_ws = config.get('INPUTS', 'metric_wind_input_folder')
        if not os.path.isdir(metric_wind_input_ws):
            logging.warning(
                 '\nHourly wind folder does not exist but calc_metric_wind_flag=True'
                 '\n  Setting calc_metric_wind_flag=False\n  {}'.format(
                     metric_wind_input_ws))
            calc_metric_wind_flag = False
    if calc_metric_etr_flag:
        metric_etr_input_ws = config.get('INPUTS', 'metric_etr_input_folder')
        if not os.path.isdir(metric_etr_input_ws):
            logging.warning(
                 '\nHourly ETr folder does not exist but calc_metric_etr_flag=True'
                 '\n  Setting calc_metric_etr_flag=False\n  {}'.format(
                     metric_etr_input_ws))
            calc_metric_etr_flag = False
    if calc_metric_tair_flag:
        metric_tair_input_ws = config.get('INPUTS', 'metric_tair_input_folder')
        if not os.path.isdir(metric_tair_input_ws):
            logging.warning(
                 '\nHourly Tair folder does not exist but calc_metric_tair_flag=True'
                 '\n  Setting calc_metric_tair_flag=False\n  {}'.format(
                     metric_tair_input_ws))
            calc_metric_tair_flag = False
    if (calc_metric_ea_flag or calc_metric_wind_flag or
            calc_metric_etr_flag or calc_metric_tair_flag):
        metric_hourly_re = re.compile(config.get('INPUTS', 'metric_hourly_re'))
        metric_daily_re = re.compile(config.get('INPUTS', 'metric_daily_re'))

    if calc_swb_ke_flag:
        awc_input_path = config.get('INPUTS', 'awc_input_path')
        etr_input_ws = config.get('INPUTS', 'etr_input_folder')
        ppt_input_ws = config.get('INPUTS', 'ppt_input_folder')
        etr_input_re = re.compile(config.get('INPUTS', 'etr_input_re'))
        ppt_input_re = re.compile(config.get('INPUTS', 'ppt_input_re'))
        if not os.path.isfile(awc_input_path):
            logging.warning(
                 '\nAWC raster does not exist but calc_swb_ke_flag=True'
                 '\n  Setting calc_swb_ke_flag=False\n  {}'.format(
                     awc_input_path))
            calc_swb_ke_flag = False
        if not os.path.isdir(etr_input_ws):
            logging.warning(
                 '\nDaily ETr folder does not exist but calc_swb_ke_flag=True'
                 '\n  Setting calc_swb_ke_flag=False\n  {}'.format(
                     etr_input_ws))
            calc_swb_ke_flag = False
        if not os.path.isdir(ppt_input_ws):
            logging.warning(
                 '\nDaily PPT folder does not exist but calc_swb_ke_flag=True'
                 '\n  Setting calc_swb_ke_flag=False\n  {}'.format(
                     ppt_input_ws))
            calc_swb_ke_flag = False

    # Build folders for support rasters
    if ((calc_fmask_common_flag or calc_refl_toa_flag or
         # calc_refl_sur_ledaps_flag or
         calc_ts_bt_flag or
         calc_metric_ea_flag or calc_metric_wind_flag or
         calc_metric_etr_flag or calc_metric_tair_flag or
         calc_swb_ke_flag) and
        not os.path.isdir(image.support_ws)):
        os.makedirs(image.support_ws)
    if calc_refl_toa_flag and not os.path.isdir(image.refl_toa_ws):
        os.makedirs(image.refl_toa_ws)
    # if calc_refl_sur_ledaps_flag and not os.path.isdir(image.refl_sur_ws):
    #     os.makedirs(image.refl_sur_ws)

    # Apply overwrite flag
    if overwrite_flag:
        overwrite_list = [
            image.fmask_cloud_raster, image.fmask_snow_raster,
            image.fmask_water_raster
            # image.elev_raster, image.landuse_raster
            # image.common_area_raster
        ]
        for overwrite_path in overwrite_list:
            try:
                python_common.remove_file(image.fmask_cloud_raster)
            except:
                pass

    # Use QA band to build common area rasters
    logging.info('\nCommon Area Raster')
    qa_ds = gdal.Open(dn_image_dict[image.qa_band], 0)
    common_geo = drigo.raster_ds_geo(qa_ds)
    common_extent = drigo.raster_ds_extent(qa_ds)
    common_proj = drigo.raster_ds_proj(qa_ds)
    common_osr = drigo.raster_ds_osr(qa_ds)
    # Initialize common_area as all non-fill QA values
    qa_array = drigo.raster_ds_to_array(qa_ds, return_nodata=False)
    common_array = qa_array != 1
    common_rows, common_cols = common_array.shape
    del qa_ds


    # First try applying user defined cloud masks
    cloud_mask_path = os.path.join(
        cloud_mask_ws, image.folder_id + '_mask.shp')
    if cloud_mask_flag and os.path.isfile(cloud_mask_path):
        logging.info('  Applying cloud mask shapefile')
        feature_path = os.path.join(
            cloud_mask_ws, (image.folder_id + '_mask.shp'))
        logging.info('    {}'.format(feature_path))
        cloud_mask_memory_ds = drigo.polygon_to_raster_ds(
            feature_path, nodata_value=0, burn_value=1,
            output_osr=common_osr, output_cs=30,
            output_extent=common_extent)
        cloud_array = drigo.raster_ds_to_array(
            cloud_mask_memory_ds, return_nodata=False)
        # DEADBEEF - If user sets a cloud mask,
        #   it is probably better than Fmask
        # Eventually change "if" calc_fmask_common_flag: to "elif"
        common_array[cloud_array == 1] = 0
        del cloud_mask_memory_ds, cloud_array

    if calc_fmask_common_flag:
        fmask_array = et_numpy.bqa_fmask_func(qa_array)
        fmask_mask = (fmask_array >= 2) & (fmask_array <= 4)
        if fmask_erode_flag:
            logging.info(
                 '  Eroding and dilating Fmask clouds, shadows, and snow '
                 '{} cells\n    to remove errantly masked pixels.'.format(
                    fmask_erode_cells))
            fmask_mask = ndimage.binary_erosion(
                fmask_mask, iterations=fmask_erode_cells,
                structure=ndimage.generate_binary_structure(2, 2))
            fmask_mask = ndimage.binary_dilation(
                fmask_mask, iterations=fmask_erode_cells,
                structure=ndimage.generate_binary_structure(2, 2))
        if fmask_buffer_flag:
            logging.info(
                ('  Buffering Fmask clouds, shadows, and snow ' +
                 '{} cells').format(fmask_buffer_cells))
            # Only buffer clouds, shadow, and snow (not water or nodata)
            if fmask_mask is None:
                fmask_mask = (fmask_array >= 2) & (fmask_array <= 4)
            fmask_mask = ndimage.binary_dilation(
                fmask_mask, iterations=fmask_buffer_cells,
                structure=ndimage.generate_binary_structure(2, 2))
        # Reset common_array for buffered cells
        common_array[fmask_mask] = 0
        del fmask_array, fmask_mask

    if common_array is not None:
        # Erode and dilate to remove fringe on edge
        # Default is to not smooth, but user can force smoothing
        if smooth_flag:
            common_array = smooth_func(common_array)
        # Check that there are some cloud free pixels
        if not np.any(common_array):
            logging.error('  ERROR: There are no cloud/snow free pixels')
            return False
        # Always overwrite common area raster
        # if not os.path.isfile(image.common_area_raster):
        drigo.array_to_raster(
            common_array, image.common_area_raster,
            output_geo=common_geo, output_proj=common_proj,
            stats_flag=stats_flag)
        # Print common geo/extent
        logging.debug('  Common geo:      {}'.format(common_geo))
        logging.debug('  Common extent:   {}'.format(common_extent))


    # Extract Fmask components as separate rasters
    if (calc_fmask_flag or calc_fmask_cloud_flag or calc_fmask_snow_flag or
            calc_fmask_water_flag):
        logging.info('\nFmask')
        fmask_array = et_numpy.bqa_fmask_func(qa_array)

        # Save Fmask data as separate rasters
        if (calc_fmask_flag and not os.path.isfile(image.fmask_output_raster)):
            drigo.array_to_raster(
                fmask_array.astype(np.uint8), image.fmask_output_raster,
                output_geo=common_geo, output_proj=common_proj,
                mask_array=None, output_nodata=255, stats_flag=stats_flag)
        if (calc_fmask_cloud_flag and
                not os.path.isfile(image.fmask_cloud_raster)):
            fmask_cloud_array = (fmask_array == 2) | (fmask_array == 4)
            drigo.array_to_raster(
                fmask_cloud_array.astype(np.uint8), image.fmask_cloud_raster,
                output_geo=common_geo, output_proj=common_proj,
                mask_array=None, output_nodata=255, stats_flag=stats_flag)
            del fmask_cloud_array
        if (calc_fmask_snow_flag and
                not os.path.isfile(image.fmask_snow_raster)):
            fmask_snow_array = (fmask_array == 3)
            drigo.array_to_raster(
                fmask_snow_array.astype(np.uint8), image.fmask_snow_raster,
                output_geo=common_geo, output_proj=common_proj,
                mask_array=None, output_nodata=255, stats_flag=stats_flag)
            del fmask_snow_array
        if (calc_fmask_water_flag and
                not os.path.isfile(image.fmask_water_raster)):
            fmask_water_array = (fmask_array == 1)
            drigo.array_to_raster(
                fmask_water_array.astype(np.uint8), image.fmask_water_raster,
                output_geo=common_geo, output_proj=common_proj,
                mask_array=None, output_nodata=255, stats_flag=stats_flag)
            del fmask_water_array
        del fmask_array

    # # Calculate elevation
    # if calc_elev_flag and not os.path.isfile(elev_path):
    #     logging.info('Elevation')
    #     elev_array, elev_nodata = drigo.raster_to_array(
    #         elev_pr_path, 1, common_extent)
    #     drigo.array_to_raster(
    #         elev_array, elev_raster,
    #         output_geo=common_geo, output_proj=env.snap_proj,
    #         mask_array=common_array, stats_flag=stats_flag)
    #     del elev_array, elev_nodata, elev_path
    #
    # # Calculate landuse
    # if calc_landuse_flag and not os.path.isfile(landuse_raster):
    #     logging.info('Landuse')
    #     landuse_array, landuse_nodata = drigo.raster_to_array(
    #         landuse_pr_path, 1, common_extent)
    #     drigo.array_to_raster(
    #         landuse_array, landuse_raster,
    #         output_geo=common_geo, output_proj=env.snap_proj,
    #         mask_array=common_array, stats_flag=stats_flag)
    #     del landuse_array, landuse_nodata, landuse_raster

    # Calculate toa reflectance
    # f32_gtype, f32_nodata = numpy_to_gdal_type(np.float32)
    if calc_refl_toa_flag:
        logging.info('Top-of-Atmosphere Reflectance')
        if os.path.isfile(image.refl_toa_raster) and overwrite_flag:
            logging.debug('  Overwriting: {}'.format(
                image.refl_toa_raster))
            python_common.remove_file(image.refl_toa_raster)
        if not os.path.isfile(image.refl_toa_raster):
            # First build empty composite raster
            drigo.build_empty_raster(
                image.refl_toa_raster, image.band_toa_cnt, np.float32, None,
                env.snap_proj, env.cellsize, common_extent)
            # cos_theta_solar_flt = et_common.cos_theta_solar_func(
            #    image.sun_elevation)

            # Process by block
            logging.info('Processing by block')
            logging.debug('  Mask  cols/rows: {}/{}'.format(
                common_cols, common_rows))
            for b_i, b_j in drigo.block_gen(common_rows, common_cols, bs):
                logging.debug('  Block  y: {:5d}  x: {:5d}'.format(b_i, b_j))
                block_data_mask = drigo.array_to_block(
                    common_array, b_i, b_j, bs).astype(np.bool)
                block_rows, block_cols = block_data_mask.shape
                block_geo = drigo.array_offset_geo(common_geo, b_j, b_i)
                block_extent = drigo.geo_extent(
                    block_geo, block_rows, block_cols)
                logging.debug('    Block rows: {}  cols: {}'.format(
                    block_rows, block_cols))
                logging.debug('    Block extent: {}'.format(block_extent))
                logging.debug('    Block geo: {}'.format(block_geo))

                # Process each TOA band
                # for band, band_i in sorted(image.band_toa_dict.items()):
                for band, dn_image in sorted(dn_image_dict.items()):
                    if band not in image.band_toa_dict.keys():
                        continue
                    # thermal_band_flag = (band == image.thermal_band)
                    # Set 0 as nodata value
                    drigo.raster_path_set_nodata(dn_image, 0)
                    # Calculate TOA reflectance
                    dn_array, dn_nodata = drigo.raster_to_array(
                        dn_image, 1, block_extent)
                    dn_array = dn_array.astype(np.float64)
                    # dn_array = dn_array.astype(np.float32)
                    dn_array[dn_array == 0] = np.nan
                    #
                    if image.type in ['Landsat4', 'Landsat5', 'Landsat7']:
                        refl_toa_array = et_numpy.l457_refl_toa_band_func(
                            dn_array, image.cos_theta_solar,
                            image.dr, image.esun_dict[band],
                            image.lmin_dict[band], image.lmax_dict[band],
                            image.qcalmin_dict[band], image.qcalmax_dict[band])
                    elif image.type in ['Landsat8']:
                        refl_toa_array = et_numpy.l8_refl_toa_band_func(
                            dn_array, image.cos_theta_solar,
                            image.refl_mult_dict[band],
                            image.refl_add_dict[band])
                    # if (image.type in ['Landsat4', 'Landsat5', 'Landsat7'] and
                    #     not thermal_band_flag):
                    #     refl_toa_array = et_numpy.l457_refl_toa_band_func(
                    #         dn_array, image.cos_theta_solar,
                    #         image.dr, image.esun_dict[band],
                    #         image.lmin_dict[band], image.lmax_dict[band],
                    #         image.qcalmin_dict[band],
                    #         image.qcalmax_dict[band])
                    #         # image.rad_mult_dict[band],
                    #         # image.rad_add_dict[band])
                    # elif (image.type in ['Landsat8'] and
                    #       not thermal_band_flag):
                    #     refl_toa_array = et_numpy.l8_refl_toa_band_func(
                    #         dn_array, image.cos_theta_solar,
                    #         image.refl_mult_dict[band],
                    #         image.refl_add_dict[band])
                    # elif (image.type in ['Landsat4', 'Landsat5', 'Landsat7'] and
                    #       thermal_band_flag):
                    #     refl_toa_array = et_numpy.l457_ts_bt_band_func(
                    #         dn_array,
                    #         image.lmin_dict[band], image.lmax_dict[band],
                    #         image.qcalmin_dict[band],
                    #         image.qcalmax_dict[band],
                    #         # image.rad_mult_dict[band],
                    #         # image.rad_add_dict[band],
                    #         image.k1_dict[band], image.k2_dict[band])
                    # elif (image.type in ['Landsat8'] and
                    #       thermal_band_flag):
                    #     refl_toa_array = et_numpy.l8_ts_bt_band_func(
                    #         dn_array,
                    #         image.rad_mult_dict[band],
                    #         image.rad_add_dict[band],
                    #         image.k1_dict[band], image.k2_dict[band])

                    # refl_toa_array = et_numpy.refl_toa_band_func(
                    #     dn_array, cos_theta_solar_flt,
                    #     image.dr, image.esun_dict[band],
                    #     image.lmin_dict[band], image.lmax_dict[band],
                    #     image.qcalmin_dict[band], image.qcalmax_dict[band],
                    #     thermal_band_flag)
                    drigo.block_to_raster(
                        refl_toa_array.astype(np.float32),
                        image.refl_toa_raster,
                        b_i, b_j, band=image.band_toa_dict[band])
                    # drigo.array_to_comp_raster(
                    #    refl_toa_array.astype(np.float32),
                    #    image.refl_toa_raster,
                    #    image.band_toa_dict[band], common_array)
                    del refl_toa_array, dn_array
            if stats_flag:
                drigo.raster_statistics(image.refl_toa_raster)

        # # Process each TOA band
        # # for band, band_i in sorted(image.band_toa_dict.items()):
        # for band, dn_image in sorted(dn_image_dict.items()):
        #     thermal_band_flag = (band == image.thermal_band)
        #     #  Set 0 as nodata value
        #     drigo.raster_path_set_nodata(dn_image, 0)
        #     #  Calculate TOA reflectance
        #     dn_array, dn_nodata = drigo.raster_to_array(
        #         dn_image, 1, common_extent)
        #     dn_array = dn_array.astype(np.float64)
        #     # dn_array = dn_array.astype(np.float32)
        #     dn_array[dn_array == 0] = np.nan
        #     #
        #     if (image.type in ['Landsat4', 'Landsat5', 'Landsat7'] and
        #         not thermal_band_flag):
        #         refl_toa_array = et_numpy.l457_refl_toa_band_func(
        #             dn_array, image.cos_theta_solar,
        #             image.dr, image.esun_dict[band],
        #             image.lmin_dict[band], image.lmax_dict[band],
        #             image.qcalmin_dict[band], image.qcalmax_dict[band])
        #             # image.rad_mult_dict[band], image.rad_add_dict[band])
        #     elif (image.type in ['Landsat4', 'Landsat5', 'Landsat7'] and
        #           thermal_band_flag):
        #         refl_toa_array = et_numpy.l457_ts_bt_band_func(
        #             dn_array, image.lmin_dict[band], image.lmax_dict[band],
        #             image.qcalmin_dict[band], image.qcalmax_dict[band],
        #             # image.rad_mult_dict[band], image.rad_add_dict[band],
        #             image.k1_dict[band], image.k2_dict[band])
        #     elif (image.type in ['Landsat8'] and
        #           not thermal_band_flag):
        #         refl_toa_array = et_numpy.l8_refl_toa_band_func(
        #             dn_array, image.cos_theta_solar,
        #             image.refl_mult_dict[band], image.refl_add_dict[band])
        #     elif (image.type in ['Landsat8'] and
        #           thermal_band_flag):
        #         refl_toa_array = et_numpy.l8_ts_bt_band_func(
        #             dn_array,
        #             image.rad_mult_dict[band], image.rad_add_dict[band],
        #             image.k1_dict[band], image.k2_dict[band])
        #     # refl_toa_array = et_numpy.refl_toa_band_func(
        #     #     dn_array, cos_theta_solar_flt,
        #     #     image.dr, image.esun_dict[band],
        #     #     image.lmin_dict[band], image.lmax_dict[band],
        #     #     image.qcalmin_dict[band], image.qcalmax_dict[band],
        #     #     thermal_band_flag)
        #     drigo.array_to_comp_raster(
        #         refl_toa_array.astype(np.float32), image.refl_toa_raster,
        #         image.band_toa_dict[band], common_array)
        #     del refl_toa_array, dn_array


    # Calculate brightness temperature
    if calc_ts_bt_flag:
        logging.info('Brightness Temperature')
        if os.path.isfile(image.ts_bt_raster) and overwrite_flag:
            logging.debug('  Overwriting: {}'.format(image.ts_bt_raster))
            python_common.remove_file(image.ts_bt_raster)
        if not os.path.isfile(image.ts_bt_raster):
            band = image.thermal_band
            thermal_dn_path = dn_image_dict[band]
            drigo.raster_path_set_nodata(thermal_dn_path, 0)
            thermal_dn_array, thermal_dn_nodata = drigo.raster_to_array(
                thermal_dn_path, 1, common_extent, return_nodata=True)
            thermal_dn_mask = thermal_dn_array != thermal_dn_nodata
            if image.type in ['Landsat4', 'Landsat5', 'Landsat7']:
                ts_bt_array = et_numpy.l457_ts_bt_band_func(
                    thermal_dn_array,
                    image.lmin_dict[band], image.lmax_dict[band],
                    image.qcalmin_dict[band], image.qcalmax_dict[band],
                    # image.rad_mult_dict[band], image.rad_add_dict[band],
                    image.k1_dict[band], image.k2_dict[band])
            elif image.type in ['Landsat8']:
                ts_bt_array = et_numpy.l8_ts_bt_band_func(
                    thermal_dn_array,
                    image.rad_mult_dict[band], image.rad_add_dict[band],
                    image.k1_dict[band], image.k2_dict[band])
            # thermal_rad_array = et_numpy.refl_toa_band_func(
            #     thermal_dn_array, image.cos_theta_solar,
            #     image.dr, image.esun_dict[band],
            #     image.lmin_dict[band], image.lmax_dict[band],
            #     image.qcalmin_dict[band], image.qcalmax_dict[band],
            #     thermal_band_flag=True)
            # ts_bt_array = et_numpy.ts_bt_func(
            #     thermal_rad_array, image.k1_dict[image.thermal_band],
            #     image.k2_dict[image.thermal_band])
            ts_bt_array[~thermal_dn_mask] = np.nan
            drigo.array_to_raster(
                ts_bt_array, image.ts_bt_raster,
                output_geo=common_geo, output_proj=env.snap_proj,
                # mask_array=common_array,
                stats_flag=stats_flag)
            # del thermal_dn_array, thermal_rad_array
            del thermal_dn_path, thermal_dn_array, ts_bt_array


    # Interpolate/project/clip METRIC hourly/daily rasters
    if (calc_metric_ea_flag or
            calc_metric_wind_flag or
            calc_metric_etr_flag):
        logging.info('METRIC hourly/daily rasters')

        # Get bracketing hours from image acquisition time
        image_prev_dt = image.acq_datetime.replace(
            minute=0, second=0, microsecond=0)
        image_next_dt = image_prev_dt + timedelta(seconds=3600)

        # Get NLDAS properties from one of the images
        input_ws = os.path.join(
            metric_etr_input_ws, str(image_prev_dt.year))
        try:
            input_path = [
                os.path.join(input_ws, file_name)
                for file_name in os.listdir(input_ws)
                for match in [metric_hourly_re.match(file_name)]
                if (match and
                    (image_prev_dt.strftime('%Y%m%d') ==
                     match.group('YYYYMMDD')))][0]
        except IndexError:
            logging.error('  No hourly file for {}'.format(
                image_prev_dt.strftime('%Y-%m-%d %H00')))
            return False
        try:
            input_ds = gdal.Open(input_path)
            input_osr = drigo.raster_ds_osr(input_ds)
            # input_proj = drigo.osr_proj(input_osr)
            input_extent = drigo.raster_ds_extent(input_ds)
            input_cs = drigo.raster_ds_cellsize(input_ds, x_only=True)
            # input_geo = input_extent.geo(input_cs)
            input_x, input_y = input_extent.origin()
            input_ds = None
        except:
            logging.error('  Could not get default input image properties')
            logging.error('    {}'.format(input_path))
            return False

        # Project Landsat scene extent to NLDAS GCS
        common_gcs_osr = common_osr.CloneGeogCS()
        common_gcs_extent = drigo.project_extent(
            common_extent, common_osr, common_gcs_osr,
            cellsize=env.cellsize)
        common_gcs_extent.buffer_extent(0.1)
        common_gcs_extent.adjust_to_snap(
            'EXPAND', input_x, input_y, input_cs)
        # common_gcs_geo = common_gcs_extent.geo(input_cs)

        def metric_weather_func(output_raster, input_ws, input_re,
                                prev_dt, next_dt,
                                resample_method=gdal.GRA_NearestNeighbour,
                                rounding_flag=False):
            """Interpolate/project/clip METRIC hourly rasters"""
            logging.debug('    Output: {}'.format(output_raster))
            if os.path.isfile(output_raster):
                if overwrite_flag:
                    logging.debug('    Overwriting output')
                    python_common.remove_file(output_raster)
                else:
                    logging.debug('    Skipping, file already exists ' +
                                  'and overwrite is False')
                    return False
            prev_ws = os.path.join(input_ws, str(prev_dt.year))
            next_ws = os.path.join(input_ws, str(next_dt.year))

            # Technically previous and next could come from different days
            # or even years, although this won't happen in the U.S.
            try:
                prev_path = [
                    os.path.join(prev_ws, input_name)
                    for input_name in os.listdir(prev_ws)
                    for input_match in [input_re.match(input_name)]
                    if (input_match and
                        (prev_dt.strftime('%Y%m%d') ==
                         input_match.group('YYYYMMDD')))][0]
                logging.debug('    Input prev: {}'.format(prev_path))
            except IndexError:
                logging.error('  No previous hourly file')
                logging.error('    {}'.format(prev_dt))
                return False
            try:
                next_path = [
                    os.path.join(next_ws, input_name)
                    for input_name in os.listdir(next_ws)
                    for input_match in [input_re.match(input_name)]
                    if (input_match and
                        (next_dt.strftime('%Y%m%d') ==
                         input_match.group('YYYYMMDD')))][0]
                logging.debug('    Input next: {}'.format(next_path))
            except IndexError:
                logging.error('  No next hourly file')
                logging.error('    {}'.format(next_dt))
                return False

            # Band numbers are 1's based
            prev_band = int(prev_dt.strftime('%H')) + 1
            next_band = int(next_dt.strftime('%H')) + 1
            logging.debug('    Input prev band: {}'.format(prev_band))
            logging.debug('    Input next band: {}'.format(next_band))

            # Read arrays
            prev_array = drigo.raster_to_array(
                prev_path, band=prev_band, mask_extent=common_gcs_extent,
                return_nodata=False)
            next_array = drigo.raster_to_array(
                next_path, band=next_band, mask_extent=common_gcs_extent,
                return_nodata=False)
            if not np.any(prev_array) or not np.any(next_array):
                logging.warning('\nWARNING: Input NLDAS array is all nodata\n')
                return None

            output_array = hourly_interpolate_func(
                prev_array, next_array,
                prev_dt, next_dt, image.acq_datetime)
            output_array = drigo.project_array(
                output_array, resample_method,
                input_osr, input_cs, common_gcs_extent,
                common_osr, env.cellsize, common_extent, output_nodata=None)

            # Apply common area mask
            output_array[~common_array] = np.nan

            # Reduce the file size by rounding to the nearest n digits
            if rounding_flag:
                output_array = np.around(output_array, rounding_digits)
            # Force output to 32-bit float
            drigo.array_to_raster(
                output_array.astype(np.float32), output_raster,
                output_geo=common_geo, output_proj=common_proj,
                stats_flag=stats_flag)
            del output_array
            return True

        # Ea - Project to Landsat scene after clipping
        if calc_metric_ea_flag:
            logging.info('  Hourly vapor pressure (Ea)')
            metric_weather_func(
                image.metric_ea_raster, metric_ea_input_ws,
                metric_hourly_re, image_prev_dt, image_next_dt,
                gdal.GRA_Bilinear, rounding_flag=True)

        # Wind - Project to Landsat scene after clipping
        if calc_metric_wind_flag:
            logging.info('  Hourly windspeed')
            metric_weather_func(
                image.metric_wind_raster, metric_wind_input_ws,
                metric_hourly_re, image_prev_dt, image_next_dt,
                gdal.GRA_NearestNeighbour, rounding_flag=False)

        # ETr - Project to Landsat scene after clipping
        if calc_metric_etr_flag:
            logging.info('  Hourly reference ET (ETr)')
            metric_weather_func(
                image.metric_etr_raster, metric_etr_input_ws,
                metric_hourly_re, image_prev_dt, image_next_dt,
                gdal.GRA_NearestNeighbour, rounding_flag=False)

        # ETr 24hr - Project to Landsat scene after clipping
        if calc_metric_etr_flag:
            logging.info('  Daily reference ET (ETr)')
            logging.debug('    Output: {}'.format(
                image.metric_etr_24hr_raster))
            if (os.path.isfile(image.metric_etr_24hr_raster) and
                    overwrite_flag):
                logging.debug('    Overwriting output')
                os.remove(image.metric_etr_24hr_raster)
            if not os.path.isfile(image.metric_etr_24hr_raster):
                etr_prev_ws = os.path.join(
                    metric_etr_input_ws, str(image_prev_dt.year))
                try:
                    input_path = [
                        os.path.join(etr_prev_ws, file_name)
                        for file_name in os.listdir(etr_prev_ws)
                        for match in [metric_daily_re.match(file_name)]
                        if (match and
                            (image_prev_dt.strftime('%Y%m%d') ==
                             match.group('YYYYMMDD')))][0]
                    logging.debug('    Input: {}'.format(input_path))
                except IndexError:
                    logging.error('  No daily file for {}'.format(
                        image_prev_dt.strftime('%Y-%m-%d')))
                    return False
                output_array = drigo.raster_to_array(
                    input_path, mask_extent=common_gcs_extent,
                    return_nodata=False)
                output_array = drigo.project_array(
                    output_array, gdal.GRA_NearestNeighbour,
                    input_osr, input_cs, common_gcs_extent,
                    common_osr, env.cellsize, common_extent,
                    output_nodata=None)
                # Apply common area mask
                output_array[~common_array] = np.nan
                # Reduce the file size by rounding to the nearest n digits
                # output_array = np.around(output_array, rounding_digits)
                drigo.array_to_raster(
                    output_array, image.metric_etr_24hr_raster,
                    output_geo=common_geo, output_proj=common_proj,
                    stats_flag=stats_flag)
                del output_array
                del input_path

        # Tair - Project to Landsat scene after clipping
        if calc_metric_tair_flag:
            logging.info('  Hourly air temperature (Tair)')
            metric_weather_func(
                image.metric_tair_raster, metric_tair_input_ws,
                metric_hourly_re, image_prev_dt, image_next_dt,
                gdal.GRA_NearestNeighbour, rounding_flag=False)

        # Cleanup
        del image_prev_dt, image_next_dt

    # Soil Water Balance
    if calc_swb_ke_flag:
        logging.info('Daily soil water balance')

        # Check if output file already exists
        logging.debug('  Ke:  {}'.format(image.ke_raster))
        if os.path.isfile(image.ke_raster):
            if overwrite_flag:
                logging.debug('    Overwriting output')
                python_common.remove_file(image.ke_raster)
            else:
                logging.debug('    Skipping, file already ' +
                              'exists and overwrite is False')
                return False
        ke_array = et_common.raster_swb_func(
            image.acq_datetime, common_osr, env.cellsize, common_extent,
            awc_input_path, etr_input_ws, etr_input_re,
            ppt_input_ws, ppt_input_re,
            spinup_days=spinup_days, min_spinup_days=min_spinup_days)
        # Apply common area mask
        ke_array[~common_array] = np.nan
        # Reduce the file size by rounding to the nearest 2 digits
        np.around(ke_array, 2, out=ke_array)

        # Force output to 32-bit float
        drigo.array_to_raster(
            ke_array.astype(np.float32), image.ke_raster,
            output_geo=common_geo, output_proj=common_proj,
            stats_flag=stats_flag)

    # Remove Landsat TOA rasters
    if not keep_dn_flag:
        for landsat_item in python_common.build_file_list(
                image.orig_data_ws, image.image_re):
            os.remove(os.path.join(image.orig_data_ws, landsat_item))
    return True


def common_area_func(image_list):
    """"""
    common_nodata = 0
    # Assume all Landsat images have same extent
    common_shape = drigo.raster_path_shape(image_list[0])
    common_array = np.ones(common_shape, dtype=np.bool)
    for i, image in enumerate(image_list):
        image_array, image_nodata = drigo.raster_to_array(image)
        common_array &= (image_array != common_nodata)
        del image_array, image_nodata
    return common_array


def smooth_func(mask_array, erode=8, dilate=8):
    """Remove edge pixels (mostly for Landsat 5 images)

    Parameters
    ----------
    mask_array : ndarray
    erode : int
    dilate : int

    Returns
    -------
    ndarray

    """
    # 3x3
    struct = ndimage.generate_binary_structure(2, 2).astype(np.uint8)

    # cross
    # struct = ndimage.generate_binary_structure(2, 1).astype(np.uint8)

    return ndimage.binary_dilation(
        ndimage.binary_erosion(mask_array, struct, erode), struct, dilate)

    # if shrink_flag and landsat_type in ['Landsat4', 'Landsat5']:
    #    return ndimage.binary_dilation(
    #        ndimage.binary_erosion(mask_array, struct, 10), struct, 8)
    # elif shrink_flag and landsat_type in ['Landsat7']:
    #    return ndimage.binary_dilation(
    #        ndimage.binary_erosion(mask_array, struct, 6), struct, 6)
    # else:
    #    return ndimage.binary_dilation(
    #        ndimage.binary_erosion(mask_array, struct, 1), struct, 1)


def hourly_interpolate_func(prev_array, next_array,
                            image_prev_dt, image_next_dt, image_dt,
                            output_extent=None):
    """Interpolate between two NLDAS hourly rasters to a new time

    Parameters
    ----------
    prev_array : ndarray
        Array at the previous timestep.
    next_array : ndarray
        Array at the next timestep.
    image_prev_dt : datetime
        Datetime of the previous array.
    image_next_dt : datetime
        Datetime of the next array.
    image_dt : datetime
        Datetime to interpolate to.
    output_extent: gdal_common.extent
        Extent to clip output.

    Returns
    -------
    ndarray

    """
    output_array = (
        (next_array - prev_array) *
        ((image_dt - image_prev_dt).total_seconds() /
         (image_next_dt - image_prev_dt).total_seconds()) +
        prev_array)
    return output_array


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Landsat scene prep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Landsat project input file', metavar='FILE')
    # parser.add_argument(
    #    '-i', '--metric_ini', required=True,
    #    help='METRIC input file', metavar='FILE')
    parser.add_argument(
        '-bs', '--blocksize', default=2048, type=int,
        help='Processing block size')
    parser.add_argument(
        '--smooth', default=False, action="store_true",
        help='Dilate and erode image to remove fringe/edge pixels')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '--debug', help='Debug level logging',
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO)
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', args.workspace))

    # Delay
    sleep(random.uniform(0, max([0, args.delay])))

    main(image_ws=args.workspace, ini_path=args.ini, bs=args.blocksize,
         smooth_flag=args.smooth, stats_flag=args.stats,
         overwrite_flag=args.overwrite)
