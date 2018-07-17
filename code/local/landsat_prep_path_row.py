#!/usr/bin/env python
#--------------------------------
# Name:         landsat_prep_path_row.py
# Purpose:      Prepare Landsat path/row data
#--------------------------------

import argparse
# from builtins import input
from datetime import datetime
import itertools
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys

import drigo
import numpy as np
from osgeo import gdal, ogr, osr

import et_common
import et_image
import python_common

gdal.UseExceptions()


def main(ini_path, tile_list=None, overwrite_flag=False, mp_procs=1):
    """Prep Landsat path/row specific data

    Parameters
    ----------
    ini_path : str
        File path of the input parameters file.
    tile_list : list, optional
        Landsat path/rows to process (i.e. [p045r043, p045r033]).
        This will override the tile list in the INI file.
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
    mp_procs : int, optional
        Number of cores to use (the default is 1).

    Returns
    -------
    None

    """
    logging.info('\nPrepare path/row data')

    # Check the GDAL_DATA environment variable
    # TODO: Move this to drigo
    if "GDAL_DATA" not in os.environ:
        raise Exception('The GDAL_DATA environment variable is not set\n')
    elif not os.path.isdir(os.getenv("GDAL_DATA")):
        raise Exception(
            'The GDAL_DATA environment folder does not exist:\n'
            '  {}'.format(os.getenv("GDAL_DATA")))

    # Open config file
    config = python_common.open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    year = config.getint('INPUTS', 'year')
    if tile_list is None:
        tile_list = python_common.read_param('tile_list', [], config, 'INPUTS')
    project_ws = config.get('INPUTS', 'project_folder')
    logging.debug('  Year: {}'.format(year))
    logging.debug('  Path/rows: {}'.format(', '.join(tile_list)))
    logging.debug('  Project: {}'.format(project_ws))

    # study_area_path = config.get('INPUTS', 'study_area_path')
    footprint_path = config.get('INPUTS', 'footprint_path')
    # For now, assume the UTM zone file is colocated with the footprints shapefile
    utm_path = python_common.read_param(
        'utm_path',
        os.path.join(os.path.dirname(footprint_path), 'wrs2_tile_utm_zones.json'),
        config, 'INPUTS')
    keep_list_path = python_common.read_param(
        'keep_list_path', '', config, 'INPUTS')
    # DEADBEEF - Remove if keep list works
    # skip_list_path = python_common.read_param(
    #     'skip_list_path', '', config, 'INPUTS')

    landsat_flag = python_common.read_param(
        'landsat_flag', True, config, 'INPUTS')
    ledaps_flag = False
    dem_flag = python_common.read_param(
        'dem_flag', True, config, 'INPUTS')
    nlcd_flag = python_common.read_param(
        'nlcd_flag', True, config, 'INPUTS')
    cdl_flag = python_common.read_param(
        'cdl_flag', False, config, 'INPUTS')
    landfire_flag = python_common.read_param(
        'landfire_flag', False, config, 'INPUTS')
    field_flag = python_common.read_param(
        'field_flag', False, config, 'INPUTS')

    tile_gcs_buffer = python_common.read_param('tile_buffer', 0.25, config)

    # Input/output folder and file paths
    if landsat_flag:
        landsat_input_ws = config.get('INPUTS', 'landsat_input_folder')
    else:
        landsat_input_ws = None
    # if ledaps_flag:
    #     ledaps_input_ws = config.get('INPUTS', 'ledaps_input_folder')
    # else:
    #     ledaps_input_ws = None

    if dem_flag:
        dem_input_ws = config.get('INPUTS', 'dem_input_folder')
        dem_tile_fmt = config.get('INPUTS', 'dem_tile_fmt')
        dem_output_ws = config.get('INPUTS', 'dem_output_folder')
        dem_output_name = python_common.read_param(
            'dem_output_name', 'dem.img', config)
        # dem_output_name = config.get('INPUTS', 'dem_output_name')
    else:
        dem_input_ws, dem_tile_fmt = None, None
        dem_output_ws, dem_output_name = None, None

    if nlcd_flag:
        nlcd_input_path = config.get('INPUTS', 'nlcd_input_path')
        nlcd_output_ws = config.get('INPUTS', 'nlcd_output_folder')
        nlcd_output_fmt = python_common.read_param(
            'nlcd_output_fmt', 'nlcd_{:04d}.img', config)
    else:
        nlcd_input_path, nlcd_output_ws, nlcd_output_fmt = None, None, None

    if cdl_flag:
        cdl_input_path = config.get('INPUTS', 'cdl_input_path')
        cdl_ag_list = config.get('INPUTS', 'cdl_ag_list')
        cdl_ag_list = list(python_common.parse_int_set(cdl_ag_list))
        # default_cdl_ag_list = range(1,62) + range(66,78) + range(204,255)
        # cdl_ag_list = python_common.read_param(
        #    'cdl_ag_list', default_cdl_ag_list, config)
        # cdl_ag_list = list(map(int, cdl_ag_list))
        # cdl_non_ag_list = python_common.read_param(
        #    'cdl_non_ag_list', [], config)
        cdl_output_ws = config.get('INPUTS', 'cdl_output_folder')
        cdl_output_fmt = python_common.read_param(
            'cdl_output_fmt', 'cdl_{:04d}.img', config)
        cdl_ag_output_fmt = python_common.read_param(
            'cdl_ag_output_fmt', 'cdl_ag_{:04d}.img', config)
    else:
        cdl_input_path, cdl_ag_list = None, None
        cdl_output_ws, cdl_output_fmt, cdl_ag_output_fmt = None, None, None

    if landfire_flag:
        landfire_input_path = config.get('INPUTS', 'landfire_input_path')
        landfire_ag_list = config.get('INPUTS', 'landfire_ag_list')
        landfire_ag_list = list(
            python_common.parse_int_set(landfire_ag_list))
        # default_landfire_ag_list = range(3960,4000)
        # landfire_ag_list = python_common.read_param(
        #    'landfire_ag_list', default_landfire_ag_list, config)
        # landfire_ag_list = list(map(int, landfire_ag_list))
        landfire_output_ws = config.get('INPUTS', 'landfire_output_folder')
        landfire_output_fmt = python_common.read_param(
            'landfire_output_fmt', 'landfire_{:04d}.img', config)
        landfire_ag_output_fmt = python_common.read_param(
            'landfire_ag_output_fmt', 'landfire_ag_{:04d}.img', config)
    else:
        landfire_input_path, landfire_ag_list = None, None
        landfire_output_ws = None
        landfire_output_fmt, landfire_ag_output_fmt = None, None

    if field_flag:
        field_input_path = config.get('INPUTS', 'field_input_path')
        field_output_ws = config.get('INPUTS', 'field_output_folder')
        field_output_fmt = python_common.read_param(
            'field_output_fmt', 'fields_{:04d}.img', config)
    else:
        field_input_path = None
        field_output_ws, field_output_fmt = None, None

    # File/folder names
    orig_data_folder_name = 'ORIGINAL_DATA'

    # Check inputs folders/paths
    logging.info('\nChecking input folders/files')
    file_check(footprint_path)
    file_check(utm_path)
    if landsat_flag:
        folder_check(landsat_input_ws)
    # if ledaps_flag:
    #     folder_check(ledaps_input_ws)
    if dem_flag:
        folder_check(dem_input_ws)
    if nlcd_flag:
        file_check(nlcd_input_path)
    if cdl_flag:
        file_check(cdl_input_path)
    if landfire_flag:
        # Landfire will likely be an ESRI grid (set as a folder)
        if not (os.path.isdir(landfire_input_path) or
                os.path.isfile(landfire_input_path)):
            logging.error('\n  {} does not exist'.format(landfire_input_path))
    if field_flag:
        file_check(field_input_path)
    if keep_list_path:
        file_check(keep_list_path)
    # DEADBEEF - Remove if keep list works
    # if skip_list_path:
    #     file_check(skip_list_path)

    # Build output folders
    if not os.path.isdir(project_ws):
        os.makedirs(project_ws)
    if dem_flag and not os.path.isdir(dem_output_ws):
        os.makedirs(dem_output_ws)
    if nlcd_flag and not os.path.isdir(nlcd_output_ws):
        os.makedirs(nlcd_output_ws)
    if cdl_flag and not os.path.isdir(cdl_output_ws):
        os.makedirs(cdl_output_ws)
    if landfire_flag and not os.path.isdir(landfire_output_ws):
        os.makedirs(landfire_output_ws)
    if field_flag and not os.path.isdir(field_output_ws):
        os.makedirs(field_output_ws)

    # For now assume path/row are two digit numbers
    tile_fmt = 'p{:03d}r{:03d}'
    tile_re = re.compile('p(\d{3})r(\d{3})')
    image_id_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(?:\w{4})_(\d{3})(\d{3})_'
        '(\d{4})(\d{2})(\d{2})_(?:\d{8})_(?:\d{2})_(?:\w{2})$')
    snap_cs = 30
    snap_xmin, snap_ymin = (15, 15)

    # Set snap environment parameters
    env = drigo.env
    env.cellsize = snap_cs
    env.snap_xmin, env.snap_ymin = snap_xmin, snap_ymin

    # Use WGSS84 (EPSG 4326) for GCS spatial reference
    # Could also use NAD83 (EPSG 4269)
    # gcs_epsg = 4326
    # gcs_osr = epsg_osr(4326)
    # gcs_proj = osr_proj(gcs_osr)

    # Landsat Footprints (WRS2 Descending Polygons)
    logging.debug('\nFootprint (WRS2 descending should be GCS84):')
    tile_gcs_osr = drigo.feature_path_osr(footprint_path)
    logging.debug('  OSR: {}'.format(tile_gcs_osr))

    # Doublecheck that WRS2 descending shapefile is GCS84
    # if tile_gcs_osr != epsg_osr(4326):
    #     logging.error('  WRS2 is not GCS84')
    #     sys.exit()

    # Get geometry for each path/row
    tile_gcs_wkt_dict = path_row_wkt_func(
        footprint_path, path_field='PATH', row_field='ROW')

    # Get UTM zone for each path/row
    # DEADBEEF - Using "eval" is considered unsafe and should be changed
    tile_utm_zone_dict = eval(open(utm_path, 'r').read())

    # Project study area geometry to GCS coordinates
    # logging.debug('\nStudy area')
    # study_area_geom = feature_path_geom_union(study_area_path)
    # study_area_gcs_geom = study_area_geom.Clone()
    # study_area_gcs_geom.TransformTo(tile_gcs_osr)

    # Get list of all intersecting Landsat path/rows
    # logging.info('\nLandsat path/rows')
    # tile_list = []
    # for tile_name, tile_gcs_wkt in tile_gcs_wkt_dict.items():
    #     tile_gcs_geom = ogr.CreateGeometryFromWkt(tile_gcs_wkt)
    #     if tile_gcs_geom.Intersects(study_area_gcs_geom):
    #         tile_list.append(tile_name)
    # for tile_name in sorted(tile_list):
    #     logging.debug('  {}'.format(tile_name))

    # Check that each path/row extent and UTM zone exist
    logging.info('\nChecking path/row list against footprint shapefile')
    for tile_name in sorted(tile_list):
        if tile_name not in tile_gcs_wkt_dict.keys():
            logging.error(
                '  {} feature not in footprint shapefile'.format(tile_name))
            continue
        elif tile_name not in tile_utm_zone_dict.keys():
            logging.error(
                '  {} UTM zone not in footprint shapefile'.format(tile_name))
            continue
        elif tile_utm_zone_dict[tile_name] == 0:
            logging.error(
                ('  UTM zone is not set for {} in ' +
                 'footprint shapefile').format(tile_name))
            continue

    # Build output folders for each path/row
    logging.info('\nBuilding path/row folders')
    for tile_name in tile_list:
        logging.debug('  {} {}'.format(year, tile_name))
        tile_output_ws = os.path.join(project_ws, str(year), tile_name)
        if ((landsat_flag or ledaps_flag) and
                not os.path.isdir(tile_output_ws)):
            os.makedirs(tile_output_ws)
        if (dem_flag and
                not os.path.isdir(os.path.join(dem_output_ws, tile_name))):
            os.makedirs(os.path.join(dem_output_ws, tile_name))
        if (nlcd_flag and
                not os.path.isdir(os.path.join(nlcd_output_ws, tile_name))):
            os.makedirs(os.path.join(nlcd_output_ws, tile_name))
        if (cdl_flag and
                not os.path.isdir(os.path.join(cdl_output_ws, tile_name))):
            os.makedirs(os.path.join(cdl_output_ws, tile_name))
        if (landfire_flag and
                not os.path.isdir(os.path.join(landfire_output_ws, tile_name))):
            os.makedirs(os.path.join(landfire_output_ws, tile_name))
        if (field_flag and
                not os.path.isdir(os.path.join(field_output_ws, tile_name))):
            os.makedirs(os.path.join(field_output_ws, tile_name))

    # Read keep list
    if keep_list_path:
        logging.debug('\nReading scene keep list')
        with open(keep_list_path) as keep_list_f:
            keep_list = keep_list_f.readlines()
            keep_list = [image_id.strip() for image_id in keep_list
                         if image_id_re.match(image_id.strip())]
    else:
        logging.debug('\nScene keep list not set in INI')
        keep_list = []

    # DEADBEEF - Remove if keep list works
    # # Read skip list
    # if (landsat_flag or ledaps_flag) and skip_list_path:
    #     logging.debug('\nReading scene skiplist')
    #     with open(skip_list_path) as skip_list_f:
    #         skip_list = skip_list_f.readlines()
    #         skip_list = [scene.strip() for scene in skip_list
    #                      if image_id_re.match(scene.strip())]
    # else:
    #     logging.debug('\nScene skip list not set in INI')
    #     skip_list = []

    # Copy and unzip raw Landsat scenes
    # Use these for thermal band, MTL file (scene time), and to run FMask
    if landsat_flag:
        logging.info('\nExtract raw Landsat scenes')
        # Process each path/row
        extract_targz_list = []
        for tile_name in tile_list:
            tile_output_ws = os.path.join(project_ws, str(year), tile_name)

            # path/row as strings with leading zeros
            path, row = map(str, tile_re.match(tile_name).groups())
            tile_input_ws = os.path.join(
                landsat_input_ws, path, row, str(year))
            if not os.path.isdir(tile_input_ws):
                continue
            logging.info('  {} {}'.format(year, tile_name))

            # Process each tar.gz file
            for input_name in sorted(os.listdir(tile_input_ws)):
                if (not image_id_re.match(input_name) and
                        not input_name.endswith('.tar.gz')):
                    continue

                # Get Landsat product ID from tar.gz file name
                image_id = input_name.split('.')[0]

                # Output workspace
                image_output_ws = os.path.join(tile_output_ws, image_id)
                orig_data_ws = os.path.join(
                    image_output_ws, orig_data_folder_name)

                if keep_list and image_id not in keep_list:
                    logging.debug('    {} - Skipping scene'.format(image_id))
                    # DEADBEEF - Should the script always remove the scene
                    #   if it is in the skip list?
                    # Maybe only if overwrite is set?
                    if os.path.isdir(image_output_ws):
                        # input('Press ENTER to delete {}'.format(image_id))
                        shutil.rmtree(image_output_ws)
                    continue

                # DEADBEEF - Remove if keep list works
                # if skip_list and image_id in skip_list:
                #     logging.debug('    {} - Skipping scene'.format(image_id))
                #     # DEADBEEF - Should the script always remove the scene
                #     #   if it is in the skip list?
                #     # Maybe only if overwrite is set?
                #     if os.path.isdir(image_output_ws):
                #         # input('Press ENTER to delete {}'.format(image_id))
                #         shutil.rmtree(image_output_ws)
                #     continue

                # If orig_data_ws doesn't exist, don't check images
                if not os.path.isdir(orig_data_ws):
                    os.makedirs(orig_data_ws)
                elif (not overwrite_flag and
                        landsat_files_check(image_output_ws)):
                    continue

                # Extract Landsat tar.gz file
                input_path = os.path.join(tile_input_ws, input_name)
                if mp_procs > 1:
                    extract_targz_list.append([input_path, orig_data_ws])
                else:
                    python_common.extract_targz_func(
                        input_path, orig_data_ws)

                # # Use a command line call
                # input_path = os.path.join(tile_input_ws, input_name)
                # if job_i % pbs_jobs != 0:
                #     job_list.append('tar -zxvf {} -C {} &\n'.format(
                #         input_path, orig_data_ws))
                # else:
                #     job_list.append('tar -zxvf {} -C {}\n'.format(
                #         input_path, orig_data_ws))
                #     # job_list.append('tar -zxvf {} -C {} &\n'.format(
                #     #     input_path, orig_data_ws))
                #     # job_list.append('wait\n')
                # job_i += 1

        # Extract Landsat tar.gz files using multiprocessing
        if extract_targz_list:
            pool = mp.Pool(mp_procs)
            results = pool.map(
                python_common.extract_targz_mp, extract_targz_list,
                chunksize=1)
            pool.close()
            pool.join()
            del results, pool

    # Get projected extent for each path/row
    # This should probably be in a function
    if (dem_flag or
            nlcd_flag or
            cdl_flag or
            landfire_flag or
            field_flag):
        tile_utm_extent_dict = gcs_to_utm_dict(
            tile_list, tile_utm_zone_dict, tile_gcs_osr,
            tile_gcs_wkt_dict, tile_gcs_buffer, snap_xmin, snap_ymin, snap_cs)

    # Mosaic DEM tiles for each path/row
    if dem_flag:
        logging.info('\nBuild DEM for each path/row')
        mosaic_mp_list = []
        for tile_name in tile_list:
            # Output folder and path
            tile_output_path = os.path.join(
                dem_output_ws, tile_name, dem_output_name)
            if not overwrite_flag and os.path.isfile(tile_output_path):
                logging.debug('    {} already exists, skipping'.format(
                    os.path.basename(tile_output_path)))
                continue
            logging.info('  {}'.format(tile_name))

            # Get the path/row geometry in GCS for selecting intersecting tiles
            tile_gcs_geom = ogr.CreateGeometryFromWkt(
                tile_gcs_wkt_dict[tile_name])
            # Apply a small buffer (in degrees) to the extent
            # DEADBEEF - Buffer fails if GDAL is not built with GEOS support
            # tile_gcs_geom = tile_gcs_geom.Buffer(tile_gcs_buffer)
            tile_gcs_extent = drigo.Extent(tile_gcs_geom.GetEnvelope())
            tile_gcs_extent = tile_gcs_extent.ogrenv_swap()
            tile_gcs_extent.buffer_extent(tile_gcs_buffer)
            # tile_gcs_extent.ymin, tile_gcs_extent.xmax = tile_gcs_extent.xmax, tile_gcs_extent.ymin

            # Offsets are needed since tile name is upper left corner of tile
            # Tile n36w120 spans -120 <-> -119 and 35 <-> 36
            lon_list = range(
                int(tile_gcs_extent.xmin) - 1, int(tile_gcs_extent.xmax))
            lat_list = range(
                int(tile_gcs_extent.ymin) + 1, int(tile_gcs_extent.ymax) + 2)

            # Get list of DEM tile rasters
            dem_tile_list = []
            for lat, lon in itertools.product(lat_list, lon_list):
                # Convert sign of lat/lon to letter
                lat = (
                    'n' + '{:02d}'.format(abs(lat)) if
                    lat >= 0 else 's' + '{:02d}'.format(abs(lat)))
                lon = (
                    'w' + '{:03d}'.format(abs(lon)) if
                    lon < 0 else 'e' + '{:03d}'.format(abs(lon)))
                dem_tile_path = os.path.join(
                    dem_input_ws, dem_tile_fmt.format(lat, lon))
                if os.path.isfile(dem_tile_path):
                    dem_tile_list.append(dem_tile_path)
            if not dem_tile_list:
                logging.warning('    WARNING: No DEM tiles were selected')
                continue

            # Mosaic tiles using mosaic function
            tile_utm_osr = drigo.epsg_osr(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_proj = drigo.epsg_proj(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_extent = tile_utm_extent_dict[tile_name]
            tile_utm_ullr = tile_utm_extent.ul_lr_swap()

            # Mosaic, clip, project using custom function
            if mp_procs > 1:
                mosaic_mp_list.append([
                    dem_tile_list, tile_output_path,
                    tile_utm_proj, snap_cs, tile_utm_extent])
            else:
                drigo.mosaic_tiles(dem_tile_list, tile_output_path,
                                 tile_utm_osr, snap_cs, tile_utm_extent)

            # Cleanup
            del tile_output_path
            del tile_gcs_geom, tile_gcs_extent, tile_utm_extent
            del tile_utm_osr, tile_utm_proj
            del lon_list, lat_list, dem_tile_list
        # Mosaic DEM rasters using multiprocessing
        if mosaic_mp_list:
            pool = mp.Pool(mp_procs)
            results = pool.map(mosaic_tiles_mp, mosaic_mp_list, chunksize=1)
            pool.close()
            pool.join()
            del results, pool

    # Project/clip NLCD for each path/row
    if nlcd_flag:
        logging.info('\nBuild NLCD for each path/row')
        project_mp_list = []
        for tile_name in tile_list:
            nlcd_output_path = os.path.join(
                nlcd_output_ws, tile_name, nlcd_output_fmt.format(year))
            if not overwrite_flag and os.path.isfile(nlcd_output_path):
                logging.debug('    {} already exists, skipping'.format(
                    os.path.basename(nlcd_output_path)))
                continue
            logging.info('  {}'.format(tile_name))

            # Set the nodata value on the NLCD raster if it is not set
            nlcd_ds = gdal.Open(nlcd_input_path, 0)
            nlcd_band = nlcd_ds.GetRasterBand(1)
            nlcd_nodata = nlcd_band.GetNoDataValue()
            nlcd_ds = None
            if nlcd_nodata is None:
                nlcd_nodata = 255

            # Clip and project
            tile_utm_osr = drigo.epsg_osr(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_proj = drigo.epsg_proj(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_extent = tile_utm_extent_dict[tile_name]
            tile_utm_ullr = tile_utm_extent.ul_lr_swap()

            if mp_procs > 1:
                project_mp_list.append([
                    nlcd_input_path, nlcd_output_path, gdal.GRA_NearestNeighbour,
                    tile_utm_proj, snap_cs, tile_utm_extent, nlcd_nodata])
            else:
                drigo.project_raster(
                    nlcd_input_path, nlcd_output_path, gdal.GRA_NearestNeighbour,
                    tile_utm_osr, snap_cs, tile_utm_extent, nlcd_nodata)

            # Cleanup
            del nlcd_output_path
            del nlcd_ds, nlcd_band, nlcd_nodata
            del tile_utm_osr, tile_utm_proj, tile_utm_extent
        # Project NLCD rasters using multiprocessing
        if project_mp_list:
            pool = mp.Pool(mp_procs)
            results = pool.map(
                drigo.project_raster_mp, project_mp_list, chunksize=1)
            pool.close()
            pool.join()
            del results, pool

    # Project/clip CDL for each path/row
    if cdl_flag:
        logging.info('\nBuild CDL for each path/row')
        project_mp_list, remap_mp_list = [], []
        for tile_name in tile_list:
            cdl_output_path = os.path.join(
                cdl_output_ws, tile_name, cdl_output_fmt.format(year))
            cdl_ag_output_path = os.path.join(
                cdl_output_ws, tile_name, cdl_ag_output_fmt.format(year))
            if not os.path.isfile(cdl_input_path):
                logging.error('\n\n  {} does not exist'.format(
                    cdl_input_path))
                sys.exit()
            if not overwrite_flag and os.path.isfile(cdl_output_path):
                logging.debug('    {} already exists, skipping'.format(
                    os.path.basename(cdl_output_path)))
                continue
            logging.info('  {}'.format(tile_name))

            # Set the nodata value on the CDL raster if it is not set
            cdl_ds = gdal.Open(cdl_input_path, 0)
            cdl_band = cdl_ds.GetRasterBand(1)
            cdl_nodata = cdl_band.GetNoDataValue()
            cdl_ds = None
            if cdl_nodata is None:
                cdl_nodata = 255

            # Clip and project
            tile_utm_osr = drigo.epsg_osr(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_proj = drigo.epsg_proj(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_extent = tile_utm_extent_dict[tile_name]
            if mp_procs > 1:
                project_mp_list.append([
                    cdl_input_path, cdl_output_path, gdal.GRA_NearestNeighbour,
                    tile_utm_proj, snap_cs, tile_utm_extent, cdl_nodata])
                remap_mp_list.append([
                    cdl_output_path, cdl_ag_output_path, cdl_ag_list])
            else:
                drigo.project_raster(
                    cdl_input_path, cdl_output_path, gdal.GRA_NearestNeighbour,
                    tile_utm_osr, snap_cs, tile_utm_extent, cdl_nodata)
                # Build a mask of CDL ag lands
                remap_mask_func(
                    cdl_output_path, cdl_ag_output_path, cdl_ag_list)

            # Cleanup
            del cdl_output_path
            del cdl_ds, cdl_band, cdl_nodata
            del tile_utm_osr, tile_utm_proj, tile_utm_extent
        # Project CDL rasters using multiprocessing
        if project_mp_list:
            pool = mp.Pool(mp_procs)
            results = pool.map(
                drigo.project_raster_mp, project_mp_list, chunksize=1)
            pool.close()
            pool.join()
            del results, pool
        if remap_mp_list:
            pool = mp.Pool(mp_procs)
            results = pool.map(remap_mask_mp, remap_mp_list, chunksize=1)
            pool.close()
            pool.join()
            del results, pool

    # Project/clip LANDFIRE for each path/row
    if landfire_flag:
        logging.info('\nBuild LANDFIRE for each path/row')
        project_mp_list, remap_mp_list = [], []
        for tile_name in tile_list:
            landfire_output_path = os.path.join(
                landfire_output_ws, tile_name,
                landfire_output_fmt.format(year))
            landfire_ag_output_path = os.path.join(
                landfire_output_ws, tile_name,
                landfire_ag_output_fmt.format(year))
            if not overwrite_flag and os.path.isfile(landfire_output_path):
                logging.debug('    {} already exists, skipping'.format(
                    os.path.basename(landfire_output_path)))
                continue
            logging.info('  {}'.format(tile_name))

            # Set the nodata value on the LANDFIRE raster if it is not set
            # landfire_ds = gdal.Open(landfire_input_path, 0)
            # landfire_band = landfire_ds.GetRasterBand(1)
            # landfire_nodata = landfire_band.GetNoDataValue()
            # landfire_ds = None
            # if landfire_nodata is None:
            #     landfire_nodata = 32767
            # del landfire_ds, landfire_band
            landfire_nodata = 32767

            # Clip and project
            tile_utm_osr = drigo.epsg_osr(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_proj = drigo.epsg_proj(
                32600 + int(tile_utm_zone_dict[tile_name]))
            tile_utm_extent = tile_utm_extent_dict[tile_name]
            if mp_procs > 1:
                project_mp_list.append([
                    landfire_input_path, landfire_output_path,
                    gdal.GRA_NearestNeighbour, tile_utm_proj, snap_cs,
                    tile_utm_extent, landfire_nodata])
                remap_mp_list.append([
                    landfire_output_path, landfire_ag_output_path,
                    landfire_ag_list])
            else:
                drigo.project_raster(
                    landfire_input_path, landfire_output_path,
                    gdal.GRA_NearestNeighbour, tile_utm_osr, snap_cs,
                    tile_utm_extent, landfire_nodata)
                # Build a mask of LANDFIRE ag lands
                remap_mask_func(
                    landfire_output_path, landfire_ag_output_path,
                    landfire_ag_list)

            # Cleanup
            del landfire_output_path
            del tile_utm_osr, tile_utm_proj, tile_utm_extent
        # Project LANDFIRE rasters using multiprocessing
        if project_mp_list:
            pool = mp.Pool(mp_procs)
            results = pool.map(
                drigo.project_raster_mp, project_mp_list, chunksize=1)
            pool.close()
            pool.join()
            del results, pool
        if remap_mp_list:
            pool = mp.Pool(mp_procs)
            results = pool.map(remap_mask_mp, remap_mp_list, chunksize=1)
            pool.close()
            pool.join()
            del results, pool

    # Convert field shapefiles to raster
    if field_flag:
        logging.info('\nBuild field rasters for each path/row')
        for tile_name in tile_list:
            logging.info('  {}'.format(tile_name))
            tile_output_ws = os.path.join(field_output_ws, tile_name)

            # Shapefile paths
            field_proj_name = (
                os.path.splitext(field_output_fmt.format(year))[0] +
                "_wgs84z{}.shp".format(tile_utm_zone_dict[tile_name]))
            field_proj_path = os.path.join(tile_output_ws, field_proj_name)
            field_output_path = os.path.join(
                tile_output_ws, field_output_fmt.format(year))
            if not overwrite_flag and os.path.isfile(field_output_path):
                logging.debug('    {} already exists, skipping'.format(
                    os.path.basename(field_output_path)))
                continue

            # The ogr2ogr spatial query is in the input spatial reference
            # Project the path/row extent to the field osr/proj
            field_input_osr = drigo.feature_path_osr(field_input_path)
            tile_utm_osr = drigo.epsg_osr(
                32600 + int(tile_utm_zone_dict[tile_name]))
            # field_input_proj = drigo.osr_proj(field_input_osr)
            # tile_utm_proj = drigo.osr_proj(tile_utm_osr)
            field_tile_extent = drigo.project_extent(
                tile_utm_extent_dict[tile_name],
                tile_utm_osr, field_input_osr, 30)

            # Project shapefile to the path/row zone
            # Clipping requires GDAL to be built with GEOS support
            subprocess.call([
                'ogr2ogr',
                '-t_srs', 'EPSG:326{}'.format(tile_utm_zone_dict[tile_name]),
                '-f', 'ESRI Shapefile', '-overwrite'] +
                ['-spat'] + list(map(str, field_tile_extent)) +
                ['-clipdst'] + list(map(str, tile_utm_extent_dict[tile_name])) +
                # ['-clipdst'] + list(map(str, tile_utm_extent_dict[tile_name])) +
                # ['-clipsrc'] + list(map(str, field_tile_extent)) +
                # ['-clipsrc'] + list(map(str, field_tile_extent)) +
                [field_proj_path, field_input_path])

            # Convert shapefile to raster
            field_mem_ds = drigo.polygon_to_raster_ds(
                field_proj_path, nodata_value=0, burn_value=1,
                output_osr=tile_utm_osr,
                output_extent=tile_utm_extent_dict[tile_name])
            field_output_driver = drigo.raster_driver(field_output_path)
            if field_output_path.lower().endswith('.img'):
                field_output_ds = field_output_driver.CreateCopy(
                    field_output_path, field_mem_ds, 0, ['COMPRESS=YES'])
            else:
                field_output_ds = field_output_driver.CreateCopy(
                    field_output_path, field_mem_ds, 0)
            field_output_ds, field_mem_ds = None, None

            # Remove field shapefile
            # try:
            #     remove_file(field_proj_path)
            # except:
            #     pass

            # Cleanup
            del tile_utm_osr, field_tile_extent, field_input_osr
            # del tile_utm_proj, field_input_proj
            del field_proj_name, field_proj_path, field_output_path

    logging.debug('\nScript complete')


def gcs_to_utm_dict(tile_list, tile_utm_zone_dict,
                    tile_gcs_osr, tile_gcs_wkt_dict, gcs_buffer=0.25,
                    snap_xmin=None, snap_ymin=None, snap_cs=None):
    """Return a dictionary of Landsat path/row GCS extents projected to UTM

    Parameters
    ----------
    tile_list : list
    tile_utm_zone_dict : dict
    tile_gcs_osr :
    tile_gcs_wkt_dict :
    gcs_buffer : float, optional
    snap_xmin : float or None, optional
    snap_ymin : float or None, optional
    snap_cs : float or None, optional

    Returns
    -------
    dict

    """
    # If parameters are not set, try to get from env
    # if snap_xmin is None and env.snap_xmin:
    #     snap_xmin = env.snap_xmin
    # if snap_ymin is None and env.snap_ymin:
    #     snap_ymin = env.snap_ymin
    # if snap_cs is None and env.cellsize:
    #     snap_cs = env.cellsize

    logging.info('\nCalculate projected extent for each path/row')
    output_dict = dict()
    for tile_name in sorted(tile_list):
        logging.info('  {}'.format(tile_name))
        # Create an OSR object from the utm projection
        tile_utm_osr = drigo.epsg_osr(32600 + int(tile_utm_zone_dict[tile_name]))
        # tile_utm_proj = drigo.osr_proj(tile_utm_osr)
        # Create utm transformation
        tile_utm_tx = osr.CoordinateTransformation(
            tile_gcs_osr, tile_utm_osr)
        tile_gcs_geom = ogr.CreateGeometryFromWkt(tile_gcs_wkt_dict[tile_name])
        # Buffer extent by 0.1 degrees
        # DEADBEEF - Buffer fails if GDAL is not built with GEOS support
        # tile_gcs_geom = tile_gcs_geom.Buffer(gcs_buffer)
        # Create gcs to utm transformer and apply it
        tile_utm_geom = tile_gcs_geom.Clone()
        tile_utm_geom.Transform(tile_utm_tx)
        tile_utm_extent = drigo.Extent(tile_utm_geom.GetEnvelope())
        tile_utm_extent = tile_utm_extent.ogrenv_swap()
        # 0.1 degrees ~ 10 km
        tile_utm_extent.buffer_extent(gcs_buffer * 100000)
        tile_utm_extent.adjust_to_snap('EXPAND', snap_xmin, snap_ymin, snap_cs)
        output_dict[tile_name] = tile_utm_extent
    return output_dict


def path_row_wkt_func(input_path, path_field='PATH', row_field='ROW',
                      tile_fmt='p{:03d}r{:03d}'):
    """Return a dictionary of Landsat path/rows and their geometries"""
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


def feature_path_geom_union(feature_path):
    """Union geometries of all feature in shapefile

    This is probably overkill since study area only has one feature
    """
    output_geom = ogr.Geometry(ogr.wkbPolygon)
    feature_ds = ogr.Open(feature_path, 0)
    feature_lyr = feature_ds.GetLayer()
    feature_ftr = feature_lyr.GetNextFeature()
    while feature_ftr:
        feature_geom = feature_ftr.GetGeometryRef()
        if output_geom.IsEmpty():
            output_geom = feature_geom.Clone()
        else:
            output_geom.Union(feature_geom)
        feature_ftr = feature_lyr.GetNextFeature()
    feature_ds = None
    return output_geom


def mosaic_tiles_mp(tup):
    """Pool multiprocessing friendly mosaic/project/clip function

    Parameters
    ----------
    tup : tuple
        Parameters that will be unpacked and passed to drigo.mosaic_tiles().

    Returns
    -------
    True if sucessful

    Notes
    -----
    mp.Pool needs all inputs packed into a single tuple.

    Tuple is unpacked and and single processing version of function is called.

    Since OSR spatial reference object can't be pickled, WKT string is passed
    in instead and converted to OSR spatial reference object.

    """
    if len(tup) == 4:
        input_list, output_raster, output_proj, output_cs = tup[:4]
        output_extent = None
    elif len(tup) == 5:
        input_list, output_raster, output_proj, output_cs, output_extent = tup[:5]

    return drigo.mosaic_tiles(
        input_list, output_raster, output_osr=drigo.proj_osr(output_proj),
        output_cs=output_cs, output_extent=output_extent)

    # return drigo.mosaic_tiles(*tup)


def remap_mask_mp(tup):
    """Remap the input raster to all 1's and return a mask raster

    Parameters
    ----------
    tup : tuple
        Parameters that will be unpacked and passed to remap_mask_func().

    Notes
    -----
    mp.Pool needs all inputs are packed into a single tuple
    Tuple is unpacked and and single processing version of function is called

    """
    return remap_mask_func(*tup)


def remap_mask_func(input_path, output_path, value_list):
    """Remap the input raster to all 1's and return a mask raster

    Parameters
    ----------
    input_path : str
        File path of the input raster.
    output_path : str
        File path of the output (mask) raster.
    value_list : list
        Values that will be set to 1.

    Returns
    -------
    True if sucessful

    """
    input_ds = gdal.Open(input_path)
    input_geo = drigo.raster_ds_geo(input_ds)
    input_proj = drigo.raster_ds_proj(input_ds)
    input_array = drigo.raster_ds_to_array(input_ds)[0]
    input_mask = np.zeros(input_array.shape, dtype=np.bool)
    for input_value in value_list:
        input_mask[input_array == input_value] = 1
    drigo.array_to_raster(input_mask, output_path, input_geo, input_proj)

    return True


def landsat_files_check(image_ws):
    """Check if Landsat folder needs to be rebuilt from tar.gz

    Parameters
    ----------
    image_ws : str
        Landsat image folder.

    Returns
    -------
    True if there are sufficient files to run METRIC
    False if files are missing and image should be extracted from tar.gz

    """
    try:
        image = et_image.Image(image_ws)
    except et_image.InvalidImage:
        return False
    if image.mtl_path is None:
        return False

    # Get list of raw digital number (DN) images from ORIGINAL_DATA folder
    dn_image_dict = et_common.landsat_band_image_dict(
        image.orig_data_ws, image.image_name_re)

    # Check if sets of rasters are present
    # Output from metric_model1
    if (os.path.isfile(image.albedo_sur_raster) and
        os.path.isfile(image.ts_raster) and
        (os.path.isfile(image.ndvi_toa_raster) or
         os.path.isfile(image.ndvi_sur_raster)) and
        (os.path.isfile(image.lai_toa_raster) or
         os.path.isfile(image.lai_sur_raster))):
        return True
    # Output from prep_scene
    elif (os.path.isfile(image.refl_toa_raster) and
          os.path.isfile(image.ts_bt_raster)):
        return True
    # Output from prep_path_row
    elif (dn_image_dict and
          set(list(image.band_toa_dict.keys()) +
                  [image.thermal_band, image.qa_band]) ==
          set(dn_image_dict.keys())):
        return True
    else:
        return False


def folder_check(folder_path):
    """"""
    if not os.path.isdir(folder_path):
        logging.error('  {} does not exist'.format(folder_path))
        return False
    else:
        return True


def file_check(file_path):
    """"""
    if not os.path.isfile(file_path):
        logging.error('  {} does not exist'.format(file_path))
        return False
    else:
        return True


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Batch Landsat path/row prep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Landsat project input file', metavar='FILE')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int,
        metavar='N', nargs='?', const=mp.cpu_count(),
        help='Number of processers to use')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
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
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini,
         overwrite_flag=args.overwrite, mp_procs=args.multiprocessing)
