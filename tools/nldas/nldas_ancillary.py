#--------------------------------
# Name:         nldas_ancillary.py
# Purpose:      Process NLDAS ancillary data
#--------------------------------

import argparse
import datetime as dt
import logging
import os
# import subprocess
import sys

import drigo
import netCDF4
import numpy as np
from osgeo import gdal, osr
# import pandas as pd

import _utils


def main(ancillary_ws=os.getcwd(), zero_elev_nodata_flag=False,
         overwrite_flag=False):
    """Process NLDAS ancillary data

    Parameters
    ----------
    ancillary_ws : str
        Folder of ancillary rasters.
    zero_elev_nodata_flag : bool, optional
        If True, set elevation nodata values to 0 (the default is False).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    """
    logging.info('\nProcess NLDAS ancillary data')

    # Site URLs
    mask_url = 'https://ldas.gsfc.nasa.gov/sites/default/files/ldas/nldas/NLDAS_masks-veg-soil.nc4'
    elev_url = 'https://ldas.gsfc.nasa.gov/sites/default/files/ldas/nldas/NLDAS_elevation.nc4'

    # Manually define the spatial reference and extent of the NLDAS data
    # This could be read in from a raster
    nldas_osr = osr.SpatialReference()
    nldas_osr.ImportFromEPSG(4326)
    if int(gdal.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        nldas_osr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    nldas_proj = drigo.osr_proj(nldas_osr)
    nldas_geo = (-125.0005,  0.125, 0., 53.0005, 0., -0.125)
    # nldas_geo = (-124.9375,  0.125, 0., 25.0625 + 224 * 0.125, 0., -0.125)
    logging.debug('  Geo: {}'.format(nldas_geo))
    nldas_nodata = -9999.0
    # logging.debug('  X/Y: {} {}'.format(gridmet_x, gridmet_y))
    # logging.debug('  Cellsize: {}'.format(gridmet_cs))

    # Build output workspace if it doesn't exist
    if not os.path.isdir(ancillary_ws):
        os.makedirs(ancillary_ws)

    # Input paths
    mask_nc = os.path.join(ancillary_ws, os.path.basename(mask_url))
    elev_nc = os.path.join(ancillary_ws, os.path.basename(elev_url))

    # Output paths
    elev_raster = os.path.join(ancillary_ws, 'nldas_elev.img')
    mask_raster = os.path.join(ancillary_ws, 'nldas_mask.img')
    lat_raster = os.path.join(ancillary_ws, 'nldas_lat.img')
    lon_raster = os.path.join(ancillary_ws, 'nldas_lon.img')

    if overwrite_flag or not os.path.isfile(mask_raster):
        logging.info('\nNLDAS Mask')
        logging.info('  Downloading')
        logging.debug('    {}'.format(mask_url))
        logging.debug('    {}'.format(mask_nc))
        _utils.url_download(mask_url, mask_nc)

        logging.info('  Extracting')
        logging.debug('  {}'.format(mask_raster))
        mask_nc_f = netCDF4.Dataset(mask_nc, 'r')
        mask_array = np.flipud(mask_nc_f.variables['NLDAS_mask'][0, :, :])
        # mask_array = np.flipud(mask_nc_f.variables['CONUS_mask'][0, :, :])
        drigo.array_to_raster(
            mask_array.astype(np.uint8), mask_raster,
            output_geo=nldas_geo, output_proj=nldas_proj)
        mask_nc_f.close()
        del mask_nc_f, mask_array
        # os.remove(mask_nc)

    if overwrite_flag or not os.path.isfile(elev_raster):
        logging.info('\nNLDAS Elevation')
        logging.info('  Downloading')
        logging.debug('    {}'.format(elev_url))
        logging.debug('    {}'.format(elev_nc))
        _utils.url_download(elev_url, elev_nc)

        logging.info('  Extracting')
        logging.debug('  {}'.format(elev_raster))
        elev_nc_f = netCDF4.Dataset(elev_nc, 'r')
        elev_ma = elev_nc_f.variables['NLDAS_elev'][0, :, :]
        # elev_ma = elev_nc_f.variables['CONUS_mask'][0, :, :]
        elev_array = np.flipud(elev_ma.data.astype(np.float32))
        # elev_nodata = float(elev_ma.fill_value)
        elev_array[
            (elev_array == elev_ma.fill_value) |
            (elev_array <= -300)] = np.nan
        if zero_elev_nodata_flag:
            elev_array[np.isnan(elev_array)] = 0
        if np.all(np.isnan(elev_array)):
            logging.error(
                '\nERROR: The elevation array is all nodata, exiting\n')
            sys.exit()

        drigo.array_to_raster(
            elev_array, elev_raster,
            output_geo=nldas_geo, output_proj=nldas_proj)
        elev_nc_f.close()
        del elev_nc_f, elev_array
        # os.remove(mask_nc)


    # Compute latitude/longitude rasters
    if ((overwrite_flag or
         not os.path.isfile(lat_raster) or
         not os.path.isfile(lat_raster)) and
        os.path.isfile(mask_raster)):
        logging.info('\nNLDAS Latitude/Longitude')
        logging.debug('    {}'.format(lat_raster))
        lat_array, lon_array = drigo.raster_lat_lon_func(mask_raster)
        # Handle the conversion to radians in the other scripts
        # lat_array *= (math.pi / 180)
        drigo.array_to_raster(
            lat_array, lat_raster, output_geo=nldas_geo,
            output_proj=nldas_proj)
        logging.debug('    {}'.format(lon_raster))
        drigo.array_to_raster(
            lon_array, lon_raster, output_geo=nldas_geo,
            output_proj=nldas_proj)
        del lat_array, lon_array

    # # Download the elevation data if necessary
    # logging.info('\nDownloading ASCII files')
    # if overwrite_flag or not os.path.isfile(input_elev_ascii):
    #     logging.info("  {}".format(os.path.basename(elev_url)))
    #     logging.debug("    {}".format(elev_url))
    #     logging.debug("    {}".format(input_elev_ascii))
    #     _utils.url_download(elev_url, input_elev_ascii)
    #
    # # The XYZ ASCII format is expecting LAT/LON/VALUE
    # # Export new asc files with just the needed columns for each raster
    # logging.debug('\nParsing elevation ASCII file')
    # logging.debug('  {}'.format(elev_ascii))
    # elev_df = pd.read_table(
    #     input_elev_ascii, header=None, sep=r"\s+", engine='python',
    #     names=['COL', 'ROW', 'LAT', 'LON', 'VALUE'])
    # elev_df = elev_df.sort_values(['LAT', 'LON'])
    # if zero_elev_nodata_flag:
    #     elev_df.loc[elev_df['VALUE'] == nldas_nodata, 'VALUE'] = 0
    # elev_df[['LON', 'LAT', 'VALUE']].to_csv(
    #     elev_ascii, header=None, index=False)
    #
    # # Remove existing rasters if necessary
    # #   -overwrite argument could be passed to gdalwarp instead
    # if overwrite_flag:
    #     logging.info('\nRemoving existing rasters')
    #     # if os.path.isfile(elev_raster):
    #     #     logging.info('  {}'.format(elev_raster))
    #     #     subprocess.call(['gdalmanage', 'delete', elev_raster])
    #
    # # Convert XYZ ascii to raster
    # logging.info('\nConverting ASCII to raster')
    # if not os.path.isfile(elev_raster):
    #     logging.info('  {}'.format(elev_ascii))
    #     subprocess.call(
    #         ['gdalwarp', '-of', 'HFA', '-t_srs', nldas_epsg,
    #          '-co', 'COMPRESSED=TRUE', elev_ascii, elev_raster,
    #          '-ot', 'Float32',
    #          '-srcnodata', str(nldas_nodata),
    #          '-dstnodata', str(drigo.numpy_type_nodata(np.float32))],
    #         cwd=ancillary_ws)
    #     # subprocess.call(
    #     #     ['gdal_translate', '-of', 'HFA', '-a_srs', nldas_epsg,
    #     #      '-co', 'COMPRESSED=TRUE', elev_ascii, elev_raster],
    #     #     cwd=ancillary_ws)
    #
    # # Cleanup
    # os.remove(elev_ascii)

    logging.debug('\nScript Complete')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/nldas
        tools:   ./pymetric/tools
        output:  ./pymetric/nldas
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    nldas_folder = os.path.join(project_folder, 'nldas')
    ancillary_folder = os.path.join(nldas_folder, 'ancillary')

    parser = argparse.ArgumentParser(
        description='Download/prep NLDAS ancillary data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ancillary', default=ancillary_folder, metavar='PATH',
        help='Ancillary raster folder path')
    parser.add_argument(
        '--zero', default=False, action="store_true",
        help='Set elevation nodata values to 0')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.ancillary and os.path.isdir(os.path.abspath(args.ancillary)):
        args.ancillary = os.path.abspath(args.ancillary)

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{:<20s} {}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{:<20s} {}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(ancillary_ws=args.ancillary, zero_elev_nodata_flag=args.zero,
         overwrite_flag=args.overwrite)
