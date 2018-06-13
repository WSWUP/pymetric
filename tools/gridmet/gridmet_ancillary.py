#--------------------------------
# Name:         gridmet_ancillary.py
# Purpose:      Process GRIDMET ancillary data
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys

import drigo
import netCDF4
import numpy as np
from osgeo import osr

import _utils


def main(ancillary_ws=os.getcwd(), zero_elev_nodata_flag=False,
         overwrite_flag=False):
    """Process GRIDMET ancillary data

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
    logging.info('\nProcess GRIDMET ancillary rasters')

    # Site URL
    elev_url = 'https://climate.northwestknowledge.net/METDATA/data/metdata_elevationdata.nc'

    # Manually define the spatial reference and extent of the GRIDMET data
    # This could be read in from a raster
    gridmet_osr = osr.SpatialReference()
    # Assume GRIDMET data is in WGS84 not NAD83 (need to check with John)
    gridmet_osr.ImportFromEPSG(4326)
    # gridmet_osr.ImportFromEPSG(4326)
    gridmet_proj = drigo.osr_proj(gridmet_osr)
    gridmet_cs = 1. / 24   # 0.041666666666666666
    gridmet_x = -125 + gridmet_cs * 5
    gridmet_y = 49 + gridmet_cs * 10
    # gridmet_y = lon_array[0,0] - 0.5 * gridmet_cs
    # gridmet_y = lat_array[0,0] + 0.5 * gridmet_cs
    # gridmet_rows, gridmet_cols = elev_array.shape
    gridmet_geo = (gridmet_x, gridmet_cs, 0., gridmet_y, 0., -gridmet_cs)
    # gridmet_extent = drigo.geo_extent(
    #     gridmet_geo, gridmet_rows, gridmet_cols)
    # Keep track of the original/full geo-transform and extent
    # gridmet_full_geo = (
    #     gridmet_x, gridmet_cs, 0., gridmet_y, 0., -gridmet_cs)
    # gridmet_full_extent = drigo.geo_extent(
    #     gridmet_geo, gridmet_rows, gridmet_cols)
    logging.debug('  X/Y: {} {}'.format(gridmet_x, gridmet_y))
    logging.debug('  Geo: {}'.format(gridmet_geo))
    logging.debug('  Cellsize: {}'.format(gridmet_cs))

    # Build output workspace if it doesn't exist
    if not os.path.isdir(ancillary_ws):
        os.makedirs(ancillary_ws)

    # Output paths
    elev_nc = os.path.join(ancillary_ws, os.path.basename(elev_url))
    elev_raster = os.path.join(ancillary_ws, 'gridmet_elev.img')
    lat_raster = os.path.join(ancillary_ws, 'gridmet_lat.img')
    lon_raster = os.path.join(ancillary_ws, 'gridmet_lon.img')

    # Compute DEM raster
    if overwrite_flag or not os.path.isfile(elev_raster):
        logging.info('\nGRIDMET DEM')
        logging.info('  Downloading')
        logging.debug('    {}'.format(elev_url))
        logging.debug('    {}'.format(elev_nc))
        _utils.url_download(elev_url, elev_nc)
        # try:
        #     urllib.urlretrieve(elev_url, elev_nc)
        # except:
        #     logging.error("  ERROR: {}\n  FILE: {}".format(
        #         sys.exc_info()[0], elev_nc))
        #     # Try to remove the file since it may not have completely downloaded
        #     os.remove(elev_nc)

        logging.info('  Extracting')
        logging.debug('    {}'.format(elev_raster))
        elev_nc_f = netCDF4.Dataset(elev_nc, 'r')
        elev_ma = elev_nc_f.variables['elevation'][0, :, :]
        elev_array = elev_ma.data.astype(np.float32)
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
            output_geo=gridmet_geo, output_proj=gridmet_proj)
        elev_nc_f.close()
        # del elev_nc_f, elev_ma, elev_array, elev_nodata
        del elev_nc_f, elev_ma, elev_array
        os.remove(elev_nc)

    # Compute latitude/longitude rasters
    if ((overwrite_flag or
         not os.path.isfile(lat_raster) or
         not os.path.isfile(lat_raster)) and
        os.path.isfile(elev_raster)):
        logging.info('\nGRIDMET Latitude/Longitude')
        logging.debug('    {}'.format(lat_raster))
        lat_array, lon_array = drigo.raster_lat_lon_func(elev_raster)
        # Handle the conversion to radians in the other GRIDMET scripts
        # lat_array *= (math.pi / 180)
        drigo.array_to_raster(
            lat_array, lat_raster, output_geo=gridmet_geo,
            output_proj=gridmet_proj)
        logging.debug('    {}'.format(lon_raster))
        drigo.array_to_raster(
            lon_array, lon_raster, output_geo=gridmet_geo,
            output_proj=gridmet_proj)
        del lat_array, lon_array

    logging.debug('\nScript Complete')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/gridmet
        tools:   ./pymetric/tools
        output:  ./pymetric/gridmet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    gridmet_folder = os.path.join(project_folder, 'gridmet')
    ancillary_folder = os.path.join(gridmet_folder, 'ancillary')

    parser = argparse.ArgumentParser(
        description='Process GRIDMET ancillary data',
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
