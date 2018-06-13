#--------------------------------
# Name:         nldas_ancillary.py
# Purpose:      Process NLDAS ancillary data
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys

import drigo
import numpy as np
import pandas as pd

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
    mask_url = 'http://ldas.gsfc.nasa.gov/nldas/asc/NLDASmask_UMDunified.asc'
    elev_url = 'http://ldas.gsfc.nasa.gov/nldas/asc/gtopomean15k.asc'

    nldas_epsg = 'EPSG:4269'
    # nldas_epsg = 'EPSG:4326'

    nldas_nodata = -9999.0

    # Site URLs
    # file_re = re.compile(
    #    'NLDAS_FORA0125_H.A(?P<YEAR>\d{4})(?P<MONTH>\d{2})(?P<DAY>\d{2}).' +
    #    '(?P<TIME>\d{4}).002.grb')
    # file_re = re.compile(
    #    'NLDAS_FORA0125_H.A(?P<DATE>\d{8}).(?P<TIME>\d{4}).002.grb')

    # Build output workspace if it doesn't exist
    if not os.path.isdir(ancillary_ws):
        os.makedirs(ancillary_ws)

    # Input paths
    input_elev_ascii = os.path.join(ancillary_ws, os.path.basename(elev_url))
    input_mask_ascii = os.path.join(ancillary_ws, os.path.basename(mask_url))

    # Output paths
    elev_ascii = os.path.join(ancillary_ws, 'nldas_elev.asc')
    mask_ascii = os.path.join(ancillary_ws, 'nldas_mask.asc')
    lat_ascii = os.path.join(ancillary_ws, 'nldas_lat.asc')
    lon_ascii = os.path.join(ancillary_ws, 'nldas_lon.asc')
    elev_raster = os.path.join(ancillary_ws, 'nldas_elev.img')
    mask_raster = os.path.join(ancillary_ws, 'nldas_mask.img')
    lat_raster = os.path.join(ancillary_ws, 'nldas_lat.img')
    lon_raster = os.path.join(ancillary_ws, 'nldas_lon.img')

    # Download the elevation data if necessary
    logging.info('\nDownloading ASCII files')
    if overwrite_flag or not os.path.isfile(input_elev_ascii):
        logging.info("  {}".format(os.path.basename(elev_url)))
        logging.debug("    {}".format(elev_url))
        logging.debug("    {}".format(input_elev_ascii))
        _utils.url_download(elev_url, input_elev_ascii)

    # Download the land/water mask if necessary
    if overwrite_flag or not os.path.isfile(input_mask_ascii):
        logging.info("  {}".format(os.path.basename(mask_url)))
        logging.debug("    {}".format(elev_url))
        logging.debug("    {}".format(input_elev_ascii))
        _utils.url_download(mask_url, input_mask_ascii)

    # The XYZ ASCII format is expecting LAT/LON/VALUE
    # Export new asc files with just the needed columns for each raster
    logging.debug('\nParsing input ASCII files')

    logging.debug('  {}'.format(elev_ascii))
    elev_df = pd.read_table(
        input_elev_ascii, header=None, sep=r"\s+", engine='python',
        names=['COL', 'ROW', 'LAT', 'LON', 'VALUE'])
    elev_df = elev_df.sort_values(['LAT', 'LON'])
    if zero_elev_nodata_flag:
        elev_df.loc[elev_df['VALUE'] == nldas_nodata, 'VALUE'] = 0
    elev_df[['LON', 'LAT', 'VALUE']].to_csv(
        elev_ascii, header=None, index=False)

    logging.debug('  {}'.format(input_mask_ascii))
    mask_df = pd.read_table(
        input_mask_ascii, header=None, sep=r"\s+", engine='python',
        names=['COL', 'ROW', 'LAT', 'LON', 'VALUE'])
    mask_df = mask_df.sort_values(['LAT', 'LON'])
    mask_df[['LON', 'LAT', 'VALUE']].to_csv(
        mask_ascii, header=None, index=False)
    mask_df[['LON', 'LAT', 'LAT']].to_csv(lat_ascii, header=None, index=False)
    mask_df[['LON', 'LAT', 'LON']].to_csv(lon_ascii, header=None, index=False)

    # Remove existing rasters if necessary
    #   -overwrite argument could be passed to gdalwarp instead
    if overwrite_flag:
        logging.info('\nRemoving existing rasters')
        if os.path.isfile(elev_raster):
            logging.info('  {}'.format(elev_raster))
            subprocess.call(['gdalmanage', 'delete', elev_raster])
        if os.path.isfile(mask_raster):
            logging.info('  {}'.format(mask_raster))
            subprocess.call(['gdalmanage', 'delete', mask_raster])
        if os.path.isfile(lat_raster):
            logging.info('  {}'.format(lat_raster))
            subprocess.call(['gdalmanage', 'delete', lat_raster])
        if os.path.isfile(lon_raster):
            logging.info('  {}'.format(lon_raster))
            subprocess.call(['gdalmanage', 'delete', lon_raster])

    # Convert XYZ ascii to raster
    logging.info('\nConverting ASCII to raster')
    if not os.path.isfile(elev_raster):
        logging.info('  {}'.format(elev_ascii))
        subprocess.call(
            ['gdalwarp', '-of', 'HFA', '-t_srs', nldas_epsg,
             '-co', 'COMPRESSED=TRUE', elev_ascii, elev_raster,
             '-ot', 'Float32',
             '-srcnodata', str(nldas_nodata),
             '-dstnodata', str(drigo.numpy_type_nodata(np.float32))],
            cwd=ancillary_ws)
        # subprocess.call(
        #     ['gdal_translate', '-of', 'HFA', '-a_srs', nldas_epsg,
        #      '-co', 'COMPRESSED=TRUE', elev_ascii, elev_raster],
        #     cwd=ancillary_ws)
    if not os.path.isfile(mask_raster):
        logging.info('  {}'.format(mask_ascii))
        subprocess.call(
            ['gdalwarp', '-of', 'HFA', '-t_srs', nldas_epsg,
             '-co', 'COMPRESSED=TRUE', mask_ascii, mask_raster],
            cwd=ancillary_ws)
    if not os.path.isfile(lat_raster):
        logging.info('  {}'.format(lat_ascii))
        subprocess.call(
            ['gdalwarp', '-of', 'HFA', '-t_srs', nldas_epsg,
             '-co', 'COMPRESSED=TRUE', lat_ascii, lat_raster],
            cwd=ancillary_ws)
    if not os.path.isfile(lon_raster):
        logging.info('  {}'.format(lon_ascii))
        subprocess.call(
            ['gdalwarp', '-of', 'HFA', '-t_srs', nldas_epsg,
             '-co', 'COMPRESSED=TRUE', lon_ascii, lon_raster],
            cwd=ancillary_ws)

    # Cleanup
    os.remove(elev_ascii)
    os.remove(mask_ascii)
    os.remove(lat_ascii)
    os.remove(lon_ascii)

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
