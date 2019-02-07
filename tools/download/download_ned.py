#--------------------------------
# Name:         download_ned.py
# Purpose:      Download NED tiles
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import sys
import zipfile

import drigo
from osgeo import ogr

import _utils as utils


def main(extent_path, output_folder, overwrite_flag=False):
    """Download NED tiles that intersect the study_area

    Parameters
    ----------
    extent_path : str 
        File path to study area shapefile.
    output_folder : str 
        Folder path where files will be saved.
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None
    
    Notes
    -----
    Script assumes DEM data is in 1x1 WGS84 degree tiles.
    Download 10m (1/3 arc-second) or 30m (1 arc-second) versions from:
        10m: rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/13/IMG
        30m: rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/1/IMG
    For this example, only download 30m DEM.

    """
    logging.info('\nDownload NED tiles')
    site_url = 'rockyftp.cr.usgs.gov'
    site_folder = 'vdelivery/Datasets/Staged/Elevation/1/IMG'
    # site_url = 'ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/1/IMG'

    # Use 1 degree snap point and "cellsize" to get 1x1 degree tiles
    tile_osr = drigo.epsg_osr(4326)
    tile_x, tile_y, tile_cs = 0, 0, 1

    buffer_cells = 0

    # Error checking
    if not os.path.isfile(extent_path):
        logging.error('\nERROR: The input_path does not exist\n')
        return False
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Check that input is a shapefile

    # Get the extent of each feature
    logging.debug('  Reading extents')
    lat_lon_list = []
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    input_ds = shp_driver.Open(extent_path, 1)
    input_osr = drigo.feature_ds_osr(input_ds)
    input_layer = input_ds.GetLayer()
    input_ftr = input_layer.GetNextFeature()
    while input_ftr:
        input_geom = input_ftr.GetGeometryRef()
        input_extent = drigo.Extent(input_geom.GetEnvelope())
        input_extent = input_extent.ogrenv_swap()
        input_ftr = input_layer.GetNextFeature()
        logging.debug('Input Extent:  {}'.format(input_extent))

        # Project study area extent to input raster coordinate system
        output_extent = drigo.project_extent(
            input_extent, input_osr, tile_osr)
        logging.debug('Output Extent: {}'.format(output_extent))

        # Extent needed to select 1x1 degree tiles
        tile_extent = output_extent.copy()
        tile_extent.adjust_to_snap(
            'EXPAND', tile_x, tile_y, tile_cs)
        logging.debug('Tile Extent:   {}'.format(tile_extent))

        # Get list of avaiable tiles that intersect the extent
        lat_lon_list.extend([
            (lat, -lon)
            for lon in range(int(tile_extent.xmin), int(tile_extent.xmax))
            for lat in range(int(tile_extent.ymax), int(tile_extent.ymin), -1)])
    lat_lon_list = sorted(list(set(lat_lon_list)))

    # Retrieve a list of files available on the FTP server (keyed by lat/lon)
    logging.debug('  Retrieving NED tile list from server')
    zip_files = {
        m.group(1): x
        for x in utils.ftp_file_list(site_url, site_folder)
        for m in [re.search('[\w]*(n\d{2}w\d{3})[\w]*.zip', x)] if m}
    # logging.debug(zip_files[:10])

    # Attempt to download the tiles
    logging.debug('\nDownloading tiles')
    logging.info('')
    for lat_lon in lat_lon_list:
        logging.info('Tile: {}'.format(lat_lon))
        lat_lon_key = 'n{:02d}w{:03d}'.format(*lat_lon)

        try:
            zip_name = zip_files[lat_lon_key]
        except KeyError:
            logging.exception(
                'Error finding zip file for {}, skipping tile'.format(lat_lon))
            continue
        zip_url = '/'.join([site_url, site_folder, zip_name])
        zip_path = os.path.join(output_folder, zip_name)

        tile_path = os.path.join(output_folder, '{}.img'.format(lat_lon_key))

        logging.debug('  {}'.format(zip_url))
        logging.debug('  {}'.format(zip_path))
        logging.debug('  {}'.format(tile_path))
        if os.path.isfile(tile_path):
            if not overwrite_flag:
                logging.debug('  tile already exists, skipping')
                continue
            else:
                logging.debug('  tile already exists, removing')
                os.remove(tile_path)

        utils.ftp_download(site_url, site_folder, zip_name, zip_path)

        logging.debug('  Extracting')
        try:
            zip_f = zipfile.ZipFile(zip_path)
            img_name = [x for x in zip_f.namelist()
                        if re.search('[\w]*(n\d{2}w\d{3})[\w]*.img$', x)][0]
            img_path = os.path.join(output_folder, img_name)
            zip_f.extract(img_name, output_folder)
            zip_f.close()
            os.rename(img_path, tile_path)
        except Exception as e:
            logging.info('  Unhandled exception: {}'.format(e))

        try:
            os.remove(zip_path)
        except Exception as e:
            logging.info('  Unhandled exception: {}'.format(e))


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/download
        tools:   ./pymetric/tools
        output:  ./pymetric/dem
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'dem', 'tiles')

    parser = argparse.ArgumentParser(
        description='Download NED',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--extent', required=True, metavar='FILE',
        help='Study area shapefile')
    parser.add_argument(
        '--output', default=output_folder, metavar='FOLDER',
        help='Output folder')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.extent and os.path.isfile(os.path.abspath(args.extent)):
        args.extent = os.path.abspath(args.extent)
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(extent_path=args.extent, output_folder=args.output,
         overwrite_flag=args.overwrite)
