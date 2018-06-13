#--------------------------------
# Name:         cimis_ancillary.py
# Purpose:      Download CIMIS data
#--------------------------------

import argparse
import datetime as dt
import gzip
import logging
import os
import subprocess
import sys
import zipfile

import drigo
import numpy as np

import _utils


def main(ancillary_ws, overwrite_flag=False):
    """Process CIMIS ancillary data

    Parameters
    ----------
    ancillary_ws : str
        Folder of ancillary rasters.
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None
    
    """
    logging.info('\nProcess CIMIS ancillary data')

    # Site URL
    site_url = 'http://cimis.casil.ucdavis.edu/cimis/'

    # DEM for air pressure calculation
    # http://topotools.cr.usgs.gov/gmted_viewer/gmted2010_global_grids.php
    elev_full_url = (
        'http://edcintl.cr.usgs.gov/downloads/sciweb1/shared/'
        'topo/downloads/GMTED/Grid_ZipFiles/mn30_grd.zip')
    elev_full_zip = os.path.join(ancillary_ws, 'mn30_grd.zip')
    elev_full_raster = os.path.join(ancillary_ws, 'mn30_grd')

    # Get CIMIS grid properties from 2010/01/01 ETo raster
    # Grid of the spatial cimis input rasters
    # cimis_extent = drigo.Extent((-410000, -660000, 610000, 460000))
    # cimis_cs = 2000
    # cimis_geo = drigo.extent_geo(cimis_extent, cimis_cs)

    # Spatial reference parameters
    cimis_proj4 = (
        "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 "
        "+y_0=-4000000 +ellps=GRS80 +datum=NAD83 +units=m +no_defs")
    cimis_osr = drigo.proj4_osr(cimis_proj4)
    # cimis_epsg = 3310  # NAD_1983_California_Teale_Albers
    # cimis_osr = drigo.epsg_osr(cimis_epsg)
    # Comment this line out if building GeoTIFF instead of IMG
    cimis_osr.MorphToESRI()
    cimis_proj = cimis_osr.ExportToWkt()

    # snap_xmin, snap_ymin = (0, 0)

    # Build output workspace if it doesn't exist
    if not os.path.isdir(ancillary_ws):
        os.makedirs(ancillary_ws)

    # File paths
    mask_url = site_url + '/2010/01/01/ETo.asc.gz'
    # mask_gz = os.path.join(ancillary_ws, 'cimis_mask.asc.gz')
    mask_ascii = os.path.join(ancillary_ws, 'cimis_mask.asc')
    mask_raster = os.path.join(ancillary_ws, 'cimis_mask.img')
    elev_raster = os.path.join(ancillary_ws, 'cimis_elev.img')
    lat_raster = os.path.join(ancillary_ws, 'cimis_lat.img')
    lon_raster = os.path.join(ancillary_ws, 'cimis_lon.img')

    # Download an ETo ASCII raster to generate the mask raster
    if overwrite_flag or not os.path.isfile(mask_raster):
        logging.info('\nCIMIS mask')
        logging.debug('  Downloading')
        logging.debug("    {}".format(mask_url))
        logging.debug("    {}".format(mask_ascii))
        _utils.url_download(mask_url, mask_ascii)

        # DEADBEEF - The files do not appeared to be compressed even though
        # logging.debug("    {}".format(mask_gz))
        # _utils.url_download(mask_url, mask_gz)
        #
        #   they are named .asc.gz
        # # Uncompress '.gz' file to a new file
        # logging.debug('  Uncompressing')
        # logging.debug('    {}'.format(mask_ascii))
        # try:
        #     input_f = gzip.open(mask_gz, 'rb')
        #     output_f = open(mask_ascii, 'wb')
        #     output_f.write(input_f.read())
        #     output_f.close()
        #     input_f.close()
        #     del input_f, output_f
        # except:
        #     logging.error("  ERROR EXTRACTING FILE")
        # os.remove(mask_gz)

        # # Set spatial reference of the ASCII files
        # if build_prj_flag:
        #     prj_file = open(mask_asc.replace('.asc','.prj'), 'w')
        #     prj_file.write(output_proj)
        #     prj_file.close()

        # Convert the ASCII raster to a IMG raster
        logging.debug('  Computing mask')
        logging.debug('    {}'.format(mask_raster))
        mask_array = drigo.raster_to_array(mask_ascii, return_nodata=False)
        cimis_geo = drigo.raster_path_geo(mask_ascii)
        # cimis_extent = drigo.raster_path_extent(mask_ascii)
        logging.debug('    {}'.format(cimis_geo))
        mask_array = np.isfinite(mask_array).astype(np.uint8)
        drigo.array_to_raster(
            mask_array, mask_raster,
            output_geo=cimis_geo, output_proj=cimis_proj, output_nodata=0)
        # drigo.ascii_to_raster(
        #     mask_ascii, mask_raster, np.float32, cimis_proj)
        os.remove(mask_ascii)

    # Compute latitude/longitude rasters
    if ((overwrite_flag or
         not os.path.isfile(lat_raster) or
         not os.path.isfile(lat_raster)) and
        os.path.isfile(mask_raster)):
        logging.info('\nCIMIS latitude/longitude')
        logging.debug('    {}'.format(lat_raster))
        lat_array, lon_array = drigo.raster_lat_lon_func(mask_raster)
        drigo.array_to_raster(
            lat_array, lat_raster, output_geo=cimis_geo,
            output_proj=cimis_proj)
        logging.debug('    {}'.format(lon_raster))
        drigo.array_to_raster(
            lon_array, lon_raster, output_geo=cimis_geo,
            output_proj=cimis_proj)

    # Compute DEM raster
    if overwrite_flag or not os.path.isfile(elev_raster):
        logging.info('\nCIMIS DEM')
        logging.debug('  Downloading GMTED2010 DEM')
        logging.debug("    {}".format(elev_full_url))
        logging.debug("    {}".format(elev_full_zip))
        if overwrite_flag or not os.path.isfile(elev_full_zip):
            _utils.url_download(elev_full_url, elev_full_zip)

        # Uncompress '.gz' file to a new file
        logging.debug('  Uncompressing')
        logging.debug('    {}'.format(elev_full_raster))
        if overwrite_flag or not os.path.isfile(elev_full_raster):
            try:
                with zipfile.ZipFile(elev_full_zip, "r") as z:
                    z.extractall(ancillary_ws)
            except:
                logging.error("  ERROR EXTRACTING FILE")
            os.remove(elev_full_zip)

        # Get the extent and cellsize from the mask
        logging.debug('  Projecting to CIMIS grid')
        cimis_cs = drigo.raster_path_cellsize(mask_raster)[0]
        cimis_extent = drigo.raster_path_extent(mask_raster)
        logging.debug('    Extent: {}'.format(cimis_extent))
        logging.debug('    Cellsize: {}'.format(cimis_cs))

        logging.info('  {}'.format(mask_ascii))
        if overwrite_flag and os.path.isfile(elev_raster):
            subprocess.call(['gdalmanage', 'delete', elev_raster])
        if not os.path.isfile(elev_raster):
            subprocess.call(
                ['gdalwarp', '-r', 'average', '-t_srs', cimis_proj4,
                 '-te', str(cimis_extent.xmin), str(cimis_extent.ymin),
                 str(cimis_extent.xmax), str(cimis_extent.ymax),
                 '-tr', str(cimis_cs), str(cimis_cs),
                 '-of', 'HFA', '-co', 'COMPRESSED=TRUE',
                 elev_full_raster, elev_raster],
                cwd=ancillary_ws)

    logging.debug('\nScript Complete')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/cimis
        tools:   ./pymetric/tools
        output:  ./pymetric/cimis
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    cimis_folder = os.path.join(project_folder, 'cimis')
    ancillary_folder = os.path.join(cimis_folder, 'ancillary')

    parser = argparse.ArgumentParser(
        description='Download/prep CIMIS ancillary data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ancillary', default=ancillary_folder, metavar='PATH',
        help='Ancillary raster folder path')
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

    main(ancillary_ws=args.ancillary, overwrite_flag=args.overwrite)
