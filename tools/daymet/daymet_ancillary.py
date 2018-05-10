#--------------------------------
# Name:         daymet_ancillary.py
# Purpose:      Process DAYMET ancillary data
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys

import drigo
import numpy as np


def main(ancillary_ws=os.getcwd(), zero_elev_nodata_flag=False,
         overwrite_flag=False):
    """Process DAYMET ancillary data

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
    logging.info('\nProcess DAYMET ancillary rasters')

    # Site URL
    # ancillary_url = 'http://daymet.ornl.gov/files/ancillary_files.tgz'

    # Build output workspace if it doesn't exist
    if not os.path.isdir(ancillary_ws):
        os.makedirs(ancillary_ws)

    # Input paths
    # ancillary_targz = os.path.join(
    #     ancillary_ws, os.path.basename(ancillary_url))
    # dem_nc = os.path.join(ancillary_ws, 'dem_data.nc')
    # mask_nc = os.path.join(ancillary_ws, 'mask_data.nc')

    # Output paths
    dem_raster = os.path.join(ancillary_ws, 'daymet_elev.img')
    lat_raster = os.path.join(ancillary_ws, 'daymet_lat.img')
    lon_raster = os.path.join(ancillary_ws, 'daymet_lon.img')
    # mask_raster = os.path.join(ancillary_ws, 'daymet_mask.img')

    # Spatial reference parameters
    daymet_proj4 = (
        "+proj=lcc +datum=WGS84 +lat_1=25 n "
        "+lat_2=60n +lat_0=42.5n +lon_0=100w")
    daymet_osr = drigo.proj4_osr(daymet_proj4)
    daymet_osr.MorphToESRI()
    daymet_proj = daymet_osr.ExportToWkt()
    daymet_cs = 1000
    # daymet_nodata = -9999

    # For now, hardcode the DAYMET extent/geo
    snap_xmin, snap_ymin = -4560750, -3090500
    daymet_rows, daymet_cols = 8075, 7814
    # snap_xmin, snap_ymin = -4659000, -3135000
    # daymet_rows, daymet_cols = 8220, 8011
    # daymet_geo = (
    #     snap_xmin, daymet_cs, 0.,
    #     snap_ymin + daymet_cs * daymet_rows, 0., -daymet_cs)
    daymet_extent = drigo.Extent([
        snap_xmin, snap_ymin,
        snap_xmin + daymet_cs * daymet_cols,
        snap_ymin + daymet_cs * daymet_rows])
    daymet_geo = daymet_extent.geo(daymet_cs)
    logging.debug("  Extent:   {}".format(daymet_extent))
    logging.debug("  Geo:      {}".format(daymet_geo))
    # logging.debug("  Cellsize: {}".format(daymet_cs))
    # logging.debug("  Shape:    {}".format(daymet_extent.shape(daymet_cs)))

    # # Download the ancillary raster tar.gz
    # if overwrite_flag or not os.path.isfile(ancillary_targz):
    #     logging.info('\nDownloading ancillary tarball files')
    #     logging.info("  {}".format(os.path.basename(ancillary_url)))
    #     logging.debug("    {}".format(ancillary_url))
    #     logging.debug("    {}".format(ancillary_targz))
    #     url_download(ancillary_url, ancillary_targz)
    #     try:
    #         urllib.urlretrieve(ancillary_url, ancillary_targz)
    #     except:
    #         logging.error("  ERROR: {}\n  FILE: {}".format(
    #             sys.exc_info()[0], ancillary_targz))
    #         os.remove(ancillary_targz)

    # # Extract the ancillary rasters
    # ancillary_list = [dem_nc]
    # # ancillary_list = [dem_nc, mask_nc]
    # if (os.path.isfile(ancillary_targz) and
    #     (overwrite_flag or
    #      not all([os.path.isfile(os.path.join(ancillary_ws, x))
    #               for x in ancillary_list]))):
    #     logging.info('\nExtracting ancillary rasters')
    #     logging.debug("  {}".format(ancillary_targz))
    #     tar = tarfile.open(ancillary_targz)
    #     for member in tar.getmembers():
    #         print member.name
    #         member.name = os.path.basename(member.name)
    #         # Strip off leading numbers from ancillary raster name
    #         member.name = member.name.split('_', 1)[1]
    #         member_path = os.path.join(ancillary_ws, member.name)
    #         if not member.name.endswith('.nc'):
    #             continue
    #         elif member_path not in ancillary_list:
    #             continue
    #         elif os.path.isfile(member_path):
    #             continue
    #         logging.debug("  {}".format(member.name))
    #         tar.extract(member, ancillary_ws)
    #     tar.close()

    # # Mask
    # if ((overwrite_flag or
    #      not os.path.isfile(mask_raster)) and
    #     os.path.isfile(mask_nc)):
    #     logging.info('\nExtracting mask raster')
    #     mask_nc_f = netCDF4.Dataset(mask_nc, 'r')
    #     logging.debug(mask_nc_f)
    #     # logging.debug(mask_nc_f.variables['image'])
    #     mask_array = mask_nc_f.variables['image'][:]
    #     mask_array[mask_array == daymet_nodata] = 255
    #     drigo.array_to_raster(
    #         mask_array, mask_raster,
    #         output_geo=daymet_geo, output_proj=daymet_proj,
    #         output_nodata=255)
    #     mask_nc_f.close()

    # # DEM
    # if ((overwrite_flag or not os.path.isfile(dem_raster)) and
    #         os.path.isfile(dem_nc)):
    #     logging.info('\nExtracting DEM raster')
    #     dem_nc_f = netCDF4.Dataset(dem_nc, 'r')
    #     logging.debug(dem_nc_f)
    #     # logging.debug(dem_nc_f.variables['image'])
    #     dem_array = dem_nc_f.variables['image'][:]
    #     # Rounding issues of the nodata value when converting to float32
    #     dem_array[dem_array == daymet_nodata] -= 1
    #     dem_array = dem_array.astype(np.float32)
    #     if zero_elev_nodata_flag:
    #         dem_array[dem_array <= daymet_nodata] = 0
    #     else:
    #         dem_array[dem_array <= daymet_nodata] = np.nan
    #     drigo.array_to_raster(
    #         dem_array, dem_raster,
    #         output_geo=daymet_geo, output_proj=daymet_proj)
    #     dem_nc_f.close()

    # Latitude/Longitude
    if (os.path.isfile(dem_raster) and
            (overwrite_flag or
             not os.path.isfile(lat_raster) or
             not os.path.isfile(lon_raster))):
        logging.info('\nDAYMET Latitude/Longitude')
        logging.debug('    {}'.format(lat_raster))
        lat_array, lon_array = drigo.raster_lat_lon_func(
            dem_raster, gcs_cs=0.05)
        drigo.array_to_raster(
            lat_array.astype(np.float32), lat_raster,
            output_geo=daymet_geo, output_proj=daymet_proj)
        logging.debug('    {}'.format(lon_raster))
        drigo.array_to_raster(
            lon_array.astype(np.float32), lon_raster,
            output_geo=daymet_geo, output_proj=daymet_proj)
        del lat_array, lon_array

    logging.debug('\nScript Complete')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pyMETRIC/tools/daymet
        tools:   ./pyMETRIC/tools
        output:  ./pyMETRIC/daymet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    daymet_folder = os.path.join(project_folder, 'daymet')

    parser = argparse.ArgumentParser(
        description='Download/prep DAYMET ancillary data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ancillary', default=os.path.join(daymet_folder, 'ancillary'),
        metavar='PATH', help='Ancillary raster folder path')
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
