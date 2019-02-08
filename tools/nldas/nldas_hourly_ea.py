#--------------------------------
# Name:         nldas_hourly_ea.py
# Purpose:      Extract NLDAS vapor pressure (ea) rasters
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import sys

import drigo
import numpy as np
from osgeo import gdal
import refet

import _utils

# np.seterr(invalid='ignore')


def main(grb_ws, ancillary_ws, output_ws, scene_list_path=None,
         start_dt=None, end_dt=None, times_str='',
         extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Extract hourly NLDAS vapour pressure rasters

    Parameters
    ----------
    grb_ws : str
        Folder of NLDAS GRB files.
    ancillary_ws : str
        Folder of ancillary rasters.
    output_ws : str
        Folder of output rasters.
    scene_list_path : str, optional
        Landsat scene keep list file path.
    start_dt : datetime, optional
        Start date.
    end_dt : datetime, optional
        End date.
    times : str, optional
        Comma separated values and/or ranges of UTC hours (i.e. "1, 2, 5-8").
        Parsed with python_common.parse_int_set().
    extent_path : str, optional
        File path defining the output extent.
    output_extent : list, optional
        Decimal degrees values defining output extent.
    stats_flag : bool, optional
        If True, compute raster statistics (the default is True).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    """
    logging.info('\nExtracting NLDAS vapour pressure rasters')

    # input_fmt = 'NLDAS_FORA0125_H.A{:04d}{:02d}{:02d}.{}.002.grb'
    input_re = re.compile(
        'NLDAS_FORA0125_H.A(?P<YEAR>\d{4})(?P<MONTH>\d{2})' +
        '(?P<DAY>\d{2}).(?P<TIME>\d{4}).002.grb$')

    # # Landsat Collection 1 Product ID
    # landsat_re = re.compile(
    #     '^(?:LT04|LT05|LE07|LC08)_\w{4}_\d{3}\d{3}_(?P<DATE>\d{8})_'
    #     '\w{8}_\w{2}_\w{2}')

    # Landsat Custom Scene ID
    landsat_re = re.compile(
        '^(?:LT04|LT05|LE07|LC08)_\d{6}_(?P<DATE>\d{8})')

    output_folder = 'ea'
    output_fmt = 'ea_{:04d}{:02d}{:02d}_hourly_nldas.img'
    # output_fmt = 'ea_{:04d}{:02d}{:02d}_{:04d}_nldas.img'

    # Only process a specific hours
    if not times_str:
        time_list = range(0, 24, 1)
    else:
        time_list = list(_utils.parse_int_set(times_str))
    time_list = ['{:02d}00'.format(t) for t in time_list]

    # Assume NLDAS is NAD83
    # input_epsg = 'EPSG:4269'

    # Ancillary raster paths
    mask_path = os.path.join(ancillary_ws, 'nldas_mask.img')
    elev_path = os.path.join(ancillary_ws, 'nldas_elev.img')

    # Process Landsat scene list and start/end input parameters
    if not scene_list_path and (not start_dt or not end_dt):
        logging.error(
            '\nERROR: A Landsat scene list or start/end dates must be set, '
            'exiting\n')
        return False
    if scene_list_path is not None and os.path.isfile(scene_list_path):
        # Build a date list from the Landsat scene keep list file
        logging.info('\nReading dates from scene keep list file')
        logging.info('  {}'.format(scene_list_path))
        with open(scene_list_path) as input_f:
            keep_list = input_f.readlines()
        date_list = sorted([
            dt.datetime.strptime(m.group('DATE'), '%Y%m%d').strftime('%Y-%m-%d')
            for image_id in keep_list
            for m in [landsat_re.match(image_id)] if m])
        logging.debug('  {}'.format(', '.join(date_list)))
    else:
        date_list = []
    if start_dt and end_dt:
        logging.debug('  Start date: {}'.format(start_dt))
        logging.debug('  End date:   {}'.format(end_dt))
    else:
        start_dt = dt.datetime.strptime(date_list[0], '%Y-%m-%d')
        end_dt = dt.datetime.strptime(date_list[-1], '%Y-%m-%d')

    # This allows GDAL to throw Python Exceptions
    # gdal.UseExceptions()
    # mem_driver = gdal.GetDriverByName('MEM')

    # Get the NLDAS spatial reference from the mask raster
    nldas_ds = gdal.Open(mask_path)
    nldas_osr = drigo.raster_ds_osr(nldas_ds)
    nldas_proj = drigo.osr_proj(nldas_osr)
    nldas_cs = drigo.raster_ds_cellsize(nldas_ds, x_only=True)
    nldas_extent = drigo.raster_ds_extent(nldas_ds)
    nldas_geo = nldas_extent.geo(nldas_cs)
    nldas_x, nldas_y = nldas_extent.origin()
    nldas_ds = None
    logging.debug('  Projection: {}'.format(nldas_proj))
    logging.debug('  Cellsize: {}'.format(nldas_cs))
    logging.debug('  Geo: {}'.format(nldas_geo))
    logging.debug('  Extent: {}'.format(nldas_extent))

    # Subset data to a smaller extent
    if output_extent is not None:
        logging.info('\nComputing subset extent & geo')
        logging.debug('  Extent: {}'.format(output_extent))
        nldas_extent = drigo.Extent(output_extent)
        nldas_extent.adjust_to_snap('EXPAND', nldas_x, nldas_y, nldas_cs)
        nldas_geo = nldas_extent.geo(nldas_cs)
        logging.debug('  Geo: {}'.format(nldas_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    elif extent_path is not None:
        logging.info('\nComputing subset extent & geo')
        if not os.path.isfile(extent_path):
            logging.error(
                '\nThe extent object not exist, exiting\n'
                '  {}'.format(extent_path))
            return False
        elif extent_path.lower().endswith('.shp'):
            nldas_extent = drigo.feature_path_extent(extent_path)
            extent_osr = drigo.feature_path_osr(extent_path)
            extent_cs = None
        else:
            nldas_extent = drigo.raster_path_extent(extent_path)
            extent_osr = drigo.raster_path_osr(extent_path)
            extent_cs = drigo.raster_path_cellsize(extent_path, x_only=True)
        nldas_extent = drigo.project_extent(
            nldas_extent, extent_osr, nldas_osr, extent_cs)
        nldas_extent.adjust_to_snap('EXPAND', nldas_x, nldas_y, nldas_cs)
        nldas_geo = nldas_extent.geo(nldas_cs)
        logging.debug('  Geo: {}'.format(nldas_geo))
        logging.debug('  Extent: {}'.format(nldas_extent))
    logging.debug('')

    # Read the NLDAS mask array if present
    if mask_path and os.path.isfile(mask_path):
        mask_array, mask_nodata = drigo.raster_to_array(
            mask_path, mask_extent=nldas_extent, fill_value=0,
            return_nodata=True)
        mask_array = mask_array != mask_nodata
    else:
        mask_array = None

    # Read elevation arrays (or subsets?)
    elev_array = drigo.raster_to_array(
        elev_path, mask_extent=nldas_extent, return_nodata=False)
    pair_array = refet.calcs._air_pressure(elev_array)

    # Build output folder
    var_ws = os.path.join(output_ws, output_folder)
    if not os.path.isdir(var_ws):
        os.makedirs(var_ws)

    # Each sub folder in the main folder has all imagery for 1 day
    # The path for each subfolder is the /YYYY/DOY

    # This approach will process files for target dates
    # for input_dt in date_range(start_dt, end_dt + dt.timedelta(1)):
    #     logging.info(input_dt.date())

    # Iterate all available files and check dates if necessary
    for root, folders, files in os.walk(grb_ws):
        root_split = os.path.normpath(root).split(os.sep)

        # If the year/doy is outside the range, skip
        if (re.match('\d{4}', root_split[-2]) and
                re.match('\d{3}', root_split[-1])):
            root_dt = dt.datetime.strptime('{}_{}'.format(
                root_split[-2], root_split[-1]), '%Y_%j')
            logging.info('{}'.format(root_dt.date()))
            if ((start_dt is not None and root_dt < start_dt) or
                    (end_dt is not None and root_dt > end_dt)):
                continue
            elif date_list and root_dt.date().isoformat() not in date_list:
                continue
        # If the year is outside the range, don't search subfolders
        elif re.match('\d{4}', root_split[-1]):
            root_year = int(root_split[-1])
            logging.info('Year: {}'.format(root_year))
            if ((start_dt is not None and root_year < start_dt.year) or
                    (end_dt is not None and root_year > end_dt.year)):
                folders[:] = []
            else:
                folders[:] = sorted(folders)
            continue
        else:
            continue

        # Create a single raster for each day with 24 bands
        # Each time step will be stored in a separate band
        output_name = output_fmt.format(
            root_dt.year, root_dt.month, root_dt.day)
        output_path = os.path.join(
            var_ws, str(root_dt.year), output_name)
        logging.debug('  {}'.format(output_path))
        if os.path.isfile(output_path):
            if not overwrite_flag:
                logging.debug('    File already exists, skipping')
                continue
            else:
                logging.debug('    File already exists, removing existing')
                os.remove(output_path)
        logging.debug('  {}'.format(root))
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        drigo.build_empty_raster(
            output_path, band_cnt=24, output_dtype=np.float32,
            output_proj=nldas_proj, output_cs=nldas_cs,
            output_extent=nldas_extent, output_fill_flag=True)

        # Iterate through hourly files
        for input_name in sorted(files):
            logging.info('  {}'.format(input_name))
            input_path = os.path.join(root, input_name)
            input_match = input_re.match(input_name)
            if input_match is None:
                logging.debug('  Regular expression didn\'t match, skipping')
                continue
            input_dt = dt.datetime(
                int(input_match.group('YEAR')),
                int(input_match.group('MONTH')),
                int(input_match.group('DAY')))
            input_doy = int(input_dt.strftime('%j'))
            time_str = input_match.group('TIME')
            band_num = int(time_str[:2]) + 1
            # if start_dt is not None and input_dt < start_dt:
            #     continue
            # elif end_dt is not None and input_dt > end_dt:
            #     continue
            # elif date_list and input_dt.date().isoformat() not in date_list:
            #     continue
            if time_str not in time_list:
                logging.debug('    Time not in list, skipping')
                continue
            logging.debug('    Time: {} {}'.format(input_dt.date(), time_str))
            logging.debug('    Band: {}'.format(band_num))

            # Determine band numbering/naming
            input_band_dict = grib_band_names(input_path)

            # Compute vapour pressure from specific humidity
            input_ds = gdal.Open(input_path)
            sph_array = drigo.raster_ds_to_array(
                input_ds, band=input_band_dict['Specific humidity [kg/kg]'],
                mask_extent=nldas_extent, return_nodata=False)
            ea_array = refet.calcs._actual_vapor_pressure(
                q=sph_array, pair=pair_array)
            # ea_array = (sph_array * pair_array) / (0.622 + 0.378 * sph_array)

            # Save the projected array as 32-bit floats
            drigo.array_to_comp_raster(
                ea_array.astype(np.float32), output_path, band=band_num)
            # drigo.block_to_raster(
            #     ea_array.astype(np.float32), output_path, band=band)
            # drigo.array_to_raster(
            #     ea_array.astype(np.float32), output_path,
            #     output_geo=nldas_geo, output_proj=nldas_proj,
            #     stats_flag=stats_flag)

            del sph_array
            input_ds = None

        if stats_flag:
            drigo.raster_statistics(output_path)

    logging.debug('\nScript Complete')


def grib_band_names(grib_path):
    """Return a dictionary of all GRIB_ELEMENT and band number items"""
    band_dict = dict()
    try:
        grib_ds = gdal.Open(grib_path)
    except IOError:
        grib_driver = gdal.GetDriverByName('GRIB')
        grib_ds = grib_driver.Open(grib_path)
    for i in range(grib_ds.RasterCount):
        band_meta = [
            x for x in grib_ds.GetRasterBand(i + 1).GetMetadata_List()
            if 'GRIB_COMMENT' in x][0]
            # if 'GRIB_ELEMENT' in x][0]
        band_dict[band_meta.split('=')[1]] = i + 1
    grib_ds = None
    return band_dict


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
    grb_folder = os.path.join(nldas_folder, 'grb')

    parser = argparse.ArgumentParser(
        description='NLDAS hourly vapour pressure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--start', default=None, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='Start date')
    parser.add_argument(
        '--end', default=None, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='End date')
    parser.add_argument(
        '--landsat', default=None, metavar='PATH',
        help='Landsat scene keep list path')
    parser.add_argument(
        '--grb', default=grb_folder, metavar='PATH',
        help='Input GRB folder path')
    parser.add_argument(
        '--ancillary', default=ancillary_folder, metavar='PATH',
        help='Ancillary raster folder path')
    parser.add_argument(
        '--output', default=nldas_folder, metavar='PATH',
        help='Output raster folder path')
    parser.add_argument(
        '--times', default='0-23', type=str,
        help='Time list and/or range (0-23 for all times)')
    parser.add_argument(
        '--extent', default=None, metavar='PATH',
        help='Subset extent path')
    parser.add_argument(
        '-te', default=None, type=float, nargs=4,
        metavar=('xmin', 'ymin', 'xmax', 'ymax'),
        help='Subset extent in decimal degrees')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.grb and os.path.isdir(os.path.abspath(args.grb)):
        args.grb = os.path.abspath(args.grb)
    if args.ancillary and os.path.isdir(os.path.abspath(args.ancillary)):
        args.ancillary = os.path.abspath(args.ancillary)
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
    if args.landsat and os.path.isfile(os.path.abspath(args.landsat)):
        args.landsat = os.path.abspath(args.landsat)
    if args.extent and os.path.isfile(os.path.abspath(args.extent)):
        args.extent = os.path.abspath(args.extent)

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{:<20s} {}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{:<20s} {}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(grb_ws=args.grb, ancillary_ws=args.ancillary, output_ws=args.output,
         start_dt=args.start, end_dt=args.end, scene_list_path=args.landsat,
         times_str=args.times, extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite)
