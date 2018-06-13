#--------------------------------
# Name:         daymet_daily_variables.py
# Purpose:      Extract DAYMET variables
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import sys

import drigo
import netCDF4
import numpy as np
from osgeo import gdal

import _utils


def main(start_dt, end_dt, netcdf_ws, ancillary_ws, output_ws,
         variables=['prcp'], extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Extract DAYMET temperature

    Parameters
    ----------
    start_dt : datetime
        Start date.
    end_dt : datetime
        End date.
    netcdf_ws : str
        Folder of DAYMET netcdf files.
    ancillary_ws : str
        Folder of ancillary rasters.
    output_ws : str
        Folder of output rasters.
    variables : list, optional
        DAYMET variables to download (the default is ['prcp']).
        Choices: 'prcp', 'srad', 'vp', 'tmmn', 'tmmx', 'all.
        Set as ['all'] to process all variables.
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
    logging.info('\nExtracting DAYMET variables')
    logging.debug('  Start date: {}'.format(start_dt))
    logging.debug('  End date:   {}'.format(end_dt))

    # Get DAYMET spatial reference from an ancillary raster
    mask_raster = os.path.join(ancillary_ws, 'daymet_mask.img')

    daymet_re = re.compile('daymet_v3_(?P<VAR>\w+)_(?P<YEAR>\d{4})_na.nc4$')

    # DAYMET rasters to extract
    var_full_list = ['prcp', 'srad', 'vp', 'tmmn', 'tmmx']
    if not variables:
        logging.error('\nERROR: variables parameter is empty\n')
        sys.exit()
    elif type(variables) is not list:
        # DEADBEEF - I could try converting comma separated strings to lists?
        logging.warning('\nERROR: variables parameter must be a list\n')
        sys.exit()
    elif 'all' in variables:
        logging.error('\nDownloading all variables\n  {}'.format(
            ','.join(var_full_list)))
        var_list = var_full_list[:]
    elif not set(variables).issubset(set(var_full_list)):
        logging.error('\nERROR: variables parameter is invalid\n  {}'.format(
            variables))
        sys.exit()
    else:
        var_list = variables[:]

    # DAYMET band name dictionary
    # daymet_band_dict = dict()
    # daymet_band_dict['prcp'] = 'precipitation_amount'
    # daymet_band_dict['srad'] = 'surface_downwelling_shortwave_flux_in_air'
    # daymet_band_dict['sph'] = 'specific_humidity'
    # daymet_band_dict['tmin'] = 'air_temperature'
    # daymet_band_dict['tmax'] = 'air_temperature'

    # Get extent/geo from mask raster
    daymet_ds = gdal.Open(mask_raster)
    daymet_osr = drigo.raster_ds_osr(daymet_ds)
    daymet_proj = drigo.osr_proj(daymet_osr)
    daymet_cs = drigo.raster_ds_cellsize(daymet_ds, x_only=True)
    daymet_extent = drigo.raster_ds_extent(daymet_ds)
    daymet_geo = daymet_extent.geo(daymet_cs)
    daymet_x, daymet_y = daymet_extent.origin()
    daymet_ds = None
    logging.debug('  Projection: {}'.format(daymet_proj))
    logging.debug('  Cellsize: {}'.format(daymet_cs))
    logging.debug('  Geo: {}'.format(daymet_geo))
    logging.debug('  Extent: {}'.format(daymet_extent))
    logging.debug('  Origin: {} {}'.format(daymet_x, daymet_y))

    # Subset data to a smaller extent
    if output_extent is not None:
        logging.info('\nComputing subset extent & geo')
        logging.debug('  Extent: {}'.format(output_extent))
        # Assume input extent is in decimal degrees
        output_extent = drigo.project_extent(
            drigo.Extent(output_extent), drigo.epsg_osr(4326), daymet_osr, 0.001)
        output_extent = drigo.intersect_extents([daymet_extent, output_extent])
        output_extent.adjust_to_snap('EXPAND', daymet_x, daymet_y, daymet_cs)
        output_geo = output_extent.geo(daymet_cs)
        logging.debug('  Geo: {}'.format(output_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    elif extent_path is not None:
        logging.info('\nComputing subset extent & geo')
        if extent_path.lower().endswith('.shp'):
            output_extent = drigo.feature_path_extent(extent_path)
            extent_osr = drigo.feature_path_osr(extent_path)
            extent_cs = None
        else:
            output_extent = drigo.raster_path_extent(extent_path)
            extent_osr = drigo.raster_path_osr(extent_path)
            extent_cs = drigo.raster_path_cellsize(extent_path, x_only=True)
        output_extent = drigo.project_extent(
            output_extent, extent_osr, daymet_osr, extent_cs)
        output_extent = drigo.intersect_extents([daymet_extent, output_extent])
        output_extent.adjust_to_snap('EXPAND', daymet_x, daymet_y, daymet_cs)
        output_geo = output_extent.geo(daymet_cs)
        logging.debug('  Geo: {}'.format(output_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    else:
        output_extent = daymet_extent.copy()
        output_geo = daymet_geo[:]
    # output_shape = output_extent.shape(cs=daymet_cs)
    xi, yi = drigo.array_geo_offsets(daymet_geo, output_geo, daymet_cs)
    output_rows, output_cols = output_extent.shape(daymet_cs)
    logging.debug('  Shape: {} {}'.format(output_rows, output_cols))
    logging.debug('  Offsets: {} {} (x y)'.format(xi, yi))

    # Process each variable
    for input_var in var_list:
        logging.info("\nVariable: {}".format(input_var))

        # Rename variables to match cimis
        if input_var == 'prcp':
            output_var = 'ppt'
        else:
            output_var = input_var

        # Build output folder
        var_ws = os.path.join(output_ws, output_var)
        if not os.path.isdir(var_ws):
            os.makedirs(var_ws)

        # Process each file in the input workspace
        for input_name in sorted(os.listdir(netcdf_ws)):
            logging.debug("{}".format(input_name))
            input_match = daymet_re.match(input_name)
            if not input_match:
                logging.debug('  Regular expression didn\'t match, skipping')
                continue
            elif input_match.group('VAR') != input_var:
                logging.debug('  Variable didn\'t match, skipping')
                continue
            year_str = input_match.group('YEAR')
            logging.info("  Year: {}".format(year_str))
            year_int = int(year_str)
            year_days = int(dt.datetime(year_int, 12, 31).strftime('%j'))
            if start_dt is not None and year_int < start_dt.year:
                logging.debug('    Before start date, skipping')
                continue
            elif end_dt is not None and year_int > end_dt.year:
                logging.debug('    After end date, skipping')
                continue

            # Build input file path
            input_raster = os.path.join(netcdf_ws, input_name)
            # if not os.path.isfile(input_raster):
            #     logging.debug(
            #         '    Input raster doesn\'t exist, skipping    {}'.format(
            #             input_raster))
            #     continue

            # Build output folder
            output_year_ws = os.path.join(var_ws, year_str)
            if not os.path.isdir(output_year_ws):
                os.makedirs(output_year_ws)

            # Read in the DAYMET NetCDF file
            input_nc_f = netCDF4.Dataset(input_raster, 'r')
            # logging.debug(input_nc_f.variables)

            # Check all valid dates in the year
            year_dates = _utils.date_range(
                dt.datetime(year_int, 1, 1), dt.datetime(year_int + 1, 1, 1))
            for date_dt in year_dates:
                if start_dt is not None and date_dt < start_dt:
                    logging.debug('  {} - before start date, skipping'.format(
                        date_dt.date()))
                    continue
                elif end_dt is not None and date_dt > end_dt:
                    logging.debug('  {} - after end date, skipping'.format(
                        date_dt.date()))
                    continue
                else:
                    logging.info('  {}'.format(date_dt.date()))

                output_path = os.path.join(
                    output_year_ws, '{}_{}_daymet.img'.format(
                        output_var, date_dt.strftime('%Y%m%d')))
                if os.path.isfile(output_path):
                    logging.debug('    {}'.format(output_path))
                    if not overwrite_flag:
                        logging.debug('    File already exists, skipping')
                        continue
                    else:
                        logging.debug('    File already exists, removing existing')
                        os.remove(output_path)

                doy = int(date_dt.strftime('%j'))
                doy_i = range(1, year_days + 1).index(doy)

                # Arrays are being read as masked array with a fill value of -9999
                # Convert to basic numpy array arrays with nan values
                try:
                    input_ma = input_nc_f.variables[input_var][
                        doy_i, yi: yi + output_rows, xi: xi + output_cols]
                except IndexError:
                    logging.info('    date not in netcdf, skipping')
                    continue
                input_nodata = float(input_ma.fill_value)
                output_array = input_ma.data.astype(np.float32)
                output_array[output_array == input_nodata] = np.nan

                # Convert Kelvin to Celsius
                if input_var in ['tmax', 'tmin']:
                    output_array -= 273.15

                # Save the array as 32-bit floats
                drigo.array_to_raster(
                    output_array.astype(np.float32), output_path,
                    output_geo=output_geo, output_proj=daymet_proj,
                    stats_flag=stats_flag)
                del input_ma, output_array
            input_nc_f.close()
            del input_nc_f

    logging.debug('\nScript Complete')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/daymet
        tools:   ./pymetric/tools
        output:  ./pymetric/daymet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    daymet_folder = os.path.join(project_folder, 'daymet')
    ancillary_folder = os.path.join(daymet_folder, 'ancillary')
    netcdf_folder = os.path.join(daymet_folder, 'netcdf')

    parser = argparse.ArgumentParser(
        description='DAYMET daily variables',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--start', required=True, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='Start date')
    parser.add_argument(
        '--end', required=True, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='End date')
    parser.add_argument(
        '--netcdf', default=netcdf_folder, metavar='PATH',
        help='Input netCDF folder path')
    parser.add_argument(
        '--ancillary', default=ancillary_folder, metavar='PATH',
        help='Ancillary raster folder path')
    parser.add_argument(
        '--output', default=daymet_folder, metavar='PATH',
        help='Output raster folder path')
    parser.add_argument(
        '--vars', default=['prcp'], nargs='+',
        choices=['prcp', 'srad', 'vp', 'tmmn', 'tmmx', 'all'],
        help='DAYMET variables to extract')
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
    if args.netcdf and os.path.isdir(os.path.abspath(args.netcdf)):
        args.netcdf = os.path.abspath(args.netcdf)
    if args.ancillary and os.path.isdir(os.path.abspath(args.ancillary)):
        args.ancillary = os.path.abspath(args.ancillary)
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
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

    main(start_dt=args.start, end_dt=args.end, netcdf_ws=args.netcdf,
         ancillary_ws=args.ancillary, output_ws=args.output,
         variables=args.vars, extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite)
