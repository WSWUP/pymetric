#--------------------------------
# Name:         daymet_climatologies.py
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


def main(netcdf_ws=os.getcwd(), ancillary_ws=os.getcwd(),
         output_ws=os.getcwd(), variables=['prcp'],
         daily_flag=False, monthly_flag=True, annual_flag=False,
         start_year=1981, end_year=2010,
         extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Extract DAYMET temperature

    Parameters
    ----------
    netcdf_ws : str
        Folder of DAYMET netcdf files.
    ancillary_ws : str
        Folder of ancillary rasters.
    output_ws : str
        Folder of output rasters.
    variables : list, optional
        DAYMET variables to download ('prcp', 'srad', 'vp', 'tmmn', 'tmmx').
        Set as ['all'] to process all variables.
    daily_flag : bool, optional
        If True, compute daily (DOY) climatologies.
    monthly_flag : bool, optional
        If True, compute monthly climatologies.
    annual_flag : bool, optional
        If True, compute annual climatologies.
    start_year : int, optional
        Climatology start year.
    end_year : int, optional
        Climatology end year.
    extent_path : str, optional
        File path a raster defining the output extent.
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
    logging.info('\nGenerating DAYMET climatologies')

    daily_fmt = 'daymet_{var}_30yr_normal_{doy:03d}.img'
    monthly_fmt = 'daymet_{var}_30yr_normal_{month:02d}.img'
    annual_fmt = 'daymet_{var}_30yr_normal.img'
    # daily_fmt = 'daymet_{var}_normal_{start}_{end}_{doy:03d}.img'
    # monthly_fmt = 'daymet_{var}_normal_{start}_{end}_{month:02d}.img'
    # annual_fmt = 'daymet_{var}_normal_{start}_{end}.img'

    # If a date is not set, process 1981-2010 climatology
    try:
        start_dt = dt.datetime(start_year, 1, 1)
        logging.debug('  Start date: {}'.format(start_dt))
    except:
        start_dt = dt.datetime(1981, 1, 1)
        logging.info('  Start date: {}'.format(start_dt))
    try:
        end_dt = dt.datetime(end_year, 12, 31)
        logging.debug('  End date:   {}'.format(end_dt))
    except:
        end_dt = dt.datetime(2010, 12, 31)
        logging.info('  End date:   {}'.format(end_dt))

    # Get DAYMET spatial reference from an ancillary raster
    mask_raster = os.path.join(ancillary_ws, 'daymet_mask.img')

    daymet_re = re.compile('daymet_v3_(?P<VAR>\w+)_(?P<YEAR>\d{4})_na.nc4$')

    # DAYMET rasters to extract
    var_full_list = ['prcp', 'tmmn', 'tmmx']
    # data_full_list = ['prcp', 'srad', 'vp', 'tmmn', 'tmmx']
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
        output_extent = drigo.project_extent(
            drigo.raster_path_extent(extent_path),
            drigo.raster_path_osr(extent_path), daymet_osr,
            drigo.raster_path_cellsize(extent_path, x_only=True))
        output_extent = drigo.intersect_extents([daymet_extent, output_extent])
        output_extent.adjust_to_snap('EXPAND', daymet_x, daymet_y, daymet_cs)
        output_geo = output_extent.geo(daymet_cs)
        logging.debug('  Geo: {}'.format(output_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    else:
        output_extent = daymet_extent.copy()
        output_geo = daymet_geo[:]
    output_shape = output_extent.shape(cs=daymet_cs)
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
        logging.debug("Output name: {}".format(output_var))

        # Build output folder
        var_ws = os.path.join(output_ws, output_var)
        if not os.path.isdir(var_ws):
            os.makedirs(var_ws)

        # Build output arrays
        logging.debug('  Building arrays')
        if daily_flag:
            daily_sum = np.full(
                (365, output_shape[0], output_shape[1]), 0, np.float64)
            daily_count = np.full(
                (365, output_shape[0], output_shape[1]), 0, np.uint8)
        if monthly_flag:
            monthly_sum = np.full(
                (12, output_shape[0], output_shape[1]), 0, np.float64)
            monthly_count = np.full(
                (12, output_shape[0], output_shape[1]), 0, np.uint8)
        if monthly_flag:
            annual_sum = np.full(
                (output_shape[0], output_shape[1]), 0, np.float64)
            annual_count = np.full(
                (output_shape[0], output_shape[1]), 0, np.uint8)

        # Process each file/year separately
        for input_name in sorted(os.listdir(netcdf_ws)):
            logging.debug("  {}".format(input_name))
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
            if not os.path.isfile(input_raster):
                logging.debug(
                    '  Input raster doesn\'t exist, skipping    {}'.format(
                        input_raster))
                continue

            # Build output folder
            if daily_flag:
                daily_ws = os.path.join(var_ws, 'daily')
                if not os.path.isdir(daily_ws):
                    os.makedirs(daily_ws)

            if monthly_flag:
                monthly_temp_sum = np.full(
                    (12, output_shape[0], output_shape[1]), 0, np.float64)
                monthly_temp_count = np.full(
                    (12, output_shape[0], output_shape[1]), 0, np.uint8)

            # Read in the DAYMET NetCDF file
            input_nc_f = netCDF4.Dataset(input_raster, 'r')
            # logging.debug(input_nc_f.variables)

            # Check all valid dates in the year
            year_dates = _utils.date_range(
                dt.datetime(year_int, 1, 1), dt.datetime(year_int + 1, 1, 1))
            for date_dt in year_dates:
                logging.debug('  {}'.format(date_dt.date()))
                # if start_dt is not None and date_dt < start_dt:
                #     logging.debug(
                #         '  {} - before start date, skipping'.format(
                #             date_dt.date()))
                #     continue
                # elif end_dt is not None and date_dt > end_dt:
                #     logging.debug('  {} - after end date, skipping'.format(
                #         date_dt.date()))
                #     continue
                # else:
                #     logging.info('  {}'.format(date_dt.date()))

                doy = int(date_dt.strftime('%j'))
                doy_i = range(1, year_days + 1).index(doy)
                month_i = date_dt.month - 1

                # Arrays are being read as masked array with a -9999 fill value
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
                output_mask = np.isfinite(output_array)

                # Convert Kelvin to Celsius
                if input_var in ['tmax', 'tmin']:
                    output_array -= 273.15

                # Save values
                if daily_flag:
                    daily_sum[doy_i, :, :] += output_array
                    daily_count[doy_i, :, :] += output_mask
                if monthly_flag:
                    monthly_temp_sum[month_i, :, :] += output_array
                    monthly_temp_count[month_i, :, :] += output_mask
                if annual_flag:
                    annual_sum[:, :] += output_array
                    annual_count[:, :] += output_mask

                # Cleanup
                # del input_ds, input_array
                del input_ma, output_array, output_mask

            # Compute mean monthly for the year
            if monthly_flag:
                # Sum precipitation
                if input_var == 'prcp':
                    monthly_sum += monthly_temp_sum
                else:
                    monthly_sum += monthly_temp_sum / monthly_temp_count
                # Is this the right count?
                monthly_count += np.any(monthly_temp_count, axis=0)
                del monthly_temp_sum, monthly_temp_count

            input_nc_f.close()
            del input_nc_f

        # Save the projected climatology arrays
        if daily_flag:
            for doy_i in range(daily_sum.shape[0]):
                daily_name = daily_fmt.format(
                    var=output_var, start=start_year, end=end_year,
                    doy=doy_i + 1)
                daily_path = os.path.join(daily_ws, daily_name)
                drigo.array_to_raster(
                    daily_sum[doy_i, :, :] / daily_count[doy_i, :, :],
                    daily_path, output_geo=output_geo,
                    output_proj=daymet_proj, stats_flag=stats_flag)
            del daily_sum, daily_count
        if monthly_flag:
            for month_i in range(monthly_sum.shape[0]):
                monthly_name = monthly_fmt.format(
                    var=output_var, start=start_year, end=end_year,
                    month=month_i + 1)
                monthly_path = os.path.join(var_ws, monthly_name)
                drigo.array_to_raster(
                    monthly_sum[month_i, :, :] / monthly_count[month_i, :, :],
                    monthly_path, output_geo=output_geo,
                    output_proj=daymet_proj, stats_flag=stats_flag)
            del monthly_sum, monthly_count
        if annual_flag:
            annual_name = annual_fmt.format(
                var=output_var, start=start_year, end=end_year)
            annual_path = os.path.join(var_ws, annual_name)
            drigo.array_to_raster(
                annual_sum / annual_count, annual_path,
                output_geo=output_geo, output_proj=daymet_proj,
                stats_flag=stats_flag)
            del annual_sum, annual_count

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
        description='DAYMET climatologies',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--netcdf', default=os.path.join(daymet_folder, 'netcdf'),
        metavar='PATH', help='Input netCDF folder path')
    parser.add_argument(
        '--ancillary', default=os.path.join(daymet_folder, 'ancillary'),
        metavar='PATH', help='Ancillary raster folder path')
    parser.add_argument(
        '--output', default=daymet_folder,
        metavar='PATH', help='Output raster folder path')
    parser.add_argument(
        '--vars', default=['prcp'], nargs='+',
        choices=['prcp', 'tmmn', 'tmmx'],
        # choices=['prcp', 'srad', 'vp', 'tmmn', 'tmmx', 'all'],
        help='DAYMET variables to process')
    parser.add_argument(
        '--daily', default=False, action="store_true",
        help='Compute daily (DOY) climatologies')
    parser.add_argument(
        '--monthly', default=False, action="store_true",
        help='Compute monthly climatologies')
    parser.add_argument(
        '--annual', default=False, action="store_true",
        help='Compute annual climatologies')
    parser.add_argument(
        '--start', default=1981, type=int,
        help='Start date (format YYYY)', metavar='DATE')
    parser.add_argument(
        '--end', default=2010, type=int,
        help='End date (format YYYY)', metavar='DATE')
    parser.add_argument(
        '--extent', default=None, metavar='PATH',
        help='Subset extent raster path')
    parser.add_argument(
        '-te', default=None, type=float, nargs=4,
        metavar=('xmin', 'ymin', 'xmax', 'ymax'),
        help='Subset extent in decimal degrees')
    parser.add_argument(
        '--no_stats', default=False, action="store_true",
        help='Don\'t compute raster statistics')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
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

    main(netcdf_ws=args.netcdf, ancillary_ws=args.ancillary,
         output_ws=args.output, variables=args.vars,
         daily_flag=args.daily, monthly_flag=args.monthly,
         annual_flag=args.annual,
         start_year=args.start, end_year=args.end,
         extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite)
