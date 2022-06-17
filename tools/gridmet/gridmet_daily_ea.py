#--------------------------------
# Name:         gridmet_daily_ea.py
# Purpose:      Extract GRIDMET vapor pressure
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
import refet

import _utils

np.seterr(invalid='ignore')


def main(start_dt, end_dt, netcdf_ws, ancillary_ws, output_ws,
         extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Extract GRIDMET temperature

    Parameters
    ----------
    start_dt : datetime
        Start date.
    end_dt : datetime
        End date.
    netcdf_ws : str
        Folder of GRIDMET netcdf files.
    ancillary_ws : str
        Folder of ancillary rasters.
    output_ws : str
        Folder of output rasters.
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
    logging.info('\nExtracting GRIDMET vapor pressure')
    logging.debug('  Start date: {}'.format(start_dt))
    logging.debug('  End date:   {}'.format(end_dt))

    # Save GRIDMET lat, lon, and elevation arrays
    elev_raster = os.path.join(ancillary_ws, 'gridmet_elev.img')

    output_fmt = '{}_{}_daily_gridmet.img'
    gridmet_re = re.compile('(?P<VAR>\w+)_(?P<YEAR>\d{4}).nc$')

    # GRIDMET band name dictionary
    gridmet_band_dict = dict()
    gridmet_band_dict['pr'] = 'precipitation_amount'
    gridmet_band_dict['srad'] = 'surface_downwelling_shortwave_flux_in_air'
    gridmet_band_dict['sph'] = 'specific_humidity'
    gridmet_band_dict['tmmn'] = 'air_temperature'
    gridmet_band_dict['tmmx'] = 'air_temperature'
    gridmet_band_dict['vs'] = 'wind_speed'

    # Get extent/geo from elevation raster
    gridmet_ds = gdal.Open(elev_raster)
    gridmet_osr = drigo.raster_ds_osr(gridmet_ds)
    gridmet_proj = drigo.osr_proj(gridmet_osr)
    gridmet_cs = drigo.raster_ds_cellsize(gridmet_ds, x_only=True)
    gridmet_extent = drigo.raster_ds_extent(gridmet_ds)
    gridmet_full_geo = gridmet_extent.geo(gridmet_cs)
    gridmet_x, gridmet_y = gridmet_extent.origin()
    gridmet_ds = None
    logging.debug('  Projection: {}'.format(gridmet_proj))
    logging.debug('  Cellsize: {}'.format(gridmet_cs))
    logging.debug('  Geo: {}'.format(gridmet_full_geo))
    logging.debug('  Extent: {}'.format(gridmet_extent))

    # Subset data to a smaller extent
    if output_extent is not None:
        logging.info('\nComputing subset extent & geo')
        logging.debug('  Extent: {}'.format(output_extent))
        gridmet_extent = drigo.Extent(output_extent)
        gridmet_extent.adjust_to_snap(
            'EXPAND', gridmet_x, gridmet_y, gridmet_cs)
        gridmet_geo = gridmet_extent.geo(gridmet_cs)
        logging.debug('  Geo: {}'.format(gridmet_geo))
        logging.debug('  Extent: {}'.format(gridmet_extent))
    elif extent_path is not None:
        logging.info('\nComputing subset extent & geo')
        if not os.path.isfile(extent_path):
            logging.error(
                '\nThe extent object does not exist, exiting\n'
                '  {}'.format(extent_path))
            return False
        elif extent_path.lower().endswith('.shp'):
            gridmet_extent = drigo.feature_path_extent(extent_path)
            extent_osr = drigo.feature_path_osr(extent_path)
            extent_cs = None
        else:
            gridmet_extent = drigo.raster_path_extent(extent_path)
            extent_osr = drigo.raster_path_osr(extent_path)
            extent_cs = drigo.raster_path_cellsize(extent_path, x_only=True)
        gridmet_extent = drigo.project_extent(
            gridmet_extent, extent_osr, gridmet_osr, extent_cs)
        gridmet_extent.adjust_to_snap(
            'EXPAND', gridmet_x, gridmet_y, gridmet_cs)
        gridmet_geo = gridmet_extent.geo(gridmet_cs)
        logging.debug('  Geo: {}'.format(gridmet_geo))
        logging.debug('  Extent: {}'.format(gridmet_extent))
    else:
        gridmet_geo = gridmet_full_geo

    # Get indices for slicing/clipping input arrays
    g_i, g_j = drigo.array_geo_offsets(
        gridmet_full_geo, gridmet_geo, cs=gridmet_cs)
    g_rows, g_cols = gridmet_extent.shape(cs=gridmet_cs)

    # row_a, row_b = 585 - g_j, 585 - (g_j + g_rows),
    # col_a, col_b = g_i + g_cols, g_i,

    # Flip row indices since GRIDMET arrays are flipped up/down
    # Hard coding GRIDMET row count for now

    # Flipping is used for other GRIDMET netcdfs but ea appears to not need (
    # TODO: Why is this? Still seems flipped but ouputting data now
    row_a, row_b = g_j, (g_j + g_rows),
    col_a, col_b = g_i, (g_i + g_cols),

    # Read the elevation array
    elev_array = drigo.raster_to_array(
        elev_raster, mask_extent=gridmet_extent, return_nodata=False)
    pair_array = refet.calcs._air_pressure(elev_array)
    del elev_array

    # Process each variable
    input_var = 'sph'
    output_var = 'ea'
    logging.info("\nVariable: {}".format(input_var))

    # Build output folder
    var_ws = os.path.join(output_ws, output_var)
    if not os.path.isdir(var_ws):
        os.makedirs(var_ws)

    # Process each file in the input workspace
    for input_name in sorted(os.listdir(netcdf_ws)):
        input_match = gridmet_re.match(input_name)
        if not input_match:
            logging.debug("{}".format(input_name))
            logging.debug('  Regular expression didn\'t match, skipping')
            continue
        elif input_match.group('VAR') != input_var:
            logging.debug("{}".format(input_name))
            logging.debug('  Variable didn\'t match, skipping')
            continue
        else:
            logging.info("{}".format(input_name))

        year_str = input_match.group('YEAR')
        logging.info("  {}".format(year_str))
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
        #         '  Input NetCDF doesn\'t exist, skipping    {}'.format(
        #             input_raster))
        #     continue

        # Create a single raster for each year with 365 bands
        # Each day will be stored in a separate band
        output_path = os.path.join(
            var_ws, output_fmt.format(output_var, year_str))
        logging.debug('  {}'.format(output_path))
        if os.path.isfile(output_path):
            if not overwrite_flag:
                logging.debug('    File already exists, skipping')
                continue
            else:
                logging.debug('    File already exists, removing existing')
                os.remove(output_path)
        drigo.build_empty_raster(
            output_path, band_cnt=366, output_dtype=np.float32,
            output_proj=gridmet_proj, output_cs=gridmet_cs,
            output_extent=gridmet_extent, output_fill_flag=True)

        # Read in the GRIDMET NetCDF file
        # Immediately clip input array to save memory
        input_nc_f = netCDF4.Dataset(input_raster, 'r')
        input_nc = input_nc_f.variables[gridmet_band_dict[input_var]][
            :, row_a: row_b, col_a: col_b].copy()
        input_nc = np.flip(input_nc, 1)
        input_nc_f.close()
        del input_nc_f

        # A numpy array is returned when slicing a masked array
        #   if there are no masked pixels
        # This is a hack to force the numpy array back to a masked array
        if type(input_nc) != np.ma.core.MaskedArray:
            input_nc = np.ma.core.MaskedArray(
                input_nc, np.zeros(input_nc.shape, dtype=bool))

        # Check all valid dates in the year
        year_dates = _utils.date_range(
            dt.datetime(year_int, 1, 1), dt.datetime(year_int + 1, 1, 1))
        for date_dt in year_dates:
            if start_dt is not None and date_dt < start_dt:
                # logging.debug('  before start date, skipping')
                continue
            elif end_dt is not None and date_dt > end_dt:
                # logging.debug('  after end date, skipping')
                continue
            logging.info('  {}'.format(date_dt.strftime('%Y_%m_%d')))

            doy = int(date_dt.strftime('%j'))
            doy_i = range(1, year_days + 1).index(doy)

            # Arrays are being read as masked array with a fill value of -9999
            # Convert to basic numpy array arrays with nan values
            try:
                input_full_ma = input_nc[doy_i, :, :]
            except IndexError:
                logging.info('    date not in netcdf, skipping')
                continue
            input_full_array = input_full_ma.data.astype(np.float32)
            input_full_nodata = float(input_full_ma.fill_value)
            input_full_array[input_full_array == input_full_nodata] = np.nan

            # Since inputs are netcdf, need to create GDAL raster
            #   datasets in order to use gdal_common functions
            # Create an in memory dataset of the full ETo array
            input_full_ds = drigo.array_to_mem_ds(
                input_full_array, output_geo=gridmet_geo,
                output_proj=gridmet_proj)

            # Then extract the subset from the in memory dataset
            sph_array = drigo.raster_ds_to_array(
                input_full_ds, 1, mask_extent=gridmet_extent,
                return_nodata=False)

            # Compute ea [kPa] from specific humidity [kg/kg]
            ea_array = (sph_array * pair_array) / (0.622 + 0.378 * sph_array)

            # Save the projected array as 32-bit floats
            drigo.array_to_comp_raster(
                ea_array.astype(np.float32), output_path,
                band=doy, stats_flag=False)
            # drigo.array_to_raster(
            #     ea_array.astype(np.float32), output_path,
            #     output_geo=gridmet_geo, output_proj=gridmet_proj,
            #     stats_flag=False)
            del sph_array, ea_array

        if stats_flag:
            drigo.raster_statistics(output_path)

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
    netcdf_folder = os.path.join(gridmet_folder, 'netcdf')

    parser = argparse.ArgumentParser(
        description='GRIDMET daily vapor pressure',
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
        '--output', default=gridmet_folder, metavar='PATH',
        help='Output raster folder path')
    parser.add_argument(
        '--extent', default=None, metavar='PATH',
        help='Subset extent shapefile or raster path')
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
         extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite)
