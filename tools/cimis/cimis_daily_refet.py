#--------------------------------
# Name:         cimis_daily_refet.py
# Purpose:      Calculate ETr from spatial CIMIS data
#--------------------------------

import argparse
import datetime as dt
import logging
import math
import os
import re
import sys

import drigo
import numpy as np
from osgeo import gdal
import refet
# from scipy import ndimage

import _utils


def main(start_dt, end_dt, img_ws, ancillary_ws, output_ws,
         etr_flag=False, eto_flag=False, extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False, use_cimis_eto_flag=False):
    """Compute daily ETr/ETo from CIMIS data

    Parameters
    ----------
    start_dt : datetime
        Start date.
    end_dt : datetime
        End date.
    img_ws : str
        Root folder of GRIDMET data.
    ancillary_ws : str
        Folder of ancillary rasters.
    output_ws : str
        Folder of output rasters.
    etr_flag : bool, optional
        If True, compute alfalfa reference ET (ETr).
    eto_flag : bool, optional
        If True, compute grass reference ET (ETo).
    extent_path : str, optional
        File path defining the output extent.
    output_extent : list, optional
        Decimal degrees values defining output extent.
    stats_flag : bool, optional
        If True, compute raster statistics (the default is True).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
    use_cimis_eto_flag : bool, optional
        If True, use CIMIS ETo raster if one of the component rasters is
        missing and ETo/ETr cannot be computed (te default is False).
        If False, this will only be applied if any of the inputs are missing
        and ETo can't be computed.

    Returns
    -------
    None

    """
    logging.info('\nComputing CIMIS ETo/ETr')
    logging.debug('  Start date: {}'.format(start_dt))
    logging.debug('  End date:   {}'.format(end_dt))

    np.seterr(invalid='ignore')

    # Compute ETr and/or ETo
    if not etr_flag and not eto_flag:
        logging.info('  ETo/ETr flag(s) not set, defaulting to ETr')
        etr_flag = True

    etr_folder = 'etr'
    eto_folder = 'eto'
    etr_fmt = 'etr_{}_daily_cimis.img'
    eto_fmt = 'eto_{}_daily_cimis.img'

    # DEM for air pressure calculation
    mask_raster = os.path.join(ancillary_ws, 'cimis_mask.img')
    dem_raster = os.path.join(ancillary_ws, 'cimis_elev.img')
    lat_raster = os.path.join(ancillary_ws, 'cimis_lat.img')
    # lon_raster = os.path.join(ancillary_ws, 'cimis_lon.img')

    # Interpolate zero windspeed pixels
    # interpolate_zero_u2_flag = False

    # Interpolate edge and coastal cells
    # interpolate_edge_flag = False

    # Resample type
    # 0 = GRA_NearestNeighbour, Nearest neighbour (select on one input pixel)
    # 1 = GRA_Bilinear,Bilinear (2x2 kernel)
    # 2 = GRA_Cubic, Cubic Convolution Approximation (4x4 kernel)
    # 3 = GRA_CubicSpline, Cubic B-Spline Approximation (4x4 kernel)
    # 4 = GRA_Lanczos, Lanczos windowed sinc interpolation (6x6 kernel)
    # 5 = GRA_Average, Average (computes the average of all non-NODATA contributing pixels)
    # 6 = GRA_Mode, Mode (selects the value which appears most often of all the sampled points)
    resample_type = gdal.GRA_CubicSpline

    # Wind speed is measured at 2m
    zw = 2

    # Output workspaces
    etr_ws = os.path.join(output_ws, etr_folder)
    eto_ws = os.path.join(output_ws, eto_folder)
    if etr_flag and not os.path.isdir(etr_ws):
        os.makedirs(etr_ws)
    if eto_flag and not os.path.isdir(eto_ws):
        os.makedirs(eto_ws)

    # Get CIMIS grid properties from mask
    cimis_mask_ds = gdal.Open(mask_raster)
    cimis_osr = drigo.raster_ds_osr(cimis_mask_ds)
    cimis_proj = drigo.osr_proj(cimis_osr)
    cimis_cs = drigo.raster_ds_cellsize(cimis_mask_ds, x_only=True)
    cimis_extent = drigo.raster_ds_extent(cimis_mask_ds)
    cimis_full_geo = cimis_extent.geo(cimis_cs)
    cimis_x, cimis_y = cimis_extent.origin()
    cimis_mask_ds = None
    logging.debug('  Projection: {}'.format(cimis_proj))
    logging.debug('  Cellsize: {}'.format(cimis_cs))
    logging.debug('  Geo: {}'.format(cimis_full_geo))
    logging.debug('  Extent: {}'.format(cimis_extent))

    # Manually set CIMIS grid properties
    # cimis_extent = drigo.Extent((-400000, -650000, 600000, 454000))
    # cimis_cs = 2000
    # cimis_geo = drigo.extent_geo(cimis_extent, cellsize)
    # cimis_epsg = 3310  # NAD_1983_California_Teale_Albers
    # cimis_x, cimis_y = (0,0)

    # Subset data to a smaller extent
    if output_extent is not None:
        logging.info('\nComputing subset extent & geo')
        logging.debug('  Extent: {}'.format(output_extent))
        cimis_extent = drigo.Extent(output_extent)
        cimis_extent.adjust_to_snap('EXPAND', cimis_x, cimis_y, cimis_cs)
        cimis_geo = cimis_extent.geo(cimis_cs)
        logging.debug('  Geo: {}'.format(cimis_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    elif extent_path is not None:
        logging.info('\nComputing subset extent & geo')
        if extent_path.lower().endswith('.shp'):
            cimis_extent = drigo.feature_path_extent(extent_path)
            extent_osr = drigo.feature_path_osr(extent_path)
            extent_cs = None
        else:
            cimis_extent = drigo.raster_path_extent(extent_path)
            extent_osr = drigo.raster_path_osr(extent_path)
            extent_cs = drigo.raster_path_cellsize(extent_path, x_only=True)
        cimis_extent = drigo.project_extent(
            cimis_extent, extent_osr, cimis_osr, extent_cs)
        cimis_extent.adjust_to_snap('EXPAND', cimis_x, cimis_y, cimis_cs)
        cimis_geo = cimis_extent.geo(cimis_cs)
        logging.debug('  Geo: {}'.format(cimis_geo))
        logging.debug('  Extent: {}'.format(cimis_extent))
    else:
        cimis_geo = cimis_full_geo

    # Latitude
    lat_array = drigo.raster_to_array(
        lat_raster, mask_extent=cimis_extent, return_nodata=False)
    lat_array = lat_array.astype(np.float32)
    lat_array *= math.pi / 180

    # Elevation data
    elev_array = drigo.raster_to_array(
        dem_raster, mask_extent=cimis_extent, return_nodata=False)
    elev_array = elev_array.astype(np.float32)

    # Process each year in the input workspace
    logging.info("")
    for year_str in sorted(os.listdir(img_ws)):
        logging.debug('{}'.format(year_str))
        if not re.match('^\d{4}$', year_str):
            logging.debug('  Not a 4 digit year folder, skipping')
            continue
        year_ws = os.path.join(img_ws, year_str)
        year_int = int(year_str)
        # year_days = int(dt.datetime(year_int, 12, 31).strftime('%j'))
        if start_dt is not None and year_int < start_dt.year:
            logging.debug('  Before start date, skipping')
            continue
        elif end_dt is not None and year_int > end_dt.year:
            logging.debug('  After end date, skipping')
            continue
        logging.info('{}'.format(year_str))

        # Output paths
        etr_raster = os.path.join(etr_ws, etr_fmt.format(year_str))
        eto_raster = os.path.join(eto_ws, eto_fmt.format(year_str))
        if etr_flag and (overwrite_flag or not os.path.isfile(etr_raster)):
            logging.debug('  {}'.format(etr_raster))
            drigo.build_empty_raster(
                etr_raster, band_cnt=366, output_dtype=np.float32,
                output_proj=cimis_proj, output_cs=cimis_cs,
                output_extent=cimis_extent, output_fill_flag=True)
        if eto_flag and (overwrite_flag or not os.path.isfile(eto_raster)):
            logging.debug('  {}'.format(eto_raster))
            drigo.build_empty_raster(
                eto_raster, band_cnt=366, output_dtype=np.float32,
                output_proj=cimis_proj, output_cs=cimis_cs,
                output_extent=cimis_extent, output_fill_flag=True)

        # Process each date in the year
        for date_str in sorted(os.listdir(year_ws)):
            logging.debug('{}'.format(date_str))
            try:
                date_dt = dt.datetime.strptime(date_str, '%Y_%m_%d')
            except ValueError:
                logging.debug(
                    '  Invalid folder date format (YYYY_MM_DD), skipping')
                continue
            if start_dt is not None and date_dt < start_dt:
                logging.debug('  Before start date, skipping')
                continue
            elif end_dt is not None and date_dt > end_dt:
                logging.debug('  After end date, skipping')
                continue
            logging.info(date_str)
            date_ws = os.path.join(year_ws, date_str)
            doy = int(date_dt.strftime('%j'))

            # Set file paths
            tmax_path = os.path.join(date_ws, 'Tx.img')
            tmin_path = os.path.join(date_ws, 'Tn.img')
            tdew_path = os.path.join(date_ws, 'Tdew.img')
            rso_path = os.path.join(date_ws, 'Rso.img')
            rs_path = os.path.join(date_ws, 'Rs.img')
            u2_path = os.path.join(date_ws, 'U2.img')
            eto_path = os.path.join(date_ws, 'ETo.img')
            # k_path = os.path.join(date_ws, 'K.img')
            # rnl_path = os.path.join(date_ws, 'Rnl.img')
            input_list = [
                tmin_path, tmax_path, tdew_path, u2_path, rs_path, rso_path]

            # If any input raster is missing, skip the day
            #   Unless ETo is present (and use_cimis_eto_flag is True)
            day_skip_flag = False
            for t_path in input_list:
                if not os.path.isfile(t_path):
                    logging.info('    {} is missing'.format(t_path))
                    day_skip_flag = True

            if (day_skip_flag and
                    use_cimis_eto_flag and
                    os.path.isfile(eto_path)):
                logging.info('    Using CIMIS ETo directly')
                eto_array = drigo.raster_to_array(
                    eto_path, 1, cimis_extent, return_nodata=False)
                eto_array = eto_array.astype(np.float32)
                if not np.any(eto_array):
                    logging.info('    {} is empty or missing'.format(eto_path))
                    logging.info('    Skipping date')
                    continue
                # ETr
                if etr_flag:
                    drigo.array_to_comp_raster(
                        1.2 * eto_array, etr_raster, band=doy,
                        stats_flag=False)
                    # drigo.array_to_raster(
                    #     1.2 * eto_array, etr_raster,
                    #     output_geo=cimis_geo, output_proj=cimis_proj,
                    #     stats_flag=stats_flag)
                # ETo
                if eto_flag:
                    drigo.array_to_comp_raster(
                        eto_array, eto_raster, band=doy, stats_flag=False)
                    # drigo.array_to_raster(
                    #     eto_array, eto_raster,
                    #     output_geo=cimis_geo, output_proj=cimis_proj,
                    #     stats_flag=stats_flag)
                del eto_array
                continue
            elif not day_skip_flag:
                # Read in rasters
                tmin_array = drigo.raster_to_array(
                    tmin_path, 1, cimis_extent, return_nodata=False)
                tmax_array = drigo.raster_to_array(
                    tmax_path, 1, cimis_extent, return_nodata=False)
                tdew_array = drigo.raster_to_array(
                    tdew_path, 1, cimis_extent, return_nodata=False)
                # rso_array = drigo.raster_to_array(
                #     rso_path, 1, cimis_extent, return_nodata=False)
                rs_array = drigo.raster_to_array(
                    rs_path, 1, cimis_extent, return_nodata=False)
                u2_array = drigo.raster_to_array(
                    u2_path, 1, cimis_extent, return_nodata=False)
                # k_array = drigo.raster_to_array(
                #     k_path, 1, cimis_extent, return_nodata=False)
                # rnl_array = drigo.raster_to_array(
                #     rnl_path, 1, cimis_extent, return_nodata=False)

                # Check that all input arrays have data
                for t_name, t_array in [
                        [tmin_path, tmin_array], [tmax_path, tmax_array],
                        [tdew_path, tdew_array], [u2_path, u2_array],
                        [rs_path, rs_array]]:
                    if not np.any(t_array):
                        logging.warning(
                            '    {} is empty or missing'.format(t_name))
                        day_skip_flag = True
                if day_skip_flag:
                    logging.warning('    Skipping date')
                    continue

                # # DEADBEEF - Some arrays have a 500m cellsize
                # # i.e. 2011-07-25, 2010-01-01 -> 2010-07-27
                # tmin_array = rescale_array_func(tmin_array, elev_array, 'tmin')
                # tmax_array = rescale_array_func(tmax_array, elev_array, 'tmax')
                # tdew_array = rescale_array_func(tdew_array, elev_array, 'tdew')
                # rso_array = rescale_array_func(rso_array, elev_array, 'rso')
                # rs_array = rescale_array_func(rs_array, elev_array, 'rs')
                # u2_array = rescale_array_func(u2_array, elev_array, 'u2')
                # # k_array = rescale_array_func(k_array, elev_array, 'k')
                # # rnl_array = rescale_array_func(rnl_array, elev_array, 'rnl')


                # Compute Ea from Tdew
                ea_array = refet.calcs._sat_vapor_pressure(tdew_array)

                # # Calculate q from tdew by first calculating ea from tdew
                # ea_array = refet.calcs._sat_vapor_pressure(tdew_array)
                # pair_array = refet.calcs.air_pressure_func(elev_array)
                # q_array = 0.622 * ea_array / (pair_array - (0.378 * ea_array))
                # del es_array, pair_array, tdew_array

                # # Calculate rhmin/rhmax from tdew
                # ea_tmax = refet._calcs._sat_vapor_pressure(tmax_array)
                # ea_tmin = refet._calcs._sat_vapor_pressure(tmin_array)
                # rhmin = ea_tdew * 2 / (ea_tmax + ea_tmin);
                # rhmax = ea_tdew * 2 / (ea_tmax + ea_tmin);
                # del ea_tmax, ea_tmin

                refet_obj = refet.Daily(
                    tmin=tmin_array, tmax=tmax_array, ea=ea_array,
                    rs=rs_array, uz=u2_array, zw=zw, elev=elev_array,
                    lat=lat_array, doy=doy, method='asce')
                # rso_type='ARRAY', rso=rso_array

                # ETr
                if etr_flag:
                    drigo.array_to_comp_raster(
                        refet_obj.etr().astype(np.float32), etr_raster,
                        band=doy, stats_flag=False)
                    # drigo.array_to_raster(
                    #     refet_obj.etr().astype(np.float32), etr_raster,
                    #     output_geo=cimis_geo, output_proj=cimis_proj,
                    #     stats_flag=stats_flag)
                # ETo
                if eto_flag:
                    drigo.array_to_comp_raster(
                        refet_obj.eto().astype(np.float32), eto_raster,
                        band=doy, stats_flag=False)
                    # drigo.array_to_raster(
                    #     refet_obj.eto().astype(np.float32), eto_raster,
                    #     output_geo=cimis_geo, output_proj=cimis_proj,
                    #     stats_flag=stats_flag)

                # Cleanup
                del tmin_array, tmax_array, u2_array, rs_array, ea_array
                # del rnl, rs, rso
            else:
                logging.info('    Skipping date')
                continue

        if stats_flag and etr_flag:
            drigo.raster_statistics(etr_raster)
        if stats_flag and eto_flag:
            drigo.raster_statistics(eto_raster)

    logging.debug('\nScript Complete')


# def rescale_array_func(input_array, test_array, input_name):
#     """
#
#     DEADBEEF - Some arrays have a 500m cellsize
#     i.e. 2011-07-25, 2010-01-01 -> 2010-07-27
#
#     """
#     a_rows, a_cols = input_array.shape
#     b_rows, b_cols = test_array.shape
#     if a_rows == b_rows and a_cols == b_cols:
#         return input_array
#     if (a_rows % b_rows == 0 and
#             a_cols % b_cols == 0 and
#             a_rows / b_rows == a_cols / b_cols):
#         scale_factor = float(b_rows) / a_rows
#     elif (b_rows % a_rows == 0 and
#             b_cols % a_cols == 0 and
#             b_rows / a_rows == b_cols / a_cols):
#         scale_factor = float(a_rows) / b_rows
#     logging.warning('    Rescaling {} array by {}'.format(
#         input_name, scale_factor))
#     return ndimage.zoom(input_array, scale_factor, order=1)


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
    img_folder = os.path.join(cimis_folder, 'input_img')
    ancillary_folder = os.path.join(cimis_folder, 'ancillary')

    parser = argparse.ArgumentParser(
        description='CIMIS daily ETr',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--start', required=True, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='Start date')
    parser.add_argument(
        '--end', required=True, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='End date')
    parser.add_argument(
        '--img', default=img_folder, metavar='PATH',
        help='Input IMG raster folder path')
    parser.add_argument(
        '--ancillary', default=ancillary_folder, metavar='PATH',
        help='Ancillary raster folder path')
    parser.add_argument(
        '--output', default=cimis_folder, metavar='PATH',
        help='Output raster folder path')
    parser.add_argument(
        '--etr', default=False, action="store_true",
        help='Compute alfalfa reference ET (ETr)')
    parser.add_argument(
        '--eto', default=False, action="store_true",
        help='Compute grass reference ET (ETo)')
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
        '--use_cimis_eto', default=False, action="store_true",
        help='Use CIMIS ETo if ETr/ETo cannot be computed')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.img and os.path.isdir(os.path.abspath(args.img)):
        args.img = os.path.abspath(args.img)
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

    main(start_dt=args.start, end_dt=args.end,
         img_ws=args.img, ancillary_ws=args.ancillary, output_ws=args.output,
         etr_flag=args.etr, eto_flag=args.eto,
         extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite,
         use_cimis_eto_flag=args.use_cimis_eto)
