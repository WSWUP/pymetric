#!/usr/bin/env python
#--------------------------------
# Name:         monte_carlo_func.py
# Purpose:      METRIC Automated Hot/Cold Pixel Selection
#--------------------------------

import argparse
from datetime import datetime
import logging
import os
import random
import re
import sys
from time import sleep

import drigo
import numpy as np

import pixel_points_func as pixel_points
import metric_model2_func as metric_model2
import auto_calibration_func as auto_calibration

import et_image
import python_common as dripy

break_line = '\n{}'.format('#' * 80)
pixel_str_fmt = '    {:<14s}  {:>14s}  {:>14s}'
pixel_flt_fmt = '    {:<14s}  {:>14.2f}  {:>14.2f}'
pixel_pct_fmt = '    {:<14s}  {:>14.2f}%  {:>13.2f}%'


def monte_carlo(image_ws, metric_ini_path, mc_ini_path, mc_iter=None,
                cold_tgt_pct=None, hot_tgt_pct=None,
                groupsize=64, blocksize=4096, multipoint_flag=False,
                shapefile_flag=False, stats_flag=False,
                overwrite_flag=False, debug_flag=False,
                no_etrf_final_plots=None, no_etrf_temp_plots=None):
    """METRIC Monte Carlo

    Parameters
    ----------
    image_ws : str
        The workspace (path) of the landsat scene folder.
    metric_ini_path : str
        The METRIC config file (path).
    mc_ini_path : str
        The Monte Carlo config file (path).
    mc_iter : int, optional
        Iteration number for Monte Carlo processing.
    cold_tgt_pct : float, optional
        Target percentage of pixels with ETrF > than cold Kc.
    hot_tgt_pct : float, optional
        Target percentage of pixels with ETrF < than hot Kc.
    groupsize : int, optional
        Script will try to place calibration point randomly
        into a labeled group of clustered values with at least n pixels.
        -1 = In the largest group
         0 = Anywhere in the image (not currently implemented)
         1 >= In any group with a pixel count greater or equal to n
    blocksize : int, optional
        Processing block size (the default is 4096).
    shapefile_flag : bool, optional
        If True, save calibration points to shapefile (the default is False).
    multipoint_flag : bool, optional
        If True, save cal. points to multipoint shapefile (the default is False).
    stats_flag : bool, optional
        If True, compute raster statistics (the default is False).
    ovewrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
    debug_flag : bool, optional
        If True, enable debug level logging (the default is False).
    no_final_plots : bool, optional
        If True, don't save final ETrF histograms (the default is None).
        This will override the flag in the INI file.
    no_temp_plots : bool
        If True, don't save temp ETrF histogram (the default is None).
        This will override the flag in the INI file.

    Returns
    -------
    None

    """
    logging.info('METRIC Automated Calibration')

    # Open config file
    config = dripy.open_ini(mc_ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    etrf_training_path = config.get('INPUTS', 'etrf_training_path')

    # Adjust Kc cold target value based on day of year
    # etrf_doy_adj_path = dripy.read_param(
    #     'etrf_doy_adj_path', None, config, 'INPUTS')

    # Intentionally set default to None, to trigger error in eval call
    kc_cold_doy_dict = dripy.read_param('kc_cold_doy_dict', None, config, 'INPUTS')
    kc_hot_doy_dict = dripy.read_param('kc_hot_doy_dict', None, config, 'INPUTS')

    # If the "no_" flags were set True, honor them and set the flag False
    # If the "no_" flags were not set by the user, use the INI flag values
    # If not set in the INI, default to False (don't save any plots)
    if no_etrf_temp_plots:
        save_etrf_temp_plots = False
    else:
        save_etrf_temp_plots = dripy.read_param(
            'save_etrf_temp_plots', False, config, 'INPUTS')
    if no_etrf_final_plots:
        save_etrf_final_plots = False
    else:
        save_etrf_final_plots = dripy.read_param(
            'save_etrf_final_plots', False, config, 'INPUTS')
    save_ndvi_plots = dripy.read_param('save_ndvi_plots', False, config, 'INPUTS')

    max_cal_iter = dripy.read_param('max_cal_iterations', 5, config, 'INPUTS')
    max_point_iter = dripy.read_param('max_point_iterations', 10, config, 'INPUTS')
    ts_diff_threshold = dripy.read_param('ts_diff_threshold', 4, config, 'INPUTS')
    etr_ws = config.get('INPUTS', 'etr_ws')
    ppt_ws = config.get('INPUTS', 'ppt_ws')
    etr_re = re.compile(config.get('INPUTS', 'etr_re'))
    ppt_re = re.compile(config.get('INPUTS', 'ppt_re'))
    awc_path = config.get('INPUTS', 'awc_path')
    spinup_days = dripy.read_param('swb_spinup_days', 5, config, 'INPUTS')
    min_spinup_days = dripy.read_param('swb_min_spinup_days', 30, config, 'INPUTS')

    log_fmt = '  {:<18s} {}'
    break_line = '\n{}'.format('#' * 80)

    env = drigo.env
    image = et_image.Image(image_ws, env)
    logging.info(log_fmt.format('Image:', image.folder_id))

    # Check inputs
    for file_path in [awc_path]:
        if not os.path.isfile(file_path):
            logging.error(
                '\nERROR: File {} does not exist'.format(file_path))
            sys.exit()
    for folder in [etr_ws, ppt_ws]:
        if not os.path.isdir(folder):
            logging.error(
                '\nERROR: Folder {} does not exist'.format(folder))
            sys.exit()
    # if (etrf_doy_adj_path and not
    #     os.path.isfile(etrf_doy_adj_path)):
    #     logging.error(
    #         '\nERROR: File {} does not exist.'.format(
    #             etrf_doy_adj_path))
    #     sys.exit()

    # Use iteration number to file iteration string
    if mc_iter is None:
        mc_str = ''
        mc_fmt = '.img'
    elif int(mc_iter) < 0:
        logging.error('\nERROR: Iteration number must be a positive integer')
        return False
    else:
        mc_str = 'MC{:02d}_'.format(int(mc_iter))
        mc_fmt = '_{:02d}.img'.format(int(mc_iter))
        logging.info('  {:<18s} {}'.format('Iteration:', mc_iter))

    # Folder names
    etrf_ws = os.path.join(image_ws, 'ETRF')
    # indices_ws = image.indices_ws
    region_ws = os.path.join(image_ws, 'PIXEL_REGIONS')
    pixels_ws = os.path.join(image_ws, 'PIXELS')
    plots_ws = os.path.join(image_ws, 'PLOTS')
    if shapefile_flag and not os.path.isdir(pixels_ws):
        os.mkdir(pixels_ws)
    if not os.path.isdir(plots_ws):
        os.mkdir(plots_ws)

    # File names
    r_fmt = '.img'
    etrf_path = os.path.join(etrf_ws, 'et_rf' + mc_fmt)
    region_path = os.path.join(region_ws, 'region_mask' + r_fmt)

    # Initialize calibration parameters dictionary
    logging.info(break_line)
    logging.info('Calibration Parameters')
    cal_dict = dict()

    logging.debug('  Reading target cold/hot Kc from INI')
    # Using eval is potentially a really bad way of reading this in
    try:
        kc_cold_doy_dict = eval('{' + kc_cold_doy_dict + '}')
    except:
        kc_cold_doy_dict = {1: 1.05, 366: 1.05}
        logging.info(
            '  ERROR: kc_cold_doy_dict was not parsed, using default values')
    try:
        kc_hot_doy_dict = eval('{' + kc_hot_doy_dict + '}')
    except:
        kc_hot_doy_dict = {1: 0.1, 366: 0.1}
        logging.info(
            '  ERROR: kc_hot_doy_dict was not parsed, using default values')
    logging.debug('  Kc cold dict: {}'.format(kc_cold_doy_dict))
    logging.debug('  Kc hot dict: {}\n'.format(kc_hot_doy_dict))
    # doy_cold, kc_cold = zip(*sorted(kc_cold_doy_dict.items()))
    cal_dict['cold_tgt_kc'] = np.interp(
        image.acq_doy, *zip(*sorted(kc_cold_doy_dict.items())),
        left=1.05, right=1.05)
    # doy_hot, kc_hot = zip(*sorted(kc_hot_doy_dict.items()))
    cal_dict['hot_tgt_kc'] = np.interp(
        image.acq_doy, *zip(*sorted(kc_hot_doy_dict.items())),
        left=0.1, right=0.1)

    # if etrf_doy_adj_path:
    #     doy_adj_df = pd.read_csv(etrf_doy_adj_path)
    #     doy_adj = float(
    #         doy_adj_df[doy_adj_df['DOY'] == image.acq_doy]['ETRF_ADJ'])
    #     cal_dict['cold_tgt_kc'] = cal_dict['cold_tgt_kc'] + doy_adj

    # Get hot/cold etrf fraction sizes
    if cold_tgt_pct is None or hot_tgt_pct is None:
        logging.info('ETrF Tail Size Percentages')
        logging.info('  Reading target tail size from file')
        cold_tgt_pct, hot_tgt_pct = auto_calibration.etrf_fractions(
            etrf_training_path)
        if cold_tgt_pct is None or hot_tgt_pct is None:
            logging.error('\nERROR: Tail sizes were not mannually set or '
                          'read from the the file\n')
            return False
    cal_dict['cold_tgt_pct'] = cold_tgt_pct
    cal_dict['hot_tgt_pct'] = hot_tgt_pct

    logging.info(pixel_str_fmt.format('', 'Cold Pixel', 'Hot Pixel'))
    logging.info(pixel_flt_fmt.format(
        'Target kc:', cal_dict['cold_tgt_kc'], cal_dict['hot_tgt_kc']))
    logging.info(pixel_pct_fmt.format(
        'Tail Size:', cal_dict['cold_tgt_pct'], cal_dict['hot_tgt_pct']))

    # # Create calibration database
    # # Set overwrite false to use existing database if it exists
    # cal_ws = os.path.join(image_ws, cal_folder)
    # if not os.path.isdir(cal_ws):
    #     os.mkdir(cal_ws)
    # cal_path = os.path.join(cal_ws, cal_name)
    # logging.info('{:<20s} {}\{}'.format(
    #     'Calibration DB:', cal_folder, cal_name))
    # calibration_database.create_calibration_database(
    #     image_ws, cal_path, overwrite_db_flag)
    # del cal_ws

    # Remove previous calibrations from database
    # logging.info(break_line)
    # calibration_database.remove_calibration_points(
    #     image_ws, cal_path, cal_initials, mc_iter)

    # Get ETrF and region mask (from pixel rating)
    # Assume they have identical extents
    try:
        region_mask = drigo.raster_to_array(region_path, return_nodata=False)
        region_mask = region_mask.astype(np.bool)
    except:
        logging.error(
            '\nERROR: Pixel regions mask does not exist or could not be read.\n'
            '  Please try re-running the METRIC Pixel Rating tool.')
        logging.debug('  {} '.format(region_path))
        return False

    # Remove previous plots
    logging.info(break_line)
    auto_calibration.remove_histograms(plots_ws, mc_iter)

    # Generate the NDVI histogram
    if save_ndvi_plots:
        logging.info(break_line)
        logging.info('NDVI Histograms')
        if os.path.isfile(image.ndvi_toa_raster):
            ndvi_array = drigo.raster_to_array(
                image.ndvi_toa_raster, return_nodata=False)
        else:
            logging.error(
                '\nERROR: NDVI raster does not exist. METRIC Model 1 may not '
                'have run successfully.')
            logging.debug('  {} '.format(image.ndvi_toa_raster))

        # Only process ag. ETrF pixels
        ndvi_array[~region_mask] = np.nan
        ndvi_sub_array = ndvi_array[region_mask]
        if np.any(ndvi_sub_array):
            auto_calibration.save_ndvi_histograms(
                ndvi_sub_array, plots_ws, mc_iter)
        else:
            logging.error(
                '\nERROR: Empty NDVI array, histogram was not built\n')

    # Place points in suggested region allowing for a number of iterations
    #  dependent on whether or not Ts meets certain criteria
    logging.info(break_line)
    pixel_point_iters = 0
    while pixel_point_iters <= max_point_iter:
        if pixel_point_iters == max_point_iter:
            logging.error(
                '\nERROR: Suitable hot and cold pixels could not be '
                'determined. The scene will not calibrate.\n')
            return False
        cold_xy, hot_xy = pixel_points.pixel_points(
            image_ws, groupsize=groupsize, blocksize=blocksize, mc_iter=mc_iter,
            shapefile_flag=shapefile_flag, multipoint_flag=multipoint_flag,
            overwrite_flag=overwrite_flag, pixel_point_iters=pixel_point_iters)
        if any(x is None for x in cold_xy) or any(x is None for x in hot_xy):
            logging.error((
                '\nPixel points coordinates are invalid.  '
                'The scene will not calibrate.'
                '\n  Cold: {}\n  Hot: {}').format(cold_xy, hot_xy))
            return False
        cold_ts = drigo.raster_value_at_xy(image.ts_raster, cold_xy)
        hot_ts = drigo.raster_value_at_xy(image.ts_raster, hot_xy)
        if cold_ts > hot_ts:
            logging.info(
                '\nThe cold pixel is hotter than the hot pixel. Placing '
                'the points again.\n')
            logging.info(break_line)
            pixel_point_iters += 1
        elif abs(hot_ts - cold_ts) < ts_diff_threshold:
            logging.info((
                '\nThere is less than a {} degree difference in Ts hot and cold. '
                'Placing the points again.\n').format(ts_diff_threshold))
            logging.info(break_line)
            pixel_point_iters += 1
            # raise et_common.TemperatureError
        else:
            break

    # Adjust Kc hot for soil water balance
    logging.info(break_line)
    cal_dict = auto_calibration.hot_kc_swb_adjust(
        cal_dict, hot_xy, env.snap_osr, image.acq_date, awc_path,
        etr_ws, etr_re, ppt_ws, ppt_re,
        spinup_days, min_spinup_days)
    # Adjust Kc cold based on NDVI
    # cal_dict['tgt_c_kc'] = auto_calibration.kc_ndvi_adjust(
    #     cal_dict['tgt_c_kc'], cold_xy, ndvi_path, 'Cold')

    # Check that Kc hot (Ke) is not too high?
    if cal_dict['hot_tgt_kc'] >= 1.0:
        logging.error(
            '\nERROR: Target Kc hot is too high for automated '
            'calibration\n  ETrF will not be computed')
        return False
    elif (cal_dict['cold_tgt_kc'] - cal_dict['hot_tgt_kc']) <= 0.05:
        logging.error(
            '\nERROR: Target Kc hot and Kc cold are too close for '
            'automated calibration\n  ETrF will not be computed')
        return False

    # Initialize Kc values at targets
    cal_dict['kc_cold'] = cal_dict['cold_tgt_kc']
    cal_dict['kc_hot'] = cal_dict['hot_tgt_kc']

    # Iterate until max calibrations is reached or error is small
    cal_flag = False
    cal_iter = 1
    while not cal_flag:
        logging.info(break_line)
        logging.info('Calibration Iteration: {}'.format(cal_iter))

        # Run METRIC Model2 for initial ETrF map
        logging.info(break_line)
        metric_model2.metric_model2(
            image_ws, metric_ini_path, mc_iter=mc_iter,
            kc_cold=cal_dict['kc_cold'], kc_hot=cal_dict['kc_hot'],
            cold_xy=cold_xy, hot_xy=hot_xy,
            overwrite_flag=overwrite_flag)

        # Read in ETrF array
        if os.path.isfile(etrf_path):
            etrf_array = drigo.raster_to_array(etrf_path, return_nodata=False)
        else:
            logging.warning((
                'WARNING: ETrF raster does not exist. METRIC Model 2 '
                'may not have run successfully.\n {}').format(etrf_path))
            break
        etrf_geo = drigo.raster_path_geo(etrf_path)

        # Only process ag. ETrF pixels
        etrf_array[~region_mask] = np.nan
        etrf_sub_array = etrf_array[np.isfinite(etrf_array)]
        if not np.any(etrf_sub_array):
            logging.error(
                '\nERROR: Empty ETrF array, scene cannot be calibrated\n')
            break

        # Calculate calibration parameters
        logging.debug(break_line)
        cal_dict = auto_calibration.calibration_params(
            cal_dict, etrf_sub_array)

        # Plot intermediates calibration histograms
        if save_etrf_temp_plots:
            logging.debug(break_line)
            auto_calibration.save_etrf_histograms(
                etrf_sub_array, plots_ws, cal_dict,
                mc_iter, cal_iter)

        # Check calibration
        logging.debug(break_line)
        cal_flag = auto_calibration.check_calibration(cal_dict)

        # Don't re-calibrate if initial calibration was suitable
        if cal_flag:
            break
        # Limit calibration attempts
        # cal_iter index is 1 based for Monte Carlo
        # cal_iter is 0 for stand alone mode
        elif cal_iter >= max_cal_iter:
            logging.info(break_line)
            logging.info((
                'All {} iteration attempts were made, '
                'the scene will not calibrate.').format(max_cal_iter))
            if os.path.isfile(etrf_path):
                os.remove(etrf_path)
            return False
            # break

        # Adjust Kc value of calibration points (instead of moving them)
        cal_dict = auto_calibration.kc_calibration_adjust(
            cal_dict, etrf_sub_array)

        # # Select new calibration points based on ETrF distribution
        # logging.info(break_line)
        # cold_xy, hot_xy = auto_calibration.build_pixel_points(
        #     etrf_array, etrf_geo, cal_dict,
        #     shapefile_flag=shapefile_flag, pixels_ws=pixels_ws)

        del etrf_array, etrf_geo

        # Increment calibration iteration counter
        cal_iter += 1

    # Only save 'final' results if the scene was calibrated
    if cal_flag and save_etrf_final_plots:
        # Plot final ETrF distribution
        # logging.info(break_line)
        auto_calibration.save_etrf_histograms(
            etrf_sub_array, plots_ws, cal_dict, mc_iter, None)

        # Save final calibration points to database
        # logging.info(break_line)
        # calibration_database.save_calibration_points(
        #     image_ws, cal_path, cal_dict, mc_iter, 0)

    return True


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Monte Carlo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-bs', '--blocksize', default=4096, type=int,
        help='Block size for selecting calibration points')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-gs', '--groupsize', default=64, type=int,
        help='Minimum group size for placing calibration points')
    parser.add_argument(
        '-mc', '--iter',
        default=None, type=int, metavar='N',
        help='Monte Carlo iteration number')
    parser.add_argument(
        '--metric_ini', required=True, type=dripy.arg_valid_file,
        help='METRIC input file', metavar='FILE')
    parser.add_argument(
        '--mc_ini', required=True, type=dripy.arg_valid_file,
        help='Monte Carlo input file', metavar='FILE')
    parser.add_argument(
        '-m', '--multipoint', default=False, action="store_true",
        help='Save calibration points to multipoint shapeifle')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '--no_final_plots', default=False, action="store_true",
        help="Don't save final ETrF histogram plots")
    parser.add_argument(
        '--no_temp_plots', default=False, action="store_true",
        help="Don't save temporary ETrF histogram plots")
    parser.add_argument(
        '-o', '--overwrite',
        default=None, action='store_true',
        help='Force overwrite of existing files')
    parser.add_argument(
        '-s', '--shapefile', default=False, action='store_true',
        help='Save calibration points to shapefile')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
    parser.add_argument(
        '-t', '--tails', default=[None, None], type=float, nargs=2,
        help='Cold and hot tail sizes', metavar=('COLD', 'HOT'))
    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.workspace and os.path.isdir(os.path.abspath(args.workspace)):
        args.workspace = os.path.abspath(args.workspace)
    if args.metric_ini and os.path.isfile(os.path.abspath(args.metric_ini)):
        args.metric_ini = os.path.abspath(args.metric_ini)
    if args.mc_ini and os.path.isfile(os.path.abspath(args.mc_ini)):
        args.mc_ini = os.path.abspath(args.mc_ini)

    return args


if __name__ == '__main__':
    args = arg_parse()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_console = logging.StreamHandler()
    log_console.setLevel(args.loglevel)
    formatter = logging.Formatter('%(message)s')
    log_console.setFormatter(formatter)
    logger.addHandler(log_console)

    if not args.no_file_logging:
        if args.iter is not None:
            log_file_name = 'mc{:02d}_log.txt'.format(int(args.iter))
        else:
            log_file_name = 'monte_carlo_log.txt'
        log_file = logging.FileHandler(
            os.path.join(args.workspace, log_file_name), "w")
        log_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

    logging.info('\n{}'.format('#' * 80))
    log_fmt = '{:<20s} {}'
    logging.info(log_fmt.format('Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_fmt.format('Current Directory:', args.workspace))
    logging.info(log_fmt.format('Script:', os.path.basename(sys.argv[0])))
    logging.info('')

    # Delay
    sleep(random.uniform(0, max([0, abs(args.delay)])))

    # METRIC Monte Carlo
    monte_carlo(image_ws=args.workspace, metric_ini_path=args.metric_ini,
                mc_ini_path=args.mc_ini, mc_iter=args.iter,
                cold_tgt_pct=args.tails[0], hot_tgt_pct=args.tails[1],
                groupsize=args.groupsize, blocksize=args.blocksize,
                multipoint_flag=args.multipoint, shapefile_flag=args.shapefile,
                stats_flag=args.stats, overwrite_flag=args.overwrite,
                debug_flag=args.loglevel==logging.DEBUG,
                no_etrf_final_plots=args.no_final_plots,
                no_etrf_temp_plots=args.no_temp_plots)
