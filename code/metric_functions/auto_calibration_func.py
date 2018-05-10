#!/usr/bin/env python
#--------------------------------
# Name:         auto_calibration_func.py
# Purpose:      METRIC Automated Calibration based on ETRF distribution
#--------------------------------

import argparse
from datetime import datetime
import logging
import os
import random
import re
import string
import sys

import drigo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import et_common
from python_common import remove_file

break_line = '\n{}'.format('#' * 80)
pixel_str_fmt = '    {:<12s}{:>14s}  {:>14s}'
pixel_flt_fmt = '    {:<12s}{:>14.2f}  {:>14.2f}'


def etrf_fractions(etrf_training_path):
    """Generate a random ETrF target percents from the training data CDF

    Parameters
    ----------
    etrf_training_path : str

    Returns
    -------
    tuple of ETrF target percentages (cold, hot)

    """
    # Load etrf training data
    try:
        etrf_data = np.recfromcsv(etrf_training_path)
    except IOError:
        logging.error(
            ('\nERROR: The ETrF training data ({}) does not exist ' +
             'or could not be read\n').format(etrf_training_path))
        return None, None

    cold_etrf_data = np.insert(np.sort(etrf_data['kc_cld_pct']), 0, 0)
    hot_etrf_data = np.insert(np.sort(etrf_data['kc_hot_pct']), 0, 0)

    def probability_array_func(data_array):
        array_len = len(data_array)
        return np.linspace(0., array_len, array_len) / array_len
    cold_etrf_prob = probability_array_func(cold_etrf_data)
    hot_etrf_prob = probability_array_func(hot_etrf_data)

    # Seed the random number generator with mc_iter number?
    # np.random.seed(mc_iter)
    # Use uniform random sample of probability to look up tail size
    cold_tgt_prob = np.random.uniform()
    hot_tgt_prob = np.random.uniform()
    # Stratified sampling (Ones = Cold, Tens = Hot)
    # cold_tgt_prob = np.random.uniform(
    #     (iteration % 100 % 10) * 0.1,
    #     (iteration % 100 % 10) * 0.1 + 0.1)
    # hot_tgt_prob = np.random.uniform(
    #     (iteration % 100 / 10) * 0.1,
    #     (iteration % 100 / 10) * 0.1 + 0.1)

    # Get hot/cold tail size (percentage)
    cold_tgt_pct = np.interp(cold_tgt_prob, cold_etrf_prob, cold_etrf_data)
    hot_tgt_pct = np.interp(hot_tgt_prob, hot_etrf_prob, hot_etrf_data)
    return cold_tgt_pct, hot_tgt_pct


def sap_float(array, percentile):
    """Score at percentile

    Parameters
    ----------
    array :
    percentile : float

    Returns
    -------
    float

    """
    # DEADBEEF - consider adding a test for an empty array
    return float(stats.scoreatpercentile(
        array[np.isfinite(array)], percentile))


def pos_float(array, score):
    """Percentile of score

    Parameters
    ----------
    array
    score : float

    Returns
    -------
    float

    """
    return float(stats.percentileofscore(array[np.isfinite(array)], score))


def calibration_params(cal_dict, etrf_array):
    """Check the calibration

    Parameters
    ----------
    cal_dict : dict
        Calibration parameters.
    etrf_array :
        The ETrF array (assuming with non-ag pixels were masked).

    Returns
    -------
    dict

    """
    logging.debug('Calibration Parameters')

    # Get pixel count
    etrf_count = int(len(etrf_array))
    logging.debug('\nPIXEL COUNT: {}'.format(etrf_count))

    # Tail percentages as the target Kc values
    cal_dict['0_kc_pct'] = pos_float(etrf_array, 0)
    # Tail percentages as the target Kc values
    cal_dict['cold_tgt_kc_pct'] = 100 - pos_float(
        etrf_array, cal_dict['cold_tgt_kc'])
    cal_dict['hot_tgt_kc_pct'] = pos_float(
        etrf_array, cal_dict['hot_tgt_kc'])
    # Kc values corresponding to the target Kc values
    cal_dict['cold_tgt_pct_kc'] = sap_float(
        etrf_array, 100 - cal_dict['cold_tgt_pct'])
    cal_dict['hot_tgt_pct_kc'] = sap_float(
        etrf_array, cal_dict['hot_tgt_pct'])
    # Max/Min Kc
    cal_dict['min_kc'] = sap_float(etrf_array, 0)
    cal_dict['max_kc'] = sap_float(etrf_array, 100)
    # Number of pixels used for histogram
    cal_dict['pixelcount'] = etrf_count

    # Print percentile/score
    logging.debug('\nETrF Percentiles')
    percentile_list = []
    percentile_list.append([0., cal_dict['min_kc'], '(min etrf)'])
    percentile_list.append([100., cal_dict['max_kc'], '(max etrf)'])
    percentile_list.append([cal_dict['0_kc_pct'], 0, '(etrf fraction <0)'])
    percentile_list.append([
        cal_dict['hot_tgt_pct'], cal_dict['hot_tgt_pct_kc'],
        '(hot threshold)'])
    percentile_list.append([
        cal_dict['hot_tgt_kc_pct'], cal_dict['hot_tgt_kc'],
        '(etrf fraction <{:4.2f})'.format(cal_dict['hot_tgt_kc'])])
    percentile_list.append([
        100 - cal_dict['cold_tgt_pct'], cal_dict['cold_tgt_pct_kc'],
        '(cold threshold)'])
    percentile_list.append([
        100 - cal_dict['cold_tgt_kc_pct'], cal_dict['cold_tgt_kc'],
        '(etrf fraction >{:4.2f})'.format(cal_dict['cold_tgt_kc'])])
    percentile_list.sort()
    for percentile, etrf, name in percentile_list:
        logging.debug(' {:6.2f}%: {:9.6f}  {}'.format(
            percentile, etrf, name))
    return cal_dict


def check_calibration(cal_dict):
    """Check the calibration

    Parameters
    ----------
    cal_dict : dict
        Calibration parameters.

    Returns
    -------
    bool

    """
    logging.debug('Check Calibration')
    c_error, h_error = 0.5, 0.5

    # Check percentiles
    if (abs((cal_dict['cold_tgt_pct']) - cal_dict['cold_tgt_kc_pct']) <= c_error and
        abs((cal_dict['hot_tgt_pct']) - cal_dict['hot_tgt_kc_pct']) <= h_error):
        logging.info(break_line)
        logging.info(
            ('Scene Calibrated - ETrF fractions are within ' +
             '{:4.2f}% (cold) and {:4.2f}% (hot)').format(c_error, h_error))
        return True
    else:
        return False


def kc_calibration_adjust(cal_dict, etrf_array):
    """Adjust Kc values to target Kc values instead of moving calibration points

    Parameters
    ----------
    cal_dict : dict
        Calibration parameters.
    etrf_array
        The ETrF array (assuming with non-ag pixels were masked).

    Returns
    -------
    dict

    """
    logging.info('Adjusting Kc values at hot/cold calibration points')

    logging.info(pixel_str_fmt.format('', 'Cold Pixel', 'Hot Pixel'))
    logging.info(pixel_flt_fmt.format(
        'Kc:', cal_dict['kc_cold'], cal_dict['kc_hot']))
    # cal_dict['kc_hot'] = cal_dict['hot_tgt_kc']
    # cal_dict['kc_cold'] = cal_dict['cold_tgt_kc']

    # To shift the histogram to the left at the hot pixel
    #   (i.e. need more pixels below Kc hot threshold)
    #   the hot pixel Kc must be set lower
    #   (i.e. scale low value pixels down more than high value pixels)

    # Stretch histogram from cold Kc
    cal_dict['kc_hot'] = (
        ((cal_dict['hot_tgt_kc'] - cal_dict['cold_tgt_kc']) /
         (cal_dict['hot_tgt_pct_kc'] - cal_dict['cold_tgt_kc'])) *
        (cal_dict['kc_hot'] - cal_dict['cold_tgt_kc']) + cal_dict['cold_tgt_kc'])
    # Stretch histogram from maximum Kc
    # cal_dict['kc_hot'] = (
    #     ((cal_dict['hot_tgt_kc'] - cal_dict['max_kc']) /
    #      (cal_dict['hot_tgt_pct_kc'] - cal_dict['max_kc'])) *
    #    (cal_dict['kc_hot'] - cal_dict['max_kc']) + cal_dict['max_kc'])

    # To shift the histogram to the right at the cold pixel
    #   (i.e. need more pixels above Kc cold threshold)
    #   the cold pixel Kc must be set higher
    #   (i.e. scall high value pixels up more than low value pixels)

    # Stretch histogram from hot Kc
    cal_dict['kc_cold'] = (
        ((cal_dict['cold_tgt_kc'] - cal_dict['hot_tgt_kc']) /
         (cal_dict['cold_tgt_pct_kc'] - cal_dict['hot_tgt_kc'])) *
        (cal_dict['kc_cold'] - cal_dict['hot_tgt_kc']) + cal_dict['hot_tgt_kc'])

    # Stretch histogram from minimum Kc
    # cal_dict['kc_cold'] = (
    #     ((cal_dict['cold_tgt_kc'] - cal_dict['min_kc']) /
    #      (cal_dict['cold_tgt_pct_kc'] - cal_dict['min_kc'])) *
    #     (cal_dict['kc_cold'] - cal_dict['min_kc']) + cal_dict['min_kc'])

    logging.info(pixel_flt_fmt.format(
        'Kc:', cal_dict['kc_cold'], cal_dict['kc_hot']))
    return cal_dict


def remove_histograms(plots_ws, mc_iter=None):
    """Delete histograms

    Parameters
    ----------
    plots_ws : str
        Folder containing histogram plots
    mc_iter : int, optional
        Current monte carlo iteration (the default is None).

    Returns
    -------
    None

    """
    logging.info('Deleting Histograms')

    # mc_iter is None when run in stand alone mode
    # mc_iter index is 1 based when run in Monte Carlo mode
    if mc_iter is None:
        mc_image = ''
    else:
        mc_image = '_{:02d}'.format(int(mc_iter))

    plot_re = re.compile('(ETrF|NDVI)({0})?(_\w)?.png'.format(mc_image), re.I)
    for item in os.listdir(plots_ws):
        if plot_re.match(item):
            # logging.info('Remove: {}'.format(item))
            try:
                os.remove(os.path.join(plots_ws, item))
            except:
                logging.info('  Could not delete {}'.format(item))


def calc_histogram_bins(value_array, bin_size=0.01):
    """Calculate histogram bins

    Parameters
    ----------
    etrf_array :
    bin_size : float, optional
        Histogram bin size (the default is 0.01).

    Returns
    -------
    NumPy array

    """
    bin_min = np.floor(np.nanmin(value_array) / bin_size) * bin_size
    bin_max = np.ceil(np.nanmax(value_array) / bin_size) * bin_size
    bin_count = round((bin_max - bin_min) / bin_size) + 1
    bin_array = np.linspace(bin_min, bin_max, bin_count)
    logging.debug('  Histogram bins')
    logging.debug('    min: {}'.format(bin_min))
    logging.debug('    max: {}'.format(bin_max))
    logging.debug('    count: {}'.format(bin_count))
    # logging.debug('    bins: {}'.format(bin_array))
    return bin_array


def save_etrf_histograms(etrf_array, plots_ws, cal_dict,
                         mc_iter=None, cal_iter=None):
    """Plot ETRF histogram

    Parameters
    ----------
    etrf_array : str
        The ETrF array (assuming with non-ag pixels were masked).
    plots_ws : str
        The folder for saving the ETrF histogram plots.
    cal_dict : dict
        Calibration parameters.
    mc_iter : int, optional
        The current monte carlo iteration (the default is None).
    cal_iter : int, optional
        The current monte carlo iteration (the default is None).

    Returns
    -------
    None

    """
    logging.debug('Save ETrF Histograms')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    # mc_iter is None when run in stand alone mode
    # mc_iter index is 1 based when run in Monte Carlo mode
    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {:02d}'.format(int(mc_iter))
        mc_image = '_{:02d}'.format(int(mc_iter))

    # cal_iter is None when run in stand alone mode
    # cal_iter index is 1 based when run in Monte Carlo mode
    if cal_iter is None or cal_iter == 0:
        cal_iter_title = ''
        cal_iter_image = ''
    else:
        # Retrieve letter using cal_iter as the index number (0: A, 1: B, etc)
        cal_iter_title = ' - {}'.format(string.ascii_uppercase[cal_iter - 1])
        cal_iter_image = '_{}'.format(string.ascii_uppercase[cal_iter - 1])

    etrf_bins = calc_histogram_bins(etrf_array, 0.01)

    # Save historgram on check run
    plt.figure()
    n, bins, patches = plt.hist(etrf_array, bins=etrf_bins)
    plt.title('ETrF - {}{}{}'.format(
              scene_id, mc_title, cal_iter_title))
    plt.xlabel('ETrF')
    plt.ylabel('# of agricultural pixels')
    plt.xlim(-0.2, 1.4)
    # plt.ylim(0,6000)
    pct_format_str = 'Pct: {:5.2f}%\nETrF: {:5.3f}'
    kc_format_str = 'ETrF: {:5.3f}\nPct: {:5.2f}%'
    plt.figtext(
        0.15, 0.825, pct_format_str.format(cal_dict['0_kc_pct'], 0),
        horizontalalignment='left', size='small')
    plt.figtext(
        0.30, 0.825, kc_format_str.format(
            cal_dict['hot_tgt_kc'], cal_dict['hot_tgt_kc_pct']),
        horizontalalignment='left', size='small')
    plt.figtext(
        0.30, 0.76, pct_format_str.format(
            cal_dict['hot_tgt_pct'], cal_dict['hot_tgt_pct_kc']),
        horizontalalignment='left', size='small')
    plt.figtext(
        0.77, 0.825, kc_format_str.format(
            cal_dict['cold_tgt_kc'], cal_dict['cold_tgt_kc_pct']),
        horizontalalignment='left', size='small')
    plt.figtext(
        0.77, 0.76, pct_format_str.format(
            cal_dict['cold_tgt_pct'], cal_dict['cold_tgt_pct_kc']),
        horizontalalignment='left', size='small')
    plt.axvline(
        cal_dict['hot_tgt_pct_kc'], ymin=0, ymax=0.8,
        linewidth=1.0, color='r', linestyle='--')
    plt.axvline(
        cal_dict['cold_tgt_pct_kc'], ymin=0, ymax=0.8,
        linewidth=1.0, color='r', linestyle='--')
    plt.axvline(
        cal_dict['hot_tgt_kc'], ymin=0, ymax=0.8,
        linewidth=1.5, color='r', linestyle='-')
    plt.axvline(
        cal_dict['cold_tgt_kc'], ymin=0, ymax=0.8,
        linewidth=1.5, color='r', linestyle='-')
    plt.savefig(os.path.join(
        plots_ws, 'ETrF{}{}.png'.format(
            mc_image, cal_iter_image)))
    # plt.show()
    plt.close()


def save_ndvi_histograms(ndvi_array, plots_ws, mc_iter=None):
    """Plot NDVI histogram

    Parameters
    ----------
    ndvi_array :
        The NDVI array (assuming non-ag pixels were masked).
    plots_ws : str
        The folder for saving the ETrF histogram plots.
    mc_iter : int, optional
        The current monte carlo iteration (default None).

    Returns
    -------
    None

    """
    logging.debug('Save NDVI Histograms')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    # mc_iter is None when run in stand alone mode
    # mc_iter index is 1 based when run in Monte Carlo mode
    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {:02d}'.format(int(mc_iter))
        mc_image = '_{:02d}'.format(int(mc_iter))

    # Save historgram on check run
    plt.figure()
    ndvi_bins = calc_histogram_bins(ndvi_array, 0.01)
    n, bins, patches = plt.hist(ndvi_array, bins=ndvi_bins)
    plt.title('NDVI - {}{}'.format(scene_id, mc_title))
    plt.xlabel('NDVI')
    plt.ylabel('# of agricultural pixels')
    plt.xlim(-0.2, 1.2)
    # plt.ylim(0,6000)
    plt.savefig(os.path.join(
        plots_ws, 'NDVI{}.png'.format(mc_image)))
    # plt.show()
    plt.close()


def build_pixel_points(etrf_array, etrf_geo, cal_dict,
                       shapefile_flag=False, pixels_ws=None, mc_iter=None):
    """Build new suggested hot/cold pixels

    Parameters
    ----------
    etrf_array :
        The ETrF array (assuming non-ag pixels were masked).
    cal_dict : dict
        Calibration parameters.
    shapefile_flag : bool
        If True, save calibration points to shapefile (the default is False)
    pixels_ws : str

    Returns
    -------

    """
    logging.info('Building new hot/cold calibration points')

    # Suggested pixel points
    cold_pixel_path = os.path.join(pixels_ws, 'cold.shp')
    hot_pixel_path = os.path.join(pixels_ws, 'hot.shp')
    if shapefile_flag:
        remove_file(cold_pixel_path)
        remove_file(hot_pixel_path)

    # #  Limit etrf raster to just correct NDVI ranges?
    # apply_ndvi_mask = True
    # if apply_ndvi_mask:
    #     etrf_cold_array = np.copy(etrf_array)
    #     etrf_hot_array = np.copy(etrf_array)
    #     etrf_cold_array[ndvi_array < 0.4] = np.nan
    #     etrf_hot_array[ndvi_array > 0.4] = np.nan

    # Select target pixel
    cold_x, cold_y = get_target_point_in_array(
        etrf_array, etrf_geo, cal_dict['tgt_c_kc'])
    hot_x, hot_y = get_target_point_in_array(
        etrf_array, etrf_geo, cal_dict['tgt_h_kc'])

    # Save pixels
    if shapefile_flag and cold_x and cold_y:
        drigo.save_point_to_shapefile(cold_pixel_path, cold_x, cold_y)
    if shapefile_flag and hot_x and hot_y:
        drigo.save_point_to_shapefile(hot_pixel_path, hot_x, hot_y)

    # Eventually don't save a shapefile and just return the coordinates
    return (cold_x, cold_y), (hot_x, hot_y)


def get_target_point_in_array(input_array, input_geo, target_value):
    """Return the indices for a point in the array with the target value

    Parameters
    ----------
    input_array :
        input numpy array
    input_geo :
        Geo transform of the input array.
    target_value :

    Returns
    -------

    """

    # Get index of cell with a value closest to target
    target_array = np.abs(input_array - float(target_value))
    yi, xi = np.where(target_array == np.nanmin(target_array))
    # Calculate random cell x/y
    return drigo.array_offsets_xy(input_geo, random.choice(list(zip(xi, yi))))


def hot_kc_swb_adjust(cal_dict, hot_xy, hot_osr, scene_date, awc_path,
                      etr_ws, etr_re, ppt_ws, ppt_re,
                      spinup_days, min_spinup_days):
    """Adjust Kc values based on the daily soil water balance

    Parameters
    ----------
    cal_dict : dict
        Calibration parameters.
    hot_xy : tuple
        Location of the hot calibration point.
    hot_osr : class:`osr.SpatialReference
        Spatial reference of the hot calibration point.
    scene_date : datetime
    awc_path : str
        File path of the available water content raster.
    etr_ws : str
        Directory path of the ETr workspace, which contains separate rasters
        for each year.
    etr_re : class:`re`
        Compiled regular expression object from the Python native 're' module
        that will match ETr rasters.
    ppt_ws : str
        Directory path of the precipitation workspace, which contains separate
        rasters for each year.
    ppt_re : class:`re`
        Compiled regular expression object from the native Python re module
        that will match precipitaiton rasters.
    spinup_days : int
        Number of days that should be used in the spinup of the model.
    min_spinup_days : int
        Minimum number of days that are needed for spinup of the model.

    Returns
    -------
    dict

    """
    logging.info('Kc SWB Adjust')
    # First calculate daily soil water balance at calibration point
    ke = et_common.point_swb_func(
        scene_date, hot_xy, hot_osr, awc_path,
        etr_ws, etr_re, ppt_ws, ppt_re,
        spinup_days, min_spinup_days)
    # Limit Ke to 0.1 - 1.05
    cal_dict['hot_tgt_kc'] = min(max(ke, 0.1), 1.05)
    # cal_dict['hot_tgt_kc'] = np.maximum(ke, 0.1)
    logging.info(pixel_str_fmt.format('', 'Cold Pixel', 'Hot Pixel'))
    logging.info(pixel_flt_fmt.format(
        'Kc:', cal_dict['cold_tgt_kc'], cal_dict['hot_tgt_kc']))
    return cal_dict

# # DEADBEEF
# # This doesn't work well in automated calibration
# # The cold calibration is selected based only on the ETrF
# # It may not have a 'good' NDVI value
# def kc_ndvi_adjust(kc, xy, ndvi_path, pnt_type='Cold'):
#     """Adjust Kc value based on NDVI
#
#     Parameters
#     ----------
#     kc :
#         Crop coefficient at the calibration point.
#     xy :
#         Location of the calibration point.
#     ndvi_path :
#     pnt_type :
#
#     Returns
#     -------
#
#
#     """
#     logging.info('Kc NDVI Adjust')
#     ndvi_value = raster_value_at_xy(ndvi_path, xy)
#     Limit Ke to 0.1 - 1.05
#     kc = min(max(1.25 * ndvi_value, 0.1), 1.05)
#     # kc = np.minimum(1.25 * ndvi_value, 1.05)
#     logging.info(pixel_str_fmt.format('', '{} Pixel'.format(pnt_type), ''))
#     logging.info(pixel_flt_fmt.format('NDVI:', ndvi, ''))
#     logging.info(pixel_flt_fmt.format('Kc:', kc, ''))
#     return kc


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Automated Calibration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-i', '--ini', required=True, metavar='PATH',
        help='Monte Carlo input file')
    parser.add_argument(
        '-mc', '--iter', default=None, type=int, metavar='N',
        help='Monte Carlo iteration number')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    # parser.add_argument('-o', '--overwrite',
    #     default=None, action='store_true',
    #     help='Force overwrite of existing files')
    parser.add_argument(
        '-s', '--shapefile', default=False, action='store_true',
        help='Save calibration points to shapefile')
    parser.add_argument(
        '-t', '--tails', type=float,
        default=[None, None], metavar=('COLD', 'HOT'), nargs=2,
        help='Cold and hot tail sizes')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.workspace and os.path.isdir(os.path.abspath(args.workspace)):
        args.workspace = os.path.abspath(args.workspace)
    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
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
        log_file_name = 'auto_calibration_log.txt'
        log_file = logging.FileHandler(
            os.path.join(args.workspace, log_file_name), "w")
        log_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

    logging.info(break_line)
    log_fmt = '{:<20s} {}'
    logging.info(log_fmt.format(
        'Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_fmt.format('Current Directory:', args.workspace))
    logging.info(log_fmt.format('Script:', os.path.basename(sys.argv[0])))

    # # DEADBEEF - For now, script cannot be run on a single scene
    # # Open Monte Carlo config file
    # config = configparser.ConfigParser()
    # try:
    #     config.read(config_file)
    # except:
    #     logging.error('\nERROR: Config file could not be read, ' +
    #                   'is not an input file, or does not exist\n' +
    #                   'ERROR: config_file = {}\n').format(monte_carlo_ini_path)
    #     sys.exit()
    # logging.debug('Reading Input File')
    #
    # # Set snap environment parameters
    # snap_epsg = int(config.get('INPUTS', 'snap_epsg'))
    # env.snap_osr = osr.SpatialReference()
    # env.snap_osr.ImportFromEPSG(snap_epsg)
    # env.snap_proj = env.snap_osr.ExportToWkt()
    # env.snap_gcs_osr = env.snap_osr.CloneGeogCS()
    # env.snap_gcs_proj = env.snap_gcs_osr.ExportToWkt()
    # env.cellsize = read_param('cellsize', 30, config)
    # env.snap_xmin, env.snap_ymin = map(
    #     int, read_param('snap_xy', (15,15), config))
    # env.snap_geo = (
    #     env.snap_xmin, env.cellsize, 0., env.snap_ymin, 0., -env.cellsize)
    #
    # # Only process scene if Model2 build an ETRF raster
    # # ETRF raster will not be built if convergence fails
    # if not os.path.isfile(os.path.join(args.workspace, 'et_rf.img')):
    #     logging.info(break_line)
    #     logging.error('ERROR: ETRF raster was not located, ' +
    #                   'scene will not be processed')
    #     sys.exit()
    #
    # # For non-Monte Carlo runs set iteration to 0
    # mc_iter, cal_iter = 0, 0
    #
    # # Get hot/cold etrf fraction sizes
    # # logging.info(break_line)
    # c_frac, h_frac = args.tails
    #
    # # Remove previous plots
    # logging.info(break_line)
    # remove_etrf_histograms(os.path.join(args.workspace, 'PLOTS'), args.iter)
    #
    # # Calibration parameters
    # cal_dict = calibration_params(
    #     etrf_array, tgt_c_pct=c_frac, tgt_h_pct=h_frac,
    #     kc_cold=1.05, kc_hot=0.1)
    # # logging.info(break_line)
    # # cal_dict = check_calibration(workspace, mc_iter, c_frac, h_frac)
    #
    # # Check calibration
    # logging.info(break_line)
    # cal_flag = check_calibration(etrf_array, mc_iter, cal_dict)
    #
    # # Plot ETRF Histogram
    # logging.info(break_line)
    # save_etrf_histograms(
    #     etrf_array, os.path.join(args.workspace, 'PLOTS'), mc_iter, cal_iter, cal_dict)
