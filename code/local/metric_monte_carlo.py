#!/usr/bin/env python
#--------------------------------
# Name:         metric_monte_carlo.py
# Purpose:      Run METRIC Monte Carlo for each Landsat scene
# Python:       2.7, 3.5, 3.6
#--------------------------------

import argparse
from datetime import datetime
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys

from python_common import open_ini, read_param, call_mp, parse_int_set


def main(ini_path, mc_iter_str='', tile_list=None,
         cold_tgt_pct=None, hot_tgt_pct=None, groupsize=64, blocksize=2048,
         multipoint_flag=True, shapefile_flag=True, stats_flag=True,
         overwrite_flag=False, mp_procs=1, delay=0, debug_flag=False,
         new_window_flag=False, no_file_logging=False,
         no_final_plots=None, no_temp_plots=None):
    """Run METRIC Monte Carlo for all Landsat scenes

    Parameters
    ----------
    ini_path : str
        File path of the input parameters file.
    mc_iter_str : str
        MonteCarlo iteration list and/or range.
    tile_list : list, optional
        Landsat path/rows to process (i.e. [p045r043, p045r033]).
        This will override the tile list in the INI file.
    cold_tgt_pct : float, optional
        Target percentage of pixels with ETrF greater than cold Kc.
    hot_tgt_pct : float, optional
        Target percentage of pixels with ETrF less than hot Kc.
    groupsize : int, optional
        Script will try to place calibration point randomly into a labeled
        group of clustered values with at least n pixels (the default is 64).
        -1 = In the largest group
         0 = Anywhere in the image (not currently implemented)
         1 >= In any group with a pixel count greater or equal to n
    blocksize : int, optional
        Processing block size (the default is 2048).
    multipoint_flag : bool, optional
        If True, save cal. points to multipoint shapefile (the default is True).
    shapefile_flag : bool, optional
        If True, save calibration points to shapefile (the default False).
    stats_flag : bool, optional
        If True, compute raster statistics (the default is True).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
    mp_procs : int, optional
        Number of cores to use (the default is 1).
    delay : float, optional
        Max random delay starting function in seconds (the default is 0).
    debug_flag : bool, optional
        If True, enable debug level logging (the default is False).
    new_window_flag : bool, optional
        If True, open each process in new terminal window (the default is False).
        Microsoft Windows only.
    no_file_logging : bool
        If True, don't write logging to file (the default is False).
    no_final_plots : bool
        If True, don't save final ETrF histograms (the default is None).
        This will override the flag in the INI file
    no_temp_plots : bool
        If True, don't save temp ETrF histograms (the default is None).
        This will override the flag in the INI file

    Returns
    -------
    None
    """
    logging.info('\nRunning METRIC Monte Carlo')

    # Open config file
    config = open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    year = config.getint('INPUTS', 'year')
    if tile_list is None:
        tile_list = read_param('tile_list', [], config, 'INPUTS')
    project_ws = config.get('INPUTS', 'project_folder')
    logging.debug('  Year: {}'.format(year))
    logging.debug('  Path/rows: {}'.format(', '.join(tile_list)))
    logging.debug('  Project: {}'.format(project_ws))

    func_path = config.get('INPUTS', 'monte_carlo_func')
    skip_list_path = read_param('skip_list_path', '', config, 'INPUTS')

    # For now, get mc_iter list from command line, not from project file
    # mc_iter_list = config.get('INPUTS', 'mc_iter_list')
    mc_iter_list = list(parse_int_set(mc_iter_str))

    # Need soemthing in mc_iter_list to iterate over
    if not mc_iter_list:
        mc_iter_list = [None]

    # For now build INI file name from template INI names
    metric_ini_name = os.path.basename(config.get('INPUTS', 'metric_ini'))
    metric_ini_name = os.path.splitext(os.path.basename(metric_ini_name))[0]
    mc_ini_name = os.path.basename(config.get('INPUTS', 'monte_carlo_ini'))
    mc_ini_name = os.path.splitext(os.path.basename(mc_ini_name))[0]

    # INI file is built as a function of year and tile_name
    metric_ini_fmt = '{}_{}_{}.ini'
    mc_ini_fmt = '{}_{}_{}.ini'

    # Only allow new terminal windows on Windows
    if os.name is not 'nt':
        new_window_flag = False

    # if len(tile_list) == 1:
    #     devel_flag = True
    # else:
    #     devel_flag = False
    # # devel_flag = True

    # Regular expressions
    # For now assume path/row are two digit numbers
    tile_re = re.compile('p\d{3}r\d{3}', re.IGNORECASE)
    image_id_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(?:\w{4})_(\d{3})(\d{3})_'
        '(\d{4})(\d{2})(\d{2})_(?:\d{8})_(?:\d{2})_(?:\w{2})$')

    # Check inputs folders/paths
    if not os.path.isdir(project_ws):
        logging.error('\n Folder {} does not exist'.format(project_ws))
        sys.exit()

    # Read skip list
    if skip_list_path:
        logging.debug('\nReading scene skiplist')
        with open(skip_list_path) as skip_list_f:
            skip_list = skip_list_f.readlines()
            skip_list = [image_id.strip() for image_id in skip_list
                         if image_id_re.match(image_id.strip())]
    else:
        logging.debug('\nSkip list not set in INI')
        skip_list = []


    mp_list = []
    for tile_name in sorted(tile_list):
        logging.debug('\nTile: {}'.format(tile_name))
        tile_ws = os.path.join(project_ws, str(year), tile_name)
        if not os.path.isdir(tile_ws) and not tile_re.match(tile_name):
            continue

        # Check that there are scene folders
        image_folder_list = [
            os.path.join(tile_ws, image_id)
            for image_id in sorted(os.listdir(tile_ws))
            if (os.path.isdir(os.path.join(tile_ws, image_id)) and
                image_id_re.match(image_id) and
                image_id not in skip_list)]
        if not image_folder_list:
            continue
        for image_id in image_folder_list:
            pixel_ws = os.path.join(image_id, 'PIXELS')
            if not os.path.isdir(pixel_ws):
                os.mkdir(pixel_ws)
            # Since the multipoint shapefile will be appended, delete it
            #  in the wrapper script
            if multipoint_flag and os.path.isdir(pixel_ws):
                for pixel_file in os.listdir(pixel_ws):
                    if re.match('\w+_\w+.shp$', pixel_file):
                        logging.info('\n Removing {}'.format(pixel_file))
                        os.remove(os.path.join(pixel_ws, pixel_file))
        logging.debug('  {} {}'.format(year, tile_name))

        # Check that there is an input file for the path/row
        metric_ini_path = os.path.join(
            tile_ws, metric_ini_fmt.format(metric_ini_name, year, tile_name))
        mc_ini_path = os.path.join(
            tile_ws, mc_ini_fmt.format(mc_ini_name, year, tile_name))
        if not os.path.join(metric_ini_path):
            logging.warning('    METRIC Input file {} does not exist'.format(
                metric_ini_path))
            continue
        elif not os.path.join(mc_ini_path):
            logging.warning(
                '    Monte Carlo Input file {} does not exist'.format(
                    mc_ini_path))
            continue

        # Setup command line argument
        # call_args = [sys.executable, mc_func_path, '-i', ini_path]
        call_args = [sys.executable, func_path,
                     '--metric_ini', metric_ini_path,
                     '--mc_ini', mc_ini_path,
                     '--groupsize', str(groupsize)]
        if cold_tgt_pct is not None and hot_tgt_pct is not None:
            call_args.extend(['-t', str(cold_tgt_pct), str(hot_tgt_pct)])
            if blocksize:
                call_args.extend(['--blocksize', str(blocksize)])
        if shapefile_flag:
            call_args.append('--shapefile')
        if multipoint_flag:
            call_args.append('--multipoint')
        if stats_flag:
            call_args.append('--stats')
        if overwrite_flag:
            call_args.append('--overwrite')
        if debug_flag:
            call_args.append('--debug')
        if no_file_logging:
            call_args.append('--no_file_logging')
        if no_final_plots:
            call_args.append('--no_final_plots')
        if no_temp_plots:
            call_args.append('--no_temp_plots')

        # Run all scenes for each Monte Carlo iteration
        for mc_iter in mc_iter_list:
            if mc_iter is not None:
                mc_args = ['-mc', str(mc_iter)]
            else:
                mc_args = []
            for image_folder in image_folder_list:
                logging.debug('  {}'.format(os.path.basename(image_folder)))
                if mp_procs > 1:
                    mp_list.append([
                        call_args + mc_args, image_folder, delay,
                        new_window_flag])
                else:
                    subprocess.call(call_args + mc_args, cwd=image_folder)

    if mp_list:
        pool = mp.Pool(mp_procs)
        results = pool.map(call_mp, mp_list, chunksize=1)
        pool.close()
        pool.join()
        del results, pool

    logging.debug('\nScript complete')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Batch METRIC Monte Carlo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Landsat project input file', metavar='PATH')
    parser.add_argument(
        '-bs', '--blocksize', default=2048, type=int,
        help='Block size for selecting calibration points')
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-gs', '--groupsize', default=64, type=int,
        help='Minimum group size for placing calibration points')
    parser.add_argument(
        '-mc', '--mc_iter', default='', type=str,
        help='MonteCarlo iteration list and/or range')
    parser.add_argument(
        '-m', '--multipoint', default=False, action="store_true",
        help='Save calibration points to multipoint shapeifle')
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int,
        metavar='N', nargs='?', const=mp.cpu_count(),
        help='Number of processers to use')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '--no_final_plots', default=False, action="store_true",
        help="Don't save final ETrF histogram plots")
    parser.add_argument(
        '--no_temp_plots', default=False, action="store_true",
        help="Don't save temporary ETrF histogram plots")
    # The "no_stats" parameter is negated below to become "stats".
    # By default, the monte_carlo function will NOT compute raster statistics.
    # If a user runs this "local" script, they probably want statistics.
    # If not, user can "turn off" statistics.
    parser.add_argument(
        '--no_stats', default=False, action="store_true",
        help='Don\'t compute raster statistics')
    parser.add_argument(
        '-o', '--overwrite', default=None, action='store_true',
        help='Force overwrite of existing files')
    parser.add_argument(
        '-pr', '--path_row', nargs="+",
        help='Landsat path/rows to process (pXXrYY)')
    parser.add_argument(
        '-s', '--shapefile', default=False, action='store_true',
        help='Save calibration points to shapefile')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
    parser.add_argument(
        '-t', '--tails', type=float,
        default=[None, None], metavar=('COLD', 'HOT'), nargs=2,
        help='Cold and hot tail sizes')
    parser.add_argument(
        '--window', default=False, action="store_true",
        help='Open each process in a new terminal (windows only)')
    args = parser.parse_args()

    # Default is to build statistics (opposite of --no_stats default=False)
    args.stats = not args.no_stats
    # Convert relative paths to absolute paths
    if os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, mc_iter_str=args.mc_iter,
         tile_list=args.path_row, groupsize=args.groupsize,
         blocksize=args.blocksize, shapefile_flag=args.shapefile,
         multipoint_flag=args.multipoint,
         cold_tgt_pct=args.tails[0], hot_tgt_pct=args.tails[1],
         stats_flag=args.stats, overwrite_flag=args.overwrite,
         mp_procs=args.multiprocessing, delay=args.delay,
         debug_flag=args.loglevel==logging.DEBUG, new_window_flag=args.window,
         no_file_logging=args.no_file_logging,
         no_final_plots=args.no_final_plots, no_temp_plots=args.no_temp_plots)
