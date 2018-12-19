#!/usr/bin/env python
#--------------------------------
# Name:         metric_model2.py
# Purpose:      Run METRIC Model 2 for all images
#--------------------------------

import argparse
from datetime import datetime
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys

import python_common as dripy


def main(ini_path, tile_list=None, blocksize=None, stats_flag=True,
         overwrite_flag=False, mp_procs=1, delay=0,
         debug_flag=False, new_window_flag=False):
    """Run METRIC Model 2 for all images

    Parameters
    ----------
    ini_path : str
        File path of the input parameters file.
    tile_list : list, optional
        Landsat path/rows to process (i.e. [p045r043, p045r033]).
        This will override the tile list in the INI file.
    blocksize : int, optional
        Processing block size (the default is None).  If set, this blocksize
        parameter will be used instead of the value in the INI file.
    stats_flag : bool, optional
        If True, compute raster statistics (the default is True).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
    mp_procs : int, optional
        Number of cores to use (the default is 1).
    delay : float, optional
        max random delay starting function in seconds (the default is 0).
    debug_flag : bool, optional
        If True, enable debug level logging (the default is False).
    new_window_flag : bool, optional
        If True, open each process in new terminal window (the default is False).
        Microsoft Windows only.

    Returns
    -------
    None

    """
    logging.info('\nRunning METRIC model 2 for all images')

    # Open config file
    config = dripy.open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    year = config.getint('INPUTS', 'year')
    if tile_list is None:
        tile_list = dripy.read_param('tile_list', [], config, 'INPUTS')
    project_ws = config.get('INPUTS', 'project_folder')
    logging.debug('  Year: {}'.format(year))
    logging.debug('  Path/rows: {}'.format(', '.join(tile_list)))
    logging.debug('  Project: {}'.format(project_ws))

    func_path = config.get('INPUTS', 'metric_model2_func')
    keep_list_path = dripy.read_param('keep_list_path', '', config, 'INPUTS')
    # skip_list_path = dripy.read_param('skip_list_path', '', config, 'INPUTS')

    # For now build INI file name from template INI names
    ini_name = os.path.basename(config.get('INPUTS', 'metric_ini'))
    ini_name = os.path.splitext(os.path.basename(ini_name))[0]

    # INI file is built as a function of year and tile_name
    ini_fmt = '{}_{}_{}.ini'

    # Calculate pixel points before running Model 2
    # pixel_points_flag = True

    # Only allow new terminal windows on Windows
    if os.name is not 'nt':
        new_window_flag = False

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

    # Read keep/skip lists
    if keep_list_path:
        logging.debug('\nReading scene keep list')
        with open(keep_list_path) as keep_list_f:
            image_keep_list = keep_list_f.readlines()
            image_keep_list = [image_id.strip() for image_id in image_keep_list
                               if image_id_re.match(image_id.strip())]
    else:
        logging.debug('\nScene keep list not set in INI')
        image_keep_list = []

    # # DEADBEEF
    # if skip_list_path:
    #     logging.debug('\nReading scene skip list')
    #     with open(skip_list_path) as skip_list_f:
    #         image_skip_list = skip_list_f.readlines()
    #         image_skip_list = [image_id.strip() for image_id in image_skip_list
    #                            if image_id_re.match(image_id.strip())]
    # else:
    #     logging.debug('\nScene skip list not set in INI')
    #     image_skip_list = []

    mp_list = []
    for tile_name in sorted(tile_list):
        tile_ws = os.path.join(project_ws, str(year), tile_name)
        if not os.path.isdir(tile_ws) and not tile_re.match(tile_name):
            logging.debug('  {} {} - invalid tile, skipping'.format(
                year, tile_name))
            continue

        # Check that there are image folders
        image_id_list = [
            image_id for image_id in sorted(os.listdir(tile_ws))
            if (image_id_re.match(image_id) and
                os.path.isdir(os.path.join(tile_ws, image_id)) and
                (image_keep_list and image_id in image_keep_list))]
        #     (image_skip_list and image_id not in image_skip_list))]
        if not image_id_list:
            logging.debug('  {} {} - no available images, skipping'.format(
                year, tile_name))
            continue
        else:
            logging.debug('  {} {}'.format(year, tile_name))

        # Check that there is an input file for the path/row
        ini_path = os.path.join(
            tile_ws, ini_fmt.format(ini_name, year, tile_name))
        if not os.path.join(ini_path):
            logging.warning('    METRIC input file {} does not exist, '
                            'skipping'.format(ini_path))
            continue

        # Setup command line argument
        call_args = [sys.executable, func_path, '-i', ini_path]
        if blocksize is not None:
            call_args.extend(['--blocksize', str(blocksize)])
        if stats_flag:
            call_args.append('--stats')
        if overwrite_flag:
            call_args.append('--overwrite')
        if debug_flag:
            call_args.append('--debug')

        # Run METRIC Model 2 for each Monte Carlo iteration
        for image_id in image_id_list:
            logging.debug('  {}'.format(image_id))
            scene_ws = os.path.join(tile_ws, image_id)
            if mp_procs > 1:
                mp_list.append(
                    [call_args, scene_ws, delay, new_window_flag])
            else:
                subprocess.call(call_args, cwd=scene_ws)

    if mp_list:
        pool = mp.Pool(mp_procs)
        results = pool.map(dripy.call_mp, mp_list, chunksize=1)
        pool.close()
        pool.join()
        del results, pool

    logging.debug('\nScript complete')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Batch METRIC Model 2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, type=dripy.arg_valid_file,
        help='Landsat project input file', metavar='FILE')
    parser.add_argument(
        '-bs', '--blocksize', default=None, type=int,
        help='Processing block size (overwrite INI blocksize parameter)')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int,
        metavar='N', nargs='?', const=mp.cpu_count(),
        help='Number of processors to use')
    # The logic is intentionally backwards for this parameter.
    # By default, the metric_model2_func will NOT compute raster statistics.
    # If a user runs this "local" script, they probably want statistics.
    # If not, user can "turn off" statistics.
    parser.add_argument(
        '--no_stats', default=False, action="store_true",
        help='Don\'t compute raster statistics')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-pr', '--path_row', nargs="+",
        help='Landsat path/rows to process (pXXrYY)')
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

    main(ini_path=args.ini, tile_list=args.path_row, blocksize=args.blocksize,
         stats_flag=args.stats, overwrite_flag=args.overwrite,
         mp_procs=args.multiprocessing, delay=args.delay,
         debug_flag=args.loglevel==logging.DEBUG, new_window_flag=args.window)
