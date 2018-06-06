#!/usr/bin/env python
#--------------------------------
# Name:         landsat_prep_scene.py
# Purpose:      Prepare Landsat Scenes
#--------------------------------

import argparse
from datetime import datetime
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys

from python_common import open_ini, read_param, call_mp


def main(ini_path, tile_list=None, blocksize=2048, smooth_flag=True,
         stats_flag=True, overwrite_flag=False, mp_procs=1, delay=0,
         debug_flag=False, new_window_flag=False):
    """Prep Landsat scenes

    Parameters
    ----------
    ini_path : str
        File path of the input parameters file.
    tile_list : list, optional
        Landsat path/rows to process (i.e. [p045r043, p045r033]).
        This will override the tile list in the INI file.
    blocksize : int, optional
        Processing block size (the default is 2048).
    smooth_flag : bool, optional
        If True, dilate/erode image to remove fringe/edge pixels
        (the Default is True).
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
    logging.info('\nPreparing Landsat scenes')

    # Open config file
    config = open_ini(ini_path)

    logging.debug('  Reading Input File')
    year = config.getint('INPUTS', 'year')
    if tile_list is None:
        tile_list = read_param('tile_list', [], config, 'INPUTS')
    project_ws = config.get('INPUTS', 'project_folder')
    logging.debug('  Year: {}'.format(year))
    logging.debug('  Path/rows: {}'.format(', '.join(tile_list)))
    logging.debug('  Project: {}'.format(project_ws))

    func_path = config.get('INPUTS', 'prep_scene_func')
    skip_list_path = read_param('skip_list_path', '', config, 'INPUTS')

    # Only allow new terminal windows on Windows
    if os.name is not 'nt':
        new_window_flag = False

    # Regular expressions
    # For now assume path/row are two digit numbers
    tile_re = re.compile('p\d{3}r\d{3}', re.IGNORECASE)
    image_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})')

    # Check inputs folders/paths
    if not os.path.isdir(project_ws):
        logging.error('\nFolder {} does not exist'.format(project_ws))
        sys.exit()

    # Setup command line argument
    call_args = [sys.executable, func_path, '-i', ini_path]
    if blocksize:
        call_args.extend(['--blocksize', str(blocksize)])
    if stats_flag:
        call_args.append('--stats')
    if overwrite_flag:
        call_args.append('--overwrite')
    if debug_flag:
        call_args.append('--debug')
    if smooth_flag:
        call_args.append('--smooth')

    # Read skip list
    if skip_list_path:
        logging.debug('\nReading scene skip list')
        with open(skip_list_path) as skip_list_f:
            skip_list = skip_list_f.readlines()
            skip_list = [image_id.strip() for image_id in skip_list
                         if image_re.match(image_id.strip())]
    else:
        logging.debug('\nSkip list not set in INI')
        skip_list = []

    # Process each image
    mp_list = []
    for tile_name in sorted(tile_list):
        logging.debug('\nTile: {}'.format(tile_name))
        tile_ws = os.path.join(project_ws, str(year), tile_name)
        if not os.path.isdir(tile_ws) and not tile_re.match(tile_name):
            logging.debug('  No image folder, skipping')
            continue

        # Check that there are scene folders
        image_id_list = [
            image_id for image_id in sorted(os.listdir(tile_ws))
            if (os.path.isdir(os.path.join(tile_ws, image_id)) and
                image_re.match(image_id) and
                image_id not in skip_list)]
        if not image_id_list:
            logging.debug('  No available images, skipping')
            continue
        logging.info('  {} {}'.format(year, tile_name))

        # Prep each Landsat scene
        for image_id in image_id_list:
            image_ws = os.path.join(tile_ws, image_id)
            if mp_procs > 1:
                mp_list.append([
                    call_args, image_ws, delay, new_window_flag])
            else:
                logging.debug('  {}'.format(image_id))
                subprocess.call(call_args, cwd=image_ws)

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
        description='Batch Landsat scenes prep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Landsat project input file', metavar='FILE')
    parser.add_argument(
        '-bs', '--blocksize', default=2048, type=int,
        help='Block size')
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int,
        metavar='N', nargs='?', const=mp.cpu_count(),
        help='Number of processers to use')
    # The "no_smooth" parameter is negated below to become "smooth".
    # By default, prep_scene will NOT dilate/erode (smooth) edge pixels.
    # If a user runs this "local" script, they probably want to smooth.
    # If not, user can "turn off" smoothing.
    parser.add_argument(
        '--no_smooth', default=False, action="store_true",
        help='Don\t dilate and erode image to remove fringe/edge pixels')
    # The "no_stats" parameter is negated below to become "stats".
    # By default, prep_scene will NOT compute raster statistics.
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
    args.smooth = not args.no_smooth

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
         smooth_flag=args.smooth, stats_flag=args.stats,
         overwrite_flag=args.overwrite, mp_procs=args.multiprocessing,
         delay=args.delay, debug_flag=args.loglevel==logging.DEBUG,
         new_window_flag=args.window)
