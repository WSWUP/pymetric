#!/usr/bin/env python
#--------------------------------
# Name:         landsat_interpolate.py
# Purpose:      Interpolate seasonal ET for each monte carlo iteration
#--------------------------------

import argparse
from datetime import datetime
import logging
from multiprocessing import cpu_count
import os
import subprocess
import sys

import python_common as dripy


def main(ini_path, rasters_flag=None, tables_flag=None,
         mc_iter_str='', tile_list=None, blocksize=None, pyramids_flag=True,
         stats_flag=True, overwrite_flag=False,
         mp_procs=1, delay=0, debug_flag=False, no_file_logging=False):
    """Run interpolater for all Landsat scenes

    Parameters
    ----------
    ini_path : str
        File path of the input parameters file.
    rasters_flag : bool, optional
        If True, override INI and interpolate rasters.
    tables_flag : bool, optional
        If True, override INI and interpolate zone tables.
    mc_iter_str : str, optional
        MonteCarlo iteration list and/or range.
    tile_list : list, optional
        Landsat path/rows to process (i.e. [p045r043, p045r033]).
        This will override the tile list in the INI file.
    blocksize : int
        Processing block size (the default is None).
    pyramids_flag : bool, optional
        If True, compute raster pyramids (the default is True).
    stats_flag : bool, optional
        If True, compute raster statistics (the default is True).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
    mp_procs : int, optional
        Number of cpu cores to use (the default is 1).
    delay : float, optional
        Max random delay starting function in seconds (the default is 0).
    debug_flag : bool, optional
        If True, enable debug level logging (the default is False).
    no_file_logging : bool, optional
        If True, don't write logging to file (the default is False).

    Returns
    -------
    None

    """
    logging.info('\nRunning Interpolator')

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

    interpolate_folder = config.get('INPUTS', 'interpolate_folder')
    logging.debug('  Folder: {}'.format(interpolate_folder))

    # If both flags were not set, read from INI
    if rasters_flag is None and tables_flag is None:
        logging.info('  Reading interpolator flags from INI file')
        if rasters_flag is None:
            rasters_flag = dripy.read_param(
                'interpolate_rasters_flag', True, config, 'INPUTS')
        if tables_flag is None:
            tables_flag = dripy.read_param(
                'interpolate_tables_flag', True, config, 'INPUTS')
    # If both flags were set false, for now, exit the script
    # It may make more sense to assumethe user wants to interpolate something
    elif rasters_flag is False and tables_flag is False:
        logging.error('Raster and table interpolator flags are both False\n')
        logging.error('  Exiting the script')
        return False
        # sys.exit()
        # logging.info('Raster and table interpolator flags are both False\n')
        # logging.info('    Defaulting to rasters_flag=True')
        # rasters_flag = True

    if rasters_flag:
        rasters_func_path = config.get('INPUTS', 'interpolate_rasters_func')
    if tables_flag:
        tables_func_path = config.get('INPUTS', 'interpolate_tables_func')

    # For now, get mc_iter list from command line, not from project file
    # mc_iter_list = config.get('INPUTS', 'mc_iter_list')
    mc_iter_list = list(dripy.parse_int_set(mc_iter_str))

    # Need soemthing in mc_iter_list to iterate over
    if not mc_iter_list:
        mc_iter_list = [None]

    # For now build INI file name from template INI names
    ini_name = os.path.basename(config.get('INPUTS', 'interpolate_ini'))
    ini_name = os.path.splitext(os.path.basename(ini_name))[0]

    # INI file is built as a function of year
    ini_fmt = '{}_{}_{}.ini'

    # Regular expressions
    # For now assume path/row are two digit numbers
    # tile_re = re.compile('p(\d{3})r(\d{3})', re.IGNORECASE)
    # image_id_re = re.compile(
    #     '^(LT04|LT05|LE07|LC08)_(?:\w{4})_(\d{3})(\d{3})_'
    #     '(\d{4})(\d{2})(\d{2})_(?:\d{8})_(?:\d{2})_(?:\w{2})$')

    # Check inputs folders/paths
    if not os.path.isdir(project_ws):
        logging.error('\n Folder {} does not exist'.format(project_ws))
        sys.exit()

    # Check that there is an input file for the year and folder
    year_ws = os.path.join(project_ws, str(year))
    ini_path = os.path.join(
        year_ws, ini_fmt.format(
            ini_name, str(year), interpolate_folder.lower()))
    if not os.path.join(ini_path):
        logging.warning('    Input file does not exist\n  {}'.format(
            ini_path))
        return False

    # Run Interpolater for each Monte Carlo iteration
    # mp_list = []
    for mc_iter in sorted(mc_iter_list):
        logging.debug('  Year: {} Iteration: {}'.format(str(year), mc_iter))
        rasters_args = []
        tables_args = []
        if rasters_flag:
            rasters_args = [
                'python', rasters_func_path, year_ws, '-i', ini_path]
        if tables_flag:
            tables_args = ['python', tables_func_path, year_ws, '-i', ini_path]
        if mc_iter is not None:
            rasters_args.extend(['-mc', str(mc_iter)])
            tables_args.extend(['-mc', str(mc_iter)])
        if pyramids_flag:
            rasters_args.append('--pyramids')
        if stats_flag:
            rasters_args.append('--stats')
        if overwrite_flag:
            rasters_args.append('--overwrite')
            tables_args.append('--overwrite')
        if debug_flag:
            rasters_args.append('--debug')
            tables_args.append('--debug')
        if delay > 0:
            rasters_args.extend(['--delay', str(delay)])
            tables_args.extend(['--delay', str(delay)])
        if no_file_logging:
            rasters_args.append('--no_file_logging')
            tables_args.append('--no_file_logging')
        if mp_procs > 1:
            rasters_args.extend(['-mp', str(mp_procs)])
            tables_args.extend(['-mp', str(mp_procs)])
        if blocksize is not None:
            rasters_args.extend(['--blocksize', str(bs)])
            tables_args.extend(['--blocksize', str(bs)])

        if rasters_flag:
            subprocess.call(rasters_args, cwd=year_ws)
        if tables_flag:
            subprocess.call(tables_args, cwd=year_ws)


    logging.debug('\nScript complete')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Batch Landsat interpolate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, type=dripy.arg_valid_file,
        help='Landsat project input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-bs', '--blocksize', default=None, type=int, metavar='N',
        help='Processing block size (overwrite INI blocksize parameter)')
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-mc', '--mc_iter', default='', type=str,
        help='MonteCarlo iteration list and/or range')
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int, nargs='?',
        metavar="[1-{}]".format(cpu_count()), const=cpu_count(),
        choices=range(1, cpu_count() + 1),
        help='Number of processors to use')
    # The "no_stats" parameter is negated below to become "stats".
    # The "no_pyramids" parameter is negated below to become "pyramids".
    # By default, the interpolate functions will NOT compute raster statistics
    #     or pyramids.
    # If a user runs this "local" script, they probably want statistics
    #     and pyramids.
    # If not, a user can "turn off" statistics and/or pyramids.
    parser.add_argument(
        '--no_pyramids', default=False, action="store_true",
        help='Don\'t compute raster pyramids')
    parser.add_argument(
        '--no_stats', default=False, action="store_true",
        help='Don\'t compute raster statistics')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-pr', '--path_row', nargs="+",
        help='Landsat path/rows to process (pXXrYY)')
    parser.add_argument(
        '--rasters', default=None, action="store_true",
        help='Run raster interpolator (override INI flag)')
    parser.add_argument(
        '--tables', default=None, action="store_true",
        help='Run zone table interpolator (override INI flag)')
    args = parser.parse_args()

    # Default is to build statistics (opposite of --no_stats default=False)
    args.stats = not args.no_stats
    args.pyramids = not args.no_pyramids

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

    main(ini_path=args.ini, rasters_flag=args.rasters, tables_flag=args.tables,
         mc_iter_str=args.mc_iter, tile_list=args.path_row, blocksize=args.blocksize,
         pyramids_flag=args.pyramids, stats_flag=args.stats,
         overwrite_flag=args.overwrite, mp_procs=args.multiprocessing,
         delay=args.delay, debug_flag=args.loglevel==logging.DEBUG,
         no_file_logging=args.no_file_logging)
