#--------------------------------
# Name:         gridmet_download.py
# Purpose:      Download GRIDMET data
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys

import _utils


def main(netcdf_ws=os.getcwd(), variables=['all'],
         start_date=None, end_date=None, overwrite_flag=False):
    """Download GRIDMET netcdf files

    Parameters
    ----------
    netcdf_ws : str
        Folder of GRIDMET netcdf files.
    variable : list, optional
        GRIDMET variables to download ('ppt', 'srad', 'sph', 'tmmn', 'tmmx', 'vs').
        Set as ['all'] to download all PPT and ETr/ETo variables.
        Set as ['etr'] or ['eto'] to download all variables needed to compute
        ASCE standardized reference ET.
    start_date : str, optional
        ISO format date (YYYY-MM-DD).
    end_date : str, optional
        ISO format date (YYYY-MM-DD).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None
    
    """
    logging.info('Downloading GRIDMET data\n')
    site_url = 'https://www.northwestknowledge.net/metdata/data'

    # If a date is not set, process 2017
    try:
        start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
        logging.debug('  Start date: {}'.format(start_dt))
    except:
        start_dt = dt.datetime(2017, 1, 1)
        logging.info('  Start date: {}'.format(start_dt))
    try:
        end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
        logging.debug('  End date:   {}'.format(end_dt))
    except:
        end_dt = dt.datetime(2017, 12, 31)
        logging.info('  End date:   {}'.format(end_dt))

    # GRIDMET rasters to extract
    data_full_list = ['pr', 'srad', 'sph', 'tmmn', 'tmmx', 'vs']
    data_etr_list = ['srad', 'sph', 'tmmn', 'tmmx', 'vs']
    if not variables:
        logging.error('\nERROR: variables parameter is empty\n')
        sys.exit()
    elif type(variables) is not list:
        # DEADBEEF - I could try converting comma separated strings to lists?
        logging.warning('\nERROR: variables parameter must be a list\n')
        sys.exit()
    elif 'all' in variables:
        logging.error('Downloading all variables\n  {}'.format(
            ','.join(data_full_list)))
        data_list = data_full_list
    elif 'eto' in variables or 'etr' in variables:
        logging.error(
            'Downloading all variables needed to compute ETr/ETo\n  {}'.format(
                ','.join(data_etr_list)))
        data_list = data_etr_list
    elif not set(variables).issubset(set(data_full_list)):
        logging.error('\nERROR: variables parameter is invalid\n  {}'.format(
            variables))
        sys.exit()
    else:
        data_list = variables

    # Build output workspace if it doesn't exist
    if not os.path.isdir(netcdf_ws):
        os.makedirs(netcdf_ws)

    # GRIDMET data is stored by year
    year_list = sorted(list(set([
        i_dt.year for i_dt in _utils.date_range(
            start_dt, end_dt + dt.timedelta(1))])))
    year_list = map(lambda x: '{:04d}'.format(x), year_list)

    # Set data types to upper case for comparison
    data_list = list(map(lambda x: x.lower(), data_list))

    # Each sub folder in the main folder has all imagery for 1 day
    # The path for each subfolder is the /YYYY/MM/DD
    logging.info('')
    for year_str in year_list:
        logging.info(year_str)

        # Process each file in sub folder
        for data_str in data_list:
            file_name = '{}_{}.nc'.format(data_str, year_str)
            file_url = '{}/{}'.format(site_url, file_name)
            save_path = os.path.join(netcdf_ws, file_name)

            logging.info('  {}'.format(file_name))
            logging.debug('    {}'.format(file_url))
            logging.debug('    {}'.format(save_path))
            if os.path.isfile(save_path):
                if not overwrite_flag:
                    logging.debug('    File already exists, skipping')
                    continue
                else:
                    logging.debug('    File already exists, removing existing')
                    os.remove(save_path)

            _utils.url_download(file_url, save_path)

    logging.debug('\nScript Complete')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pyMETRIC/tools/gridmet
        tools:   ./pyMETRIC/tools
        output:  ./pyMETRIC/gridmet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    gridmet_folder = os.path.join(project_folder, 'gridmet')

    parser = argparse.ArgumentParser(
        description='Download daily GRIDMET data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--netcdf', default=os.path.join(gridmet_folder, 'netcdf'),
        metavar='PATH', help='Output netCDF folder path')
    parser.add_argument(
        '--vars', default=['all'], nargs='+',
        choices=['all', 'etr', 'eto', 'pr', 'srad', 'sph', 'tmmn', 'tmmx', 'vs'],
        help='GRIDMET variables to download')
    parser.add_argument(
        '--start', default='2017-01-01', type=_utils.valid_date,
        help='Start date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--end', default='2017-12-31', type=_utils.valid_date,
        help='End date (format YYYY-MM-DD)', metavar='DATE')
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
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{:<20s} {}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{:<20s} {}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(netcdf_ws=args.netcdf, variables=args.vars,
         start_date=args.start, end_date=args.end,
         overwrite_flag=args.overwrite)
