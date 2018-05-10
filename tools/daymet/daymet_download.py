#--------------------------------
# Name:         daymet_download.py
# Purpose:      Download DAYMET data
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys

import _utils


def main(netcdf_ws=os.getcwd(), variables=['all'],
         start_date=None, end_date=None,
         overwrite_flag=False):
    """Download DAYMET netcdf files

    Data is currently only available for 1980-2017

    Parameters
    ----------
    netcdf_ws : str
        Root folder of DAYMET data.
    variables : list, optional
        DAYMET variables to download ('prcp', 'srad', 'vp', 'tmmn', 'tmmx').
        Set as ['all'] to download all available variables.
    start_date : str, optional
        ISO format date (YYYY-MM-DD).
    end_date : str, optional
        ISO format date (YYYY-MM-DD).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    Notes
    -----
    https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/1328/catalog.html

    """
    logging.info('\nDownloading DAYMET data')
    site_url = 'http://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/1328'

    try:
        start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
        logging.debug('  Start date: {}'.format(start_dt))
    except Exception as e:
        start_dt = dt.datetime(2017, 1, 1)
        logging.info('  Start date: {}'.format(start_dt))
        logging.debug(e)
    try:
        end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
        logging.debug('  End date:   {}'.format(end_dt))
    except Exception as e:
        end_dt = dt.datetime(2017, 12, 31)
        logging.info('  End date:   {}'.format(end_dt))
        logging.debug(e)

    # DAYMET rasters to extract
    var_full_list = ['prcp', 'srad', 'vp', 'tmin', 'tmax']
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
        var_list = var_full_list
    elif not set(variables).issubset(set(var_full_list)):
        logging.error('\nERROR: variables parameter is invalid\n  {}'.format(
            variables))
        sys.exit()
    else:
        var_list = variables[:]

    # Build output workspace if it doesn't exist
    if not os.path.isdir(netcdf_ws):
        os.makedirs(netcdf_ws)

    # DAYMET data is stored by year
    year_list = sorted(list(set([
        i_dt.year for i_dt in _utils.date_range(
            start_dt, end_dt + dt.timedelta(1))])))
    year_list = list(map(lambda x: '{:04d}'.format(x), year_list))

    # Set data types to upper case for comparison
    var_list = list(map(lambda x: x.lower(), var_list))

    # Each sub folder in the main folder has all imagery for 1 day
    # The path for each subfolder is the /YYYY/MM/DD
    logging.info('')
    for year_str in year_list:
        logging.info(year_str)

        # Process each file in sub folder
        for variable in var_list:
            file_name = 'daymet_v3_{}_{}_na.nc4'.format(variable, year_str)
            file_url = '{}/{}/{}'.format(site_url, year_str, file_name)
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
        scripts: ./pyMETRIC/tools/daymet
        tools:   ./pyMETRIC/tools
        output:  ./pyMETRIC/daymet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    daymet_folder = os.path.join(project_folder, 'daymet')

    parser = argparse.ArgumentParser(
        description='Download daily DAYMET data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--netcdf', default=os.path.join(daymet_folder, 'netcdf'),
        metavar='PATH', help='Output netCDF folder path')
    parser.add_argument(
        '--vars', default=['all'], nargs='+',
        choices=['all', 'prcp', 'srad', 'vp', 'tmin', 'tmax'],
        help='DAYMET variables to download')
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
        '--debug', default=logging.INFO, const=logging.DEBUG,
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
