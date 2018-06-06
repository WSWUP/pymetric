#--------------------------------
# Name:         nldas_download.py
# Purpose:      Download Hourly NLDAS data
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import sys

import requests

import _utils


class BadCredentialsException(BaseException):
    pass


def main(username, password, grb_ws=os.getcwd(), keep_list_path=None,
         start_date=None, end_date=None, overwrite_flag=False):
    """Download hourly NLDAS data

    Parameters
    ----------
    username : str
        Earthdata username.
    password : str
        Earthdata password.
    grb_ws : str, optional
        Folder of NLDAS data (the default is the current working directory).
    keep_list_path : str, optional
        Landsat scene keep list file path.
    start_date : str, optional
        ISO format date string (YYYY-MM-DD).
    end_date : str, optional
        ISO format date string (YYYY-MM-DD).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    """
    logging.info('\nDownloading NLDAS data')

    # Site URL
    data_url = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.002'

    file_fmt = 'NLDAS_FORA0125_H.A{:04d}{:02d}{:02d}.{}.002.grb'
    time_list = ['{:02d}00'.format(i) for i in range(0, 24, 1)]

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

    # Build output workspace if it doesn't exist
    if not os.path.isdir(grb_ws):
        os.makedirs(grb_ws)

    #
    session = requests.Session()

    # Build a date list from the Landsat scene keep list file
    date_list = []
    if keep_list_path is not None and os.path.isfile(keep_list_path):
        logging.info('\nReading dates from scene keep list file')
        logging.info('  {}'.format(keep_list_path))
        landsat_re = re.compile(
            '^(?:LT04|LT05|LE07|LC08)_(?:\d{3})(?:\d{3})_' +
            '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})')
        with open(keep_list_path) as input_f:
            keep_list = input_f.readlines()
        keep_list = [image_id.strip() for image_id in keep_list
                     if landsat_re.match(image_id.strip())]
        date_list = [
            dt.datetime.strptime(image_id[12:20], '%Y%m%d').strftime('%Y-%m-%d')
            for image_id in keep_list]
        logging.debug('  {}'.format(', '.join(date_list)))

    # DEADBEEF
    # # Build a date list from landsat_ws scene folders or tar.gz files
    # date_list = []
    # if landsat_ws is not None and os.path.isdir(landsat_ws):
    #     logging.info('\nReading dates from Landsat IDs')
    #     logging.info('  {}'.format(landsat_ws))
    #     landsat_re = re.compile(
    #         '^(?:LT04|LT05|LE07|LC08)_(?:\d{3})(?:\d{3})_' +
    #         '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})')
    #     for root, dirs, files in os.walk(landsat_ws, topdown=True):
    #         # If root matches, don't explore subfolders
    #         try:
    #             landsat_match = landsat_re.match(os.path.basename(root))
    #             date_list.append(dt.datetime.strptime(
    #                 '_'.join(landsat_match.groups()), '%Y_%m_%d').date().isoformat())
    #             dirs[:] = []
    #         except:
    #             pass
    #
    #         for file in files:
    #             try:
    #                 landsat_match = landsat_re.match(file)
    #                 date_list.append(dt.datetime.strptime(
    #                     '_'.join(landsat_match.groups()), '%Y_%m_%d').date().isoformat())
    #             except:
    #                 pass
    #     date_list = sorted(list(set(date_list)))

    # Each sub folder in the main folder has all imagery for 1 day
    # The path for each subfolder is the /YYYY/DOY
    logging.info('')
    for input_date in _utils.date_range(start_dt, end_dt + dt.timedelta(1)):
        # Separate folder for each year/DOY if necessary
        doy_ws = os.path.join(
            grb_ws, input_date.strftime("%Y"), input_date.strftime("%j"))

        # Skip the day
        if date_list and input_date.date().isoformat() not in date_list:
            logging.debug('{}, date not in Landsat list, skipping'.format(
                input_date.date()))
            # logging.info('{}, removing'.format(input_date.date()))
            # try:
            #     shutil.rmtree(doy_ws)
            # except:
            #     pass
            continue
        else:
            logging.info('{}'.format(input_date.date()))

        # Build list of files to download
        save_path_list = []
        for time_str in time_list:
            # Download each hourly file
            # Build input file URL
            file_name = file_fmt.format(
                input_date.year, input_date.month, input_date.day, time_str)
            # File path for saving locally
            save_path = os.path.join(doy_ws, file_name)
            if os.path.isfile(save_path):
                if not overwrite_flag:
                    logging.debug('  File already exists, skipping')
                    continue
                else:
                    logging.debug('  File already exists, removing existing')
                    os.remove(save_path)
            # Save the file for download
            save_path_list.append(save_path)
        if not save_path_list:
            continue

        # Build output folders if necessary
        if not os.path.isdir(doy_ws):
            os.makedirs(doy_ws)

        date_url = data_url + '/' + input_date.strftime("%Y/%j")
        logging.debug('  {}'.format(date_url))

        # Download each hourly file
        for save_path in save_path_list:
            logging.info('  {}'.format(os.path.basename(save_path)))
            logging.debug('    {}'.format(save_path))
            file_url = '{}/{}'.format(date_url, os.path.basename(save_path))

            # with requests.Session() as session:
            r1 = session.request('get', file_url)
            r = session.get(r1.url, stream=True, auth=(username, password))
            logging.debug('  HTTP Status: {}'.format(r.status_code))

            logging.debug('  Beginning download')
            with open(save_path, "wb") as output_f:
                if 'Access denied' in r.text:
                    raise BadCredentialsException('Check EarthData credentials.')
                if r.text.startswith('<!DOCTYPE html>'):
                    raise BadCredentialsException(
                        'Check "NASA GES DISC" is authorized. '
                        'Instructions: https://disc.gsfc.nasa.gov/earthdata-login')
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:  # filter out keep-alive new chunks
                        output_f.write(chunk)
            logging.debug('  Download complete')

    logging.debug('\nScript Complete')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pyMETRIC/tools/nldas
        tools:   ./pyMETRIC/tools
        output:  ./pyMETRIC/nldas
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    nldas_folder = os.path.join(project_folder, 'nldas')

    parser = argparse.ArgumentParser(
        description='Download hourly NLDAS data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('username', help='Earthdata Username')
    parser.add_argument('password', help='Earthdata Password')
    parser.add_argument(
        '--grb', default=os.path.join(nldas_folder, 'grb'),
        metavar='PATH', help='Output GRB folder path')
    parser.add_argument(
        '--landsat', default=None,
        metavar='PATH', help='Landsat scene keep list path')
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
    if args.grb and os.path.isdir(os.path.abspath(args.grb)):
        args.grb = os.path.abspath(args.grb)
    if args.landsat and os.path.isdir(os.path.abspath(args.landsat)):
        args.landsat = os.path.abspath(args.landsat)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#'*80))
    logging.info('{:<20s} {}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{:<20s} {}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(username=args.username, password=args.password,
         grb_ws=args.grb, keep_list_path=args.landsat,
         start_date=args.start, end_date=args.end,
         overwrite_flag=args.overwrite)
