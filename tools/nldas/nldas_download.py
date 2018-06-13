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


def main(username, password, grb_ws, scene_list_path=None,
         start_dt=None, end_dt=None, overwrite_flag=False):
    """Download hourly NLDAS data

    Parameters
    ----------
    username : str
        Earthdata username.
    password : str
        Earthdata password.
    grb_ws : str, optional
        Folder of NLDAS data (the default is the current working directory).
    scene_list_path : str, optional
        Landsat scene keep list file path.
    start_dt : datetime, optional
        Start date.
    end_dt : datetime, optional
        End date.
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

    # # Landsat Collection 1 Product ID
    # landsat_re = re.compile(
    #     '^(?:LT04|LT05|LE07|LC08)_\w{4}_\d{3}\d{3}_(?P<DATE>\d{8})_'
    #     '\w{8}_\w{2}_\w{2}')

    # Landsat Custom Scene ID
    landsat_re = re.compile(
        '^(?:LT04|LT05|LE07|LC08)_\d{6}_(?P<DATE>\d{8})')

    # Process Landsat scene list and start/end input parameters
    if not scene_list_path and (not start_dt or not end_dt):
        logging.error(
            '\nERROR: A Landsat scene list or start/end dates must be set, '
            'exiting\n')
        return False
    if scene_list_path is not None and os.path.isfile(scene_list_path):
        # Build a date list from the Landsat scene keep list file
        logging.info('\nReading dates from scene keep list file')
        logging.info('  {}'.format(scene_list_path))
        with open(scene_list_path) as input_f:
            keep_list = input_f.readlines()
        date_list = sorted([
            dt.datetime.strptime(m.group('DATE'), '%Y%m%d').strftime('%Y-%m-%d')
            for image_id in keep_list
            for m in [landsat_re.match(image_id)] if m])
        logging.debug('  {}'.format(', '.join(date_list)))
    else:
        date_list = []
    if start_dt and end_dt:
        logging.debug('  Start date: {}'.format(start_dt))
        logging.debug('  End date:   {}'.format(end_dt))
    else:
        start_dt = dt.datetime.strptime(date_list[0], '%Y-%m-%d')
        end_dt = dt.datetime.strptime(date_list[-1], '%Y-%m-%d')

    # Build output workspace if it doesn't exist
    if not os.path.isdir(grb_ws):
        os.makedirs(grb_ws)

    # Login to the NLDAS data site
    session = requests.Session()

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
        scripts: ./pymetric/tools/nldas
        tools:   ./pymetric/tools
        output:  ./pymetric/nldas
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    nldas_folder = os.path.join(project_folder, 'nldas')
    grb_folder = os.path.join(nldas_folder, 'grb')

    parser = argparse.ArgumentParser(
        description='Download hourly NLDAS data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('username', help='Earthdata Username')
    parser.add_argument('password', help='Earthdata Password')
    parser.add_argument(
        '--grb', default=grb_folder, metavar='PATH',
        help='Output GRB folder path')
    parser.add_argument(
        '--landsat', default=None, metavar='PATH',
        help='Landsat scene keep list path')
    parser.add_argument(
        '--start', default=None, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='Start date')
    parser.add_argument(
        '--end', default=None, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='End date')
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
    if args.landsat and os.path.isfile(os.path.abspath(args.landsat)):
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

    main(username=args.username, password=args.password, grb_ws=args.grb,
         start_dt=args.start, end_dt=args.end, scene_list_path=args.landsat,
         overwrite_flag=args.overwrite)
