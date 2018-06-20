#--------------------------------
# Name:         download_landsat.py
# Purpose:      Download Landsat tar.gz files
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import requests
import shutil
import sys
import tarfile

import _utils


def main(scene_list_path, output_folder, start_dt=None, end_dt=None,
         overwrite_flag=False):
    """Download Landsat tar.gz files

    Parameters
    ----------
    scene_list_path : str
        Landsat scene keep list file path.
    output_folder : str
        Folder path where files will be saved.
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

    base_url = 'http://storage.googleapis.com/gcp-public-data-landsat'
    url_fmt = '{url}/{sensor}/{collection}/{path}/{row}/{id}/{file}'

    # Landsat Collection 1 Product ID
    landsat_re = re.compile(
        '^(?P<SENSOR>LT04|LT05|LE07|LC08)_(?P<DATA_TYPE>\w{4})_'
        '(?P<PATH>\d{3})(?P<ROW>\d{3})_(?P<ACQ_DATE>\d{8})_(?:\w{8})'
        '_(?P<NUMBER>\w{2})_(?P<CATEGORY>\w{2})')
    pre_c1_re = re.compile('^(LT04|LT05|LE07|LC08)_\d{6}_\d{8}')

    logging.info('\nReading dates from scene keep list file')
    logging.info('  {}\n'.format(scene_list_path))
    if not os.path.isfile(scene_list_path):
        logging.error('\nLandsat scene list does not exist, exiting')
        return False
    with open(scene_list_path) as input_f:
        product_id_list = input_f.readlines()
    product_id_list = [image_id.strip() for image_id in product_id_list
                       if landsat_re.match(image_id.strip())]

    # Apply start/end date filters
    if start_dt:
        logging.debug('Start date: {}'.format(start_dt.strftime('%Y-%m-%d')))
        product_id_list = [id for id in product_id_list
                           if id[17:25] >= start_dt.strftime('%Y%m%d')]
    if end_dt:
        logging.debug('End date:   {}'.format(end_dt.strftime('%Y-%m-%d')))
        product_id_list = [id for id in product_id_list
                           if id[17:25] <= end_dt.strftime('%Y%m%d')]
    logging.debug('\n{}\n'.format(', '.join(product_id_list)))

    bands = {
        'LT05': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF',
                 'B6.TIF', 'B7.TIF', 'BQA.TIF', 'MTL.txt'],
        'LE07': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF',
                 'B6_VCID_1.TIF', 'B6_VCID_2.TIF', 'B7.TIF', 'B8.TIF',
                 'BQA.TIF', 'MTL.txt'],
        'LC08': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF',
                 'B6.TIF', 'B7.TIF', 'B8.TIF', 'B9.TIF', 'B10.TIF', 'B11.TIF',
                 'BQA.TIF', 'MTL.txt'],
    }

    for product_id in product_id_list:
        logging.info(product_id)

        id_match = landsat_re.match(product_id)
        if not id_match and pre_c1_re.match(product_id):
            logging.error(
                '\nThe scene list does appear to contain LANDSAT_PRODUCT_IDs'
                '  (i.e. LE07_L1TP_043030_20150101_20160905_01_T1)'
                '  Exiting')
            return False

        sensor, type, path, row, date, number, category = id_match.groups()
        # print(sensor, type, path, row, date, number, category)

        year_folder = os.path.join(
            output_folder, str(int(path)), str(int(row)), date[:4])
        product_folder = os.path.join(year_folder, product_id)
        if not os.path.isdir(product_folder):
            os.makedirs(product_folder)

        for band in bands[product_id[:4]]:
            logging.debug('  Band {}'.format(band))
            file_name = '{}_{}'.format(product_id, band)
            file_url = url_fmt.format(
                url=base_url, sensor=sensor, collection=number, path=path,
                row=row, id=product_id, file=file_name)
            file_path = os.path.join(
                output_folder, str(int(path)), str(int(row)), date[:4], product_id,
                file_name)
            # logging.info('  {}'.format(image_name))
            logging.debug('  {}'.format(file_url))
            logging.debug('  {}'.format(file_path))

            if overwrite_flag or not os.path.isfile(file_path):
                _fetch_image(file_url, file_path)

        output_path = os.path.join(year_folder, product_id + '.tar.gz')
        if ((overwrite_flag or not os.path.isfile(output_path)) and
                os.path.isdir(product_folder)):
            logging.info('  Zipping'.format(band))
            logging.debug('  {}'.format(output_path))
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(product_folder, arcname='.')

        if os.path.isdir(product_folder) and os.path.isfile(output_path):
            shutil.rmtree(product_folder)


# Copied from Landsat578
class BadRequestsResponse(Exception):
    pass

def _fetch_image(url, destination_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):
                    f.write(chunk)
        elif response.status_code > 399:
            logging.info('  Code {} on {}'.format(response.status_code, url))
            raise BadRequestsResponse(Exception)
    except BadRequestsResponse:
        pass


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/download
        tools:   ./pymetric/tools
        output:  ./pymetric/landsat
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'landsat')

    parser = argparse.ArgumentParser(
        description='Download Landsat',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('scene_list', help='Landsat scene keep list path')
    parser.add_argument(
        '--start', default=None, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='Start date')
    parser.add_argument(
        '--end', default=None, type=_utils.valid_date, metavar='YYYY-MM-DD',
        help='End date')
    parser.add_argument(
        '--output', default=output_folder, metavar='FOLDER',
        help='Output folder')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
    if args.scene_list and os.path.isfile(os.path.abspath(args.scene_list)):
        args.scene_list = os.path.abspath(args.scene_list)

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')

    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(scene_list_path=args.scene_list, output_folder=args.output,
         start_dt=args.start, end_dt=args.end, overwrite_flag=args.overwrite)
