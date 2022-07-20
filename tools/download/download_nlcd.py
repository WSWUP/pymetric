#--------------------------------
# Name:         download_nlcd.py
# Purpose:      Download NLCD raster
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys
import zipfile

import _utils


def main(output_folder, year='2016', overwrite_flag=False):
    """Download NLCD raster

    Parameters
    ----------
    output_folder : str
        Folder path where files will be saved.
    year : {2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019}; optional
        NLCD year (the default is 2016).
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    Notes
    -----
    https://s3-us-west-2.amazonaws.com/mrlc/NLCD_2016_Land_Cover_L48_20190424.zip

    """

    download_url = f'https://s3-us-west-2.amazonaws.com/mrlc/' \
                   f'nlcd_{year}_land_cover_l48_20210604.zip'

    zip_name = download_url.split('/')[-1]
    zip_path = os.path.join(output_folder, zip_name)

    output_name = zip_name.replace('.zip', '.img')
    # output_path = os.path.join(output_folder, output_name)
    output_path = os.path.join(
        output_folder, os.path.splitext(zip_name)[0], output_name)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if ((not os.path.isfile(zip_path) and not os.path.isfile(output_path)) or
            overwrite_flag):
        logging.info('\nDownloading NLCD')
        logging.info('  {}'.format(download_url))
        logging.info('  {}'.format(zip_path))
        _utils.url_download(download_url, zip_path)
    else:
        logging.info('\nNLCD raster already downloaded')

    if ((overwrite_flag or not os.path.isfile(output_path)) and
            os.path.isfile(zip_path)):
        logging.info('\nExtracting NLCD files')
        logging.debug('  {}'.format(output_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_folder)
    else:
        logging.info('\nNLCD raster already extracted')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/download
        tools:   ./pymetric/tools
        output:  ./pymetric/nlcd
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'nlcd')

    parser = argparse.ArgumentParser(
        description='Download NLCD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output', default=output_folder, metavar='FOLDER',
        help='Output folder')
    parser.add_argument(
        '-y', '--year', metavar='YEAR', default='2016',
        choices=['2001', '2004', '2006', '2008', '2011', '2013', '2016', '2019'],
        help='NLCD Year (2001, 2004, 2006, 2008, 2011, 2013, or 2016, 2019)')
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

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(output_folder=args.output, year=args.year,
         overwrite_flag=args.overwrite)
