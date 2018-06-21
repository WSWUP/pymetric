#--------------------------------
# Name:         download_cdl.py
# Purpose:      Download national CDL zips
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys
import zipfile

import _utils


def main(year, output_folder, overwrite_flag=False):
    """Download national CDL zips

    Parameters
    ----------
    year : str
        4 digit year.
    output_folder : str
        Folder path where files will be saved.
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    """
    site_url = 'ftp.nass.usda.gov'
    site_folder = 'download/res'
    zip_name = '{}_30m_cdls.zip'.format(year)
    zip_path = os.path.join(output_folder, zip_name)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if not os.path.isfile(zip_path) or overwrite_flag:
        logging.info('\nDownload CDL files')
        logging.info('  {}'.format(
            '/'.join([site_url, site_folder, zip_name])))
        logging.info('  {}'.format(zip_path))
        _utils.ftp_download(site_url, site_folder, zip_name, zip_path)
    else:
        logging.info('\nCDL raster already downloaded')

    if os.path.isfile(zip_path):
        logging.info('\nExtracting CDL files')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_folder)
    else:
        logging.info('\nCDL raster already extracted')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/download
        tools:   ./pymetric/tools
        output:  ./pymetric/cdl
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'cdl')

    parser = argparse.ArgumentParser(
        description='Download CDL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-y', '--year', help='Year', metavar='YEAR', required=True)
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

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(year=args.year, output_folder=args.output,
         overwrite_flag=args.overwrite)
