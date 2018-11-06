#--------------------------------
# Name:         download_landfire.py
# Purpose:      Download LANDFIRE vegetation type raster
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import shutil
import sys
import zipfile

import _utils


def main(output_folder, version='140', overwrite_flag=False):
    """Download LANDFIRE vegetation type raster

    Parameters
    ----------
    output_folder : str
        Folder path where files will be saved.
    version : {'105', '110', '120', '130', '140'}
        LANDFIRE version string (the default is '140').
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    """
    version = str(version).replace('.', '')

    base_url = 'http://www.landfire.gov/bulk/downloadfile.php?FNAME='
    zip_dict = {
        '140': 'US_{0}_mosaic-US_{0}EVT_20180618.zip&TYPE=landfire'.format(version),
        '130': 'US_{0}_Mosaic-US_{0}_EVT_04232015.zip&TYPE=landfire'.format(version),
        '120': 'US_{0}_Mosaic-US_{0}_EVT_06142017.zip&TYPE=landfire'.format(version),
        '110': 'US_{0}_mosaic_Refresh-US_{0}EVT_05312018.zip&TYPE=landfire'.format(version),
        '105': 'US_{0}_mosaic_Refresh-US_{0}evt_09122104.zip&TYPE=landfire'.format(version),
    }
    download_url = base_url + zip_dict[version]

    output_name = 'US_{}_EVT'.format(version)
    zip_path = os.path.join(output_folder, output_name + '.zip')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if not os.path.isfile(zip_path) or overwrite_flag:
        logging.info('\nDownloading LANDFIRE vegetation type')
        logging.info('  {}'.format(download_url))
        logging.info('  {}'.format(zip_path))
        _utils.url_download(download_url, zip_path)
    else:
        logging.info('\nLANDFIRE zip file already downloaded')

    if os.path.isfile(zip_path):
        logging.info('\nExtracting LANDFIRE files')
        with zipfile.ZipFile(zip_path) as zf:
            # Extract files using zip naming and folder structure
            # zf.extractall(output_folder)

            # Ignore top level zip folder name
            for member in zf.namelist():
                # Replace root folder and switch to OS separator
                output_path = list(member.split('/'))
                output_path[0] = output_name
                output_path = os.sep.join(output_path)

                # Standardize the naming of the "Grid" folder
                output_path = output_path.replace('grid1', 'Grid')\
                    .replace('grid2', 'Grid')\
                    .replace('grid', 'Grid')\
                    .replace('Grid2', 'Grid')

                output_ws = os.path.join(
                    output_folder, os.path.dirname(output_path))

                # Skip directories
                if not os.path.basename(output_path):
                    continue

                # Only process the "Grid" (or "grid", "Grid1", "Grid2") folder
                if 'grid' not in os.path.dirname(output_path).lower():
                    continue

                # Build output directories
                if not os.path.isdir(output_ws):
                    os.makedirs(output_ws)

                # Extract
                logging.debug('  {}'.format(output_path))
                source = zf.open(member)
                target = open(os.path.join(output_folder, output_path), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
    else:
        logging.info('\nLANDFIRE zip file not present')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pymetric/tools/download
        tools:   ./pymetric/tools
        output:  ./pymetric/landfire
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'landfire')

    parser = argparse.ArgumentParser(
        description='Download LANDFIRE veg. type',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-v', '--version', metavar='VERSION', default='140',
        choices=['105', '110', '120', '130', '140'],
        help='Version (105, 110, 120, 130, or 140)')
    parser.add_argument(
        '--output', help='Output folder', metavar='FOLDER',
        default=output_folder)
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert output folder to an absolute path
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(output_folder=args.output, version=args.version,
        overwrite_flag=args.overwrite)
