#!/usr/bin/env python
#--------------------------------
# Name:         pixel_points.py
# Purpose:      METRIC Place Hot/Cold Pixels randomly in feature
#--------------------------------

import argparse
from datetime import datetime
import logging
import os
import random
import sys

import drigo
import numpy as np
from osgeo import gdal, ogr, osr
from scipy import ndimage

import et_image
import python_common


def pixel_points(image_ws, groupsize=1, blocksize=2048,
                 mc_iter=None, shapefile_flag=False,
                 multipoint_flag=False, pixel_point_iters=0,
                 overwrite_flag=None):
    """Place hot/cold pixels within .\PIXEL_REGIONS\pixel_suggestion.shp

    Parameters
    ----------
    image_ws : str
        Image folder path.
    groupsize : int, optional
        Script will try to place calibration point randomly into a
        labeled group of clustered values with at least n pixels.
        -1 = In the largest group
         0 = Anywhere in the image (not currently implemented)
         1 >= In any group with a pixel count greater or equal to n
    blocksize : int, optional
        Processing block size (the default is 2048).
    shapefile_flag : bool, optional
        If True, save calibration points to shapefile (the default is False).
    multipoing_flag : bool, optional
        If True, save cal. points to multipoint shapefile (the default is False).
    pixel_point_iters : int, optional
        Number of iterations (the default is 0).
    ovewrite_flag : bool, optional
        If True, overwrite existing files (the default is None).

    Returns
    -------
    tuple of cold coordinates, tuple of hot coordinates

    """
    logging.info('Placing hot/cold pixels in suggested regions')

    env = drigo.env
    image = et_image.Image(image_ws, env)
    np.seterr(invalid='ignore')
    common_ds = drigo.raster_path_ds(image.common_area_raster, read_only=True)
    env.snap_proj = drigo.raster_ds_proj(common_ds)
    common_ds = None

    # Open config file
    # config = open_ini(ini_path)

    # Get input parameters
    # logging.debug('  Reading Input File')
    pixels_folder = 'PIXELS'
    if mc_iter:
        cold_pixel_name = 'cold_{:02d}'.format(int(mc_iter))
        hot_pixel_name = 'hot_{:02d}'.format(int(mc_iter))
    else:
        cold_pixel_name = 'cold'
        hot_pixel_name = 'hot'
    hot_cold_pixels_name = 'hot_cold'
    r_fmt = '.img'
    s_fmt = '.shp'
    # json_format = '.geojson'

    # pixels_folder = read_param(pixels_folder, 'PIXELS', config)
    # cold_pixel_name = read_param(cold_pixel_name, 'cold', config)
    # hot_pixel_name = read_param(hot_pixel_name, 'hot', config)

    # Create Pixels Folder and Scratch Folder
    pixels_ws = os.path.join(image_ws, pixels_folder)
    region_ws = os.path.join(image_ws, 'PIXEL_REGIONS')
    if not os.path.isdir(pixels_ws):
        os.mkdir(pixels_ws)

    # Generate pixels shapefiles if they don't exist
    cold_pixel_path = os.path.join(pixels_ws, cold_pixel_name + s_fmt)
    hot_pixel_path = os.path.join(pixels_ws, hot_pixel_name + s_fmt)
    hot_cold_pixel_path = os.path.join(
        pixels_ws, hot_cold_pixels_name + s_fmt)
    if shapefile_flag and overwrite_flag:
        python_common.remove_file(cold_pixel_path)
        python_common.remove_file(hot_pixel_path)
        # python_common.remove_file(hot_cold_pixel_path)

    # Place points in the suggested pixel location raster
    cold_region_raster = os.path.join(
        region_ws, 'cold_pixel_suggestion' + r_fmt)
    hot_region_raster = os.path.join(
        region_ws, 'hot_pixel_suggestion' + r_fmt)
    if not os.path.isfile(cold_region_raster):
        logging.error(
            ('\nERROR: The cold pixel suggestion raster {} does ' +
             'not exist\n').format(os.path.basename(cold_region_raster)))
        sys.exit()
    if not os.path.isfile(hot_region_raster):
        logging.error(
            ('\nERROR: The hot pixel suggestion raster {} does ' +
             'not exist\n').format(os.path.basename(hot_region_raster)))
        sys.exit()

    # Check placement_mode value
    # Make sure it is between -1 and max pixelcount?

    # Pixel placement mode
    logging.info('Calibration point placement method:')
    if type(groupsize) is not int:
        logging.error('\nGroupsize must be an integer\n')
        sys.exit()
    elif groupsize <= -1:
        logging.info('  Randomly within largest suggested region polygon')
    elif groupsize == 0:
        logging.info('  Randomly anywhere in the image')
        sys.exit()
    elif groupsize >= 1:
        logging.info(
            ('  Randomly within suggested region polygons with '
             'more than {} pixels').format(groupsize))

    # Select random pixel
    cold_x, cold_y = get_random_point_in_raster(
        cold_region_raster, groupsize, blocksize)
    hot_x, hot_y = get_random_point_in_raster(
        hot_region_raster, groupsize, blocksize)

    # Save pixels
    if shapefile_flag and cold_x and cold_y:
        drigo.save_point_to_shapefile(
            cold_pixel_path, cold_x, cold_y, env.snap_proj)
    if shapefile_flag and hot_x and hot_y:
        drigo.save_point_to_shapefile(
            hot_pixel_path, hot_x, hot_y, env.snap_proj)
    if multipoint_flag and cold_x and cold_y:
        drigo.multipoint_shapefile(
            hot_cold_pixel_path, cold_x, cold_y,
            'COLD_{:02d}_{:02d}'.format(int(mc_iter), int(pixel_point_iters)),
            id_=mc_iter, input_proj=env.snap_proj)
    if multipoint_flag and hot_x and hot_y:
        drigo.multipoint_shapefile(
            hot_cold_pixel_path, hot_x, hot_y,
            'HOT_{:02d}_{:02d}'.format(int(mc_iter), int(pixel_point_iters)),
            id_=mc_iter, input_proj=env.snap_proj)

    # Eventually don't save a shapefile and just return the coordinates
    return (cold_x, cold_y), (hot_x, hot_y)


def get_random_point_in_raster(raster_path, groupsize, blocksize):
    """

    Parameters
    ----------
    raster_path
    groupsize
    blocksize

    Returns
    -------

    """
    # Check that raster exists
    if not os.path.isfile(raster_path):
        logging.error(
            '\nERROR: Pixel region ({}) does not exist\n'.format(raster_path))
        return None

    # Get full geo transform and shape from input raster
    raster_ds = gdal.Open(raster_path)
    raster_geo = drigo.raster_ds_geo(raster_ds)
    raster_rows, raster_cols = drigo.raster_ds_shape(raster_ds)
    raster_ds = None

    xy_list = []

    # Process blocks randomly
    logging.info('\nProcessing by block')
    logging.debug('  Raster cols/rows: {}/{}'.format(
        raster_cols, raster_rows))
    for b_i, b_j in drigo.block_gen_random(raster_rows, raster_cols, blocksize):
        logging.info('  Block  y: {:5d}  x: {:5d}'.format(b_i, b_j))
        block_array = drigo.raster_to_block(
            raster_path, b_i, b_j, blocksize, return_nodata=False)
        block_rows, block_cols = block_array.shape
        block_geo = drigo.array_offset_geo(raster_geo, b_j, b_i)
        block_extent = drigo.geo_extent(block_geo, block_rows, block_cols)
        logging.debug('    Block rows: {}  cols: {}'.format(
            block_rows, block_cols))
        logging.debug('    Block extent: {}'.format(block_extent))
        logging.debug('    Block geo: {}'.format(block_geo))

        # Check that region mask is not all nodata / 0
        if not np.any(block_array):
            logging.debug('  Empty block')
            continue
            # logging.warning('  Empty region mask, automated calibration ' +
            #                    'will not be able to calculate ETrF')
            # return None, None

        # Group cells if touching
        label_array, label_cnt = ndimage.label(
            block_array, structure=ndimage.generate_binary_structure(2, 2))
            # block_array, structure=ndimage.generate_binary_structure(2, 1))
        # label_mask = label_array > 0
        del block_array

        # For each group, calculate number of pixels and replace label value
        # count_array = np.copy(label_array)
        # blobs = ndimage.find_objects(label_array)
        # for i, blob_slice in enumerate(blobs):
        for i, blob_slice in enumerate(ndimage.find_objects(label_array)):
            label_array[blob_slice] = np.where(
                label_array[blob_slice] == (i+1),
                np.sum(label_array[blob_slice] == (i + 1)),
                label_array[blob_slice])
            # count_array[blob_slice] = np.sum(label_array[blob_slice]==(i+1))
        groupsize_max = np.max(label_array)
        logging.debug('  Max block groupsize: {}'.format(groupsize_max))

        # Blocks should be p randomly
        # If a target group is not found or looking for the largest
        #   read all blocks and track largest found group
        # If a target group is found, don't process other blocks
        if groupsize == -1 or groupsize_max < groupsize:
            yi, xi = np.where(label_array >= groupsize_max)
            x, y = drigo.array_offsets_xy(
                block_geo, random.choice(list(zip(xi, yi))))
            xy_list.append([groupsize_max, x, y])
        elif groupsize_max >= groupsize:
            yi, xi = np.where(label_array >= groupsize)
            x, y = drigo.array_offsets_xy(
                block_geo, random.choice(list(zip(xi, yi))))
            xy_list = [[groupsize_max, x, y]]
            break

    # If there are multiple points, return the point associated with the
    #   largest group
    if xy_list:
        groupsize_max, x, y = sorted(xy_list, reverse=True)[0]
        logging.debug('  Max groupsize: {}'.format(groupsize_max))
        return x, y
    else:
        return None, None

    # # Load raster array
    # raster_ds = gdal.Open(raster_path, 0)
    # raster_array = drigo.raster_ds_to_array(raster_ds, return_nodata=False)
    # raster_geo = drigo.raster_ds_geo(raster_ds)
    # raster_ds = None
    # del raster_ds
    #
    # # Check that region mask is not all nodata / 0
    # if not np.any(raster_array):
    #     logging.warning('  Empty region mask, automated calibration ' +
    #                         'will not be able to calculate ETrF')
    #     return None, None
    #
    # # Group cells if touching
    # # struct = ndimage.generate_binary_structure(2,1)
    # struct = ndimage.generate_binary_structure(2,2)
    # label_array, label_cnt = ndimage.label(raster_array, structure=struct)
    # label_mask = label_array > 0
    # del raster_array
    #
    # # For each group, calculate number of pixels and replace label value
    # # count_array = np.copy(label_array)
    # # blobs = ndimage.find_objects(label_array)
    # # for i, blob_slice in enumerate(blobs):
    # for i, blob_slice in enumerate(ndimage.find_objects(label_array)):
    #     label_array[blob_slice] = np.where(
    #         label_array[blob_slice]==(i+1),
    #         np.sum(label_array[blob_slice]==(i+1)),
    #         label_array[blob_slice])
    #     # count_array[blob_slice] = np.sum(label_array[blob_slice]==(i+1))
    #
    # # Get index of random cell in largest group
    # if placement_mode <= -1:
    #     yi, xi = np.where(label_array == np.max(label_array))
    #
    # # Get index of random cell in any group larger than placement_mode
    # else:
    #     yi, xi = np.where(label_array > placement_mode)
    #
    # # Calculate random cell x/y
    # return drigo.array_offsets_xy(raster_geo, choice(zip(xi,yi)))
    # # return array_offsets_xy(raster_geo, (xi[0], yi[0]))


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Pixel Points',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    # parser.add_argument(
    #     '-i', '--ini', required=True,
    #     help='METRIC input file', metavar='FILE')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-bs', '--blocksize', default=2048, type=int,
        help='Block size')
    parser.add_argument(
        '-gs', '--groupsize', default=1, type=int,
        help='Minimum group size for placing calibration points')
    parser.add_argument(
        '-m', '--multipoint', default=False, action="store_true",
        help='Save calibration points to multipoint shapeifle')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-p', '--pixel-point-iters', default=0,
        help='Pixel point iteration number from the MonteCarlo calibration')
    parser.add_argument(
        '-s', '--shapefile', default=False, action="store_true",
        help='Save calibration points to shapefile')
    args = parser.parse_args()

    # Convert input file to an absolute path
    if os.path.isdir(os.path.abspath(args.workspace)):
        args.workspace = os.path.abspath(args.workspace)
    # if args.ini and os.path.isfile(os.path.abspath(args.ini)):
    #     args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_console = logging.StreamHandler()
    log_console.setLevel(args.loglevel)
    formatter = logging.Formatter('%(message)s')
    log_console.setFormatter(formatter)
    logger.addHandler(log_console)

    if not args.no_file_logging:
        log_file_name = 'pixel_selection_log.txt'
        log_file = logging.FileHandler(
            os.path.join(args.workspace, log_file_name), "w")
        log_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

    logging.info('\n{}'.format('#' * 80))
    log_fmt = '{:<20s} {}'
    logging.info(log_fmt.format('Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_fmt.format('Current Directory:', args.workspace))
    logging.info(log_fmt.format('Script:', os.path.basename(sys.argv[0])))
    logging.info('')

    # Pixel Points
    pixel_points(image_ws=args.workspace,
                 groupsize=args.groupsize, blocksize=args.blocksize,
                 shapefile_flag=args.shapefile, multipoint_flag=args.multipoint,
                 pixel_point_iters=args.pixel_point_iters,
                 overwrite_flag=args.overwrite)
