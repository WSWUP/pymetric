#--------------------------------
# Name:         et_common.py
# Purpose:      Common ET support functions
#--------------------------------

from calendar import monthrange
from collections import defaultdict
import datetime as dt
import logging
import math
import os
import re
import sys

import drigo
import numpy as np
# Used by soil water balance point_swb_func
from osgeo import gdal, ogr, osr

import python_common


def landsat_folder_split(landsat_ws):
    """Return the ID portion of a full Landsat scene folder"""
    return landsat_id_split(os.path.basename(landsat_ws))


# def landsat_name_split(folder_name):
#     """Split Landsat folder name into components (Landsat, path, row, year, month, year)
#
#     """
#     # DEADBEEF - Scenes will not be merged, so it is unnecessary to
#     #   to return a row_start and row_end
#     landsat_pre_re = re.compile('^(LT4|LT5|LE7|LC8)\d{3}\d{3}\d{7}')
#     landsat_c1_re = re.compile('^(LT04|LT05|LE07|LC08)\d{3}_\d{6}_\d{8}')
#     if landsat_pre_re.match(folder_name):
#         landsat = folder_name[0: 3]
#         path = folder_name[3: 6]
#         row_start = folder_name[7: 10]
#         row_end = folder_name[10: 13]
#         year = folder_name[14: 18]
#         doy = folder_name[18: 21]
#     elif landsat_c1_re.match(folder_name):
#         landsat = folder_name[0: 3]
#         path = folder_name[3: 6]
#         row_start = folder_name[6: 9]
#         row_end = folder_name[6: 9]
#         year = folder_name[9:13]
#         month = folder_name[13:16]
#         day = folder_name[13:16]
#     # elif landsat_cloud_re.match(folder_name):
#     #     landsat = folder_name[0: 3]
#     #     path = folder_name[3: 6]
#     #     row_start = folder_name[7: 10]
#     #     row_end = folder_name[10: 13]
#     #     year = folder_name[14: 18]
#     #     doy = folder_name[18: 21]
#     else:
#         logging.error(
#             'ERROR: Could not parse landsat folder {}'.format(folder_name))
#     return landsat, path, row_start, row_end, year, month, day


def landsat_id_split(landsat_id):
    """Split Landsat ID into components (Landsat, path, row, year, DOY)

    Parameters
    ----------
    landsat_id : str

    Returns
    -------
    tuple of the Landsat ID components

    """
    landsat_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(?:\w{4})_(\d{3})(\d{3})_'
        '(\d{4})(\d{2})(\d{2})_(?:\d{8})_(?:\d{2})_(?:\w{2})$')
    if landsat_re.match(landsat_id):
        m_groups = landsat_re.match(landsat_id).groups()
        satellite, path, row, year, month, day = m_groups[0:6]
    else:
        logging.error(
            'ERROR: Could not parse landsat folder {}'.format(landsat_id))
        landsat, path, row, year, month, day = None, None, None, None, None, None

    return satellite, path, row, year, month, day


def band_dict_to_array(data_dict, band_dict):
    """
    
    Parameters
    ----------
    data_dict : dict
    band_dict: dict

    Returns
    -------
    ndarray

    """
    return np.array(
        [v for k, v in sorted(data_dict.items())
         if k in band_dict.keys()]).astype(np.float32)


def landsat_band_image_dict(ws, landsat_re):
    """Return a dictionary of Landsat images and band strings

    Copied from landsat_prep_scene_func.py
    Consider moving into et_common.py or making part of image class

    Parameters
    ----------
    ws :
    landsat_re :

    Returns
    -------
    dict

    """
    if not os.path.isdir(ws):
        return dict()
    output_dict = dict()
    for item in os.listdir(ws):
        if not os.path.isfile(os.path.join(ws, item)):
            continue
        landsat_match = landsat_re.match(item)
        if not landsat_match:
            continue
        band = landsat_match.group('band').replace('B', '')
        # Only keep first thermal band from Landsat 7: "B6_VCID_1" -> "6"
        band = band.replace('_VCID_1', '')
        output_dict[band] = os.path.join(ws, item)

    return output_dict


def doy_range_func(landsat_doy_list, year, min_days_in_month=10):
    """Calculate DOY Range

    Parameters
    ----------
    landsat_doy_list : list
    year : int
    min_days_in_month : int, optional

    Returns
    -------
    list
    
    """
    # logging.info('\nDOY List: {}'.format(landsat_doy_list))
    year = int(year)
    doy_start = int(landsat_doy_list[0])
    doy_end = int(landsat_doy_list[-1])
    doy_zero_dt = dt.datetime(year, 1, 1) + dt.timedelta(-1)
    doy_start_dt = doy_zero_dt + dt.timedelta(doy_start)
    doy_end_dt = doy_zero_dt + dt.timedelta(doy_end)
    # First day of current start month and last day of current end month
    month_start_dt = dt.datetime(
        year, python_common.doy2month(year, doy_start), 1)
    month_end_dt = dt.datetime(
        year, python_common.doy2month(year, doy_end),
        monthrange(year, python_common.doy2month(year, doy_end))[-1])
    # Won't work for December because datetime doesn't accept month 13
    # month_end_dt = dt.datetime(year, month_end + 1, 1) + dt.timedelta(-1)
    # First day of next start month and last day of prior end month
    month_start_next_dt = dt.datetime(
        year, python_common.doy2month(year, doy_start)+1, 1)
    month_end_prev_dt = dt.datetime(
        year, python_common.doy2month(year, doy_end), 1) + dt.timedelta(-1)
    # Count of number of days between doy and inner month endpoints
    month_start_day_count = (month_start_next_dt - doy_start_dt).days
    month_end_day_count = (doy_end_dt - month_end_prev_dt).days
    # Check that there are enough days in start/end months
    if month_start_day_count < min_days_in_month:
        doy_start = (month_start_next_dt - (doy_zero_dt)).days
        doy_start_dt = doy_zero_dt + dt.timedelta(doy_start)
        logging.info(
            ('\nFirst day set to DOY: {:>3d}  ({})\n since there are '
             'only {} days of data in the previous month').format(
                 doy_start, doy_start_dt, month_start_day_count))
    else:
        doy_start = (month_start_dt - (doy_zero_dt)).days
        doy_start_dt = doy_zero_dt + dt.timedelta(doy_start)
        logging.info(('\nFirst day set to DOY: {:>3d}  ({})').format(
            doy_start, doy_start_dt))
    if month_end_day_count < min_days_in_month:
        doy_end = (month_end_prev_dt - (doy_zero_dt)).days
        doy_end_dt = doy_zero_dt + dt.timedelta(doy_end)
        logging.info(
            ('Last day set to DOY:  {:>3d}  ({})\n  There are '
             'only {} days of data in the next month').format(
                 doy_end, doy_end_dt, month_end_day_count))
    else:
        doy_end = (month_end_dt - (doy_zero_dt)).days
        doy_end_dt = doy_zero_dt + dt.timedelta(doy_end)
        logging.info(('Last day set to DOY:  {:>3d}  ({})').format(
            doy_end, doy_end_dt))
    return range(doy_start, doy_end+1)


def read_refet_instantaneous_func(refet_file, year, doy, localtime=None,
                                  ref_type='ETR'):
    """Read in instantaneous RefET data

    Parameters
    ----------
    refet_file : str
    year : int
    doy : int
        Day of year.
    localtime :
    ref_type: {'ETR' (default), 'ETO'}, optional
        Reference surface type.

    Returns
    -------
    tuple of floats: dew_point, wind_speed, ea, etr, & etr_24hr

    """
    logging.debug('  RefET: {}'.format(refet_file))

    # Field names
    year_field = 'Yr'
    month_field = 'Mo'
    day_field = 'Day'
    doy_field = 'DoY'
    hrmn_field = 'HrMn'
    tmax_field = 'Tmax'
    tmin_field = 'Tmin'
    rs_field = 'Rs'
    wind_field = 'Wind'
    dewp_field = 'DewP'
    if ref_type.upper() == 'ETO':
        etr_field = 'ASCE_stPM_ETo'
    else:
        etr_field = 'ASCE_stPM_ETr'

    # Field properties
    field_dict = dict()
    field_dict[month_field] = ('i8', '{:>2d}', '{:>2s}')
    field_dict[day_field] = ('i8', '{:>3d}', '{:>3s}')
    field_dict[year_field] = ('i8', '{:>4d}', '{:>4s}')
    field_dict[doy_field] = ('i8', '{:>3d}', '{:>3s}')
    field_dict[hrmn_field] = ('i8', '{:>4d}', '{:>4s}')
    field_dict[tmax_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[tmin_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[rs_field] = ('f8', '{:>4.0f}', '{:>4s}')
    field_dict[wind_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[dewp_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[etr_field] = ('f8', '{:>5.2f}', '{:>5s}')

    # If localtime is not set, return daily means
    if localtime is None:
        daily_flag = True
    # If localtime is set, check that localtime value is valid
    elif not (0 <= localtime <= 24):
        logging.error((
            '\nERROR: localtime must be between 0 and 24.\n'
            'ERROR: value {} is invalid').format(localtime))
        sys.exit()
    else:
        daily_flag = False

    # Read in RefET file
    with open(refet_file, 'r') as refet_f:
        refet_list = refet_f.readlines()
    refet_f.close

    # Get line number where data starts
    header_split_line = 'RESULTS (SI Units):'
    refet_strip_list = [line.strip() for line in refet_list]
    try:
        header_line = refet_strip_list.index(header_split_line.strip())
        data_line = header_line + 6
    except IndexError:
        logging.error(
            '\nERROR: The line "RESULTS (SI Units):" could not be found in the RefET file'
            '\nERROR: This line is used to determine where to read data from the RefET file'
            '\nERROR: The units may not be metric or the file may be empty\n')
        sys.exit()
    # Split RefET file into header and data sections
    refet_header_list = refet_strip_list[header_line+2:data_line]
    refet_data_list = refet_list[data_line:]
    del refet_list, refet_strip_list

    # Filter spaces and newline characters at beginning and end
    # refet_list = [line.strip() for line in refet_list]
    refet_header_list = [line.strip() for line in refet_header_list]

    # This splits on whitespace
    # refet_list = [re.findall(r'[^\s]+', line) for line in refet_list]
    refet_header_list = [
        re.findall(r'[^\s]+', line) for line in refet_header_list]

    # Get field names
    refet_field_list = list(map(list, zip(*refet_header_list)))

    # join with spaces, remove '-', remove leading/trailing whitespace
    # Last, to match genfromtxt, replace ' ' with '_'
    refet_field_name_list = [
        ' '.join(l[:3]).replace('-', '').strip().replace(' ', '_')
        for l in refet_field_list]
    refet_field_unit_list = [
        l[3].replace('-', '') for l in refet_field_list]
    refet_field_count = len(refet_field_list)
    logging.debug(
        '  Field name list:\n    {}'.format(refet_field_name_list))
    logging.debug(
        '  Field unit list:\n    {}'.format(refet_field_unit_list))

    # Check that date fields exist
    if year_field not in refet_field_name_list:
        logging.error(
            ('\nERROR: Year field {} was not found in the '
             'RefET file\n').format(year_field))
        sys.exit()
    if (month_field in refet_field_name_list and
        day_field in refet_field_name_list):
        doy_flag = False
    elif doy_field in refet_field_name_list:
        doy_flag = True
    else:
        logging.error((
            '\nERROR: Month field {} and Day field {} or DOY field '
            '{} were not found in the RefET file\n').format(
                month_field, day_field, doy_field))
        sys.exit()
    refet_field_name_list = [
        f for f in refet_field_name_list if f in field_dict.keys()]
    dtype_list = ','.join([field_dict[f][0] for f in refet_field_name_list])

    # Read data as record array
    refet_data = np.genfromtxt(
        refet_data_list, names=refet_field_name_list,
        dtype=dtype_list)

    # Build doy_array if necessary
    year_array = refet_data[year_field].astype(np.int)
    if not doy_flag:
        month_array = refet_data[month_field].astype(np.int)
        day_array = refet_data[day_field].astype(np.int)
        dt_array = np.array([
            dt.datetime(int(year), int(month), int(day))
            for year, month, day in zip(year_array, month_array, day_array)])
        doy_array = np.array([d.timetuple().tm_yday for d in dt_array])
        del month_array, day_array
        del dt_array
    else:
        doy_array = refet_data[doy_field].astype(np.int)
    doy_mask = (doy_array == doy) & (year_array == year)

    # Check that there is data for year/doy
    if not np.any(doy_mask):
        logging.error(
            '\nERROR: No data for Year {} and DOY {}\n'.format(
                year, doy))
        sys.exit()

    # Print daily data
    refet_data_subset = refet_data[doy_mask]
    del refet_data, doy_mask
    logging.debug('  ' + ' '.join(
        field_dict[f][2].format(f) for f in refet_field_name_list))
    for row in refet_data_subset:
        # DEADBEEF - In a try/except since it crashes for NumPy 1.6.1
        # The default for ArcGIS 10.1 is NumPy 1.6.1
        try:
            logging.debug('  ' + ' '.join(
                field_dict[f][1].format(value)
                for f, value in zip(refet_field_name_list, row)))
        except:
            pass

    # Extract sub arrays for interpolating
    hrmn_array = refet_data_subset[hrmn_field].astype(np.float32)

    # If times go from 1,2,...22,23,0 in a day, interpolation will break
    if hrmn_array[-1] == 0:
        hrmn_array[-1] = 2400

    # Convert HHMM to a float HH.MM to match localtime
    hrmn_array *= 0.01
    tmin_array = refet_data_subset[tmin_field].astype(np.float32)
    tmax_array = refet_data_subset[tmax_field].astype(np.float32)
    # rs_array = refet_data_subset[rs_field].astype(np.float32)
    wind_array = refet_data_subset[wind_field].astype(np.float32)
    dewp_array = refet_data_subset[dewp_field].astype(np.float32)
    etr_array = refet_data_subset[etr_field].astype(np.float32)

    # Calculate vapor pressure
    ea_array = saturation_vapor_pressure_func(dewp_array)
    # Interpolate dewpoint from RefET data
    # Add 0.5 hours because RefET data is an average of
    #   the previous hour

    # This may need to be set by the user or adjusted
    tair_inst = float(np.interp(
        [localtime + 0.5], hrmn_array, tmax_array)[0])
    dew_point = float(np.interp(
        [localtime + 0.5], hrmn_array, dewp_array)[0])
    ea_inst = float(np.interp(
        [localtime + 0.5], hrmn_array, ea_array)[0])
    wind_speed = float(np.interp(
        [localtime + 0.5], hrmn_array, wind_array)[0])
    etr_inst = float(np.interp(
        [localtime + 0.5], hrmn_array, etr_array)[0])

    # ETr 24hr (mm)
    etr_24hr = float(np.sum(etr_array))

    return dew_point, wind_speed, ea_inst, etr_inst, etr_24hr


def read_refet_daily_func(refet_list, year, doy_range, ref_type='ETR'):
    """Read in daily RefET data

    Parameters
    ----------
    refet_list :
    year : int
    doy_range : list
    ref_type: {'ETR' (default), 'ETO'}, optional
        Reference surface type.

    Returns
    -------
    dict of DOY,ETr key/values

    """
    # Initialize ETr dictionary
    # doy_etr_dict = dict([(doy, 0) for doy in range(1,367)])
    doy_etr_dict = defaultdict(float)

    if ref_type.upper() == 'ETO':
        etr_field = 'ETo'
    else:
        etr_field = 'ETr'

    # Remove header information, everything above RESULTS
    # This re checks that any whitespace character can separate the words
    refet_results_re = re.compile('RESULTS\s+\(SI\s+Units\):')
    for i, refet_line in enumerate(refet_list):
        if refet_results_re.match(refet_line):
            refet_list[0:i+2] = []
            refet_header_list = refet_list[0:4]
            logging.debug('\n  RefET Data:')
            for refet_header_line in refet_header_list[0:4]:
                # logging.debug('    {}'.format(refet_header_line))
                refet_split_line = re.findall(r'[^\s]+', refet_header_line)
                logging.debug('    ' + ' '.join(
                    ['{:<5}'.format(i) for i in refet_split_line]))
            break
    try:
        len(refet_header_list)
    except NameError:
        logging.error(
            '\nERROR: The line "RESULTS (SI Units):" could not be found in the RefET file'
            '\nERROR: This line is used to determine where to read data from the RefET file'
            '\nERROR: The units may not be metric or the file may be empty\n')
        sys.exit()

    # From header rows, determine index for necessary fields
    for refet_header_line in refet_header_list:
        refet_split_line = re.findall(r'[^\s]+', refet_header_line)
        try:
            refet_yr_col = refet_split_line.index('Yr')
            refet_header_col_count = len(refet_split_line)
        except ValueError:
            pass
        try:
            refet_doy_col = refet_split_line.index('DoY')
        except ValueError:
            pass
        try:
            refet_etr_col = refet_split_line.index(etr_field)
        except ValueError:
            pass
    if not refet_yr_col or not refet_doy_col or not refet_etr_col:
        logging.error('\nERROR: One of the necessary fields was not '
                      'found in the RefET file\n')
        sys.exit()

    # Calculate daily refet
    for refet_line in refet_list:
        # re finds every character that is not a whitespace character
        #   and splits on the whitespace
        refet_split_line = re.findall(r'[^\s]+', refet_line)
        if refet_split_line[refet_yr_col] == str(year):
            if refet_header_col_count != len(refet_split_line):
                logging.info('    {}'.format(refet_line))
                logging.error('\nERROR: The RefET file may be missing data\n'
                              'ERROR: The # of columns in the header '
                              'does not equal the # of columns of data')
                sys.exit()
            doy = int(refet_split_line[refet_doy_col])
            doy_etr_dict[doy] += float(refet_split_line[refet_etr_col])

    if not set(doy_range).issubset(doy_etr_dict.keys()):
        logging.error(
            ('\nERROR: The RefET file does not have ' +
             'data for year {}').format(year))
        sys.exit()
    return doy_etr_dict


def read_nvet_daily_func(nvet_list, year, doy_range):
    """Read in daily NVET data

    Parameters
    ----------
    nvet_list
    year : int
    doy_range

    Returns
    -------
    dict of DOY,ETr key/values

    """
    # Initialize RefET dictionary
    doy_etr_dict = dict([(doy, 0) for doy in range(1, 367)])

    # Remove header information, everything above RESULTS
    # This re checks that any whitespace character can separate the words
    nvet_header_list = nvet_list[0:5]
    nvet_list[0:5] = []
    logging.info('  NVET Header:')
    for nvet_header_line in nvet_header_list[0:5]:
        logging.info('    {}'.format(nvet_header_line))
    for nvet_line in nvet_list[0:3]:
        logging.info('    {}'.format(nvet_line))
    nvet_list[0:5] = []

    # Column numbers are hardcoded here
    logging.warning('\n  NVET columns are hardcoded and need to be checked')
    nvet_yr_col = 2
    nvet_doy_col = 3
    nvet_etr_col = 14
    logging.warning('    Year Column:  {:2d}'.format(nvet_yr_col+1))
    logging.warning('    DOY Column:   {:2d}'.format(nvet_doy_col+1))
    logging.warning('    RefET Column: {:2d}'.format(nvet_etr_col+1))

    # Calculate daily refet
    for nvet_line in nvet_list:
        # re finds every character that is not a whitespace character
        #   and splits on the whitespace
        nvet_split_line = re.findall(r'[^\s]+', nvet_line)
        if nvet_split_line[nvet_yr_col] == year:
            doy = int(nvet_split_line[nvet_doy_col])
            etr = float(nvet_split_line[nvet_etr_col])
            doy_etr_dict[doy] = etr

    # ETr must be greater than 0 to be valid?
    doy_valid_etr_list = [doy for doy in doy_range if doy_etr_dict[doy] > 0]

    # Check that there is ETr data for each DOY in doy_range
    if len(doy_valid_etr_list) == 0:
        logging.error('\nERROR: The CSV ETr file does not contain data '
                      'for the year {}\n'.format(year))
        sys.exit()
    elif set(doy_range) - set(doy_valid_etr_list):
        logging.error(
            '\nERROR: The CSV ETr appears to have missing data'
            '\n  The following days are missing:\n  {}'.format(sorted(list(
                map(int, list(set(doy_range)-set(doy_valid_etr_list)))))))
        sys.exit()
    return doy_etr_dict


def read_csv_etr_daily_func(csv_etr_list, year, doy_range):
    """Read in daily ETr from a CSV file

    Parameters
    ----------
    csv_etr_list :
    year : int
    doy_range :

    Returns
    -------
    dict

    """
    # Initialize RefET dictionary
    doy_etr_dict = dict([(doy, 0) for doy in range(1, 367)])

    # Remove header information, everything above RESULTS
    # This re checks that any whitespace character can separate the words
    header_line = csv_etr_list[0]
    data_list = csv_etr_list[1:]
    logging.info('  CSV ETr data:')
    logging.info('    {}'.format(header_line))

    # Print the first few lines as a check
    for data_line in data_list[0:3]:
        logging.info('    {}'.format(data_line))

    # Column names are hardcoded here
    year_field = 'YEAR'
    doy_field = 'DOY'
    etr_field = 'ETR'
    field_i_dict = dict()

    # Figure out column index for each field name
    split_line = [s.upper() for s in header_line.split(',')]
    for field in [year_field, doy_field, etr_field]:
        try:
            field_i_dict[field] = split_line.index(field.upper())
            logging.info('    {} Column:  {:>2d}'.format(
                field, field_i_dict[field]+1))
        except ValueError:
            logging.error(
                ('\nERROR: {} field does not exist in '
                 'CSV ETr file').format(field))
            sys.exit()

    # Calculate daily refet
    for data_line in data_list:
        # re finds every character that is not a whitespace character
        #   and splits on the whitespace
        split_line = data_line.split(',')
        if int(split_line[field_i_dict[year_field]]) == int(year):
            doy = int(split_line[field_i_dict[doy_field]])
            etr = float(split_line[field_i_dict[etr_field]])
            doy_etr_dict[doy] = etr

    # ETr must be greater than 0 to be valid?
    doy_valid_etr_list = [doy for doy in doy_range if doy_etr_dict[doy] > 0]

    # Check that there is ETr data for each DOY in doy_range
    if len(doy_valid_etr_list) == 0:
        logging.error(('\nERROR: The CSV ETr file does not contain data '
                       'for the year {}\n').format(year))
        sys.exit()
    elif set(doy_range) - set(doy_valid_etr_list):
        logging.error(
            '\nERROR: The CSV ETr appears to have missing data'
            '\n  The following days are missing:\n  {}'.format(sorted(list(
                map(int, list(set(doy_range) - set(doy_valid_etr_list)))))))
        sys.exit()
    return doy_etr_dict


def fixed_etr_data_func(etr, year, doy_range):
    """Assign a fixed ETr value to all doys in doy_range

    Parameters
    ----------
    etr
    year
    doy_range

    Returns
    -------
    dict

    """
    return dict([(doy, etr) for doy in range(1, 367) if doy in doy_range])


def u_star_station_func(wind_speed_height, station_roughness,
                        wind_speed_mod):
    """U* at the station [m/s]

    Parameters
    ----------
    wind_speed_height : float
    station_roughness : float
    wind_speed_mod : float

    Returns
    -------
    float

    """
    return ((wind_speed_mod * 0.41) /
            math.log(wind_speed_height / station_roughness))


def u3_func(u_star_station, z3, station_roughness):
    """U at blending height (200m) [m/s]

    Parameters
    ----------
    u_star_station : float
    z3 : float
    station_roughness : float

    Returns
    -------
    float

    """
    return (u_star_station * math.log(z3 / station_roughness)) / 0.41


def saturation_vapor_pressure_func(temperature):
    """Saturation vapor pressure [kPa] from temperature

    Parameters
    ----------
    temperature : array_like
        Air temperature [C].

    Returns
    -------
    es : ndarray

    Notes
    -----
    es = 0.6108 * exp(17.27 * temperature / (temperature + 237.3))

    """
    es = np.array(temperature, copy=True, ndmin=1).astype(np.float64)
    es += 237.3
    np.reciprocal(es, out=es)
    es *= temperature
    es *= 17.27
    np.exp(es, out=es)
    es *= 0.6108
    return es.astype(np.float32)


def doy_fraction_func(doy):
    """Fraction of the DOY in the year [radians]

    Parameters
    ----------
    doy : float

    Returns
    -------
    float

    """
    return doy * (2 * math.pi / 365.)


def delta_func(doy):
    """Earth declination [radians]

    Parameters
    ----------
    doy : array_like

    Returns
    -------
    ndarray

    """
    return 0.409 * np.sin(doy_fraction_func(doy) - 1.39)


def air_pressure_func(elevation):
    """Air pressure [kPa]

    Parameters
    ----------
    elevation : array_like
        Elevation [m].

    Returns
    -------
    pair : ndarray

    Notes
    -----
    pair = 101.3 * (((293.15 - 0.0065 * elev) / 293.15) ** 5.26)

    """
    pair = np.array(elevation, copy=True, ndmin=1).astype(np.float64)
    pair *= -0.0065
    pair += 293.15
    pair /= 293.15
    np.power(pair, 5.26, out=pair)
    pair *= 101.3
    return pair.astype(np.float32)


def precipitable_water_func(pair, ea):
    """Precipitable water in the atmopshere

    Parameters
    ----------
    pair : array_like or float
        Air pressure [kPa].
    ea : array_like or float
        Vapor pressure [kPa].

    Returns
    -------
    array_like or float

    References
    ----------
    .. [1] Garrison, J. and Adler, G. (1990). Estimation of precipitable water
       over the United States for application to the division of solar
       radiation into its direct and diffuse components. Solar Energy, 44(4).
       https://doi.org/10.1016/0038-092X(90)90151-2
    .. [2] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    return pair * 0.14 * ea + 2.1


def dr_func(doy):
    """Inverse square of the Earth-Sun Distance

    Parameters
    ----------
    doy: array_like
        Day of year.

    Returns
    -------
    ndarray

    Notes
    -----
    This function returns 1 / d^2, not d, for direct use in radiance to
        TOA reflectance calculation.
    pi * L * d^2 / (ESUN * cos(theta)) -> pi * L / (ESUN * cos(theta) * d)

    """
    return 1.0 + 0.033 * np.cos(doy_fraction_func(doy))


def ee_dr_func(doy):
    """Earth-Sun Distance values used by Earth Engine

    Parameters
    ----------
    doy: array_like
        Day of year.

    Returns
    -------
    ndarray

    """
    return 0.033 * np.cos(doy_fraction_func(doy)) + 1.0


def seasonal_correction_func(doy):
    """Seasonal correction for solar time [hour]

    Parameters
    ----------
    doy: array_like
        Day of year.

    Returns
    -------
    ndarray

    """
    b = 2 * math.pi * (doy - 81.) / 364.
    return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.0250 * np.sin(b)


def solar_time_rad_func(lon, time, sc):
    """Solar time [hours]

    Parameters
    ----------
    lon : array_like
        UTC hour [radians].
    time : array_like
        UTC hour [hours].
    sc : array_like
        seasonal correction [hours].

    Returns
    -------
    ndarray

    """
    return time + (lon * 24 / (2 * math.pi)) + sc - 12


# def solar_time_func(lon, time, sc):
#     """Solar time (seconds) with longitude in degrees"""
#     return time + (lon / 15.) + sc


# def solar_time_deg_func(lon, time, sc):
#     """Solar time (seconds) with longitude in degrees"""
#     return time + (lon / 15.) + sc


def omega_func(solar_time):
    """Hour angle [radians]

    Parameters
    ----------
    solar_time : array_like
        UTC hour.

    Returns
    -------
    omega : ndarray

    """
    omega = (2 * math.pi / 24.0) * solar_time

    # Need to adjust omega so that the values go from -pi to pi
    # Values outside this range are wrapped (i.e. -3*pi/2 -> pi/2)
    omega = wrap_func(omega, -math.pi, math.pi)

    return omega


def wrap_func(x, x_min, x_max):
    """Wrap floating point values into range

    Parameters
    ----------
    x : array_like
        array of values to wrap.
    x_min : float
        Minimum value in output range.
    x_max : float
        Maximum value in output range.

    Returns
    -------
    ndarray

    """
    return np.mod((x - x_min), (x_max - x_min)) + x_min


def omega_sunset_func(lat, delta):
    """Sunset hour angle [radians] (Eqn 59)

    Parameters
    ----------
    lat : array_like
        Latitude [radians].
    delta : array_like
        Earth declination [radians].

    Returns
    -------
    ndarray

    """
    return np.arccos(-np.tan(lat) * np.tan(delta))


def ra_daily_func(lat, doy):
    """Daily extraterrestrial radiation [MJ m-2 d-1]

    Parameters
    ----------
    lat : array_like
        Latitude [radians].
    doy : array_like
        Day of year.

    Returns
    -------
    ndarray

    Notes
    -----
    This function  is only being called by et_numpy.rn_24_func().
    That function could be changed to use the refet.calcs._ra_daily() function
    instead, in which case this function could be removed.

    """

    delta = delta_func(doy)
    omegas = omega_sunset_func(lat, delta)
    theta = (omegas * np.sin(lat) * np.sin(delta) +
             np.cos(lat) * np.cos(delta) * np.sin(omegas))

    return (24. / math.pi) * 4.92 * dr_func(doy) * theta


def cos_theta_solar_func(sun_elevation):
    """Cosine of theta at a point given sun elevation angle"""
    return math.sin(sun_elevation * math.pi / 180.)


def cos_theta_centroid_func(t, doy, dr, lon_center, lat_center):
    """Cosine of theta at a point

    Parameters
    ----------
    t :
    doy :
    dr :
    lon_center :
    lat_center :

    Returns
    -------
    cos_theta : float

    """
    # Solar time seasonal correction
    sc = seasonal_correction_func(doy)
    # Earth declination
    delta = delta_func(doy)
    # Latitude in radians
    solar_time = solar_time_rad_func(lon_center, t, sc)
    omega = omega_func(solar_time)
    # Cosine of theta for a flat earth
    cos_theta = ((math.sin(delta) * math.sin(lat_center)) +
                 (math.cos(delta) * math.cos(lat_center) * math.cos(omega)))
    log_f = '  {:<18s} {}'
    logging.debug('\n' + log_f.format(
        'Latitude Center:', (lat_center * math.pi / 180)))
    logging.debug(log_f.format(
        'Longitude Center:', (lon_center * math.pi / 180)))
    logging.debug(log_f.format('Delta:', delta))
    logging.debug(log_f.format('Sc [hour]:', sc))
    logging.debug(log_f.format('Sc [min]:', sc*60))
    logging.debug(log_f.format('Phi:', lat_center))
    logging.debug(log_f.format('SolarTime [hour]:', solar_time))
    logging.debug(log_f.format('SolarTime [min]:', solar_time*60))
    logging.debug(log_f.format('Omega: ', omega))
    logging.debug(log_f.format('cos_theta:', cos_theta))
    # return (env.mask_array * cos_theta).astype(np.float32)

    return cos_theta


def cell_value_set(test_raster, test_name, cold_xy, hot_xy, log_level='INFO'):
    """Extract the raster values at the cold and hot calibration points

    X and Y coordinates need to be in the same spatial reference as the raster

    Parameters
    ----------
    test_raster : str
        File path of the raster to be sampled.
    test_name : str
        Display name of the raster (for logging).
    cold_xy : tuple
        x and y coordinate of the cold calibration point.
    hot_xy : tuple
        x and y coordinate of the cold calibration point.
    log_level : str
        Logging level (INFO, DEBUG).

    Returns
    -------
    tuple of the values at the calibration points

    """
    cold_flt = drigo.raster_value_at_xy(test_raster, cold_xy)
    hot_flt = drigo.raster_value_at_xy(test_raster, hot_xy)
    log_str = '    {:<14s}  {:14.8f}  {:14.8f}'.format(
        test_name + ':', cold_flt, hot_flt)
    if log_level == 'DEBUG':
        logging.debug(log_str)
    else:
        logging.info(log_str)

    return cold_flt, hot_flt



def raster_swb_func(output_dt, output_osr, output_cs, output_extent,
                    awc_path, etr_ws, etr_re, ppt_ws, ppt_re,
                    spinup_days=30, min_spinup_days=5):
    """Compute the daily soil water balance for a raster for a single date

    Parameters
    ----------
    output_dt : datetime
        Target date.
    output_osr : class:`osr.SpatialReference
        Spatial reference.
    output_cs : float
        Cellsize.
    output_extent :
        Extent
    awc_path : str
        File path of the available water content raster.
    etr_ws : str
        Directory path of the ETr workspace, which contains separate rasters
        for each year.
    etr_re (:class:`re`):
        Compiled regular expression object from the Python native 're' module
        that will match ETr rasters.
    ppt_ws : str
        Directory path of the precipitation workspace, which contains separate
        rasters for each year.
    ppt_re : :class:`re`
        Compiled regular expression object from the native Python re module
        that will match precipitaiton rasters.
    spinup_days : int, optional
        Number of days that should be used in the spinup of the model
        (the default is 30).
    min_spinup_days : int, optional
        Minimum number of days that are needed for spinup of the model
        (the default is 5).

    Returns
    -------
    array: :class:`numpy.array`: soil evaporation coeficient (Ke)

    Notes
    -----
    Calculations will be done in AWC spatial reference & cellsize.
    Final Ke will be projected to output spatial reference & cellsize.

    References
    ----------
    .. [1] Allen, R., Pereira, L., Smith, M., Raes, D., & Wright, J. (2005).
       FAO-56 dual crop coefficient method for estimating evaporation from soil
       and application extensions.
       Jourlnal of Irrigation and Drainage Engineering, 131(1).
       10.1061/(ASCE)0733-9437(2005)131:1(2)
    .. [2] Allen, R. (2011). Skin layer evaporation to account for small
       precipitation events — an enhancement to the FAO-56 evaporation model.
       Agricultural Water Management, 99.
       https://doi.org/10.1016/j.agwat.2011.08.008

    """
    # DEADBEEF - There is probably a better way to handle the daterange input.
    #  Perhaps something like setting a minimum spinup and maximum spinup
    #  days and allowing the code to take however many etr and ppt rasters
    #  it can find within that range is good. Also, we should probably
    #  add in a flag for dry vs wet starting point (when it comes to
    #  total evaporative water [tew])
    # logging.debug('Daily Soil Water Balance')

    # Compute list of dates needed for spinup
    # date_range function doesn't return end date so add 1 day to end
    dt_list = sorted(python_common.date_range(
        output_dt - dt.timedelta(days=spinup_days),
        output_dt + dt.timedelta(days=1)))
    year_list = sorted(list(set([d.year for d in dt_list])))

    # Get all available ETr and PPT paths in date range
    if not os.path.isdir(etr_ws):
        logging.error('  ETr folder does not exist\n    {}'.format(
            etr_ws))
        sys.exit()
    if not os.path.isdir(ppt_ws):
        logging.info('  PPT folder does not exist\n    {}'.format(
            ppt_ws))
        sys.exit()

    # DEADBEEF - What specific errors should be caught here?
    etr_path_dict = dict()
    ppt_path_dict = dict()
    for etr_name in os.listdir(etr_ws):
        try:
            test_year = etr_re.match(etr_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            etr_path_dict[str(test_year)] = os.path.join(etr_ws, etr_name)
    for ppt_name in os.listdir(ppt_ws):
        try:
            test_year = ppt_re.match(ppt_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            ppt_path_dict[str(test_year)] = os.path.join(ppt_ws, ppt_name)
    if not etr_path_dict:
        logging.error('  No ETr rasters were found for the point SWB\n')
        sys.exit()
    elif not ppt_path_dict:
        logging.error('  No PPT rasters were found for the point SWB\n')
        sys.exit()

    # Get raster properties from one of the rasters
    # Project Landsat scene extent to ETr/PPT rasters
    logging.debug('  ETr: {}'.format(etr_path_dict[str(output_dt.year)]))
    etr_ds = gdal.Open(etr_path_dict[str(output_dt.year)], 0)
    etr_osr = drigo.raster_ds_osr(etr_ds)
    etr_cs = drigo.raster_ds_cellsize(etr_ds, x_only=True)
    etr_x, etr_y = drigo.raster_ds_origin(etr_ds)
    etr_extent = drigo.project_extent(
        output_extent, output_osr, etr_osr, cellsize=output_cs)
    etr_extent.buffer_extent(etr_cs * 2)
    etr_extent.adjust_to_snap('EXPAND', etr_x, etr_y, etr_cs)
    etr_ds = None

    logging.debug('  PPT: {}'.format(ppt_path_dict[str(output_dt.year)]))
    ppt_ds = gdal.Open(ppt_path_dict[str(output_dt.year)], 0)
    ppt_osr = drigo.raster_ds_osr(ppt_ds)
    ppt_cs = drigo.raster_ds_cellsize(ppt_ds, x_only=True)
    ppt_x, ppt_y = drigo.raster_ds_origin(ppt_ds)
    ppt_extent = drigo.project_extent(
        output_extent, output_osr, ppt_osr, cellsize=output_cs)
    ppt_extent.buffer_extent(ppt_cs * 2)
    ppt_extent.adjust_to_snap('EXPAND', ppt_x, ppt_y, ppt_cs)
    ppt_ds = None

    # Get AWC raster properties
    # Project Landsat scene extent to AWC raster
    logging.debug('  AWC: {}'.format(awc_path))
    awc_ds = gdal.Open(awc_path, 0)
    awc_osr = drigo.raster_ds_osr(awc_ds)
    awc_cs = drigo.raster_ds_cellsize(awc_ds, x_only=True)
    awc_x, awc_y = drigo.raster_ds_origin(awc_ds)
    awc_extent = drigo.project_extent(
        output_extent, output_osr, awc_osr, cellsize=output_cs)
    awc_extent.buffer_extent(awc_cs * 4)
    awc_extent.adjust_to_snap('EXPAND', awc_x, awc_y, awc_cs)
    awc_ds = None

    # SWB computations will be done in the AWC OSR, cellsize, and extent
    awc = drigo.raster_to_array(
        awc_path, band=1, mask_extent=awc_extent,
        return_nodata=False).astype(np.float32)
    # Clip/project AWC to Landsat scene
    # awc = clip_project_raster_func(
    #     awc_path, 1, gdal.GRA_NearestNeighbour,
    #     awc_osr, awc_cs, awc_extent,
    #     output_osr, output_cs, output_extent)

    # Convert units from cm/cm to mm/m
    # awc *= 1000
    # Scale available water capacity by 1000
    # Scale field capacity and wilting point from percentage to decimal
    # fc *= 0.01
    # wp *= 0.01

    # Initialize Soil Water Balance parameters
    # Readily evaporable water (mm)
    rew = 54.4 * awc + 0.8
    # rew = (54.4 * awc / 1000) + 0.8

    # Total evaporable water (mm)
    tew = 166.0 * awc - 3.7
    # tew = (166.0 * awc / 1000) - 3.7

    # Total evaporable water (mm)
    # tew = (fc - 0.5 * wp) * (0.1 * 1000)

    # Difference of TEW and REW
    # tew_rew = tew - rew

    # Dry initial Depletion
    de = np.copy(tew)
    d_rew = np.copy(rew)
    # de = np.copy(tew)
    # d_rew = np.copy(rew)
    # Half Initial Depletion
    # de = 0.5 * tew
    # d_rew = 0.5 * rew
    # Wet Initial Depletion
    # de = 0
    # d_rew = 0

    # Coefficients for Skin Layer Retention Efficiency, Allen (2011)
    c0 = 0.8
    c1 = 2 * (1 - c0)
    # ETr ke max
    ke_max = 1.0
    # ETo Ke max
    # ke_max = 1.2

    # Spinup model up to test date, iteratively calculating Ke
    # Pass doy as band number to raster_value_at_xy
    for spinup_dt in dt_list:
        logging.debug('    {}'.format(spinup_dt.date().isoformat()))
        etr = clip_project_raster_func(
            etr_path_dict[str(spinup_dt.year)],
            # int(spinup_dt.strftime('%j')), gdal.GRA_NearestNeighbour,
            int(spinup_dt.strftime('%j')), gdal.GRA_Bilinear,
            etr_osr, etr_cs, etr_extent, awc_osr, awc_cs, awc_extent)
        ppt = clip_project_raster_func(
            ppt_path_dict[str(spinup_dt.year)],
            # int(spinup_dt.strftime('%j')), gdal.GRA_NearestNeighbour,
            int(spinup_dt.strftime('%j')), gdal.GRA_Bilinear,
            ppt_osr, ppt_cs, ppt_extent, awc_osr, awc_cs, awc_extent)
        ke, de, d_rew = daily_swb_func(
            etr, ppt, de, d_rew, rew, tew, c0, c1, ke_max)

    # Project to output spatial reference, cellsize, and extent
    ke = drigo.project_array(
        # ke, gdal.GRA_NearestNeighbour,
        ke, gdal.GRA_Bilinear,
        awc_osr, awc_cs, awc_extent,
        output_osr, output_cs, output_extent,
        output_nodata=None)

    return ke


def clip_project_raster_func(input_raster, band, resampling_type,
                             input_osr, input_cs, input_extent,
                             ouput_osr, output_cs, output_extent):
    """Clip and then project an input raster
    
    Parameters
    ----------
    input_raster
    band
    resampling_type
    input_osr
    input_cs
    input_extent
    ouput_osr
    output_cs
    output_extent

    Returns
    -------
    ndarray

    Notes
    -----
    This function is only called by the raster_swb_func and could eventually
    be moved to the drigo module.

    """
    # Read array from input raster using input extent
    input_array = drigo.raster_to_array(
        input_raster, band=band, mask_extent=input_extent,
        return_nodata=False).astype(np.float32)

    # Convert nan to a nodata value so a copy isn't made in project_array
    input_array[np.isnan(input_array)] = drigo.numpy_type_nodata(
        input_array.dtype)

    # Project and clip array to block
    output_array = drigo.project_array(
        input_array, resampling_type,
        input_osr, input_cs, input_extent,
        ouput_osr, output_cs, output_extent)
    return output_array


def point_swb_func(test_dt, test_xy, test_osr, awc_path,
                   etr_ws, etr_re, ppt_ws, ppt_re,
                   spinup_days=30, min_spinup_days=5):
    """Compute the daily soil water balance at a point for a single date

    Parameters
    ----------
    test_dt : class:`datetime.datetime`
        Target date.
    test_xy : tuple
        Tuple of the x and y coordinates for which the soil water balance is
        to be calculated. Must be in the same projection as the test_osr.
    test_osr : class:`osr.SpatialReference
        Spatial reference of the text_xy point coordinates.
    awc_path : str
        Filepath of the available water content raster.
    etr_ws : str
        Directory path of the ETr workspace, which Contains separate rasters 
        for each year.
    etr_re : class:`re`
        Compiled regular expression object from the Python native 're' module
        that will match ETr rasters.
    ppt_ws : str
        Directory path of the precipitation workspace, which contains separate
        rasters for each year.
    ppt_re : class:`re`
        Compiled regular expression object from the native Python re module
        that will match precipitaiton rasters.
    spinup_days : int, optional
        Number of days that should be used in the spinup of the model
        (the default is 30).
    min_spinup_days : int, optinal
        Minimum number of days that are needed for spinup of the model
        (the default is 5).

    Returns
    -------
    ke : float
        Soil evaporation coefficient.

    Notes
    -----
    Spinup SWB model for N spinup dates and calculate the Ke (soil evaporation
    coefficient) for the desired x/y coordinates.

    References
    ----------
    .. [1] Allen, R., Pereira, L., Smith, M., Raes, D., & Wright, J. (2005).
       FAO-56 dual crop coefficient method for estimating evaporation from soil
       and application extensions.
       Jourlnal of Irrigation and Drainage Engineering, 131(1).
       10.1061/(ASCE)0733-9437(2005)131:1(2)
    .. [2] Allen, R. (2011). Skin layer evaporation to account for small
       precipitation events — an enhancement to the FAO-56 evaporation model.
       Agricultural Water Management, 99.
       https://doi.org/10.1016/j.agwat.2011.08.008

    """
    # DEADBEEF - There is probably a better way to handle the date range input.
    #  Perhaps something like setting a minimum spinup and maximum spinup
    #  days and allowing the code to take however many etr and ppt rasters
    #  it can find within that range is good. Also, we should probably
    #  add in a flag for dry vs wet starting point (when it comes to
    #  total evaporative water [tew])
    logging.debug('Daily Soil Water Balance')
    logging.debug('  Test Point: {} {}'.format(*test_xy))

    # Compute list of dates needed for spinup
    # date_range function doesn't return end date so add 1 day to end
    dt_list = sorted(python_common.date_range(
        test_dt - dt.timedelta(days=spinup_days),
        test_dt + dt.timedelta(days=1)))
    year_list = sorted(list(set([d.year for d in dt_list])))

    # Get all available ETr and PPT paths in date range
    etr_path_dict = dict()
    ppt_path_dict = dict()
    if not os.path.isdir(etr_ws):
        logging.error('  ETr folder does not exist\n    {}'.format(
            etr_ws))
        sys.exit()
    if not os.path.isdir(ppt_ws):
        logging.info('  PPT folder does not exist\n    {}'.format(
            ppt_ws))
        sys.exit()

    # DEADBEEF - What specific errors should be caught here?
    for etr_name in os.listdir(etr_ws):
        try:
            test_year = etr_re.match(etr_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            etr_path_dict[str(test_year)] = os.path.join(etr_ws, etr_name)
    for ppt_name in os.listdir(ppt_ws):
        try:
            test_year = ppt_re.match(ppt_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            ppt_path_dict[str(test_year)] = os.path.join(ppt_ws, ppt_name)
    if not etr_path_dict:
        logging.error('  No ETr rasters were found for the point SWB\n')
        sys.exit()
    elif not ppt_path_dict:
        logging.error('  No PPT rasters were found for the point SWB\n')
        sys.exit()

    # for year in year_list:
    #     etr_year_ws = os.path.join(etr_ws, str(year))
    #     if os.path.isdir(etr_year_ws):
    #         for etr_name in os.listdir(etr_year_ws):
    #             try:
    #                 test_dt = dt.datetime.strptime(
    #                     etr_re.match(etr_name).group('YYYYMMDD'), '%Y%m%d')
    #             except:
    #                 continue
    #             if test_dt in dt_list:
    #                 etr_path_dict[test_dt.date().isoformat()] = os.path.join(
    #                     etr_year_ws, etr_name)
    #     else:
    #         logging.info('  ETr year folder does not exist\n    {}'.format(
    #             etr_year_ws))

    #     ppt_year_ws = os.path.join(ppt_ws, str(year))
    #     if os.path.isdir(ppt_year_ws):
    #         for ppt_name in os.listdir(ppt_year_ws):
    #             try:
    #                 test_dt = dt.datetime.strptime(
    #                     ppt_re.match(ppt_name).group('YYYYMMDD'), '%Y%m%d')
    #             except:
    #                 continue
    #             if test_dt in dt_list:
    #                 ppt_path_dict[test_dt.date().isoformat()] = os.path.join(
    #                     ppt_year_ws, ppt_name)
    #     else:
    #         logging.info('  PPT year folder does not exist\n    {}'.format(
    #             ppt_year_ws))

    # DEADBEEF - Need a different way to check for spin up dates
    # # Check the number of available ETr/PPT images
    # etr_spinup_days = len(etr_path_dict.keys()) - 1
    # ppt_spinup_days = len(ppt_path_dict.keys()) - 1
    # if etr_spinup_days < spinup_days:
    #     logging.warning('  Only {}/{} ETr spinup days available'.format(
    #         etr_spinup_days, spinup_days))
    #     if etr_spinup_days <= min_spinup_days:
    #         logging.error('    Exiting')
    #         exit()
    # if ppt_spinup_days < spinup_days:
    #     logging.warning('  Only {}/{} PPT spinup days available'.format(
    #         ppt_spinup_days, spinup_days))
    #     if ppt_spinup_days <= min_spinup_days:
    #         logging.error('    Exiting')
    #         sys.exit()

    # Project input point to AWC coordinate system
    awc_pnt = ogr.Geometry(ogr.wkbPoint)
    awc_pnt.AddPoint(test_xy[0], test_xy[1])
    awc_pnt.Transform(osr.CoordinateTransformation(
        test_osr, drigo.raster_path_osr(awc_path)))
    logging.debug('  AWC Point: {} {}'.format(
        awc_pnt.GetX(), awc_pnt.GetY()))

    # Project input point to ETr coordinate system
    etr_pnt = ogr.Geometry(ogr.wkbPoint)
    etr_pnt.AddPoint(test_xy[0], test_xy[1])
    etr_pnt.Transform(osr.CoordinateTransformation(
        test_osr, drigo.raster_path_osr(list(etr_path_dict.values())[0])))
    logging.debug('  ETr Point: {} {}'.format(
        etr_pnt.GetX(), etr_pnt.GetY()))

    # Project input point to PPT coordinate system
    ppt_pnt = ogr.Geometry(ogr.wkbPoint)
    ppt_pnt.AddPoint(test_xy[0], test_xy[1])
    ppt_pnt.Transform(osr.CoordinateTransformation(
        test_osr, drigo.raster_path_osr(list(ppt_path_dict.values())[0])))
    logging.debug('  PPT Point: {} {}'.format(
        ppt_pnt.GetX(), ppt_pnt.GetY()))

    # Read in soil properties
    awc = drigo.raster_value_at_point(awc_path, awc_pnt)
    # Convert units from cm/cm to mm/m
    # awc *= 1000
    # fc = drigo.raster_value_at_point(fc_path, test_pnt)
    # wp = drigo.raster_value_at_point(wp_path, test_pnt)
    # Scale available water capacity by 1000
    # Scale field capacity and wilting point from percentage to decimal
    # fc *= 0.01
    # wp *= 0.01

    # Initialize Soil Water Balance parameters
    # Readily evaporable water (mm)
    rew = 54.4 * awc + 0.8
    # rew = (54.4 * awc / 1000) + 0.8
    # Total evaporable water (mm)
    tew = 166.0 * awc - 3.7
    # tew = (166.0 * awc / 1000) - 3.7
    # Total evaporable water (mm)
    # tew = (fc - 0.5 * wp) * (0.1 * 1000)
    # Difference of TEW and REW
    # tew_rew = tew - rew

    # Dry initial Depletion
    de = float(tew)
    d_rew = float(rew)
    # de = np.copy(tew)
    # d_rew = np.copy(rew)
    # Half Initial Depletion
    # de = 0.5 * tew
    # d_rew = 0.5 * rew
    # Wet Initial Depletion
    # de = 0
    # d_rew = 0

    # Coefficients for Skin Layer Retention Efficiency, Allen (2011)
    c0 = 0.8
    c1 = 2 * (1 - c0)
    # ETr ke max
    ke_max = 1.0
    # ETo Ke max
    # ke_max = 1.2

    logging.debug('  AWC: {}'.format(awc))
    # logging.debug('  FC:  {}'.format(fc))
    # logging.debug('  WP:  {}'.format(wp))
    logging.debug('  REW: {}'.format(rew))
    logging.debug('  TEW: {}'.format(tew))
    logging.debug('  de:  {}'.format(de))
    logging.debug(
        '\n  {:>10s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}'.format(
            *'DATE,ETR,PPT,KE,DE,D_REW'.split(',')))

    # Spinup model up to test date, iteratively calculating Ke
    # Pass doy as band number to raster_value_at_point
    for spinup_dt in dt_list:
        etr, ppt = 0., 0.
        try:
            etr = drigo.raster_value_at_point(
                etr_path_dict[str(spinup_dt.year)], etr_pnt,
                band=int(spinup_dt.strftime('%j')))
        except KeyError:
            logging.debug(
                '  ETr raster for date {} does not exist'.format(
                    spinup_dt.date().isoformat()))
        try:
            ppt = drigo.raster_value_at_point(
                ppt_path_dict[str(spinup_dt.year)], ppt_pnt,
                band=int(spinup_dt.strftime('%j')))
        except KeyError:
            logging.debug(
                '  PPT raster for date {} does not exist'.format(
                    spinup_dt.date().isoformat()))

        ke, de, d_rew = map(float, daily_swb_func(
            etr, ppt, de, d_rew, rew, tew, c0, c1, ke_max))
        logging.debug((
            '  {:>10s} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}').format(
                spinup_dt.date().isoformat(), etr, ppt, ke, de, d_rew))
        
    return ke


def array_swb_func(etr, ppt, awc):
    """Iteratively compute the daily soil water balance through an array stack

    Parameters
    ----------
    etr : ndarray
        Daily reference ET [mm].
    ppt : ndarray
        Daily Precipitaiton [mm].
    awc : array_like
        Available water content [mm].

    Returns
    -------
    ke : ndarray
        Soil evaporation coefficient.

    Notes
    -----
    Script will assume the 0th axis of the input arrays is time.

    Spinup days are assumed to be in the data.

    References
    ----------
    .. [1] Allen, R., Pereira, L., Smith, M., Raes, D., & Wright, J. (2005).
       FAO-56 dual crop coefficient method for estimating evaporation from soil
       and application extensions.
       Jourlnal of Irrigation and Drainage Engineering, 131(1).
       10.1061/(ASCE)0733-9437(2005)131:1(2)
    .. [2] Allen, R. (2011). Skin layer evaporation to account for small
       precipitation events — an enhancement to the FAO-56 evaporation model.
       Agricultural Water Management, 99.
       https://doi.org/10.1016/j.agwat.2011.08.008

    """
    # logging.debug('Daily Soil Water Balance')
    ke = np.full(etr.shape, np.nan, np.float32)

    # Initialize Soil Water Balance parameters
    # Readily evaporable water (mm)
    rew = 54.4 * awc + 0.8
    # rew = (54.4 * awc / 1000) + 0.8

    # Total evaporable water (mm)
    tew = 166.0 * awc - 3.7
    # tew = (166.0 * awc / 1000) - 3.7

    # Total evaporable water (mm)
    # tew = (fc - 0.5 * wp) * (0.1 * 1000)

    # Difference of TEW and REW
    # tew_rew = tew - rew

    # Half Initial Depletion
    # de = 0.5 * tew
    # d_rew = 0.5 * rew

    # Dry initial Depletion
    de = np.copy(tew)
    d_rew = np.copy(rew)

    # Wet Initial Depletion
    # de = 0
    # d_rew = 0

    # Coefficients for Skin Layer Retention Efficiency, Allen (2011)
    c0 = 0.8
    c1 = 2 * (1 - c0)

    # ETr ke max
    ke_max = 1.0

    for i in range(etr.shape[0]):
        ke[i], de, d_rew = daily_swb_func(
            etr[i], ppt[i], de, d_rew, rew, tew, c0, c1, ke_max)
    return ke


def daily_swb_func(etr, ppt, de_prev, d_rew_prev, rew, tew,
                   c0=0.8, c1=0.4, ke_max=1.0):
    """Compute the daily soil water balance for a single time step

    Parameters
    ----------
    etr : array_like
        Daily reference ET [mm].
    ppt : array_like
        Precipitation [mm].
    de_prev : 
    d_rew_prev : 
    rew : 
    tew : 
    c0 :
        (the default is 0.8).
    c1 :
        (the default is 0.4).
    ke_max :
        (the default is 1.0).

    Returns
    -------
    tuple: numpy.arrays (ke, de, d_rew)

    References
    ----------
    .. [1] Allen, R., Pereira, L., Smith, M., Raes, D., & Wright, J. (2005).
       FAO-56 dual crop coefficient method for estimating evaporation from soil
       and application extensions.
       Jourlnal of Irrigation and Drainage Engineering, 131(1).
       https://10.1061/(ASCE)0733-9437(2005)131:1(2)
    .. [2] Allen, R. (2011). Skin layer evaporation to account for small
       precipitation events — an enhancement to the FAO-56 evaporation model.
       Agricultural Water Management, 99.
       https://doi.org/10.1016/j.agwat.2011.08.008

    """
    # Stage 1 evaporation (Eqn 1)
    e1 = np.array(etr, copy=True, ndmin=1)
    e1 *= ke_max

    # Evaporation reduction coefficient (Eqn 5b)
    kr = np.clip((tew - de_prev) / (tew - rew), 0, 1)

    # Fraction of time interval residing in stage 1 (Eqn 10b)
    # Don't calculate ".min(1)" here, wait until in Es calc
    ft = np.clip(np.nan_to_num((rew - d_rew_prev) / e1), 0, 1)

    # Total evaporation from the soil (Eqn 11)
    es = np.clip((1 - ft) * kr * e1 - d_rew_prev + rew, 0, e1)

    # Infiltration efficiency factor (Eqn 13)
    ceff = np.clip(c1 * ((tew - de_prev) / tew) + c0, 0, 1)

    # Depletion of the skin layer
    # With skin evap calculation (Eqn 12)
    # Modified to remove fb adjustment
    d_rew = np.copy(np.clip((d_rew_prev - (ceff * ppt) + es), 0, rew))
    # d_rew = np.clip((d_rew_prev - (ceff * ppt) + es), 0, rew)

    # Without skin evap (Eqn )
    # var d_rew = rew
    # Depth of evaporation of the TEW surface soil layer (Eqn 9)
    # Modified to remove fb adjustment
    de = np.copy(np.clip(de_prev - ppt + es, 0, tew))
    # de = np.clip(de_prev - ppt + es, 0, tew)

    # # Save current as previous for next iteration
    # de_prev = de
    # d_rew_prev = d_rew

    # Evaporation coefficient (Eqn )
    ke = np.clip(np.nan_to_num(es / etr), 0, 1)
    
    return ke, de, d_rew
