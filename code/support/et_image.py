#--------------------------------
# Name:         et_image.py
# Purpose:      ET image class
#--------------------------------

import datetime as dt
import logging
import math
import os
import re

import drigo
import numpy as np
from osgeo import gdal, ogr, osr

import et_common


class InvalidImage(Exception):
    """ Easy to understand naming conventions work best! """
    pass


class Image:
    InvalidImage = InvalidImage

    def __init__(self, image_folder, env=None):
        """

        Parameters
        ----------
        image_folder : str
        env : drigo.env
            Environment class used in gdal_common

        References
        ----------
        .. [1] Chander, G., Markham, B., Helder, D. (2009).
            Summary of current radiometric calibration coefficients for
            Landsat MSS, TM, ETM+, and EO-1 ALI sensors.
            Remote Sensing of Environment, 113(5)
            https://doi.org/10.1016/j.rse.2009.01.007
        .. [2] Tasumi, M., Allen, R., and Trezza, R. (2008).
            At-surface reflectance and albedo from satellite for operational
            calculation of land surface energy balance.
            Journal of Hydrologic Engineering, 13(2), 51-63.
            https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)

        """
        # Initial image properties can be set just based on the image folder
        self.folder_id = os.path.split(image_folder)[-1]
        folder_re = re.compile(
            '^(?P<prefix>\w{4})_(?P<path>\d{3})(?P<row>\d{3})_'
            '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})$')
        # landsat_re = re.compile(
        #     '^(LT04|LT05|LE07|LC08)_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})')
        # mtl_re = '%s\D{3}\d{2}_MTL.txt' % self.folder_id

        prefix_list = ['LT04', 'LT05', 'LE07', 'LC08']

        folder_match = folder_re.match(self.folder_id)
        if (not folder_match or
                folder_match.group('prefix') not in prefix_list):
            logging.error(
                '\nERROR: The sensor type could not be determined from '
                'the folder name: {}\nERROR: Only Landsat 4, 5, 7, and 8 '
                'are currently supported.\n'.format(self.folder_id))
            raise InvalidImage
        self.prefix = folder_match.group('prefix')
        self.qa_band = 'QA'
        if self.prefix == 'LT04':
            self.type = 'Landsat4'
            self.thermal_band = '6'
            self.band_toa_dict = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 6}
            self.band_sur_dict = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 6}
        elif self.prefix == 'LT05':
            self.type = 'Landsat5'
            self.thermal_band = '6'
            self.band_toa_dict = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 6}
            self.band_sur_dict = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 6}
        elif self.prefix == 'LE07':
            self.type = 'Landsat7'
            self.thermal_band = '6'
            self.band_toa_dict = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 6}
            self.band_sur_dict = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 6}
        elif self.prefix == 'LC08':
            self.type = 'Landsat8'
            self.thermal_band = '10'
            self.band_toa_dict = {
                '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6}
            self.band_sur_dict = {
                '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6}

        # Initially match input TIF images using folder ID
        # Use a really simple regular expression to support different
        # Landsat ID formats
        self.image_re = re.compile('L\w+_(?P<band>B(?:\w+)).TIF')

        self.band_toa_cnt = len(self.band_toa_dict.keys())
        self.band_sur_cnt = len(self.band_sur_dict.keys())
        # Mask of SUR bands in TOA array
        # This is used when TOA and SUR arrays have different bands
        self.band_toa_sur_mask = np.array([
            (v-1) for k, v in sorted(self.band_toa_dict.items())
            if k in self.band_sur_dict.keys()])

        # Image date (from folder name)
        self.folder_year = int(folder_match.group('year'))
        self.folder_month = int(folder_match.group('month'))
        self.folder_day = int(folder_match.group('day'))
        self.folder_dt = dt.datetime(
            self.folder_year, self.folder_month, self.folder_day)
        self.folder_doy = int(self.folder_dt.strftime('%j'))

        # TOA reflectance, at-surface reflectance and albedo coefficients
        self.lmax_dict = {}
        self.lmin_dict = {}
        self.qcalmax_dict = {}
        self.qcalmin_dict = {}
        # Only used by Landsat 8, but in newer versions of all MTL files
        self.refl_mult_dict = {}
        self.refl_add_dict = {}
        self.rad_mult_dict = {}
        self.rad_add_dict = {}
        self.k1_dict = {}
        self.k2_dict = {}

        if self.type in ['Landsat4', 'Landsat5', 'Landsat7']:
            # "c" coefficients from Chander et al. 2009
            # "wb" coefficients from Trezza et al. 2008
            self.c1_dict = {'1': 0.987, '2': 2.319, '3': 0.951,
                            '4': 0.375, '5': 0.234, '7': 0.365}
            self.c2_dict = {'1': -0.000710, '2': -0.000164, '3': -0.000329,
                            '4': -0.000479, '5': -0.001012, '7': -0.000966}
            self.c3_dict = {'1': 0.000036, '2': 0.000105, '3': 0.00028,
                            '4': 0.005018, '5': 0.004336, '7': 0.004296}
            self.c4_dict = {'1': 0.0880, '2': 0.0437, '3': 0.0875,
                            '4': 0.1355, '5': 0.0560, '7': 0.0155}
            self.c5_dict = {'1': 0.0789, '2': -1.2697, '3': 0.1014,
                            '4': 0.6621, '5': 0.7757, '7': 0.639}
            self.cb_dict = {'1': 0.640, '2': 0.31, '3': 0.286,
                            '4': 0.189, '5': 0.274, '7': -0.186}
            self.wb_dict = {'1': 0.254, '2': 0.149, '3': 0.147,
                            '4': 0.311, '5': 0.103, '7': 0.036}
        elif self.type in ['Landsat8']:
            # "c" coefficients from
            # "wb" coefficients from
            self.c1_dict = {'2': 0.987, '3': 2.148, '4': 0.942,
                            '5': 0.248, '6': 0.260, '7': 0.315}
            self.c2_dict = {'2': -0.000727, '3': -0.000199, '4': -0.000261,
                            '5': -0.000410, '6': -0.001084, '7': -0.000975}
            self.c3_dict = {'2': 0.000037, '3': 0.000058, '4': 0.000406,
                            '5': 0.000563, '6': 0.000675, '7': 0.004012}
            self.c4_dict = {'2': 0.0869, '3': 0.0464, '4': 0.0928,
                            '5': 0.2256, '6': 0.0632, '7': 0.0116}
            self.c5_dict = {'2': 0.0788, '3': -1.0962, '4': 0.1125,
                            '5': 0.7991, '6': 0.7549, '7': 0.6906}
            self.cb_dict = {'2': 0.640, '3': 0.310, '4': 0.286,
                            '5': 0.189, '6': 0.274, '7': -0.186}
            self.wb_dict = {'2': 0.254, '3': 0.149, '4': 0.147,
                            '5': 0.311, '6': 0.103, '7': 0.036}

        # Convert dictionaries to arrays
        self.c1 = et_common.band_dict_to_array(self.c1_dict, self.band_sur_dict)
        self.c2 = et_common.band_dict_to_array(self.c2_dict, self.band_sur_dict)
        self.c3 = et_common.band_dict_to_array(self.c3_dict, self.band_sur_dict)
        self.c4 = et_common.band_dict_to_array(self.c4_dict, self.band_sur_dict)
        self.c5 = et_common.band_dict_to_array(self.c5_dict, self.band_sur_dict)
        self.cb = et_common.band_dict_to_array(self.cb_dict, self.band_sur_dict)
        self.wb = et_common.band_dict_to_array(self.wb_dict, self.band_sur_dict)

        # Coefficients from Chander 2009
        # DEADBEEF - Aren't these in the collection 1 MTL file for all Landsat?
        #   It may not be necessary to set them again here.
        if self.type in ['Landsat4']:
            self.k1_dict = {self.thermal_band: 671.62}
            self.k2_dict = {self.thermal_band: 1284.30}
            self.esun_dict = {
                '1': 1983., '2': 1795., '3': 1539., '4': 1028.,
                '5': 219.8, '6': 1., '7': 83.49}
        elif self.type in ['Landsat5']:
            self.k1_dict = {self.thermal_band: 607.76}
            self.k2_dict = {self.thermal_band: 1260.56}
            self.esun_dict = {
                '1': 1983., '2': 1796., '3': 1536., '4': 1031.,
                '5': 220., '6': 1., '7': 83.44}
        elif self.type in ['Landsat7']:
            self.k1_dict = {self.thermal_band: 666.09}
            self.k2_dict = {self.thermal_band: 1282.71}
            self.esun_dict = {
                '1': 1997., '2': 1812., '3': 1533., '4': 1039.,
                '5': 230.8, '6': 1., '7': 84.9}
                # '5': 230.8, '6_VCID_1': 1., '6_VCID_2': 1., '7': 84.9}
        elif self.type in ['Landsat8']:
            # For Landsat 8, K is read from MTL and ESUN is not currently used
            pass

        # Set folder paths
        self.set_folder_paths(image_folder, self.type)

        # Read Landsat MTL file
        self.image_id = None
        self.mtl_path = None
        if os.path.isdir(self.orig_data_ws):
            # Use a really simple regular expression to support
            # different Landsat ID formats
            mtl_list = [
                os.path.join(self.orig_data_ws, item)
                for item in os.listdir(self.orig_data_ws)
                if re.match('L\w+_MTL.txt', item)]
            if mtl_list:
                self.mtl_path = mtl_list[0]
                self.image_id = os.path.basename(self.mtl_path).split('_MTL')[0]
                logging.debug('  {:<18s} {}'.format(
                    'MTL File:', os.path.basename(self.mtl_path)))
                self.read_mtl(self.mtl_path)
                self.process_mtl()

        # After finding/reading MTL file, update image regular expression
        if self.image_id is not None:
            if self.prefix in ['LT04', 'LT05']:
                # 0? outside band capturing group is for old style ID
                #   where band numbers was two digits
                self.image_re = re.compile(
                    '%s_(?P<band>B(?:[1-7]|QA)).TIF$' % self.image_id)
            elif self.prefix == 'LE07':
                self.image_re = re.compile(
                    '%s_(?P<band>B(?:[1-7]|QA)(?:_VCID_1)?).TIF$' % self.image_id)
            elif self.prefix == 'LC08':
                self.image_re = re.compile(
                    '%s_(?P<band>B(?:[234567]|10|QA)).TIF$' % self.image_id)

        # Set snap parameters to environment
        if env is not None:
            env.cellsize = 30
            env.snap_xmin, env.snap_ymin = 15, 15
            env.snap_osr = osr.SpatialReference()
            # From Landsat MTL file
            if self.mtl_path is not None:
                env.snap_osr.ImportFromEPSG(self.epsg)
                env.snap_proj = env.snap_osr.ExportToWkt()
                env.snap_gcs_osr = env.snap_osr.CloneGeogCS()
                env.snap_gcs_proj = env.snap_gcs_osr.ExportToWkt()
            env.snap_geo = (
                env.snap_xmin, env.cellsize, 0.,
                env.snap_ymin, 0., -env.cellsize)

        # Set raster names
        self.set_raster_paths(image_folder)

    def read_mtl(self, mtl_path):
        """

        Parameters
        ----------
        mtl_path

        Returns
        -------

        """
        with open(mtl_path, 'r') as mtl_f:
            mtl_lines = mtl_f.readlines()

        # Use capturing groups to get the band number/name string
        # Landsat 4, 5, & 7
        # For older Landsat 5 and 7, process band "N0" as "N"
        # For Landsat 7, process band "6_VCID_1" as "6" and ignore "6_VCID_2"
        lmax_re = re.compile(
            '^(?:RADIANCE_MAXIMUM_BAND_|LMAX_BAND)' +
            '(?P<band>[1-7])(?:0|1|_VCID_[1])?')
        lmin_re = re.compile(
            '^(?:RADIANCE_MINIMUM_BAND_|LMIN_BAND)' +
            '(?P<band>[1-7])(?:0|1|_VCID_[1])?')
        qcalmax_re = re.compile(
            '^(?:QUANTIZE_CAL_MAX_BAND_|QCALMAX_BAND)' +
            '(?P<band>[1-7])(?:0|1|_VCID_[1])?')
        qcalmin_re = re.compile(
            '^(?:QUANTIZE_CAL_MIN_BAND_|QCALMIN_BAND)' +
            '(?P<band>[1-7])(?:0|1|_VCID_[1])?')
        # gain_re = re.compile(
        #     '^GAIN_BAND_(?P<band>[1-7])(?:0|1|_VCID_[12])?')

        # Landsat 8
        refl_mult_re = re.compile('^REFLECTANCE_MULT_BAND_(?P<band>\d+)')
        refl_add_re = re.compile('^REFLECTANCE_ADD_BAND_(?P<band>\d+)')
        rad_mult_re = re.compile('^RADIANCE_MULT_BAND_(?P<band>\d+)')
        rad_add_re = re.compile('^RADIANCE_ADD_BAND_(?P<band>\d+)')
        k1_re = re.compile('^K1_CONSTANT_BAND_(?P<band>10|11)')
        k2_re = re.compile('^K2_CONSTANT_BAND_(?P<band>10|11)')

        # Read header by line
        for mtl_line in mtl_lines:
            mtl_line = [i.strip() for i in mtl_line.split('=')]
            if mtl_line[0] == 'END':
                break
            elif ('DATA_TYPE' == mtl_line[0] or
                  'PRODUCT_TYPE' == mtl_line[0]):
                self.correction = mtl_line[1].replace('"', '')
            # DEADBEEF
            # elif (('DATA_TYPE' == mtl_line[0] or
            #        'PRODUCT_TYPE' == mtl_line[0]) and
            #       mtl_line[1] != '"L1T"'):
            #     logging.error(
            #         ('\nERROR: Scene is not georeferenced '+
            #          '(type: {})\n').format(mtl_line[1]))
            #     sys.exit()
            elif 'WRS_PATH' == mtl_line[0]:
                self.path = int(mtl_line[1])
            elif 'WRS_ROW' == mtl_line[0]:
                self.row = int(mtl_line[1])
            elif 'SUN_AZIMUTH' == mtl_line[0]:
                self.sun_azimuth = float(mtl_line[1])
            elif 'SUN_ELEVATION' == mtl_line[0]:
                self.sun_elevation = float(mtl_line[1])
            elif ('DATE_ACQUIRED' == mtl_line[0] or
                  'ACQUISITION_DATE' == mtl_line[0]):
                acq_date_list = mtl_line[1].split('-')
                # self.acq_year = int(acq_date_list[0])
                # self.acq_mon = int(acq_date_list[1])
                # self.acq_day = int(acq_date_list[2])
            elif ('SCENE_CENTER_TIME' in mtl_line[0] or
                  'SCENE_CENTER_SCAN_TIME' in mtl_line[0]):
                acq_time_list = mtl_line[1].strip('Z" ').split(':')
                # self.acq_hour = int(acq_time_list[0])
                # self.acq_min  = int(acq_time_list[1])
                # self.acq_sec  = int(acq_time_list[2].split('.')[0])
                # self.acq_msec = int(acq_time_list[2].split('.')[1][:6])
            # Landsat 4, 5, & 7
            elif lmax_re.match(mtl_line[0]):
                band_str = lmax_re.match(mtl_line[0]).group('band')
                self.lmax_dict[band_str] = float(mtl_line[1])
            elif lmin_re.match(mtl_line[0]):
                band_str = lmin_re.match(mtl_line[0]).group('band')
                self.lmin_dict[band_str] = float(mtl_line[1])
            elif qcalmax_re.match(mtl_line[0]):
                band_str = qcalmax_re.match(mtl_line[0]).group('band')
                self.qcalmax_dict[band_str] = float(mtl_line[1])
            elif qcalmin_re.match(mtl_line[0]):
                band_str = qcalmin_re.match(mtl_line[0]).group('band')
                self.qcalmin_dict[band_str] = float(mtl_line[1])
            # Landsat 8
            elif refl_mult_re.match(mtl_line[0]):
                band_str = refl_mult_re.match(mtl_line[0]).group('band')
                self.refl_mult_dict[band_str] = float(mtl_line[1])
            elif refl_add_re.match(mtl_line[0]):
                band_str = refl_add_re.match(mtl_line[0]).group('band')
                self.refl_add_dict[band_str] = float(mtl_line[1])
            elif rad_mult_re.match(mtl_line[0]):
                band_str = rad_mult_re.match(mtl_line[0]).group('band')
                self.rad_mult_dict[band_str] = float(mtl_line[1])
            elif rad_add_re.match(mtl_line[0]):
                band_str = rad_add_re.match(mtl_line[0]).group('band')
                self.rad_add_dict[band_str] = float(mtl_line[1])
            elif k1_re.match(mtl_line[0]):
                band_str = k1_re.match(mtl_line[0]).group('band')
                self.k1_dict[band_str] = float(mtl_line[1])
            elif k2_re.match(mtl_line[0]):
                band_str = k2_re.match(mtl_line[0]).group('band')
                self.k2_dict[band_str] = float(mtl_line[1])
            #
            elif 'SPACECRAFT_ID' == mtl_line[0]:
                self.satellite = mtl_line[1].strip('\' ')
            elif ('UTM_ZONE' == mtl_line[0] or
                  'ZONE_NUMBER' == mtl_line[0]):
                self.utm_zone = int(mtl_line[1].strip('\' '))
        # Save acquistion date and time to datetime object
        self.acq_datetime = dt.datetime(*map(int, [
            acq_date_list[0], acq_date_list[1], acq_date_list[2],
            acq_time_list[0], acq_time_list[1],
            acq_time_list[2].split('.')[0],
            acq_time_list[2].split('.')[1][:6]]))

    def process_mtl(self):
        """

        Returns
        -------

        """
        # Acquisition year
        self.acq_year = self.acq_datetime.year

        # Calculate GMT image acquisition date
        self.acq_date = self.acq_datetime.replace(
            hour=0, minute=0, second=0, microsecond=0)

        # Standard clock time of satelite overpass
        #   GMT in hours (exp 14:30 = 14.5)
        self.acq_time = ((self.acq_datetime - self.acq_date).seconds) / 3600.

        # Day of year
        self.acq_doy = int(self.acq_datetime.timetuple().tm_yday)

        # Sun-Earth Distance
        self.dr = et_common.dr_func(self.acq_doy)

        # Calculate sun elevation cos_theta
        self.cos_theta_solar = et_common.cos_theta_solar_func(
            self.sun_elevation)

        # All US Landsat scenes are WGS 84 Zone XXN -> EPSG 326XX
        self.epsg = 32600 + int(self.utm_zone)
        # Display Header Information
        logging.debug('  MTL File Variables')
        log_f = '  {:<18s} {}'
        logging.debug(log_f.format('Sensor (Folder):', self.type))
        logging.debug(log_f.format('Sensor (Header):', self.satellite))
        logging.debug(log_f.format('Scene Path:', self.path))
        logging.debug(log_f.format('Scene Row:', self.row))
        logging.debug(log_f.format('Acq. Date:', self.acq_datetime.date()))
        logging.debug(log_f.format('Acq. DOY:', self.acq_doy))
        logging.debug(log_f.format('Acq. Time:', self.acq_datetime.time()))
        logging.debug(log_f.format('Acq. Time [hour]:', self.acq_time))
        logging.debug(log_f.format('Acq. Time [min]:', self.acq_time * 60))
        logging.debug(log_f.format('Sun Azimuth:', self.sun_azimuth))
        logging.debug(log_f.format('Sun Elevation:', self.sun_elevation))
        logging.debug(log_f.format('dr:', self.dr))
        logging.debug(log_f.format('Cos(theta) Solar:', self.cos_theta_solar))
        logging.debug(log_f.format('UTM Zone:', self.utm_zone))
        logging.debug(log_f.format('EPSG:', self.epsg))
        logging.debug(log_f.format('Correction:', self.correction))

    # DEADBEEF - This might be better labeled as "METRIC" folder paths
    def set_folder_paths(self, image_folder, image_type):
        """

        Parameters
        ----------
        image_folder
        image_type

        Returns
        -------

        """
        self.orig_data_ws = os.path.join(image_folder, 'ORIGINAL_DATA')
        self.support_ws = os.path.join(image_folder, 'SUPPORT_RASTERS')
        self.refl_toa_ws = os.path.join(image_folder, 'REFLECTANCE_TOA')
        self.refl_sur_ws = os.path.join(image_folder, 'REFLECTANCE_SUR')
        self.indices_ws = os.path.join(image_folder, 'INDICES')

    # DEADBEEF - This might be better labeled as "METRIC" raster paths
    def set_raster_paths(self, image_folder):
        """Set raster names relative to image folder

        Parameters
        ----------
        image_folder

        Returns
        -------

        """
        r_fmt = '.img'
        self.common_area_raster = os.path.join(
            self.support_ws, 'common_area' + r_fmt)
        self.refl_sur_tasumi_raster = os.path.join(
            self.refl_sur_ws, 'refl_sur_tasumi' + r_fmt)
        self.refl_sur_ledaps_raster = os.path.join(
            self.refl_sur_ws, 'refl_sur_ledaps' + r_fmt)
        self.ndvi_sur_raster = os.path.join(
            self.indices_ws, 'ndvi' + r_fmt)
        self.ndwi_sur_raster = os.path.join(
            self.indices_ws, 'ndwi' + r_fmt)
        self.lai_sur_raster = os.path.join(
            self.indices_ws, 'lai' + r_fmt)
        self.albedo_sur_raster = os.path.join(
            image_folder, 'albedo_at_sur' + r_fmt)
        self.ts_raster = os.path.join(image_folder, 'ts' + r_fmt)
        self.ke_raster = os.path.join(self.support_ws, 'ke' + r_fmt)

        # self.elev_raster = os.path.join(
        #     self.support_ws, elev_name + r_fmt)
        # self.landuse_raster = os.path.join(
        #     self.support_ws, landuse_name + r_fmt)
        self.fmask_output_raster = os.path.join(
            self.support_ws, 'fmask' + r_fmt)
        self.fmask_cloud_raster = os.path.join(
            self.support_ws, 'cloud_mask_fmask' + r_fmt)
        self.fmask_snow_raster = os.path.join(
            self.support_ws, 'snow_mask_fmask' + r_fmt)
        self.fmask_water_raster = os.path.join(
            self.support_ws, 'water_mask_fmask' + r_fmt)
        self.refl_toa_raster = os.path.join(
            self.refl_toa_ws, 'refl_toa' + r_fmt)
        self.ts_bt_raster = os.path.join(image_folder, 'ts_bt' + r_fmt)
        self.qa_raster = os.path.join(self.support_ws, 'qa' + r_fmt)

        # DEADBEEF - is this the most logical place to put these?
        self.ndvi_toa_raster = os.path.join(
            self.indices_ws, 'ndvi_toa' + r_fmt)
        self.ndwi_toa_raster = os.path.join(
            self.indices_ws, 'ndwi_toa' + r_fmt)
        self.lai_toa_raster = os.path.join(
            self.indices_ws, 'lai_toa' + r_fmt)

        # Hourly/daily weather rasters
        self.metric_ea_raster = os.path.join(
            self.support_ws, 'ea_metric' + r_fmt)
        self.metric_wind_raster = os.path.join(
            self.support_ws, 'wind_metric' + r_fmt)
        self.metric_etr_raster = os.path.join(
            self.support_ws, 'etr_metric' + r_fmt)
        self.metric_etr_24hr_raster = os.path.join(
            self.support_ws, 'etr_24hr_metric' + r_fmt)
        self.metric_tair_raster = os.path.join(
            self.support_ws, 'tair_metric' + r_fmt)

    def convert_solar_time(self, gmt_offset=0):
        """

        Parameters
        ----------
        gmt_offset

        Returns
        -------

        Notes
        -----
        Based on: http://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time

        """
        gdal.AllRegister()
        time_layer = gdal.Open(self.lst_view_time_raster)
        if time_layer is None:
            logging.error('ERROR: Unable to open file {}'.format(
                self.lst_view_time_raster))
            raise SystemExit()

        # Read LST solar time as an array
        time_array = time_layer.ReadAsArray()
        # Set the time array to nan when it's equal to no data value
        time_array = time_array.astype(float)
        time_array[time_array == 255] = np.nan
        self.time_array = time_array
        del time_array

        # Use gdal_common to extract/calculate that lat/long per grid cell
        lat_array, lon_array = drigo.raster_ds_lat_lon_func(time_layer)
        self.lat_array = lat_array
        self.lon_array = lon_array
        del lat_array, lon_array

        # Calculate the local standard time meridian
        lstm = 15 * gmt_offset
        # Calculate B
        b = (360/365) * (self.acq_doy - 81)
        # Calculate the "equation of time"
        eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
        # Time correction factor
        tc = 4 * (self.lon_array - lstm) + eot
        # Calculate local time
        self.local_time_array = self.time_array - (tc/60)

        local_time_ma = np.ma.masked_array(
            self.local_time_array, np.isnan(self.local_time_array))
        self.acq_time = np.ma.extras.median(local_time_ma)
        del local_time_ma

        self.acq_hour = int(self.acq_time)
        self.acq_min = int(round(
            float(self.acq_hour - int(self.acq_hour)) * 60))

        self.acq_datetime = (
            dt.datetime(self.acq_year, 1, 1, self.acq_hour, self.acq_min, 0) +
            dt.timedelta(self.acq_doy - 1))
        self.acq_date = dt.date(
            self.acq_datetime.year, self.acq_datetime.month,
            self.acq_datetime.day)
