#--------------------------------
# Name:         python_common.py
# Purpose:      Common Python Support Functions
# Python:       2.7, 3.5, 3.6
#--------------------------------

import argparse
from calendar import monthrange
import configparser
import datetime as dt
from ftplib import FTP
import glob
import gzip
import logging
import os
import random
import re
import subprocess
import sys
import tarfile
from time import sleep

import requests


def isfloat(s):
    """Test if an object is a float"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def shuffle(input_list):
    """Return a randomly shuffled copy of a list"""
    output_list = list(input_list)
    random.shuffle(output_list)
    return output_list


def count_digits_func(input_value):
    """Return the number of digits in a number"""
    return len(str(abs(input_value)))


def list_flatten(iter_list):
    """Return a flattened copy of a nested list"""
    return list(item for iter_ in iter_list for item in iter_)


def build_file_list(ws, test_re, test_other_re=None):
    """Return a list of files in a folder matching a regular expression"""
    if test_other_re is None:
        test_other_re = re.compile('a^')
    if os.path.isdir(ws):
        return sorted([os.path.join(ws, item) for item in os.listdir(ws)
                       if (os.path.isfile(os.path.join(ws, item)) and
                           test_re.match(item) or test_other_re.match(item))])
    else:
        return []


def date_range(start_date, end_date):
    """Yield datetimes within a date range"""
    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


def month_range_func(doy_range, year):
    """Return a list of months for a given day of year range"""
    month_start = doy2month(year, doy_range[0])
    month_end = doy2month(year, doy_range[-1])
    return list(range(month_start, month_end+1))


def doy2month(year, doy):
    """Return the month for a given day of year and year"""
    doy_dt = dt.datetime(int(year),1,1) + dt.timedelta(doy-1)
    return doy_dt.month


def month2doy(year, month):
    """Return a list of the day of years in a month"""
    start_doy = dt.datetime(year, month, 1).timetuple().tm_yday
    end_doy = start_doy + monthrange(year, month)[1]
    return list(range(start_doy, end_doy))


def valid_date(input_date):
    """Check that a date string is ISO format (YYYY-MM-DD)

    This function is used to check the format of dates entered as command
        line arguments.
    DEADBEEF - It would probably make more sense to have this function
        parse the date using dateutil parser (http://labix.org/python-dateutil)
        and return the ISO format string.

    Parameters
    ----------
    input_date : str

    Returns
    -------
    str

    Raises
    ------
    ArgParse ArgumentTypeError

    """
    try:
        return dt.datetime.strptime(input_date, "%Y-%m-%d").date().isoformat()
    except ValueError:
        msg = "Not a valid date: '{}'.".format(input_date)
        raise argparse.ArgumentTypeError(msg)


def open_ini(ini_path):
    """Open config file"""
    log_fmt = '  {:<18s} {}'
    logging.info(log_fmt.format('INI File:', os.path.basename(ini_path)))
    config = configparser.ConfigParser()
    try:
        config.read(ini_path)
    except IOError:
        logging.error(('\nERROR: INI file does not exist\n'
                       '  {}\n').format(ini_path))
        sys.exit()
    except configparser.MissingSectionHeaderError:
        logging.error('\nERROR: INI file is missing a section header\n'
                      '    Please make sure the following line is at the '
                      'beginning of the file\n[INPUTS]\n')
        sys.exit()
    except Exception as e:
        logging.error(('\nERROR: Unhandled exception reading INI file:\n'
                       '  {}\n').format(ini_path, e))
        logging.error('{}\n'.format(e))
        sys.exit()
    return config


def read_param(param_str, param_default, config, section='INPUTS'):
    """"""
    param_type = type(param_default)
    try:
        if param_type is float:
            param_value = config.getfloat(section, param_str)
        elif param_type is int:
            param_value = config.getint(section, param_str)
        elif param_type is bool:
            param_value = config.getboolean(section, param_str)
        elif param_type is list or param_type is tuple:
            param_value = [
                # i for i in re.split('\W+', config.get('INPUTS', param_str)) if i]
                i.strip() for i in config.get(section, param_str).split(',') if i]
        elif param_type is str or param_default is None:
            param_value = config.get(section, param_str)
            if param_value.upper() == 'NONE':
                param_value = None
        elif param_type is unicode or param_default is None:
            # Intentionally check unicode after str
            # In Python 3 all strings are unicode so this won't be executed
            param_value = str(config.get(section, param_str))
            if param_value.upper() == 'NONE':
                param_value = None
        else:
            logging.error('ERROR: Unknown Input Type: {}'.format(param_type))
            sys.exit()
    except configparser.NoOptionError as e:
        # logging.debug('Missing Parameter: {}'.format(e))
        param_value = param_default
        if type(param_default) is str and param_value.upper() == 'NONE':
            param_value = None
        logging.debug('  NOTE: {} = {}'.format(param_str, param_value))
    except Exception as e:
        logging.exception('\nUnhandled Exception\n{}'.format(e))
        sys.exit()
        # param_value = param_default
        # if type(param_default) is str and param_value.upper() == 'NONE':
        #     param_value = None
        # logging.debug('  NOTE: {} = {}'.format(param_str, param_value))
    return param_value


def call_mp(tup):
    """Pool multiprocessing friendly subprocess call function

    mp.Pool needs all inputs are packed into a single tuple
    Tuple is unpacked and and single processing version of function is called
    """
    return call_sp(*tup)

def call_sp(call_args, call_ws, delay=0, new_window_flag=False,
            call_env=None, shell_flag=False):
    """
    Parameters
    ----------
    call_args: list of command line arguments
    call_ws: path to set subprocess current working directory
    delay: integer, delay each process call randomly 0-n seconds
    new_window_flag: boolean, start process in new window
        Windows only
    call_env: environment parameter to pass to subprocess
    shell_flag: boolean

    Returns
    -------
    bool

    """
    sleep(random.uniform(0, max([0, delay])))

    # Windows needs shell=True
    # Only allow windows to spawn new terminal windows
    if new_window_flag and os.name is 'nt':
        p = subprocess.Popen(
            ['start', 'cmd', '/c'] + call_args,
            cwd=call_ws, env=call_env, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE)
        p.communicate()
    else:
        subprocess.call(
            call_args, cwd=call_ws, env=call_env, shell=shell_flag)
    return True


def parse_int_set(nputstr=""):
    """Return list of numbers given a string of ranges

    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
    """
    selection = set()
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token)-1]
                    for x in range(first, last+1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    # print "Invalid set: " + str(invalid)
    return selection


def remove_file(file_path):
    """Remove a feature/raster and all of its anciallary files"""
    file_ws = os.path.dirname(file_path)
    for file_name in glob.glob(os.path.splitext(file_path)[0]+".*"):
        os.remove(os.path.join(file_ws, file_name))


def extract_targz_mp(tup):
    """Pool multiprocessing friendly tar.gz extract function

    mp.Pool needs all inputs are packed into a single tuple
    Tuple is unpacked and and single processing version of function is called
    """
    return extract_targz_func(*tup)


def extract_targz_func(input_path, output_ws):
    """"""
    print('    {}'.format(os.path.basename(input_path)))
    try:
        input_tar = tarfile.open(input_path, 'r:gz')
        input_tar.extractall(output_ws)
        input_tar.close()
    except IOError:
        print('    IOError')


def extract_hdfgz_mp(tup):
    """Pool multiprocessing friendly hdf extract function

    mp.Pool needs all inputs are packed into a single tuple
    Tuple is unpacked and and single processing version of function is called
    """
    return extract_hdfgz_func(*tup)


def extract_hdfgz_func(input_path, output_path):
    """"""
    print('    {} {}'.format(os.path.basename(input_path), output_path))
    try:
        input_f = gzip.open(input_path, 'rb')
        with open(output_path, 'wb') as output_f:
            output_f.write(input_f.read())
        input_f.close()
    except IOError:
        print('    IOError')


def url_download(download_url, output_path, verify=True):
    """Download file from a URL using requests module"""
    response = requests.get(download_url, stream=True, verify=verify)
    if response.status_code != 200:
        logging.error('  HTTPError: {}'.format(response.status_code))
        return False

    logging.debug('  Beginning download')
    with (open(output_path, "wb")) as output_f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:  # filter out keep-alive new chunks
                output_f.write(chunk)
    logging.debug('  Download complete')
    return True


def ftp_download(site_url, site_folder, file_name, output_path):
    """"""
    try:
        ftp = FTP()
        ftp.connect(site_url)
        ftp.login()
        ftp.cwd('{}'.format(site_folder))
        logging.debug('  Beginning download')
        ftp.retrbinary('RETR %s' % file_name, open(output_path, 'wb').write)
        logging.debug('  Download complete')
        ftp.quit()
    except Exception as e:
        logging.info('  Unhandled exception: {}'.format(e))
