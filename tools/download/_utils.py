import argparse
import datetime as dt
from ftplib import FTP
import logging

import requests


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


def ftp_file_list(site_url, site_folder):
    """"""
    try:
        ftp = FTP()
        ftp.connect(site_url)
        ftp.login()
        ftp.cwd('{}'.format(site_folder))
        files = ftp.nlst()
        ftp.quit()
    except Exception as e:
        logging.info('  Unhandled exception: {}'.format(e))
        files = []
    return files


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
    datetime

    Raises
    ------
    ArgParse ArgumentTypeError

    """
    try:
        return dt.datetime.strptime(input_date, "%Y-%m-%d")
        # return dt.datetime.strptime(input_date, "%Y-%m-%d").date().isoformat()
    except ValueError:
        msg = "Not a valid date: '{}'.".format(input_date)
        raise argparse.ArgumentTypeError(msg)
