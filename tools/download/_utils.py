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
