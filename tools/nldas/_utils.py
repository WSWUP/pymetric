import argparse
import datetime as dt
import logging

import requests


def date_range(start_date, end_date):
    """Yield datetimes within a date range"""
    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


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