import re
import csv
import logging
import math
import glob
# import argparse
import numpy as np
import os
import pandas as pd
import time
import datetime
import drms
import urllib
# import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.units as u
import telegram_handler
# import warnings
import sunpy.wcs
import sunpy.map
import pickle
import telepot
from colorlog import ColoredFormatter
from astropy.coordinates import SkyCoord
# from astropy.io import fits
# from astropy.time import Time
# from datetime import timedelta
# from sunpy.coordinates import frames
# from astropy.coordinates import SkyCoord
from tg_tqdm import tg_tqdm
# from tqdm import tqdm
# warnings.filterwarnings("ignore")


# define constants
EMAIL = 'emal@email.ru'
SAVE_PATH = 'dataset'
tg_bot_token = 'TOKEN'
tm_chat_id = 1234
ik_chat_id = 1234
sun_group_id = -1234
DATE_DELIMIT = '2010-06-28'
TG_LOGGER = False
FILE_DELETE = False
LOGGER_LEVEL = logging.WARNING
# LOGGER_LEVEL = logging.DEBUG
VERBOSE = True
PERIOD = 300
START_DATE = '1996-04-01'
CROP_DATE = '2017-11-01'
SLEEP = 0.1
PROGRESS = 10


# logging.basicConfig(filename='futs_parse.log', level=logging.INFO)


def set_logger(level=logging.WARNING, name='logger', telegram=False):
    """Return a logger with a default ColoredFormatter."""

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(funcName)s - %(message)s")
    stream_formatter = ColoredFormatter(
        "%(asctime)s [%(log_color)s%(levelname)-8s%(reset)s: %(funcName)s] %(white)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    log_handler = logging.FileHandler("fits_parse.log")
    log_handler.setFormatter(file_formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    if telegram:
        tg_handler = telegram_handler.TelegramHandler(tg_bot_token, sun_group_id)
        tg_formatter = telegram_handler.HtmlFormatter()
        tg_handler.setFormatter(tg_formatter)
        logger.addHandler(tg_handler)

    logger.setLevel(level)

    return logger


logger = set_logger(level=LOGGER_LEVEL, name='sun_logger', telegram=TG_LOGGER)


def check_dataset_directory():

    if not os.path.exists('HMIdataset/fragments'):
        logger.warning('HMIdataset folders not exist, create them')
        os.makedirs('HMIdataset/fragments')

    if not os.path.exists('MDIdataset/fragments'):
        logger.warning('MDIdataset folders not exist, create them')
        os.makedirs('MDIdataset/fragments')

    return True


def clean_folder(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

    return True


def message_of_start(token=tg_bot_token, id=sun_group_id):
    bot = telepot.Bot(token)
    bot.sendMessage(id, 'Start parsing fits on remote server')


def message_of_start_cropping(token=tg_bot_token, id=sun_group_id):
    bot = telepot.Bot(token)
    bot.sendMessage(id, '-' * 30)
    bot.sendMessage(id, 'Start cropping regions')
    bot.sendMessage(id, '-' * 30)


def hook_for_download_fits(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def request_mfits_by_date_MDI(moment, email=EMAIL, path_to_save='MDIdataset', verbose=False):
    """
    Function for request fits from JSOC database
    moment: pd.datetime object
    return: filepath to the magnetogram
    """

    filename = 'mdi.fd_m_96m_lev182.' + moment.strftime('%Y%m%d_%H%M%S_TAI.data.fits')
    filepath = os.path.join(path_to_save, filename)

    if os.path.exists(filepath):
        pass
    else:

        c = drms.Client(email=email, verbose=verbose)
        str_for_query = 'mdi.fd_M_96m_lev182' + moment.strftime('[%Y.%m.%d_%H:%M:%S_TAI]')
        logger.info('Magnetogram: {} will be downloaded ... '.format(str_for_query))
        r = c.export(str_for_query, method='url', protocol='fits')
        logger.debug(r)

        try:
            r.wait()
            logger.info(r.request_url)
        except Exception as e:
            logger.warning('Can not wait anymore, skip this. Get Exception: {}'.format(e))

        try:
            logger.info("Download data and save to path {}".format(filepath))
            r.download(path_to_save, verbose=verbose)
        except Exception as e:
            logger.error('Get error while trying download: {}'.format(e))
            logger.warning('Skip this date')

    return filepath


def request_batch_mfits_by_date(moment,
                                period_of_days=30, email=EMAIL,
                                path_to_save='dataset',
                                verbose=False,
                                type_mag='MDI',
                                token=tg_bot_token,
                                chat_id=sun_group_id):
    '''Request batch fits for a period of days and return:
    request url
    period of days that was apply
    first date of batch
    last date of batch
    '''

    c = drms.Client(email=email, verbose=verbose)

    def set_str_for_query(period_of_days=period_of_days):
        if type_mag == 'MDI':
            str_for_query = 'mdi.fd_M_96m_lev182' + moment.strftime('[%Y.%m.%d_%H:%M:%S_TAI/{}d@24h]'.format(period_of_days))
            filename_to_check = 'mdi.fd_m_96m_lev182.' + moment.strftime('%Y%m%d_%H%M%S_TAI.data.fits')
            path_to_save = 'MDIdataset'
        if type_mag == 'HMI':
            str_for_query = 'hmi.m_720s' + moment.strftime('[%Y.%m.%d_%H:%M:%S_TAI/{}d@24h]'.format(period_of_days))
            path_to_save = 'HMIdataset'
            filename_to_check = 'hmi.m_720s.' + moment.strftime('%Y%m%d_%H%M%S_TAI.magnetogram.fits')

        return str_for_query, path_to_save, filename_to_check

    str_for_query, path_to_save, filename_to_check = set_str_for_query()
    logger.debug('{}\n{}\n{}'.format(str_for_query, path_to_save, filename_to_check))
    if os.path.exists(os.path.join(path_to_save, filename_to_check)):
        period_of_days = 10
        logger.info('Files already exists. Skip downloads this batch size of {}'.format(period_of_days))
        return None, period_of_days, moment, moment + datetime.timedelta(days=period_of_days), period_of_days

    logger.info('Magnetogram: {} will be downloaded ... '.format(str_for_query))

    r = c.export(str_for_query, protocol='fits')
    logger.debug(r)
    logger.debug(r.has_failed())

    treshold = round(math.log(period_of_days) ** 2 / 2)
    while r.has_failed():
        period_of_days -= round(treshold)
        if period_of_days < round(treshold / 2):
            logger.warning('Period of days is too small, skip this request to 10 days')
            logger.warning('Export request was {}: '.format(str_for_query))
            period_of_days = 10
            return None, period_of_days, moment, moment + datetime.timedelta(days=period_of_days), period_of_days
        time.sleep(1)
        logger.info('Export request has failed. Reduce number of days in it on {}. Now days in request {}'.format(int(treshold), period_of_days))
        str_for_query, _, _ = set_str_for_query(period_of_days=period_of_days)
        logger.debug('Request string: {}'.format(str_for_query))
        r = c.export(str_for_query, protocol='fits')

    logger.debug(r)
    logger.debug(len(r.data))

    try:
        r.wait(sleep=10, retries_notfound=10)
    except Exception as e:
        logger.error('Can not wait anymore, skip this. Get Exception: {}'.format(e))

    logger.info("Download data and save to path {}".format(path_to_save))

    first_date_batch = r.urls[0:]['record'].values[0].replace('[', ' ').split()[1].split('_')[0].replace('.', '-')
    last_date_batch = r.urls[-1:]['record'].values[0].replace('[', ' ').split()[1].split('_')[0].replace('.', '-')

    with tg_tqdm(r.urls.index, token=token, chat_id=chat_id, desc='DOWNLOAD BATCH',
                 postfix='start_date = {}, end_date = {}'.format(first_date_batch, last_date_batch)) as batch_d:
        for ind in batch_d:
            try:
                # file_name = '.'.join(r.urls.filename[ind].split('.')[:3] + r.urls.filename[ind].split('.')[4:])
                urllib.request.urlretrieve(r.urls.url[ind], os.path.join(path_to_save, r.urls.filename[ind]))
            except Exception as e:
                logger.error('Get error while trying download {}: {}'.format(r.urls.url[ind], repr(e)))
                logger.warning('Skip this file')

    len_batch = len(r.urls)

    return r.request_url, period_of_days, first_date_batch, last_date_batch, len_batch


def request_mfits_by_date_HMI(moment, email=EMAIL, path_to_save='HMIdataset', verbose=False):
    """
    Function for request fits from JSOC database
    moment: pd.datetime object
    return: filepath to the magnetogram
    """

    filename = 'hmi.m_720s.' + moment.strftime('%Y%m%d_%H%M%S_TAI.magnetogram.fits')
    filepath = os.path.join(path_to_save, filename)

    if os.path.exists(filepath):
        pass
    else:

        c = drms.Client(email=email, verbose=verbose)
        str_for_query = 'hmi.m_720s' + moment.strftime('[%Y.%m.%d_%H:%M:%S_TAI]{magnetogram}')
        logger.info('Magnetogram: {} will be downloaded ... '.format(str_for_query))
        r = c.export(str_for_query, method='url', protocol='fits')
        logger.debug(r)

        try:
            r.wait()
            logger.info(r.request_url)
        except Exception as e:
            logger.warning('Can not wait anymore, skip this. Get Exception: {}'.format(e))

        try:
            logger.info("Download data and save to path {}".format(filepath))
            r.download(path_to_save, verbose=verbose)
        except Exception as e:
            logger.error('Get error while trying download: {}'.format(e))
            logger.warning('Skip this date')

    return filepath


def read_fits_to_map(filepath, plot_show=False, ln=False):
    """
    read fits to sunpy object and plot in logariphmic scale
    return
    mymap: sunpy object
    """

    mymap = sunpy.map.Map(filepath)

    if plot_show:
        plt.figure(figsize=(12, 12))
        if ln:
            data = np.sign(mymap.data) * np.log1p(np.abs(mymap.data))
        data = mymap.data
        plt.imshow(data, cmap='gray')

    return mymap


def region_coord_list(datestr, sunspots_df, limit_deg=45):
    """
    Function for working with sunspot_1996_2017.pkl dataframe,
    return list of tuples: (datestr, NOAA number, location)
    used in cropping

    args:
    datestr: string for date in the format used in dataframe '2001-04-30'
    sunspots_df: dataframe from file sunspot_1996_2017.pkl

    return: list of tuples
    """

    date_df = sunspots_df.loc[datestr]
    date_df.index = date_df.index.droplevel()
    rc_list = []
    for index, row in date_df.iterrows():
        try:
            restriction_degree = (abs(float(row.location[1:3]) <= limit_deg)) and (abs(float(row.location[4:])) <= limit_deg)
            if restriction_degree:
                rc_list.append((pd.to_datetime(datestr, format='%Y-%m-%d'), index, row.location))
        except ValueError as e:
            if TG_LOGGER:
                time.sleep(SLEEP)
            logger.warning('Some error with read location {} in degree for date {}: {}'.format(row.location, datestr, e))
        except Exception as e:
            if TG_LOGGER:
                time.sleep(SLEEP)
            logger.error('Some error with read location {} in degree for date {}: {}'.format(row.location, datestr, e))

    return rc_list


def return_pixel_from_map(mag_map, record, limit_deg=45):
    '''
    convert lon lat coordinate to coordinate in pixel in sun map and return it
    '''

    pattern = re.compile("[NS]\d{2}[EW]\d{2}")
    assert bool(pattern.match(record)), 'Pattern should be in the same format as N20E18'
    assert (abs(float(record[1:3]) <= limit_deg)) and (abs(float(record[4:])) <= limit_deg), 'Consider only regions between -{}, +{} degree'.format(limit_deg)
    if record[0] == 'N':
        lat = float(record[1:3])
    else:
        lat = -float(record[1:3])
    if record[3] == 'W':
        lon = float(record[4:])
    else:
        lon = -float(record[4:])

    hpc_coord = sunpy.wcs.convert_hg_hpc(lon, lat, b0_deg=mag_map.meta['crlt_obs'])
    coord = SkyCoord(hpc_coord[0] * u.arcsec, hpc_coord[1] * u.arcsec, frame=mag_map.coordinate_frame)
    # pixel_pos = mag_map.world_to_pixel(coord)
    pixel_pos = mag_map.world_to_pixel(coord) * u.pixel
    # pixel_pos = pixel_pos.to_value()

    return pixel_pos


def crop_regions(mag_map, rc_list, type_mag, delta=100, plot_rec=False, plot_crop=False, limit_deg=45, ln=False):
    '''
    Crop region by size delta and save it to disk,
    if plot_rec, plot rectangle of regions on disk,
    if plot_crop, plot only crop regions
    '''
    if ln:
        data = np.sign(mag_map.data) * np.log1p(np.abs(mag_map.data))
    data = mag_map.data

    if type_mag == 'MDI':
        delta = 100
    if type_mag == 'HMI':
        delta = 200

    if plot_rec:
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.matshow(data)
        plt.gray()
        ax.set_title('{} magnetogram at '.format(type_mag) + rc_list[0][0].strftime('%Y-%m-%d %H:%M'))

        for record in rc_list:
            try:
                pxs = return_pixel_from_map(mag_map, record[2], limit_deg).to_value()
            except Exception as e:
                logger.error('Some error with get pixel coordinates from map: {}. Skip it'.format(e))
                continue
            rect = patches.Rectangle((pxs[0] - 1.25 * delta, pxs[1] - delta), 2.5 * delta, 2 * delta, linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.annotate('{}.AR'.format(type_mag) + str(record[1]), xy=(pxs[0], pxs[1]), xytext=(pxs[0], pxs[1] - 50), color='yellow', fontsize='xx-large')

        plt.show()

    submaps = []
    for record in rc_list:

        filename = '{}.{}.AR{}.fits'.format(type_mag, record[0].strftime('%Y-%m-%d_%H%M%S'), record[1])
        filepath = os.path.join('{}dataset/fragments'.format(type_mag), filename)
        try:
            pxs = return_pixel_from_map(mag_map, record[2], limit_deg)
        except Exception as e:
            logger.error('Some error with get pixel coordinates from map: {}. Skip it'.format(e))
            continue
        bot_l = [pxs[0] - delta * 1.25 * u.pixel, pxs[1] - delta * u.pixel]
        top_r = [pxs[0] + delta * 1.25 * u.pixel, pxs[1] + delta * u.pixel]

        submap = mag_map.submap(bot_l * u.pixel, top_r * u.pixel)

        if plot_crop:
            submap.peek()

        try:
            submap.save(filepath)
        except Exception as e:
            if TG_LOGGER:
                time.sleep(SLEEP)
            logger.info('Could not save fits {} cause: {}. Skip it'.format(filename, e))

        submaps.append(submap)

    return submaps


def date_compare(date):
    return date < datetime.datetime.fromtimestamp(time.mktime(time.strptime(DATE_DELIMIT, '%Y-%m-%d')))


if __name__ == '__main__':

    check_dataset_directory()
    message_of_start()

    try:
        sunspots = pickle.load(urllib.request.urlopen('https://raw.githubusercontent.com/iknyazeva/FitsProcessing/master/sunspot_1996_2017.pkl'))
        logger.info('Load sunspot dataframe is successful!')
    except Exception as e:
        logger.error('Can not load sunspot dataframe, halt parsing! Get Exception: {}'.format(e))
        raise(e)

    requests_urls = []
    if START_DATE:
        try:
            start_moment = sunspots[(sunspots.index.get_level_values(0) > START_DATE)].index.get_level_values(0)[0]
        except IndexError as e:
            logger.info('Index out of bound. Possibly the table is ended: {}'.format(e))
            start_moment = START_DATE
        except Exception as e:
            logger.error('Some error then get start_moment for first iteration: {}'.format(e))
    else:
        start_moment = sunspots.index.get_level_values(0)[0]
    logger.debug(start_moment)
    count_of_days_left = len(sunspots[(sunspots.index.get_level_values(0) >= start_moment)].groupby(level=0))
    logger.debug(count_of_days_left)

    with tg_tqdm(sunspots[(sunspots.index.get_level_values(0) > start_moment)].groupby(level=0),
                 token=tg_bot_token, chat_id=sun_group_id, desc='MAIN PROGRESS DOWNLOAD') as tgm:
        number_batch = 1
        while count_of_days_left > 0:
            tgm.set_postfix(batch=number_batch)
            if date_compare(start_moment):
                request_url,\
                    period_of_days,\
                    first_date_batch,\
                    last_date_batch,\
                    len_batch = request_batch_mfits_by_date(start_moment, period_of_days=PERIOD,
                                                            email=EMAIL, type_mag='MDI', verbose=VERBOSE)
            else:
                request_url,\
                    period_of_days,\
                    first_date_batch,\
                    last_date_batch,\
                    len_batch = request_batch_mfits_by_date(start_moment, period_of_days=PERIOD,
                                                            email=EMAIL, type_mag='HMI', verbose=VERBOSE)

            logger.debug('Returned period of days {}'.format(period_of_days))
            # requests_urls.append(request_url)
            try:
                start_moment = sunspots[(sunspots.index.get_level_values(0) > last_date_batch)].index.get_level_values(0)[0]
            except IndexError as e:
                logger.info('Index out of bound. Possibly the table is ended: {}'.format(e))
            except Exception as e:
                logger.error('Some error then get start_moment for next iteration: {}'.format(e))
            count_of_days_left = len(sunspots[(sunspots.index.get_level_values(0) >= start_moment)])
            number_batch += 1

            with open('requests_urls.csv', 'a', newline='') as file:
                csv.writer(file).writerow(request_url)

            tgm.update(len_batch)

    # with open('requests_urls.csv', 'w') as file:
    #     csv.writer(file, delimiter='\n').writerow(requests_urls)

    message_of_start_cropping()

    if CROP_DATE:
        crop_df = sunspots[(sunspots.index.get_level_values(0) > CROP_DATE)]
    else:
        crop_df = sunspots
    with tg_tqdm(range(1), tg_bot_token, sun_group_id,
                 total=len(crop_df.groupby(level=0)), desc='CROPPING PROGRESS') as tgt:

        def is_progress(acc, total, progress=PROGRESS, tqdm_instanse=tgt):
            if (acc % PROGRESS == 0):
                logger.debug('In if acc = {}'.format(acc))
                time.sleep(SLEEP)
                tgt.update(PROGRESS)
            elif (acc >= total):
                logger.debug('In if acc = {}'.format(acc))
                time.sleep(SLEEP)
                tgt.update(total % PROGRESS)

            return True

        acc = 0
        total = len(crop_df.groupby(level=0))
        logger.debug(total)
        for date, df in crop_df.groupby(level=0):

            rc_list = region_coord_list(str(date), df, limit_deg=45)

            if not rc_list:
                acc += 1
                time.sleep(SLEEP)
                is_progress(acc, total)
                logger.debug('rc_list is empty - {}, acc = {}'.format(rc_list, acc))
                continue

            if date_compare(date):
                filename = 'mdi.fd_m_96m_lev182.' + date.strftime('%Y%m%d_%H%M%S_TAI') + '*.fits'
                path = 'MDIdataset/'
                try:
                    filepath = glob.glob(path + filename)[0]
                    if TG_LOGGER:
                        time.sleep(SLEEP)
                    logger.debug('filepath: {}'.format(filepath))
                except IndexError as e:
                    logger.info('File with this date {} is not exist'.format(str(date)))
                    acc += 1
                    is_progress(acc, total)
                    continue
                except Exception as e:
                    logger.error('Some error with glob:'.format(e))
                    acc += 1
                    is_progress(acc, total)
                    continue
                type_mag = 'MDI'

            else:
                filename = 'hmi.m_720s.' + date.strftime('%Y%m%d_%H%M%S_TAI') + '*.fits'
                path = 'HMIdataset/'
                try:
                    filepath = glob.glob(path + filename)[0]
                    if TG_LOGGER:
                        time.sleep(SLEEP)
                    logger.debug('filepath: {}'.format(filepath))
                except IndexError as e:
                    if TG_LOGGER:
                        time.sleep(SLEEP)
                    logger.info('File with this date {} is not exist'.format(str(date)))
                    acc += 1
                    is_progress(acc, total)
                    continue
                except Exception as e:
                    if TG_LOGGER:
                        time.sleep(SLEEP)
                    logger.error('Some error with glob:'.format(e))
                    acc += 1
                    is_progress(acc, total)
                    continue
                type_mag = 'HMI'

            try:
                sun_map = read_fits_to_map(filepath, plot_show=False)
                crop_regions(sun_map, rc_list, plot_rec=False, plot_crop=False, type_mag=type_mag)
            except ValueError as e:
                if TG_LOGGER:
                    time.sleep(SLEEP)
                logger.info('Get Exception while reading: {}'.format(e))
                logger.info('Doing active farther, skip it.')
                # acc += 1
                # continue
            except Exception as e:
                if TG_LOGGER:
                    time.sleep(SLEEP)
                logger.error('Get Exception while reading: {}'.format(e))
                logger.warning('Doing active farther, skip it.') 
               # acc += 1
                # continue

            # tgt.update()
            acc += 1
            logger.debug('acc = {}'.format(acc))
            is_progress(acc, total)

    if FILE_DELETE:
        clean_folder('MDIdataset')
        clean_folder('HMIdataset')
