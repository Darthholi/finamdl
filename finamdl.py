#!/usr/bin/env python
# !finam-export

import sys
import time
import os.path
import datetime
import logging
import pandas as pd
from operator import attrgetter
from functools import partial
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=6)

import click
from click_datetime import Datetime

from finam import (Exporter,
                   Timeframe,
                   Market,
                   FinamExportError,
                   FinamObjectNotFoundError)
from finam.utils import click_validate_enum

"""
Helper script to download a set of assets
"""

logging.getLogger("finam").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


from tqdm import tqdm


def _arg_split(ctx, param, value):
    if value is None:
        return value

    try:
        items = value.split(',')
    except ValueError:
        raise click.BadParameter('comma-separated {} is required, got {}'
                                 .format(param, value))
    return items


def make_date_ranges(start, end, delta):
    curr = start
    yield curr, curr+delta
    while curr + delta < end:
        curr += delta
        yield curr, curr+delta
    yield curr, end


timedeltas_by_setting = {Timeframe.MINUTES1: datetime.timedelta(minutes=1),
Timeframe.MINUTES5: datetime.timedelta(minutes=5),
Timeframe.MINUTES10: datetime.timedelta(minutes=10),
Timeframe.MINUTES15: datetime.timedelta(minutes=15),
Timeframe.MINUTES30: datetime.timedelta(minutes=30),
Timeframe.DAILY: datetime.timedelta(days=1),
                         }


def datepoint_to_date(x):
    return datetime.datetime.strptime(str(x["<DATE>"]) + " " + x["<TIME>"], "%Y%m%d %H:%M:%S")


def sanitize_df(df):
    if "<DATE>" in df:
        df.drop_duplicates(inplace=True, subset=["<DATE>", "<TIME>"])
        
        dt_string = df["<DATE>"].astype(str) + " " + df["<TIME>"]
        del df["<DATE>"]
        del df["<TIME>"]
        
        def datepoint_to_date(x):
            return datetime.datetime.strptime(x, "%Y%m%d %H:%M:%S")
        
        index = pd.DatetimeIndex(dt_string.parallel_apply(datepoint_to_date))
        # index = pd.DatetimeIndex(dt_string.apply(datepoint_to_date))
        df.index = index
        assert df.index.duplicated(keep='first').any() == False
    
    df.rename(columns={"<OPEN>": "open",
                       "<CLOSE>": "close",
                       "<HIGH>": "high",
                       "<LOW>": "low",
                       "<VOL>": "vol",
                       }, inplace=True)

    return df[~df.index.duplicated(keep='first')]


@click.command()
@click.option('--destdir',
              help='Destination directory name',
              required=True,
              type=click.Path(exists=True, file_okay=False, writable=True,
                              resolve_path=True))
def sanitize_downloaded_parquets(destdir):
    onlyfiles = [f for f in listdir(destdir) if isfile(join(destdir, f)) and os.path.splitext(f)[-1] == ".parquet"]
    for pfile in tqdm(onlyfiles):
        df = pd.read_parquet(pfile)
        df = sanitize_df(df)
        df.to_parquet(pfile)


@click.command()
@click.option('--contracts',
              help='Contracts to lookup',
              required=False,
              callback=_arg_split)
@click.option('--market',
              help='Market to lookup',
              callback=partial(click_validate_enum, Market),
              required=False)
@click.option('--timeframe',
              help='Timeframe to use (DAILY, HOURLY, MINUTES30 etc)',
              default=Timeframe.MINUTES1.name,
              callback=partial(click_validate_enum, Timeframe),
              required=False)
@click.option('--destdir',
              help='Destination directory name',
              required=True,
              type=click.Path(exists=True, file_okay=False, writable=True,
                              resolve_path=True))
@click.option('--skiperr',
              help='Continue if a download error occurs. False by default',
              required=False,
              default=True,
              type=bool)
@click.option('--delay',
              help='Seconds to sleep between requests',
              type=click.IntRange(0, 600),
              default=0)
@click.option('--startdate', help='Start date',
              type=Datetime(format='%Y-%m-%d'),
              default='2010-01-01',
              required=False)
@click.option('--enddate', help='End date',
              type=Datetime(format='%Y-%m-%d'),
              default=datetime.date.today().strftime('%Y-%m-%d'),
              required=False)
@click.option('--skip_existing', help='Skip already existing files?',
              type=bool,
              default=True,
              required=False)
@click.option('--update_existing', help='Update already existing files?',
              type=bool,
              default=True,
              required=False)
def download(contracts, market, timeframe, destdir,
         delay, startdate, enddate, skiperr, skip_existing, update_existing, sanitize=True, daysdelta=20):
    exporter = Exporter()

    if not any((contracts, market)):
        raise click.BadParameter('Neither contracts nor market is specified')

    market_filter = dict()
    if market:
        market_filter.update(market=Market[market])
        contracts_df = exporter.lookup(**market_filter)
        contracts_df = contracts_df.reset_index()
        destpath = os.path.join(destdir, '{}.{}'.format(market, "csv"))
        contracts_df.to_csv(destpath)
    else:
        contracts_list = []
        for contract_code in contracts:
            try:
                contracts_list.append(exporter.lookup(code=contract_code, **market_filter))
            except FinamObjectNotFoundError:
                logger.info('unknown contract "{}"'.format(contract_code))
                if not skiperr:
                    sys.exit(1)
        contracts_df = pd.concat(contracts_list)
            
    date_ranges = list(make_date_ranges(startdate, enddate, datetime.timedelta(days=daysdelta)))
    if not contracts:
        contracts_to_dl = sorted(contracts_df['code'].to_list())
    else:
        contracts_to_dl = sorted(contracts)
    
    t = tqdm(contracts_to_dl, position=0)
    for contract_code in t:
        this_contract = contracts_df[contracts_df['code'] == contract_code].iloc[0]
        destpath = os.path.join(destdir, '{}-{}.{}'
                                .format(this_contract.code, timeframe, "parquet"))
        data_orig = None
        last_datetime = None
        if os.path.exists(destpath):
            if skip_existing:
                continue
            if update_existing:
                data_orig = pd.read_parquet(destpath)
                last_datetime = data_orig.index[-1]

        # logger.info(u'Downloading contract {}'.format(contract))
        if t:
            t.set_description(f" {contract_code} {this_contract['name']}")
            t.refresh()  # to show immediately the update

        all_data = []
        for date_range in tqdm(date_ranges, position=1):
            if last_datetime is not None:
                if date_range[1] < last_datetime:
                    continue
                if date_range[1] - last_datetime <= pd.Timedelta(days=daysdelta):
                    date_range = (last_datetime, date_range[1])
            try:
                data = exporter.download(this_contract['id'],
                                         start_date=date_range[0],
                                         end_date=date_range[1],
                                         timeframe=Timeframe[timeframe],
                                         market=Market(this_contract['market']))
                # if len(all_data) > 0 and len(data) <=0:
                #    raise ValueError("returned nothing")
                """
                if len(all_data) > 0 and (datepoint_to_date(data.iloc[0]) - datepoint_to_date(all_data[-1].iloc[-1]) > timedeltas_by_setting[Timeframe[timeframe]]):
                    raise ValueError("returned a hole")
                """
                if len(data) > 0:
                    all_data.append(data)
                time.sleep(delay)
            except FinamExportError as e:
                if skiperr:
                    logger.error(repr(e))
                    continue
                else:
                    raise
        if len(all_data) > 0:
            data = pd.concat(all_data)
            if sanitize:
                data = sanitize_df(data)

            if data_orig is not None and last_datetime is not None:
                data = data[data.index > last_datetime]
                data = pd.concat([data_orig, data])
                if len(data.index) >0 and data.index.duplicated().any():
                    print(f"{contract_code} has duplicates?")
                    continue
                    
            if sanitize:
                data.to_parquet(destpath)
            else:
                data.to_parquet(destpath, index=False)
            
            if delay > 0:
                logger.info('Sleeping for {} second(s)'.format(delay))
                time.sleep(delay)


@click.group()
def clisap():
    pass

clisap.add_command(download)
clisap.add_command(sanitize_downloaded_parquets)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clisap()