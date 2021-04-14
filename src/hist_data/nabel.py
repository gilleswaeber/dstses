"""
Retrieve data from the NABEL network (16 stations in Switzerland)
More information on the BAFU website: https://www.bafu.admin.ch/bafu/en/home/topics/air/state/data.html

Start date varies by station, earliest is 01.01.2001 (Härkingen A1). PM10 for 13/16 stations in 2001.

Timezone is MEZ/CET (= UTC+1)

Author: Gilles Waeber

NABEL data
==========
Stations:               PM10  PM2.5
 - Bern-Bollwerk        Yes   Yes
 - Lausanne-César-Roux  Yes   Yes
 - Lugano-Università    Yes   Yes
 - Zürich-Kaserne       Yes   Yes
 - Basel-Binningen      Yes   Yes
 - Dübendorf-Empa       Yes   Yes
 - Härkingen-A1         Yes   Yes
 - Sion-Aéroport-A9     Yes   Yes
 - Magadino-Cadenazzo   Yes   Yes
 - Payerne              Yes   Yes
 - Tänikon              Yes   Yes
 - Beromünster          Yes   Yes
 - Chaumont             Yes   No
 - Rigi-Seebodenalp     Yes   Yes
 - Davos-Seehornwald    Yes   No
 - Jungfraujoch         Yes   Yes
Rainfall and temperature available for all

Measures available and units:
 - Ozone                                  O₃          µg/m³
 - Nitrogen dioxide                       NO₂         µg/m³
 - Sulfur dioxide                         SO₂         µg/m³
 - Carbon monoxide                        CO          µg/m³
 - Particulate matter                     PM10        µg/m³
 - Particulate matter                     PM2.5       µg/m³
 - Soot                                   EC in PM2.5 µg/m³
 - Particle number concentration          CPC         1/cm³
 - Non-methane volatile organic compounds NMVOC       ppm
 - Nitrogen oxides                        NOₓ         µg/m³ eq. NO₂
 - Temperature                            T           °C
 - Precipitation                          PREC        mm
 - Global radiation                       RAD         W/m²
"""
import re
from datetime import datetime
from io import StringIO

import pandas
import requests
from dateutil.tz import tz
from sqlalchemy.engine import Connection
from tqdm import tqdm

from hist_data.common import add_columns, table_exists

NABEL_TABLE = 'nabel'

TYPE_PM10 = 6
TYPE_PM2_5 = 7
TYPE_TEMPERATURE = 9
TYPE_RAINFALL = 10

tz_local = tz.gettz('UTC+1')

START_YEAR = 2001


def nabel_download(con: Connection, from_time, to_time):
    tqdm.write(f' {from_time} to {to_time}')

    df_pm10 = nabel_hourly_data(from_time, to_time, measure_id=TYPE_PM10).add_suffix('.PM10')
    df_pm2_5 = nabel_hourly_data(from_time, to_time, measure_id=TYPE_PM2_5).add_suffix('.PM2.5')
    df_temperature = nabel_hourly_data(from_time, to_time, measure_id=TYPE_TEMPERATURE).add_suffix('.Temperature')
    df_rainfall = nabel_hourly_data(from_time, to_time, measure_id=TYPE_RAINFALL).add_suffix('.Rainfall')

    df = df_pm10
    df = df.merge(df_pm2_5, how='outer', on='date')
    df = df.merge(df_temperature, how='outer', on='date')
    df = df.merge(df_rainfall, how='outer', on='date')

    add_columns(NABEL_TABLE, df, con)
    df.to_sql(NABEL_TABLE, if_exists='append', con=con)


def nabel_download_all(con: Connection):
    print(f'Retrieving data for the city of Zurich starting in {START_YEAR}')
    con.execute(f'DROP TABLE IF EXISTS {NABEL_TABLE}')
    current_year = datetime.now(tz_local).year

    for year in tqdm(range(START_YEAR, current_year), desc='Past years'):
        nabel_download(con, datetime(year, 1, 1, 0, 0, 0, 0), datetime(year, 12, 31, 0, 0, 0, 0))

    nabel_download(con, datetime(current_year, 1, 1, 0, 0, 0, 0), datetime.now(tz_local))
    print('Done!')


def nabel_hourly_data(from_time, to_time, measure_id) -> pandas.DataFrame:
    params = dict(webgrab='no', schadstoff=measure_id, station=1, datentyp='stunden', zeitraum='1monat',
                  von=to_time.strftime('%Y-%m-%d'),
                  bis=from_time.strftime('%Y-%m-%d'),
                  ausgabe='csv', abfrageflag='true', nach='schadstoff')
    params['stationsliste[]'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    r = requests.get('https://bafu.meteotest.ch/nabel/index.php/ausgabe/index/english', params=params)
    with_skip = re.sub(r'^.*Date/time', 'Date/time', r.text, count=1, flags=re.DOTALL)
    text = StringIO(with_skip)
    df = pandas.read_csv(text, sep=';', parse_dates=[0], dayfirst=True)
    df.rename(columns={'Date/time': 'date'}, inplace=True)
    df['date'] = df['date'].dt.tz_localize(tz_local).dt.tz_convert('UTC')
    df.set_index('date', inplace=True)
    df.index.name = 'date'
    return df


def nabel_update_recent(con: Connection, from_time: datetime):
    year = datetime.now(tz_local).year
    print(f'Retrieving data for the city of Zurich for {year}')
    if table_exists(NABEL_TABLE, con):
        con.execute(f"DELETE FROM {NABEL_TABLE} WHERE date >= '{from_time.isoformat()}'")

    nabel_download(con, from_time, datetime.now(tz_local))
    print('Done!')
