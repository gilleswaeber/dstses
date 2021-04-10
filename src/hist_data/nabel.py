"""
Retrieve data from the NABEL network (16 stations in Switzerland)
More information on the BAFU website: https://www.bafu.admin.ch/bafu/en/home/topics/air/state/data.html

The latest 18 months are available.

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

from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas
import requests
from dateutil.tz import tz
from sqlalchemy.engine import Connection

NABEL_TABLE = 'nabel'

TYPE_PM10 = 6
TYPE_PM2_5 = 7
TYPE_TEMPERATURE = 9
TYPE_RAINFALL = 10

tzlocal = tz.gettz('UTC+1')


def nabel_download(con: Connection):
    from_time = datetime.now(tzlocal)
    to_time = from_time - timedelta(days=18 * 30)

    print(
        f'Retrieving PM10, PM2.5, temperature, and rainfall data from NABEL for the date range {to_time} to {from_time}')

    df_pm10 = nabel_hourly_data(from_time, to_time, measure_id=TYPE_PM10).add_suffix('.PM10')
    df_pm2_5 = nabel_hourly_data(from_time, to_time, measure_id=TYPE_PM2_5).add_suffix('.PM2.5')
    df_temperature = nabel_hourly_data(from_time, to_time, measure_id=TYPE_TEMPERATURE).add_suffix('.Temperature')
    df_rainfall = nabel_hourly_data(from_time, to_time, measure_id=TYPE_RAINFALL).add_suffix('.Rainfall')

    df = df_pm10
    df = df.merge(df_pm2_5, how='outer', on='date')
    df = df.merge(df_temperature, how='outer', on='date')
    df = df.merge(df_rainfall, how='outer', on='date')

    df.to_sql('nabel', if_exists='replace', con=con)


def nabel_hourly_data(from_time, to_time, measure_id) -> pandas.DataFrame:
    params = dict(webgrab='no', schadstoff=measure_id, station=1, datentyp='stunden', zeitraum='1monat',
                  von=to_time.strftime('%Y-%m-%d'),
                  bis=from_time.strftime('%Y-%m-%d'),
                  ausgabe='csv', abfrageflag='true', nach='schadstoff')
    params['stationsliste[]'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    r = requests.get('https://bafu.meteotest.ch/nabel/index.php/ausgabe/index/english', params=params)
    text = StringIO(r.text)
    df = pandas.read_csv(text, sep=';', skiprows=6, parse_dates=[0], dayfirst=True)
    df.rename(columns={'Date/time': 'date'}, inplace=True)
    df['date'] = df['date'].dt.tz_localize(tzlocal).dt.tz_convert('UTC')
    df.set_index('date', inplace=True)
    df.index.name = 'date'
    return df
