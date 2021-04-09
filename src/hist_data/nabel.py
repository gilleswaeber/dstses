"""
Retrieve data from the NABEL network (16 stations in Switzerland)
More information on the BAFU website: https://www.bafu.admin.ch/bafu/en/home/topics/air/state/data.html

The latest 18 months are available.

Timezone is MEZ/CET (= UTC+1)

Author: Gilles Waeber
"""

from datetime import datetime, timedelta
from io import StringIO
from itertools import chain

import pandas
import requests
from dateutil.tz import tz
from sqlalchemy.engine import Connection

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

    df = nabel_hourly_data(from_time, to_time, measure_id=TYPE_PM10)
    df.to_sql('nabel_pm10', if_exists='replace', index=False, con=con)
    df = nabel_hourly_data(from_time, to_time, measure_id=TYPE_PM2_5)
    df.to_sql('nabel_pm2_5', if_exists='replace', index=False, con=con)
    df = nabel_hourly_data(from_time, to_time, measure_id=TYPE_TEMPERATURE)
    df.to_sql('nabel_temperature', if_exists='replace', index=False, con=con)
    df = nabel_hourly_data(from_time, to_time, measure_id=TYPE_RAINFALL)
    df.to_sql('nabel_rainfall', if_exists='replace', index=False, con=con)


def nabel_hourly_data(from_time, to_time, measure_id) -> pandas.DataFrame:
    params = dict(webgrab='no', schadstoff=measure_id, station=1, datentyp='stunden', zeitraum='1monat',
                  von=to_time.strftime('%Y-%m-%d'),
                  bis=from_time.strftime('%Y-%m-%d'),
                  ausgabe='csv', abfrageflag='true', nach='schadstoff')
    params['stationsliste[]'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    r = requests.get('https://bafu.meteotest.ch/nabel/index.php/ausgabe/index/english', params=params)
    text = StringIO(r.text)
    df = pandas.read_csv(text, sep=';', skiprows=6, parse_dates=[0], dayfirst=True)
    df = df.rename(columns={'Date/time': 'date'})
    df['date'] = df['date'].dt.tz_localize(tzlocal).dt.tz_convert('UTC')
    return df


def nabel_upload_influx_db(con: Connection):
    from config_local import org, token, bucket
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS
    client = InfluxDBClient(url="https://eu-central-1-1.aws.cloud2.influxdata.com", token=token)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    places_pm10 = set(con.execute('SELECT * FROM nabel_pm10 LIMIT 1').keys())
    places_pm10.remove('date')
    places_pm2_5 = set(con.execute('SELECT * FROM nabel_pm2_5 LIMIT 1').keys())
    places_pm2_5.remove('date')
    places_temperature = set(con.execute('SELECT * FROM nabel_temperature LIMIT 1').keys())
    places_temperature.remove('date')
    places_rainfall = set(con.execute('SELECT * FROM nabel_rainfall LIMIT 1').keys())
    places_rainfall.remove('date')

    places = sorted(set(chain(places_pm10, places_pm2_5, places_temperature, places_rainfall)))

    q = " || '\n' || ".join(
        f"""'pollution,place={p} ' || SUBSTR(""" +
        (f"""IFNULL(',pm10=' || pm10."{p}", '') || """ if p in places_pm10 else '') +
        (f"""IFNULL(',pm2.5=' || pm2_5."{p}", '') || """ if p in places_pm2_5 else '') +
        f"""'', 2) || STRFTIME(' %s000000000', date) || '\n' || """

        f"""'weather,place={p} ' || SUBSTR(""" +
        (f"""IFNULL(',temperature=' || temperature."{p}", '') || """ if p in places_temperature else '') +
        (f"""IFNULL(',rainfall=' || rainfall."{p}", '') || """ if p in places_rainfall else '') +
        f"""'', 2) || STRFTIME(' %s000000000', date)"""

        for p in places
    )

    query = f"""SELECT {q} AS line_data
    FROM nabel_pm2_5 pm2_5
    INNER JOIN nabel_pm10 pm10 USING (date)
    INNER JOIN nabel_temperature temperature USING (date)
    INNER JOIN nabel_rainfall rainfall USING (date)
    """

    res = con.execute(query)

    sequence = [l for row in res for l in row['line_data'].split('\n') if '  ' not in l]

    print('SQL query:', query)
    # print('\n'.join(sequence))

    write_api.write(bucket, org, sequence)

    print('Done!')
