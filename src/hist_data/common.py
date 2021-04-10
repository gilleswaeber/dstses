from itertools import chain
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

"""
For the InfluxDB config, create a file config_local.py with: ```py
token = "WRITE_ACESS_TOKEN_CODE"
org = "INFLUX_DB_ORG_NAME"
bucket = "BUCKET_NAME"
```
"""

HIST_DATA_SQLITE_DB = Path(__file__).parent.absolute() / 'hist_data.sqlite'


def get_engine() -> Connection:
    return create_engine(f'sqlite:///{HIST_DATA_SQLITE_DB}', echo=False)


def upload_influx_db(con: Connection, table: str):
    from config_local import org, token, bucket
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS
    client = InfluxDBClient(url="https://eu-central-1-1.aws.cloud2.influxdata.com", token=token)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    cols = set(con.execute(f'SELECT * FROM {table} LIMIT 1').keys())
    places = set(p.split('.', 1)[0] for p in cols)
    places_pm10 = set(p for p in places if f'{p}.PM10' in cols)
    places_pm2_5 = set(p for p in places if f'{p}.PM2.5' in cols)
    places_temperature = set(p for p in places if f'{p}.Temperature' in cols)
    places_rainfall = set(p for p in places if f'{p}.Rainfall' in cols)

    places = sorted(set(chain(places_pm10, places_pm2_5, places_temperature, places_rainfall)))

    q = " || '\n' || ".join(
        f"""'pollution,place={p} ' || SUBSTR(""" +
        (f"""IFNULL(',pm10=' || "{p}.PM10", '') || """ if p in places_pm10 else '') +
        (f"""IFNULL(',pm2.5=' || "{p}.PM2.5", '') || """ if p in places_pm2_5 else '') +
        f"""'', 2) || STRFTIME(' %s000000000', date) || '\n' || """ +

        (
            f"""'weather,place={p} ' || SUBSTR(""" +
            (f"""IFNULL(',temperature=' || "{p}.Temperature", '') || """ if p in places_temperature else '') +
            (f"""IFNULL(',rainfall=' || "{p}.Rainfall", '') || """ if p in places_rainfall else '') +
            f"""'', 2) || STRFTIME(' %s000000000', date)"""
            if p in places_temperature or p in places_rainfall else """''"""
        )

        for p in places
    )

    query = f"""SELECT {q} AS line_data FROM {table}"""

    res = con.execute(query)

    sequence = [l for row in res for l in row['line_data'].split('\n') if '  ' not in l]

    print('SQL query:', query)
    # print('\n'.join(sequence))

    write_api.write(bucket, org, sequence)

    print('Done!')
