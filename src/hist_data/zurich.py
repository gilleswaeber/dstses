"""
PM10 and PM2.5 data for the city of Zurich.

First recorded PM10: 03.01.2001
First recorded PM2.5: 01.01.2017

Stations:
 - Zch_Stampfenbachstrasse (Zürich Stampfenbachstrasse, ZSTA)
   LV95: X=2683148, Y=1249020
   WGS84: lat=47.3868, lng=8.5398
   Address: Stampfenbachstrasse 144, 8006 Zürich
   Elevation: 445m
   Description: Die Messstation steht an zentraler Lage in der Stadt Zürich. Sie repräsentiert eine städtische Lage mit mässigem Verkehr.
   Measures: PM10 since 2001, PM2.5 since 2017
 - Zch_Schimmelstrasse (Zürich Schimmelstrasse, ZSIM)
   LV95: X=2681943, Y=1247245
   WGS84: lat=47.371, lng=8.5235
   Address: Schimmelstrasse 21, 8003 Zürich
   Elevation: 413m
   Description: Die Messstation steht an einer städtischen Hauptverkehrsachse in zentraler Lage in einem Wohn- und Geschäftsquartier in der Stadt Zürich. Sie repräsentiert eine städtische Lage mit sehr starkem Verkehr.
   Measures: PM10 since 2002, PM2.5 since 2017
 - Zch_Rosengartenstrasse (Zürich Rosengartenstrasse, ZROS)
   LV95: X=2682106, Y=1249935
   WGS84: lat=47.3952, lng=8.5261
   Address: Rosengartenstrasse 24, 8037 Zürich
   Elevation: 433m
   Description: Die Messstation steht an einer städtischen Hauptverkehrsachse in zentraler Lage in einem Wohn- und Geschäftsquartier in der Stadt Zürich. Sie repräsentiert eine städtische Lage mit sehr starkem Verkehr.
   Measures: PM10 since 2013, PM2.5 since 2018
 - Zch_Heubeeribüel (Zürich Heubeeribüel, ZHEU)
   LV95: X=2685137, Y=1248473
   WGS84: lat=47.3815, lng=8.5659
   Address: Heubeeriweg 30, 8044 Zürich
   Elevation: 610m
   Description: Die Messstation liegt in einem Schulareal an erhöhter Hanglage am Siedlungsrand der Stadtt Zürich angrenzend an ein offenes Feld gegen den Zürichberg-Wald. Der Standort ist repräsentativ für wenig verkehrsbeinflusste Wohn- und Erholungs-Gebiete am Rand der grossen Agglomerationen.
   Measures: PM10 n/a, PM2.5 n/a

Measures available:
 - CO    mg/m³
 - SO₂   µg/m³
 - NOₓ   ppb
 - NO    µg/m³
 - NO₂   µg/m³
 - O₃    µg/m³
 - PM10  µg/m³
 - PM2.5 µg/m³
"""

from io import StringIO

import pandas
import requests
from sqlalchemy.engine import Connection
from tqdm import tqdm

START_YEAR = 2001
END_YEAR = 2021

ZURICH_TABLE = 'zurich'


def zurich_hourly_data(year):
    r = requests.get(
        f'https://data.stadt-zuerich.ch/dataset/ugz_luftschadstoffmessung_stundenwerte/download/ugz_ogd_air_h1_{year}.csv')
    text = StringIO(r.content.decode('utf-8'))
    df = pandas.read_csv(text, sep=',', parse_dates=[0])
    df = df.drop(columns=["Intervall", "Einheit", "Status"])
    df = df.rename(columns={
        "Datum": "date",
        "Standort": "place",
        "Parameter": "measure",
        "Wert": "value",
    })
    df = df[df['measure'].isin(('PM10', 'PM2.5'))]
    df['item'] = df['place'] + '.' + df['measure']
    df = df.pivot(index='date', columns='item', values='value')
    df.index.name = 'date'
    return df


def zurich_download(con: Connection):
    print(f'Retrieving data for the city of Zurich from {START_YEAR} to {END_YEAR}')
    con.execute(f'DROP TABLE IF EXISTS {ZURICH_TABLE}')

    for year in tqdm(range(START_YEAR, END_YEAR + 1)):
        df = zurich_hourly_data(year)
        r = con.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ZURICH_TABLE}'")
        if r.first() is not None:
            cols = set(con.execute(f'SELECT * FROM {ZURICH_TABLE} LIMIT 1').keys())
            new_cols = set(df.columns) - cols
            for col in new_cols:
                alter = f'ALTER TABLE {ZURICH_TABLE} ADD COLUMN "{col}" FLOAT'
                con.execute(alter)
        df.to_sql(ZURICH_TABLE, if_exists='append', con=con)

    print('Done!')
