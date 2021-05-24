"""
PM10 and PM2.5 data for the city of Zurich.

References:
- https://data.stadt-zuerich.ch/dataset/ugz_luftschadstoffmessung_stundenwerte
- https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte
- https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte/resource/9e650b11-b384-4dff-a1f2-d639c5e94d6e

First recorded PM10: 03.01.2001
First recorded PM2.5: 01.01.2017

Data is stored for the timezone UTC+1.

Stations:
- Zch_Stampfenbachstrasse (Zürich Stampfenbachstrasse, ZSTA)
  LV95: X=2683148, Y=1249020
  WGS84: lat=47.3868, lng=8.5398
  Address: Stampfenbachstrasse 144, 8006 Zürich
  Elevation: 445m
  Description: Die Messstation steht an zentraler Lage in der Stadt Zürich. Sie repräsentiert eine städtische Lage mit
  mässigem Verkehr.
  Measures: PM10 since 2001, PM2.5 since 2017

- Zch_Schimmelstrasse (Zürich Schimmelstrasse, ZSIM)
  LV95: X=2681943, Y=1247245
  WGS84: lat=47.371, lng=8.5235
  Address: Schimmelstrasse 21, 8003 Zürich
  Elevation: 413m
  Description: Die Messstation steht an einer städtischen Hauptverkehrsachse in zentraler Lage in einem Wohn- und
  Geschäftsquartier in der Stadt Zürich. Sie repräsentiert eine städtische Lage mit sehr starkem Verkehr.
  Measures: PM10 since 2002, PM2.5 since 2017

- Zch_Rosengartenstrasse (Zürich Rosengartenstrasse, ZROS)
  LV95: X=2682106, Y=1249935
  WGS84: lat=47.3952, lng=8.5261
  Address: Rosengartenstrasse 24, 8037 Zürich
  Elevation: 433m
  Description: Die Messstation steht an einer städtischen Hauptverkehrsachse in zentraler Lage in einem Wohn- und
  Geschäftsquartier in der Stadt Zürich. Sie repräsentiert eine städtische Lage mit sehr starkem Verkehr.
  Measures: PM10 since 2013, PM2.5 since 2018

- Zch_Heubeeribüel (Zürich Heubeeribüel, ZHEU)
  LV95: X=2685137, Y=1248473
  WGS84: lat=47.3815, lng=8.5659
  Address: Heubeeriweg 30, 8044 Zürich
  Elevation: 610m
  Description: Die Messstation liegt in einem Schulareal an erhöhter Hanglage am Siedlungsrand der Stadtt Zürich
  angrenzend an ein offenes Feld gegen den Zürichberg-Wald. Der Standort ist repräsentativ für wenig
  verkehrsbeinflusste Wohn- und Erholungs-Gebiete am Rand der grossen Agglomerationen.
  Measures: PM10 n/a, PM2.5 n/a

Air quality measures available:
 - CO    mg/m³
 - SO₂   µg/m³
 - NOₓ   ppb
 - NO    µg/m³
 - NO₂   µg/m³
 - O₃    µg/m³
 - PM10  µg/m³
 - PM2.5 µg/m³

Weather measures available:
- Lufttemperatur                   T        °C
   Physikalisch betrachtet ist die Lufttemperatur ein Mass für den Wärmezustand eines Luftvolumens. Dieser wird bestimmt
   durch die mittlere kinetische Energie der ungeordneten Molekularbewegung in der Luft. Je grösser die mittlere
   Geschwindigkeit aller Moleküle in einem Luftvolumen ist, um so höher ist auch seine Lufttemperatur.

- Luftdruck                        p        hPa
   Mit Luftdruck wird der von der Masse der Luft unter der Wirkung der Erdanziehung ausgeübte Druck bezeichnet. Er ist
   definiert als das Gewicht der Luftsäule pro Flächeneinheit vom Erdboden bis zur äusseren Grenze der Atmosphäre.

- Windrichtung                     WD       °
   Die Windrichtung ist die Richtung, aus welcher der Wind weht. Sie wird bestimmt nach dem Polarwinkel (Azimut). Zur
   Richtungsangabe benutzt man die 360 Grad Skala des Kreises. Alle Richtungsangaben in Grad sind rechtweisend auf
   geographisch Nord bezogen, d.h. Ost = 90°, Süd = 180°, West=270° und Nord=360°.

- Windgeschwindigkeit (vektoriell) WVv      m/s
   Unter Windgeschwindigkeit ist die horizontale Verlagerungsgeschwindigkeit der Luftteilchen zu verstehen

- Windgeschwindigkeit (skalar)     WVs      m/s
   Unter Windgeschwindigkeit ist die horizontale Verlagerungsgeschwindigkeit der Luftteilchen zu verstehen

- relative Luftfeuchtigkeit        Hr       %
   Die relative Luftfeuchtigkeit gibt das Gewichtsverhältnis des momentanen Wasserdampfgehalts zu dem Wasserdampfgehalt
   an, der für die aktuelle Temperatur und den aktuellen Druck maximal möglich ist.

- Niederschlagsdauer               RainDur  min
   Anzahl der Minuten in denen es im Mittelungsintervall geregnet hat.

- Globalstrahlung                  StrGlo   W/m-2
   Die Globalstrahlung ist die am Boden von einer horizontalen Ebene empfangene Sonnenstrahlung und setzt sich aus der
   direkten Strahlung (der Schatten werfenden Strahlung) und der gestreuten Sonnenstrahlung (diffuse Himmelsstrahlung)
   aus der Himmelshalbkugel zusammen.
"""

from datetime import datetime
from io import StringIO

import pandas
import requests
from dateutil.tz import tz
from hist_data.common import add_columns, table_exists
from sqlalchemy.engine import Connection
from tqdm import tqdm

START_YEAR = 2001
END_YEAR = 2021

tz_local = tz.gettz('UTC+1')

ZURICH_TABLE = 'zurich'


def zurich_hourly_air_data(year):
	r = requests.get(
		'https://data.stadt-zuerich.ch/dataset/ugz_luftschadstoffmessung_stundenwerte/download/'
		f'ugz_ogd_air_h1_{year}.csv')
	text = StringIO(r.content.decode('utf-8'))
	df = pandas.read_csv(text, sep=',', parse_dates=[0])
	df.drop(columns=["Intervall", "Einheit", "Status"], inplace=True)
	df.rename(columns={
		"Datum": "date",
		"Standort": "place",
		"Parameter": "measure",
		"Wert": "value",
	}, inplace=True)
	df = df[df['measure'].isin(('PM10', 'PM2.5'))]
	df['item'] = df['place'] + '.' + df['measure']
	df['date'] = df['date'].dt.tz_convert('UTC')
	df = df.pivot(index='date', columns='item', values='value')
	df.index.name = 'date'
	return df


def zurich_hourly_weather_data(year):
	r = requests.get(
		'https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte/download/'
		f'ugz_ogd_meteo_h1_{year}.csv')
	text = StringIO(r.content.decode('utf-8'))
	df = pandas.read_csv(text, sep=',', parse_dates=[0])
	df.drop(columns=["Intervall", "Einheit", "Status"], inplace=True)
	df.rename(columns={
		"Datum": "date",
		"Standort": "place",
		"Parameter": "measure",
		"Wert": "value",
	}, inplace=True)
	df = df[df['measure'].isin(('T', 'rH', 'Hr', 'p'))]
	df['item'] = df['place'] + '.' + df['measure'].map({
		"T": "Temperature",
		"rH": "Humidity",
		"Hr": "Humidity",
		"p": "Pressure",
	})
	df['date'] = df['date'].dt.tz_convert('UTC')
	df = df.pivot(index='date', columns='item', values='value')
	df.index.name = 'date'
	return df


def zurich_hourly_data(year):
	df_air = zurich_hourly_air_data(year)
	df_weather = zurich_hourly_weather_data(year)
	return df_air.merge(df_weather, how='outer', on='date')


def zurich_download_all(con: Connection):
	print(f'Retrieving data for the city of Zurich from {START_YEAR} to {END_YEAR}')
	con.execute(f'DROP TABLE IF EXISTS {ZURICH_TABLE}')

	for year in tqdm(range(START_YEAR, END_YEAR + 1)):
		df = zurich_hourly_data(year)
		add_columns(ZURICH_TABLE, df, con)
		df.to_sql(ZURICH_TABLE, if_exists='append', con=con)

	print('Done!')


def zurich_update_current_year(con: Connection):
	year = datetime.now(tz_local).year
	print(f'Retrieving data for the city of Zurich for {year}')
	year_start = datetime(year, 1, 1, 0, 0, 0, 0, tzinfo=tz_local)
	df = zurich_hourly_data(year)
	if table_exists(ZURICH_TABLE, con):
		add_columns(ZURICH_TABLE, df, con)
		con.execute(f"DELETE FROM {ZURICH_TABLE} WHERE date >= '{year_start.isoformat()}'")
		df.to_sql(ZURICH_TABLE, if_exists='append', con=con)
	else:
		df.to_sql(ZURICH_TABLE, con=con)
	print('Done!')
