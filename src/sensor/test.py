import configparser
import datetime
from time import sleep

import Adafruit_DHT
from influxdb_client import InfluxDBClient
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from sps30 import SPS30

config = configparser.ConfigParser()
config.read('/home/pi/app/config.ini')

NC05 = "nc0p5"
PM1 = "pm1p0"
NC1 = "nc1p0"
PM25 = "pm2p5"
NC25 = "nc2p5"
PM4 = "pm4p0"
NC4 = "nc4p0"
PM10 = "pm10p0"
NC10 = "nc10p0"
TYP = "typical"

# You can generate a Token from the "Tokens Tab" in the UI
bucket = "actual_weather_data"
client = InfluxDBClient.from_config_file("/home/pi/app/config.ini")

sps = SPS30(1)

if sps.read_article_code() == sps.ARTICLE_CODE_ERROR:
    raise Exception("ARTICLE CODE CRC ERROR!")
else:
    print("ARTICLE CODE: " + str(sps.read_article_code()))

if sps.read_device_serial() == sps.SERIAL_NUMBER_ERROR:
    raise Exception("SERIAL NUMBER CRC ERROR!")
else:
    print("DEVICE SERIAL: " + str(sps.read_device_serial()))

sps.set_auto_cleaning_interval(604800)  # default 604800, set 0 to disable auto-cleaning

sps.device_reset()  # device has to be powered-down or reset to check new auto-cleaning interval

if sps.read_auto_cleaning_interval() == sps.AUTO_CLN_INTERVAL_ERROR:  # or returns the interval in seconds
    raise Exception("AUTO-CLEANING INTERVAL CRC ERROR!")
else:
    print("AUTO-CLEANING INTERVAL: " + str(sps.read_auto_cleaning_interval()))

sleep(5)

sps.start_measurement()

sleep(5)

while not sps.read_data_ready_flag():
    sleep(0.25)
    if sps.read_data_ready_flag() == sps.DATA_READY_FLAG_ERROR:
        raise Exception("DATA-READY FLAG CRC ERROR!")

if sps.read_measured_values() == sps.MEASURED_VALUES_ERROR:
    raise Exception("MEASURED VALUES CRC ERROR!")
else:
    print("PM1.0 Value in ug/m3: " + str(sps.dict_values[PM1]))
    print("PM2.5 Value in ug/m3: " + str(sps.dict_values[PM25]))
    print("PM4.0 Value in ug/m3: " + str(sps.dict_values[PM4]))
    print("PM10.0 Value in ug/m3: " + str(sps.dict_values[PM10]))
    print("NC0.5 Value in 1/cm3: " + str(sps.dict_values[NC05]))  # NC: Number of Concentration
    print("NC1.0 Value in 1/cm3: " + str(sps.dict_values[NC1]))
    print("NC2.5 Value in 1/cm3: " + str(sps.dict_values[NC25]))
    print("NC4.0 Value in 1/cm3: " + str(sps.dict_values[NC4]))
    print("NC10.0 Value in 1/cm3: " + str(sps.dict_values[NC10]))
    print("Typical Particle Size in um: " + str(sps.dict_values[TYP]))

sps.stop_measurement()
sps.start_fan_cleaning()

if int(config['sensor']['humidity']) == 22:
    sensor = Adafruit_DHT.DHT22
elif int(config['sensor']['humidity']) == 11:
    sensor = Adafruit_DHT.DHT11
else:
    raise Exception("No sensor chosen!")

# Set to your GPIO pin
pin = int(config['sensor']['humidity_pin'])  # 4
humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)


def get_pollution_data(name):
    data = sps.dict_values[name]
    return round(data, 2)


temp = Point("weather") \
    .tag("sensor", int(config['sensor']['humidity'])) \
    .tag("device", int(config['device']['id'])) \
    .tag("place", int(config['device']['place'])) \
    .field("temperature", round(temperature, 2)) \
    .field("humidity", round(humidity, 2)) \
    .time(datetime.datetime.utcnow(), WritePrecision.NS)

pollution = Point("pollution") \
    .tag("device", int(config['device']['id'])) \
    .tag("place", int(config['device']['place'])) \
    .field("pm1", get_pollution_data(PM1)) \
    .field("nc1", get_pollution_data(NC1)) \
    .field("pm2.5", get_pollution_data(PM25)) \
    .field("nc2.5", get_pollution_data(NC25)) \
    .field("pm4.0", get_pollution_data(PM4)) \
    .field("nc4.0", get_pollution_data(NC4)) \
    .field("pm10", get_pollution_data(PM10)) \
    .field("nc10", get_pollution_data(NC10)) \
    .time(datetime.datetime.utcnow(), WritePrecision.NS)

print("Humidity: " + str(humidity))
print("Temperature: " + str(temperature))
print(f'Writing to InfluxDB cloud: ...')

write_api = client.write_api(write_options=SYNCHRONOUS)
write_api.write(bucket, record=[temp, pollution])

write_api.close()
client.close()