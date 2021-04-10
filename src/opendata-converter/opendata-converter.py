
"""
	This script normalises the air quality and meteo datasets from opendata.swiss into two much smaller files.
	
	
"""

import logger

from pandas import read_csv, DataFrame
from os import listdir
from dateutil.parser import isoparse
from datetime import datetime
from time import time

logger = logger.Logger("OpenData Converter")
start_script: float = time()


"""
	Filters a list of filenames for a specific year and returns the meteo and the air quality filename for that year
"""
def getFilesForYear(year: int, files: list[str]) -> (str, str):
	air_quality: list[str] = list(filter(lambda name: "_air_" in name and str(year) in name, files))
	meteo: list[str] = list(filter(lambda name: "_meteo_" in name and str(year) in name, files))
	return (meteo[0] if len(meteo) > 0 else None, air_quality[0] if len(air_quality) > 0 else None)

"""
	Finds the earliest and the latest year that is given in the list of files
"""
def getMinAndMaxYears(files: list[str]) -> (int, int):
	min: int = 1000000
	max: int = 0
	for fname in files:
		year: int = int(fname[-8:-4])
		if min > year: min = year
		if max < year: max = year
	
	return (min, max)
	
"""
	Reads all data in a given directory into a single large list of records
"""
def read_data(dir: str) -> list[tuple]:
	files = [ (dir if dir.endswith("/") else (dir + "/")) + filename for filename in listdir(dir) ]
	# find the first and last year of given data
	min_year, max_year = getMinAndMaxYears(files)
	# initialise list that will contain all records
	data: list[(datetime, str, str, str, str, float, str)] = []
	
	# go through all years
	for year in range(min_year, max_year+1, 1):
		meteo, quality = getFilesForYear(year, files)
		
		# read all data from the meteo file if it exists
		if meteo != None:
			meteo_records: list[tuple] = read_csv(meteo).to_records(index=False)
			data += [ (isoparse(rec[0]), rec[1], rec[2], rec[3], rec[4], float(rec[5]), rec[6]) for rec in list(meteo_records) ]
		
		# read all data from the qir quality file if it exists
		if quality != None:
			quality_records: list[tuple] = read_csv(quality).to_records(index=False)
			data += [ (isoparse(rec[0]), rec[1], rec[2], rec[3], rec[4], float(rec[5]), rec[6]) for rec in list(quality_records) ]
	
	return data

"""
	Determines an id for a value that may occur multiple time in a table
"""
def getIdOfValue(meta_info: list[(int, str)], value) -> int:
	ids = list(filter(lambda loc: loc[1] == value, meta_info))
	if len(ids) == 1:
		return ids[0][0]
	else:
		id = len(meta_info)
		meta_info.append( (id, value) )
		return id

# read data into array
logger.info("Reading Data...")
start_read: float = time()
raw_data: list[tuple] = read_data("../../Project/Datasets/Raw/")
end_read: float = time()
no_raw: int = len(raw_data)

# filter out all measurements we are not interested in
logger.info("Filtering Data...")
start_filter: float = time()
filtered_data: list[tuple] = list(filter(lambda rec: rec[2] in [ "PM1", "PM2.5", "PM10", "Hr", "T", "p" ], raw_data))
end_filter: float = time()
no_filter: int = len(filtered_data)

# normalize data (combine multiple measurements from the same timestamp into one large table)
# the format is: *datetime, location*, PM2.5, PM10, Hr, T, p
logger.info("Normalising Data...")
start_norm: float = time()
norm_data: list[(datetime, str, float, float, float, float, float)] = []
norm_data_map: dict[(datetime, str),(datetime, str, float, float, float, float, float)] = {}

for (timestamp, location, param, interval, unit, val, state) in filtered_data:
	if not (timestamp, location) in norm_data_map:
		new_record = (datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour), location[4:], None, None, None, None, None)
		norm_data_map[(timestamp, location)] = new_record
	
	tmp: (datetime, str, float, float, float, float, float) = norm_data_map[(timestamp, location)]
	if param == "PM2.5": norm_data_map[(timestamp, location)] = tmp[:2] + tuple([val]) + tmp[3:]
	if param == "PM10": norm_data_map[(timestamp, location)] = tmp[:3] + tuple([val]) + tmp[4:]
	if param == "Hr": norm_data_map[(timestamp, location)] = tmp[:4] + tuple([val]) + tmp[5:]
	if param == "T": norm_data_map[(timestamp, location)] = tmp[:5] + tuple([val]) + tmp[6:]
	if param == "p": norm_data_map[(timestamp, location)] = tmp[:6] + tuple([val])

norm_data = list(norm_data_map.values())
end_norm: float = time()
no_records: int = len(norm_data)



# output the data into single single csv
start_output_complete: float = time()
dataframe: DataFrame = DataFrame.from_records(norm_data)
dataframe.to_csv("dataset_complete_normalized.csv", index=False, na_rep="N/A", header=["Timestamp", "Location", "PM2.5 [µg/m3]", "PM10 [µg/m3]", "Relative Humidity [%]", "Temperature [°C]", "Pressure [hPa]"])
end_output_complete: float = time()

#output the data containing measurments for PM10
start_output_pm10: float = time()
dataframe: DataFrame = DataFrame.from_records(list(filter(lambda x: x[3] != None, norm_data)))
dataframe.to_csv("dataset_pm10_normalized.csv", index=False, na_rep="N/A", header=["Timestamp", "Location", "PM2.5 [µg/m3]", "PM10 [µg/m3]", "Relative Humidity [%]", "Temperature [°C]", "Pressure [hPa]"])
end_output_pm10: float = time()

logger.info("Done.")
logger.info("Read {:d} raw measurements in {:.3f}s".format(no_raw, end_read - start_read))
logger.info("Filtered {:d} measurements in {:.3f}s".format(no_filter, end_filter - start_filter))
logger.info("Normalised {:d} records in {:.3f}s".format(no_records, end_norm - start_norm))
logger.info("Written complete normalised dataset in {:.3f}s".format(end_output_complete - start_output_complete))
logger.info("Written normalised pm10 dataset in {:.3f}s".format(end_output_pm10 - start_output_pm10))
logger.info("Entire process took {:.3f}s".format(time() - start_script))
