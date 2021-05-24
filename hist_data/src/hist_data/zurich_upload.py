from hist_data.common import upload_influx_db, get_engine
from hist_data.zurich import ZURICH_TABLE


def run():
	with get_engine().connect() as con:
		upload_influx_db(con, ZURICH_TABLE)


if __name__ == '__main__':
	run()
