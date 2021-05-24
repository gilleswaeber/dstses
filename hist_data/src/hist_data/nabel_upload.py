from hist_data.common import get_engine, upload_influx_db
from hist_data.nabel import NABEL_TABLE


def run():
	with get_engine().connect() as con:
		upload_influx_db(con, NABEL_TABLE)


if __name__ == '__main__':
	run()
