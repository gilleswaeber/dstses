from datetime import datetime
from dateutil.tz import gettz

from hist_data.common import upload_influx_db, get_engine
from hist_data.zurich import ZURICH_TABLE, zurich_update_current_year

tz_utc = gettz('UTC')


def run():
    with get_engine().connect() as con:
        zurich_update_current_year(con)

        year = datetime.now().year
        year_start = datetime(year, 1, 1, 0, 0, 0, 0, tzinfo=tz_utc)
        upload_influx_db(con, ZURICH_TABLE, year_start)


if __name__ == '__main__':
    run()
