from datetime import datetime, timedelta

from dateutil.tz import gettz

from hist_data.common import upload_influx_db, get_engine
from hist_data.nabel import nabel_update_recent, NABEL_TABLE

tz_utc = gettz('UTC')

NUM_DAYS = 10


def run():
    with get_engine().connect() as con:
        recent = datetime.utcnow() - timedelta(days=NUM_DAYS)
        nabel_update_recent(con, recent)
        upload_influx_db(con, NABEL_TABLE, recent)


if __name__ == '__main__':
    run()
