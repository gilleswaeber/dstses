from sqlalchemy import create_engine

from hist_data.nabel import nabel_upload_influx_db


def run():
    engine = create_engine('sqlite:///hist_data.sqlite', echo=False)

    with engine.connect() as con:
        nabel_upload_influx_db(con)


if __name__ == '__main__':
    run()
