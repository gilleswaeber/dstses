from sqlalchemy import create_engine

from hist_data.nabel import nabel_download


def run():
    engine = create_engine('sqlite:///hist_data.sqlite', echo=False)

    with engine.connect() as con:
        nabel_download(con)


if __name__ == '__main__':
    run()
