from hist_data.common import get_engine
from hist_data.nabel import nabel_download_all


def run():
    with get_engine().connect() as con:
        nabel_download_all(con)


if __name__ == '__main__':
    run()
