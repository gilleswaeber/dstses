from hist_data.common import get_engine
from hist_data.zurich import zurich_download


def run():
    with get_engine().connect() as con:
        zurich_download(con)


if __name__ == '__main__':
    run()
