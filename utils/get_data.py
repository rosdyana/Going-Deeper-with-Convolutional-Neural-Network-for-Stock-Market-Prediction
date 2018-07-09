import argparse
import arrow
import pandas as pd
import datetime as dt
from pandas_datareader import data, wb
import os
import fix_yahoo_finance as yf
import time
# fixed pandas_datareader can't download from yahoo finance
yf.pdr_override()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sd', '--start_date', type=str,
                        default='1990-01-01', help='Start date parameter value - format YYYY-MM-DD')
    parser.add_argument('-ed', '--end_date', type=str,
                        default=arrow.now().format('YYYY-MM-DD'), help='End date parameter - format YYYY-MM-DD')
    parser.add_argument('-t', '--ticker', nargs='+',
                        help='<Required> Set flag', required=True)
    parser.add_argument(
        '-s', '--source', help='<Required> set source', required=True)
    parser.add_argument('-a', '--attempt',
                        help='set max attempt to download', default=10)
    parser.add_argument(
        '-e', '--exist', help='check exist stock history file', default=False)
    parser.add_argument('-p', '--prefix', help='add prefix in output name')
    args = parser.parse_args()
    # # fetch all data

    prefix_name = ""
    # make sure output folder is exist
    if not os.path.isdir("../stockdatas"):
        os.mkdir("../stockdatas")
    if len(args.prefix) > 1:
        prefix_name = args.prefix
    if args.source == "tiingo":
        for ticker in set(args.ticker):
            fetch_tiingo_data(ticker, args.start_date, args.end_date,
                              "../stockdatas/{}_{}.csv".format(ticker, prefix_name))
    elif args.source == "yahoo":
        for ticker in set(args.ticker):
            fetch_yahoo_data(ticker, args.start_date, args.end_date,
                             "../stockdatas/{}_{}.csv".format(ticker, prefix_name), args.attempt, args.exist)


def fetch_tiingo_data(ticker, start_date, end_date, fname):
    url = "https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={token}"
    token = "ca5a6f47a99ae61051e4de63b26f727b1709a01d"
    data = pd.read_json(url.format(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        token=token
    ))
    data.to_csv(fname, columns=["date", "open", "close",
                                "high", "low", "volume", "adjClose"], index=False)


def fetch_yahoo_data(ticker, start_date, end_date, fname, max_attempt, check_exist):
    if (os.path.exists(fname) == True) and check_exist:
        print("file exist")
    else:
        # remove exist file
        if os.path.exists(fname):
            os.remove(fname)
        for attempt in range(max_attempt):
            time.sleep(2)
            try:
                dat = data.get_data_yahoo(''.join("{}".format(
                    ticker)),  start=start_date, end=end_date)
                dat.to_csv(fname)
            except Exception as e:
                if attempt < max_attempt - 1:
                    print('Attempt {}: {}'.format(attempt + 1, str(e)))
                else:
                    raise
            else:
                break


if __name__ == '__main__':
    main()
