from core.utils import *

class deepcandle:

    def __init__(self, config_path):
        config = read_config(config_path)
        self.label = config['base']['label']
        self.img_size = int(config['base']['img_size'])
        self.tickers = config['base']['tickers']
        self.train_start_date = config['base']['train_start'] if config['base']['train_start'] else None
        self.train_end_date = config['base']['train_end'] if config['base']['train_end'] else None
        self.test_start_date = config['base']['test_start'] if config['base']['test_start'] else None
        self.test_end_date = config['base']['test_end'] if config['base']['test_end'] else None
        self.window_size = int(config['base']['window_size'])
        self.clean = int(config['base']['clean'])

    def clean_dataset(self):
        import shutil
        try:
            shutil.rmtree('dataset')
        except:
            pass

    def generate_dataset(self, ticker):
        print(f'===== Generating {ticker} ====')
        df_train = get_ticker_data(
            ticker,
            self.train_start_date,
            self.train_end_date)
        df_test = get_ticker_data(
            ticker,
            self.test_start_date,
            self.test_end_date)
        if df_train is None or df_test is None:
            return None, None
        return df_train, df_test

    def add_label(self, df_train, df_test):
        """
        Add Label column in dataframe
        Current close > previous close price ? true : false
        """
        df_train[self.label] = df_train.Close > df_train.Close.shift()
        df_test[self.label] = df_test.Close > df_test.Close.shift()
        return df_train, df_test
        

    def convert_to_candlestick(self, ticker, df_train, df_test):
        """
        Method to convert from timeseries dataframe to candlestick chart
        Label taken from the next day data.
        Example: 5 windows size will generate candlestick chart from day 1 to day 5,
        and label will be the day 6.
        """
        for data_type in ['train', 'test']:
            ticker_data = df_train if data_type == 'train' else df_test
            groups = split_dataframe(ticker_data, self.window_size)
            for idx, group in enumerate(groups):
                if len(group) == self.window_size and groups[idx].iloc[0].Close:
                    label = '1' if groups[idx].iloc[0].Label else '0'
                    dir_path = 'dataset/' + data_type + '/' + label + '/'
                    file_name = ticker + '_' + str(idx) + '.jpg'
                    save_to_candlestick(group,
                                        self.img_size,
                                        dir_path,
                                        file_name)

    def start(self):
        """ 
        Start preprocessing data.
        1. Add label from ticker data based on config
        2. Convert to Candlestick chart and save to path based on the label.
        3. Run CNN model
        """
        if self.clean:
            self.clean_dataset()
            print(f'tickers : {self.tickers}')
            for ticker in self.tickers.split(','):
                df_train, df_test = self.generate_dataset(ticker)
                if df_train is None or df_test is None:
                    continue
                df_train, df_test = self.add_label(df_train, df_test)
                self.convert_to_candlestick(ticker, df_train, df_test)

        from core.cnn import cnn
        cnn_model = cnn('config.ini')
        cnn_model.run_model()

# RUN
deepcandle('config.ini').start()
