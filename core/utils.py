def get_ticker_data(ticker: str, start: str = None, end: str = None):
    import yfinance as yf
    try:
        return yf.download(ticker, start, end)
    except:
        return None


def read_config(config_path):
    import configparser

    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def split_dataframe(df, chunk_size=5):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def save_to_candlestick(df, dimension, save_path, file_name):
    import matplotlib.pyplot as plt
    from mplfinance.original_flavor import candlestick2_ochl

    plt.style.use('dark_background')
    my_dpi = 96
    fig = plt.figure(figsize=(dimension / my_dpi,
                     dimension / my_dpi), dpi=my_dpi)
    ax1 = fig.add_subplot(1, 1, 1)
    candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'],
                      df['Low'], width=1,
                      colorup='#77d879', colordown='#db3f3f')
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis('off')
    from pathlib import Path
    Path(save_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + file_name, pad_inches=0, transparent=False)
    plt.close(fig)


def dataset(base_dir, n):
    import os
    from collections import defaultdict
    import numpy as np
    from skimage import io

    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            label = os.path.split(root)[-1]
            d[label].append(file_path)

    tags = sorted(d.keys())
    x = []
    y = []

    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            img = io.imread(filename)
            height, width, chan = img.shape
            assert chan == 3
            x.append(img)
            y.append(class_index)

    x = np.array(x).astype(np.float32)
    y = np.array(y)

    return x, y, tags


def build_dataset(data_directory, img_width):
    from tensorflow.keras import utils
    
    x, y, tags = dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    feature = x
    label = utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes