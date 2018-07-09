import subprocess
import os
import sys

formatters = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'END': '\033[0m',
}

symbol = sys.argv[1]
testing_start_date = "2017-01-01"
testing_end_date = "2018-06-14"
training_end_date = "2016-12-31"
training_start_date = "2000-01-01"
windows_length = sys.argv[2]
dimension = sys.argv[3]
download_data = False
onlytesting = True
onlytraining = True

try:
    if download_data:
        if onlytesting:
            # get data testing
            print('{RED}\nGet Testing Data{END}'.format(**formatters))
            subprocess.call(
                'python utils/get_data.py -sd {} -ed {} -t {} -s yahoo -p testing'.format(testing_start_date, testing_end_date, symbol), shell=True)
            print('{GREEN}Get Testing Data Done\n{END}'.format(**formatters))
        if onlytraining:
            # get data testing
            print('{RED}\nGet Training Data{END}'.format(**formatters))
            subprocess.call(
                'python utils/get_data.py -sd {} -ed {} -t {} -s yahoo -p training'.format(training_start_date, training_end_date, symbol), shell=True)
            print('{GREEN}Get Training Data Done\n{END}'.format(**formatters))
except Exception as identifier:
    print(identifier)

try:
    if onlytraining:
        # create label training
        print('{RED}\nCreate Label Training Data{END}'.format(**formatters))
        subprocess.call('python preproccess_binclass.py -m createLabel -l {} -i stockdatas/{}_training.csv'.format(
            windows_length, symbol), shell=True)
        print('{GREEN}Create Label Training Data Done\n{END}!'.format(**formatters))
    if onlytesting:
        # create label testing
        print('{RED}\nCreate Label Testing Data{END}'.format(**formatters))
        subprocess.call('python preproccess_binclass.py -m createLabel -l {} -i stockdatas/{}_testing.csv'.format(
            windows_length, symbol), shell=True)
        print('{GREEN}Create Label Testing Data Done\n{END}'.format(**formatters))
except Exception as identifier:
    print(identifier)

try:
    if onlytraining:
        # convert to candlestick chart training data
        print('{RED}\nConvert Training Data to Candlestik{END}'.format(**formatters))
        subprocess.call('python preproccess_binclass.py -m ohlc2cs -l {} -i stockdatas/{}_training.csv -t training -d {}'.format(
            windows_length, symbol, dimension), shell=True)
        print('{GREEN}Convert Training Data to Candlestik Done\n{END}'.format(
            **formatters))
    if onlytesting:
        # convert to candlestick chart testing data
        print('{RED}\nConvert Testing Data to Candlestik{END}'.format(**formatters))
        subprocess.call('python preproccess_binclass.py -m ohlc2cs -l {} -i stockdatas/{}_testing.csv -t testing -d {}'.format(
            windows_length, symbol, dimension), shell=True)
        print('{GREEN}Convert Testing Data to Candlestik Done\n{END}'.format(
            **formatters))
except Exception as identifier:
    print(identifier)

try:
    if onlytraining:
        # labelling data training
        print('{RED}\nLabelling Training Data{END}'.format(**formatters))
        subprocess.call('python preproccess_binclass.py -m img2dt -i dataset/{}_{}/{}/training -lf {}_training_label_{}.txt'.format(
            windows_length, dimension, symbol, symbol, windows_length), shell=True)
        print('{GREEN}Labelling Training Data Done\n{END}'.format(**formatters))
    if onlytesting:
        # labelling data testing
        print('{RED}\nLabelling Testing Data{END}'.format(**formatters))
        subprocess.call('python preproccess_binclass.py -m img2dt -i dataset/{}_{}/{}/testing -lf {}_testing_label_{}.txt'.format(
            windows_length, dimension, symbol, symbol, windows_length), shell=True)
        print('{GREEN}Labelling Testing Data Done\n{END}'.format(**formatters))
except Exception as identifier:
    print(identifier)


# print('{RED}Last step please resize images with your own.{END}'.format(**formatters))
# find . -maxdepth 4 -iname "*.png" | xargs -L1 -I{} convert -flatten +matte -adaptive-resize 200x200! "{}" "{}"
# find . -iname "*.png" | xargs -L1 -I{} convert -flatten +matte -adaptive-resize 200x200! "{}" "{}"

# find . -name "*.png" -exec convert "{}" -alpha off "{}" \;
# find . -name "*.png" -exec convert "{}" -adaptive-resize 20x20! "{}" \;
# os.system('spd-say --voice-type female3 "your program has finished"')
