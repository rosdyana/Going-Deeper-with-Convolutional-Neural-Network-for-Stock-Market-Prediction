from tensorflow.python import keras
from tensorflow.python.keras.layers.core import Dropout
from core.utils import read_config, build_dataset


class cnn:
    def __init__(self, config_path):
        config = read_config(config_path)
        self.input_dir = config['cnn']['dataset']
        self.img_size = int(config['base']['img_size'])
        self.channel = int(config['base']['img_channel'])
        self.epochs = int(config['cnn']['epochs'])
        self.batch_size = int(config['cnn']['batch_size'])
        self.output = config['cnn']['model_name']

    def build_model(self, shape, seed=None):
        import numpy as np
        from tensorflow.keras import layers
        import tensorflow as tf

        if seed:
            np.random.seed(seed)

        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(32,
                                (3, 3),
                                activation='relu',
                                padding='same',
                                kernel_initializer='glorot_uniform',
                                input_shape=(shape)))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(48,
                                (3, 3),
                                activation='relu',
                                padding='same',
                                kernel_initializer='glorot_uniform'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(layers.Conv2D(64,
                                (3, 3),
                                activation='relu',
                                padding='same',
                                kernel_initializer='glorot_uniform'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(96,
                                (3, 3),
                                activation='relu',
                                padding='same',
                                kernel_initializer='glorot_uniform'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))

        return model

    def run_model(self):
        import math
        from tensorflow.keras.optimizers import Adam
        import numpy as np
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from datetime import datetime

        now = datetime.now()

        shape = (self.img_size, self.img_size, self.channel)
        model = self.build_model(shape)
        print(model.summary())
        model.compile(
            optimizer=Adam(learning_rate=1.0e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        x_train, y_train, nb_classes = build_dataset(
            "{}/train".format(self.input_dir),
            self.img_size)
        x_test, y_test, nb_classes = build_dataset(
            "{}/test".format(self.input_dir),
            self.img_size)

        model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs)

        predicted = model.predict(x_test)
        y_pred = np.argmax(predicted, axis=1)
        y_test = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        if tp == 0:
            tp = 1
        if tn == 0:
            tn = 1
        if fp == 0:
            fp = 1
        if fn == 0:
            fn = 1
        TPR = float(tp)/(float(tp)+float(fn))
        FPR = float(fp)/(float(fp)+float(tn))
        accuracy = round((float(tp) + float(tn))/(float(tp) +
                                                  float(fp) + float(fn) + float(tn)), 3)
        specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
        sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)
        mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
            (float(tp)+float(fp))
            * (float(tp)+float(fn))
            * (float(tn)+float(fp))
            * (float(tn)+float(fn))
        ), 3)

        f_output = open(self.output + '_output.txt', 'a')
        f_output.write('=======\n')
        f_output.write(now.strftime("%d/%m/%Y %H:%M:%S") + '\n')
        f_output.write('{}epochs_{}batch_cnn\n'.format(
            self.epochs, self.batch_size))
        f_output.write('TN: {}\n'.format(tn))
        f_output.write('FN: {}\n'.format(fn))
        f_output.write('TP: {}\n'.format(tp))
        f_output.write('FP: {}\n'.format(fp))
        f_output.write('TPR: {}\n'.format(TPR))
        f_output.write('FPR: {}\n'.format(FPR))
        f_output.write('accuracy: {}\n'.format(accuracy))
        f_output.write('specitivity: {}\n'.format(specitivity))
        f_output.write("sensitivity : {}\n".format(sensitivity))
        f_output.write("mcc : {}\n".format(mcc))
        f_output.write("{}".format(report))
        f_output.write('=======\n')
        f_output.close()
