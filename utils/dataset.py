from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from collections import defaultdict
import numpy as np
import scipy.misc


def dataset(base_dir, n):
    # print("base_dir : {}, n : {}".format(base_dir, n))
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)

    tags = sorted(d.keys())
    # print("classes : {}".format(tags))

    X = []
    y = []

    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:

            img = scipy.misc.imread(filename)
            height, width, chan = img.shape
            assert chan == 3

            X.append(img)
            y.append(class_index)

    X = np.array(X).astype(np.float32)
    y = np.array(y)

    return X, y, tags
