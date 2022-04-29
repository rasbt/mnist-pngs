# Sebastian Raschka, 2022

import struct
import gzip
import os
import urllib.request
import numpy as np
from PIL import Image


urls = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
]
filenames = []

for url in urls:
    filename = os.path.basename(url)
    filename = os.path.join(".", filename)
    filename = filename.replace("t10k", "test")
    filenames.append(filename)
    urllib.request.urlretrieve(url, filename)


zipped_mnist = [f for f in os.listdir() if f.endswith("ubyte.gz")]
for z in filenames:
    with gzip.GzipFile(z, mode="rb") as decompressed, open(z[:-3], "wb") as outfile:
        outfile.write(decompressed.read())


def load_mnist(path, kind="train"):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte")

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


image_name_counter = 0

for kind in ("train", "test"):
    if not os.path.isdir(kind):
        os.mkdir(kind)
    for i in range(10):
        if not os.path.isdir(os.path.join(kind, str(i))):
            os.mkdir(os.path.join(kind, str(i)))

    images, labels = load_mnist(path=".", kind=kind)

    for image, label in zip(images, labels):
        image = image.reshape(28, 28)
        name = f"{image_name_counter}.png"
        image_name_counter += 1
        path = os.path.join(kind, str(label), name)
        im = Image.fromarray(image)
        im.save(path)
