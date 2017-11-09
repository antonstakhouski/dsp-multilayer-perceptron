#!/usr/bin/env python3

import numpy as np
import os
import matplotlib.image as mpimg
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MultilayerPerceptron:
    def __init__(self):
        self.p = 5
        self.side = 6
        self.m = 5
        self.h = 3
        self.a = 0.4
        self.b = 0.7
        self.max_cup = 0.08
        self.gs = gridspec.GridSpec(4, 5)

        self.x = np.zeros(self.side ** 2)
        self.g = np.zeros(self.h)
        self.y = np.zeros(self.m)

        self.v = np.zeros((len(self.x), self.h))
        self.w = np.zeros((self.h, self.m))
        self.ref_out = np.zeros((self.p, self.p))

        self.q = np.zeros(self.h)
        self.t = np.zeros(self.m)

        self.dklist = np.zeros(self.m)

        self.test_images = np.zeros((self.p, self.side, self.side))
        self.test_dir = "original/"
        self.rec_dir = "noized/"
        self.out_dir = "res/"

    def load_test_images(self):
        for _, _, files in os.walk(self.test_dir):
            i = 0
            for _file in files:
                f = mpimg.imread(self.test_dir + _file)

                plt.subplot(self.gs[0, i])
                plt.imshow(f)
                title = "Image " + str(i)
                plt.title(title)
                f = f[:, :, 0]

                self.test_images[i] = f
                self.ref_out[i, i] = 1.0
                i += 1

    def out_g(self, j):
        s = 0
        for i in range(0, len(self.x)):
            s += self.v[i, j] * self.x[i]
        return self.activate(s + self.q[j])

    def out_y(self, k):
        s = 0
        for j in range(0, self.h):
            s += self.w[j, k] * self.g[j]
        return self.activate(s + self.t[k])

    def activate(self, x):
        return 1 / (1 + np.e ** (-x))

    def init_net(self):
        for i in range(0, len(self.v)):
            for j in range(0, len(self.v[0])):
                self.v[i, j] = random.uniform(-1, 1)

        for i in range(0, len(self.w)):
            for j in range(0, len(self.w[0])):
                self.w[i, j] = random.uniform(-1, 1)

        for i in range(0, len(self.q)):
            self.q[i] = random.uniform(-1, 1)

        for i in range(0, len(self.t)):
            self.t[i] = random.uniform(-1, 1)

    def calc_neurons(self):
        for j in range(0, len(self.g)):
            self.g[j] = self.out_g(j)

        for k in range(0, len(self.y)):
            self.y[k] = self.out_y(k)

    def d(self, k, yr):
        return yr[k] - self.y[k]

    def f(self, k):
        return self.y[k] * (1 - self.y[k])

    def e(self, j, yr):
        s = 0
        for k in range(0, self.m):
            s += self.dklist[k] * self.f(k) * self.w[j, k]
        return s

    def make_corrections(self, yr):
        for k in range(len(self.y)):
            self.dklist[k] = self.d(k, yr)

        for j in range(len(self.w)):
            for k in range(len(self.w[0])):
                self.w[j, k] = self.w[j, k] + self.a * self.y[k] * (1 - self.y[k]) * self.dklist[k] * self.g[j]

        for k in range(len(self.t)):
            self.t[k] = self.t[k] + self.a * self.y[k] * (1 - self.y[k]) * self.dklist[k]

        for i in range(len(self.v)):
            for j in range(len(self.v[0])):
                self.v[i, j] = self.v[i, j] + self.b * self.g[j] * (1 - self.g[j]) * self.e(j, yr) * self.x[i]

        for j in range(len(self.q)):
            self.q[j] = self.q[j] + self.b * self.g[j] * (1 - self.g[j]) * self.e(j, yr)

    def train(self):
        self.init_net()

        while True:
            maximum = 0
            for image_num in range(0, self.p):
                self.dklist = np.zeros(self.m)
                self.x = self.test_images[image_num].ravel()

                self.calc_neurons()
                self.make_corrections(self.ref_out[image_num])

                max_val = max(list(map((lambda x: abs(x)), self.dklist)))
                if max_val >= maximum:
                    maximum = max_val

            if maximum < self.max_cup:
                break
            else:
                maximum = 0

    def play(self, name, image):
        self.x = np.array(image.ravel())
        self.calc_neurons()
        print(name)
        values = list(map((lambda x: float("%.2f" % (x * 100))), self.y))
        print(values)
        return np.argmax(values)

    def recognize(self):
        for _, _, files in os.walk(self.rec_dir):
            i = 1
            j = 0
            lst = []
            cnt = 0
            for _file in files:
                f = mpimg.imread(self.rec_dir + _file)

                plt.subplot(self.gs[i, j])
                plt.imshow(f)
                title = _file
                plt.title(title)
                f = f[:, :, 0]

                if self.play(_file, f) == j:
                    cnt += 1

                if i < 3:
                    i += 1
                else:
                    lst.append(cnt / 3 * 100)
                    cnt = 0
                    i = 1
                    j += 1
            print(lst)

    def run(self):
        self.load_test_images()
        self.train()
        self.recognize()
        plt.show()


if __name__ == "__main__":
    net = MultilayerPerceptron()
    net.run()
