#!/usr/bin/env python3

import numpy as np
import os
import matplotlib.image as mpimg
import random
import scipy.spatial.distance as dist


class MultilayerPerceptron:
    def __init__(self):
        self.p = 5
        self.side = 6
        self.m = 5
        self.h = 3
        self.a = 0.4
        self.b = 0.5
        self.max_cup = 0.8

        self.x = np.zeros(self.side ** 2)
        self.g = np.zeros(self.h)
        self.y = np.zeros(self.m)

        self.v = np.zeros((len(self.x), self.h))
        self.w = np.zeros((self.h, self.m))
        self.ref_out = np.zeros((self.p, self.p))

        self.q = np.zeros(self.h)
        self.t = np.zeros(self.m)

        self.test_images = np.zeros((self.p, self.side, self.side))
        self.test_dir = "original/"
        self.rec_dir = "noized/"
        self.out_dir = "res/"

    def load_test_images(self):
        for _, _, files in os.walk(self.test_dir):
            i = 0
            for _file in files:
                f = mpimg.imread(self.test_dir + _file)[:, :, 0]
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

    def find_winner(self):
        minimum = 0
        min_pos = 0
        for j in range(0, len(self.y)):
            value = self.victories[j] * dist.euclidean(self.x, self.w[:, j])
            if j == 0:
                minimum = value
            if value <= minimum:
                minimum = value
                min_pos = j
        self.victories[min_pos] += 1
        return min_pos

    def find_cluster(self):
        minimum = 0
        min_pos = 0
        for j in range(0, len(self.y)):
            value = dist.euclidean(self.x, self.w[:, j])
            if j == 0:
                minimum = value
            if value <= minimum:
                minimum = value
                min_pos = j
        return min_pos

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
            s += self.d(k, yr) * self.f(k) * self.w[j, k]
        return s

    def make_corrections(self, yr):
        for j in range(len(self.w)):
            for k in range(len(self.w[0])):
                self.w[j, k] = self.w[j, k] + self.a * self.y[k] * (1 - self.y[k]) * self.d(k, yr) * self.g[j]

        for k in range(len(self.t)):
            self.t[k] = self.t[k] + self.a * self.y[k] * (1 - self.y[k]) * self.d(k, yr)

        for i in range(len(self.v)):
            for j in range(len(self.v[0])):
                self.v[i, j] = self.v[i, j] + self.b * self.g[j] * (1 - self.g[j]) * self.e(j, yr) * self.x[i]

        for j in range(len(self.q)):
            self.q[j] = self.q[j] + self.b * self.g[j] * (1 - self.g[j]) * self.e(j, yr)
        pass

    def train(self):
        self.init_net()

        maximum = 0
        while True:
            for image_num in range(0, self.p):
                errors = []
                self.x = self.test_images[image_num].ravel()

                self.calc_neurons()
                self.make_corrections(self.ref_out[image_num])

                for k in range(0, self.m):
                    errors.append(np.abs(self.d(k, self.ref_out[image_num])))
                if np.argmax(errors) >= maximum:
                    maximum = np.argmax(errors)

            print(maximum)
            if maximum < self.max_cup:
                break

    def play(self, image):
        self.x = np.array(image)
        self.calc_neurons()
        return self.find_cluster()

    def recognize(self):
        for _, _, files in os.walk(self.rec_dir):
            for _file in files:
                f = mpimg.imread(self.rec_dir + _file)[:, :, 0]
                cluster = self.play(f.ravel())
                res = np.zeros((f.shape[0], f.shape[1], 3))
                res[:, :, 0] = f
                res[:, :, 1] = f
                res[:, :, 2] = f
                mpimg.imsave(self.out_dir + str(cluster) + "/" + _file, res)

    def run(self):
        self.load_test_images()
        self.train()
        #  self.recognize()


if __name__ == "__main__":
    net = MultilayerPerceptron()
    net.run()
