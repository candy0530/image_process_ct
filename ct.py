# -*- coding: utf-8 -*-
import cv2, copy
import tkFileDialog
from Tkinter import *
import numpy as np
from matplotlib import pyplot as plt

class ImageProcessing:

    def __init__(self):
        self.root = Tk()
        self.fileName = ''
        self.origin_img = []
        self.img = None
        self.cols = 0
        self.rows = 0
        self.init_window()

    def corp_image(self):
        self.origin_img.append(copy.deepcopy(self.img))
        origin = copy.deepcopy(self.img)
        alter_img = []
        for x in range(self.rows*33/100, self.rows*52/100):
            alter_img.append(self.img[x])
        self.img = np.array(alter_img, dtype=np.uint8)
        self.rows, self.cols = self.img.shape[:2]
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def rotate_image(self):
        self.origin_img.append(copy.deepcopy(self.img))
        origin = copy.deepcopy(self.img)
        alter_img = []
        origin_x = self.rows - 1
        origin_y = 0
        for x in xrange(self.cols):
            alter_img.append([])
            for y in xrange(self.rows):
                alter_img[x].append(self.img[origin_x, origin_y])
                origin_x -= 1
            origin_x = self.rows - 1
            origin_y += 1
        self.img = np.array(alter_img, dtype=np.uint8)
        self.rows, self.cols = self.img.shape[:2]
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def image_nagetives(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x]/255.0
                gray = red*0.299+green*0.587+blue*0.114
                I, H, S = self._RGB_to_IHS(gray, gray, gray)
                self.img[y, x] = self._IHS_to_RGB(1-I, np.nan_to_num(H), S)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def peak_and_valley_filter(self):
        self.origin_img.append(copy.deepcopy(self.img))
        origin_R = []
        origin_G = []
        origin_B = []
        for x in xrange(self.rows):
            origin_R.append([])
            origin_G.append([])
            origin_B.append([])
            for y in xrange(self.cols):
                red, green, blue = self.img[x, y]
                origin_R[x].append(red)
                origin_G[x].append(green)
                origin_B[x].append(blue)

        peak_filter_R = copy.deepcopy(origin_R)
        peak_filter_G = copy.deepcopy(origin_G)
        peak_filter_B = copy.deepcopy(origin_B)
        for x in xrange(self.rows):
            for y in xrange(self.cols):
                if x == 0:
                    if y == 0:
                        biggest_R = max(origin_R[x+1][y], origin_R[x+1][y+1], origin_R[x][y+1])
                        biggest_G = max(origin_G[x+1][y], origin_G[x+1][y+1], origin_G[x][y+1])
                        biggest_B = max(origin_B[x+1][y], origin_B[x+1][y+1], origin_B[x][y+1])
                    elif y == self.cols - 1:
                        biggest_R = max(origin_R[x+1][y-1], origin_R[x+1][y], origin_R[x][y-1])
                        biggest_G = max(origin_G[x+1][y-1], origin_G[x+1][y], origin_G[x][y-1])
                        biggest_B = max(origin_B[x+1][y-1], origin_B[x+1][y], origin_B[x][y-1])
                    else:
                        biggest_R = max(origin_R[x+1][y+1], origin_R[x+1][y], origin_R[x+1][y-1], origin_R[x][y+1], origin_R[x][y-1])
                        biggest_G = max(origin_G[x+1][y+1], origin_G[x+1][y], origin_G[x+1][y-1], origin_G[x][y+1], origin_G[x][y-1])
                        biggest_B = max(origin_B[x+1][y+1], origin_B[x+1][y], origin_B[x+1][y-1], origin_B[x][y+1], origin_B[x][y-1])
                elif x == self.rows -1:
                    if y == 0:
                        biggest_R = max(origin_R[x-1][y], origin_R[x-1][y+1], origin_R[x][y+1])
                        biggest_G = max(origin_G[x-1][y], origin_G[x-1][y+1], origin_G[x][y+1])
                        biggest_B = max(origin_B[x-1][y], origin_B[x-1][y+1], origin_B[x][y+1])
                    elif y == self.cols - 1:
                        biggest_R = max(origin_R[x-1][y-1], origin_R[x-1][y], origin_R[x][y-1])
                        biggest_G = max(origin_G[x-1][y-1], origin_G[x-1][y], origin_G[x][y-1])
                        biggest_B = max(origin_B[x-1][y-1], origin_B[x-1][y], origin_B[x][y-1])
                    else:
                        biggest_R = max(origin_R[x-1][y+1], origin_R[x-1][y], origin_R[x-1][y-1], origin_R[x][y+1], origin_R[x][y-1])
                        biggest_G = max(origin_G[x-1][y+1], origin_G[x-1][y], origin_G[x-1][y-1], origin_G[x][y+1], origin_G[x][y-1])
                        biggest_B = max(origin_B[x-1][y+1], origin_B[x-1][y], origin_B[x-1][y-1], origin_B[x][y+1], origin_B[x][y-1])
                else:
                    if y == 0:
                        biggest_R = max(origin_R[x+1][y], origin_R[x+1][y+1], origin_R[x][y+1], origin_R[x-1][y], origin_R[x-1][y+1])
                        biggest_G = max(origin_G[x+1][y], origin_G[x+1][y+1], origin_G[x][y+1], origin_G[x-1][y], origin_G[x-1][y+1])
                        biggest_B = max(origin_B[x+1][y], origin_B[x+1][y+1], origin_B[x][y+1], origin_B[x-1][y], origin_B[x-1][y+1])
                    elif y == self.cols - 1:
                        biggest_R = max(origin_R[x+1][y-1], origin_R[x+1][y], origin_R[x][y-1], origin_R[x-1][y-1], origin_R[x-1][y])
                        biggest_G = max(origin_G[x+1][y-1], origin_G[x+1][y], origin_G[x][y-1], origin_G[x-1][y-1], origin_G[x-1][y])
                        biggest_B = max(origin_B[x+1][y-1], origin_B[x+1][y], origin_B[x][y-1], origin_B[x-1][y-1], origin_B[x-1][y])
                    else:
                        biggest_R = max(origin_R[x+1][y+1], origin_R[x+1][y], origin_R[x+1][y-1], origin_R[x][y+1], origin_R[x][y-1], origin_R[x-1][y+1], origin_R[x-1][y], origin_R[x-1][y-1])
                        biggest_G = max(origin_G[x+1][y+1], origin_G[x+1][y], origin_G[x+1][y-1], origin_G[x][y+1], origin_G[x][y-1], origin_G[x-1][y+1], origin_G[x-1][y], origin_G[x-1][y-1])
                        biggest_B = max(origin_B[x+1][y+1], origin_B[x+1][y], origin_B[x+1][y-1], origin_B[x][y+1], origin_B[x][y-1], origin_B[x-1][y+1], origin_B[x-1][y], origin_B[x-1][y-1])

                if biggest_R < peak_filter_R[x][y]:
                    peak_filter_R[x][y] = biggest_R
                if biggest_G < peak_filter_G[x][y]:
                    peak_filter_G[x][y] = biggest_G
                if biggest_B < peak_filter_B[x][y]:
                    peak_filter_B[x][y] = biggest_B

        
        valley_filter_R = copy.deepcopy(peak_filter_R)
        valley_filter_G = copy.deepcopy(peak_filter_G)
        valley_filter_B = copy.deepcopy(peak_filter_B)
        for x in xrange(self.rows):
            for y in xrange(self.cols):
                if x == 0:
                    if y == 0:
                        smallest_R = min(peak_filter_R[x+1][y], peak_filter_R[x+1][y+1], peak_filter_R[x][y+1])
                        smallest_G = min(peak_filter_G[x+1][y], peak_filter_G[x+1][y+1], peak_filter_G[x][y+1])
                        smallest_B = min(peak_filter_B[x+1][y], peak_filter_B[x+1][y+1], peak_filter_B[x][y+1])
                    elif y == self.cols - 1:
                        smallest_R = min(peak_filter_R[x+1][y-1], peak_filter_R[x+1][y], peak_filter_R[x][y-1])
                        smallest_G = min(peak_filter_G[x+1][y-1], peak_filter_G[x+1][y], peak_filter_G[x][y-1])
                        smallest_B = min(peak_filter_B[x+1][y-1], peak_filter_B[x+1][y], peak_filter_B[x][y-1])
                    else:
                        smallest_R = min(peak_filter_R[x+1][y+1], peak_filter_R[x+1][y], peak_filter_R[x+1][y-1], peak_filter_R[x][y+1], peak_filter_R[x][y-1])
                        smallest_G = min(peak_filter_G[x+1][y+1], peak_filter_G[x+1][y], peak_filter_G[x+1][y-1], peak_filter_G[x][y+1], peak_filter_G[x][y-1])
                        smallest_B = min(peak_filter_B[x+1][y+1], peak_filter_B[x+1][y], peak_filter_B[x+1][y-1], peak_filter_B[x][y+1], peak_filter_B[x][y-1])
                elif x == self.rows -1:
                    if y == 0:
                        smallest_R = min(peak_filter_R[x-1][y], peak_filter_R[x-1][y+1], peak_filter_R[x][y+1])
                        smallest_G = min(peak_filter_G[x-1][y], peak_filter_G[x-1][y+1], peak_filter_G[x][y+1])
                        smallest_B = min(peak_filter_B[x-1][y], peak_filter_B[x-1][y+1], peak_filter_B[x][y+1])
                    elif y == self.cols - 1:
                        smallest_R = min(peak_filter_R[x-1][y-1], peak_filter_R[x-1][y], peak_filter_R[x][y-1])
                        smallest_G = min(peak_filter_G[x-1][y-1], peak_filter_G[x-1][y], peak_filter_G[x][y-1])
                        smallest_B = min(peak_filter_B[x-1][y-1], peak_filter_B[x-1][y], peak_filter_B[x][y-1])
                    else:
                        smallest_R = min(peak_filter_R[x-1][y+1], peak_filter_R[x-1][y], peak_filter_R[x-1][y-1], peak_filter_R[x][y+1], peak_filter_R[x][y-1])
                        smallest_G = min(peak_filter_G[x-1][y+1], peak_filter_G[x-1][y], peak_filter_G[x-1][y-1], peak_filter_G[x][y+1], peak_filter_G[x][y-1])
                        smallest_B = min(peak_filter_B[x-1][y+1], peak_filter_B[x-1][y], peak_filter_B[x-1][y-1], peak_filter_B[x][y+1], peak_filter_B[x][y-1])
                else:
                    if y == 0:
                        smallest_R = min(peak_filter_R[x+1][y], peak_filter_R[x+1][y+1], peak_filter_R[x][y+1], peak_filter_R[x-1][y], peak_filter_R[x-1][y+1])
                        smallest_G = min(peak_filter_G[x+1][y], peak_filter_G[x+1][y+1], peak_filter_G[x][y+1], peak_filter_G[x-1][y], peak_filter_G[x-1][y+1])
                        smallest_B = min(peak_filter_B[x+1][y], peak_filter_B[x+1][y+1], peak_filter_B[x][y+1], peak_filter_B[x-1][y], peak_filter_B[x-1][y+1])
                    elif y == self.cols - 1:
                        smallest_R = min(peak_filter_R[x+1][y-1], peak_filter_R[x+1][y], peak_filter_R[x][y-1], peak_filter_R[x-1][y-1], peak_filter_R[x-1][y])
                        smallest_G = min(peak_filter_G[x+1][y-1], peak_filter_G[x+1][y], peak_filter_G[x][y-1], peak_filter_G[x-1][y-1], peak_filter_G[x-1][y])
                        smallest_B = min(peak_filter_B[x+1][y-1], peak_filter_B[x+1][y], peak_filter_B[x][y-1], peak_filter_B[x-1][y-1], peak_filter_B[x-1][y])
                    else:
                        smallest_R = min(peak_filter_R[x+1][y+1], peak_filter_R[x+1][y], peak_filter_R[x+1][y-1], peak_filter_R[x][y+1], peak_filter_R[x][y-1], peak_filter_R[x-1][y+1], peak_filter_R[x-1][y], peak_filter_R[x-1][y-1])
                        smallest_G = min(peak_filter_G[x+1][y+1], peak_filter_G[x+1][y], peak_filter_G[x+1][y-1], peak_filter_G[x][y+1], peak_filter_G[x][y-1], peak_filter_G[x-1][y+1], peak_filter_G[x-1][y], peak_filter_G[x-1][y-1])
                        smallest_B = min(peak_filter_B[x+1][y+1], peak_filter_B[x+1][y], peak_filter_B[x+1][y-1], peak_filter_B[x][y+1], peak_filter_B[x][y-1], peak_filter_B[x-1][y+1], peak_filter_B[x-1][y], peak_filter_B[x-1][y-1])

                if smallest_R > valley_filter_R[x][y]:
                    valley_filter_R[x][y] = smallest_R
                if smallest_G > valley_filter_G[x][y]:
                    valley_filter_G[x][y] = smallest_G
                if smallest_B > valley_filter_B[x][y]:
                    valley_filter_B[x][y] = smallest_B

        for x in xrange(self.rows):
            for y in xrange(self.cols):
                self.img[x, y] = (valley_filter_R[x][y], valley_filter_G[x][y], valley_filter_B[x][y])
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def median_filter(self):
        self.origin_img.append(copy.deepcopy(self.img))
        origin_R = []
        origin_G = []
        origin_B = []
        for x in xrange(self.rows):
            origin_R.append([])
            origin_G.append([])
            origin_B.append([])
            for y in xrange(self.cols):
                red, green, blue = self.img[x, y]
                origin_R[x].append(red)
                origin_G[x].append(green)
                origin_B[x].append(blue)

        median_filter_R = copy.deepcopy(origin_R)
        median_filter_G = copy.deepcopy(origin_G)
        median_filter_B = copy.deepcopy(origin_B)
        for x in xrange(self.rows):
            for y in xrange(self.cols):
                if x == 0:
                    if y == 0:
                        sort_R = [origin_R[x+1][y], origin_R[x+1][y+1], origin_R[x][y+1]]
                        sort_G = [origin_G[x+1][y], origin_G[x+1][y+1], origin_G[x][y+1]]
                        sort_B = [origin_B[x+1][y], origin_B[x+1][y+1], origin_B[x][y+1]]
                    elif y == self.cols - 1:
                        sort_R = [origin_R[x+1][y-1], origin_R[x+1][y], origin_R[x][y-1]]
                        sort_G = [origin_G[x+1][y-1], origin_G[x+1][y], origin_G[x][y-1]]
                        sort_B = [origin_B[x+1][y-1], origin_B[x+1][y], origin_B[x][y-1]]
                    else:
                        sort_R = [origin_R[x+1][y+1], origin_R[x+1][y], origin_R[x+1][y-1], origin_R[x][y+1], origin_R[x][y-1]]
                        sort_G = [origin_G[x+1][y+1], origin_G[x+1][y], origin_G[x+1][y-1], origin_G[x][y+1], origin_G[x][y-1]]
                        sort_B = [origin_B[x+1][y+1], origin_B[x+1][y], origin_B[x+1][y-1], origin_B[x][y+1], origin_B[x][y-1]]
                elif x == self.rows -1:
                    if y == 0:
                        sort_R = [origin_R[x-1][y], origin_R[x-1][y+1], origin_R[x][y+1]]
                        sort_G = [origin_G[x-1][y], origin_G[x-1][y+1], origin_G[x][y+1]]
                        sort_B = [origin_B[x-1][y], origin_B[x-1][y+1], origin_B[x][y+1]]
                    elif y == self.cols - 1:
                        sort_R = [origin_R[x-1][y-1], origin_R[x-1][y], origin_R[x][y-1]]
                        sort_G = [origin_G[x-1][y-1], origin_G[x-1][y], origin_G[x][y-1]]
                        sort_B = [origin_B[x-1][y-1], origin_B[x-1][y], origin_B[x][y-1]]
                    else:
                        sort_R = [origin_R[x-1][y+1], origin_R[x-1][y], origin_R[x-1][y-1], origin_R[x][y+1], origin_R[x][y-1]]
                        sort_G = [origin_G[x-1][y+1], origin_G[x-1][y], origin_G[x-1][y-1], origin_G[x][y+1], origin_G[x][y-1]]
                        sort_B = [origin_B[x-1][y+1], origin_B[x-1][y], origin_B[x-1][y-1], origin_B[x][y+1], origin_B[x][y-1]]
                else:
                    if y == 0:
                        sort_R = [origin_R[x+1][y], origin_R[x+1][y+1], origin_R[x][y+1], origin_R[x-1][y], origin_R[x-1][y+1]]
                        sort_G = [origin_G[x+1][y], origin_G[x+1][y+1], origin_G[x][y+1], origin_G[x-1][y], origin_G[x-1][y+1]]
                        sort_B = [origin_B[x+1][y], origin_B[x+1][y+1], origin_B[x][y+1], origin_B[x-1][y], origin_B[x-1][y+1]]
                    elif y == self.cols - 1:
                        sort_R = [origin_R[x+1][y-1], origin_R[x+1][y], origin_R[x][y-1], origin_R[x-1][y-1], origin_R[x-1][y]]
                        sort_G = [origin_G[x+1][y-1], origin_G[x+1][y], origin_G[x][y-1], origin_G[x-1][y-1], origin_G[x-1][y]]
                        sort_B = [origin_B[x+1][y-1], origin_B[x+1][y], origin_B[x][y-1], origin_B[x-1][y-1], origin_B[x-1][y]]
                    else:
                        sort_R = [origin_R[x+1][y+1], origin_R[x+1][y], origin_R[x+1][y-1], origin_R[x][y+1], origin_R[x][y-1], origin_R[x-1][y+1], origin_R[x-1][y], origin_R[x-1][y-1]]
                        sort_G = [origin_G[x+1][y+1], origin_G[x+1][y], origin_G[x+1][y-1], origin_G[x][y+1], origin_G[x][y-1], origin_G[x-1][y+1], origin_G[x-1][y], origin_G[x-1][y-1]]
                        sort_B = [origin_B[x+1][y+1], origin_B[x+1][y], origin_B[x+1][y-1], origin_B[x][y+1], origin_B[x][y-1], origin_B[x-1][y+1], origin_B[x-1][y], origin_B[x-1][y-1]]

                sort_R.sort()
                sort_G.sort()
                sort_B.sort()
                median_filter_R[x][y] = sort_R[len(sort_R)/2]
                median_filter_G[x][y] = sort_G[len(sort_G)/2]
                median_filter_B[x][y] = sort_B[len(sort_B)/2]

        for x in xrange(self.rows):
            for y in xrange(self.cols):
                self.img[x, y] = (median_filter_R[x][y], median_filter_G[x][y], median_filter_B[x][y])
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def histogram_equalization(self):
        self.origin_img.append(copy.deepcopy(self.img))
        intensity = {}
        for index in range(256):
            intensity[index] = 0

        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x]/255.0
                I, H, S = self._RGB_to_IHS(red, green, blue)
                intensity[I*255//1] += 1

        # Calculate histogram
        for index in range(1, 256):
            intensity[index] += intensity[index-1]
        for index in range(256):
            intensity[index] = intensity[index]*1.0/(self.cols*self.rows)

        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x]/255.0
                I, H, S = self._RGB_to_IHS(red, green, blue)
                self.img[y, x] = self._IHS_to_RGB(intensity[I*255//1], np.nan_to_num(H), S)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def bright_image(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x]/255.0
                I, H, S = self._RGB_to_IHS(red, green, blue)
                self.img[y, x] = self._IHS_to_RGB(I*1.5 if I*1.5 < 1 else 1, np.nan_to_num(H), S)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def dark_image(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x]/255.0
                I, H, S = self._RGB_to_IHS(red, green, blue)
                self.img[y, x] = self._IHS_to_RGB(I*0.5, np.nan_to_num(H), S)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def binary_image(self):
        self.origin_img.append(copy.deepcopy(self.img))
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2YCR_CB)
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x]/255.0
                I, H, S = self._RGB_to_IHS(red, green, blue)
                if I < 240.0/255.0:
                # # red, green, blue = self.img[y, x]
                # Y, Cr, Cb = self.img[y, x]
                # # Y, Cb, Cr = self._RGB_to_YCbCr(red, green, blue)
                # if Y >= 60 and Y <= 255 and Cb >= 90 and Cb <= 135 and Cr >= 135 and Cr <= 170:
                    self.img[y, x] = (0, 0, 0)
                else:
                    self.img[y, x] = (255, 255, 255)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        # self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        # self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        # self.img = cv2.erode(self.img, kernel)
        # self.img = cv2.dilate(self.img, kernel)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def erode_dilate(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def dilate_erode(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def _RGB_to_IHS(self, R, G, B):
        # Return (I, H, S)
        I = (R+G+B) / 3.0
        S = 1.0 - min(R, G, B)/I if I>0 else 1.0
        H = np.arccos((R-G+R-B)/2.0/np.sqrt((R-G)*(R-G)+(R-B)*(G-B))) / np.pi * 180 if np.sqrt((R-G)*(R-G)+(R-B)*(G-B)) > 0.0 else np.arccos(0) / np.pi * 180
        if B>G :
            H = 360 - H
        return I, H, S

    def _IHS_to_RGB(self, intensity, hue, saturation):
        # Return (R, G, B)
        if hue>=0 and hue<=120:
            b = (1-saturation)/3.0
            r = (1+(saturation*np.cos(hue/180*np.pi)/np.cos((60-hue)/180*np.pi)))/3.0
            g = 1.0-(r+b)
        elif hue>120 and hue<=240:
            hue = hue-120
            r = (1-saturation)/3.0
            g = (1+(saturation*np.cos(hue/180*np.pi)/np.cos((60-hue)/180*np.pi)))/3.0
            b = 1.0-(r+g)
        elif hue>240 and hue<=360:
            hue = hue-240
            g = (1-saturation)/3.0
            b = (1+(saturation*np.cos(hue/180*np.pi)/np.cos((60-hue)/180*np.pi)))/3.0
            r = 1.0-(g+b)
        else:
            print hue
        return (3*intensity*r*255 if 3*intensity*r<1 else 255, 3*intensity*g*255//1 if 3*intensity*g<1 else 255, 3*intensity*b*255//1 if 3*intensity*b<1 else 255)

    def _RGB_to_YCbCr(self, R, G, B):
        delta = 128.0
        Y  = int( 0.299   * R + 0.587   * G + 0.114   * B);
        Cb = int(-0.16874 * R - 0.33126 * G + 0.50000 * B);
        Cr = int( 0.50000 * R - 0.41869 * G - 0.08131 * B);
        return Y, Cb, Cr

    def _YCbCr_to_RGB(self, Y, Cb, Cr):
        pass

    def RGB_to_Gray(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x]
                gray = red*0.299+green*0.587+blue*0.114
                self.img[y,x] = (gray, gray, gray)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def red_color_transform(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                r, g, b = self.img[y, x]
                self.img[y, x] = (r*1.2 if r*1.2 < 255 else 255, g, b)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def green_color_transform(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                r, g, b = self.img[y, x]
                self.img[y, x] = (r, g*1.2 if g*1.2 < 255 else 255, b)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def blue_color_transform(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                r, g, b = self.img[y, x]
                self.img[y, x] = (r, g, b*1.2 if b*1.2 < 255 else 255)
        self.show_img()
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def load_img(self):
        # file_name = 'data/' + self.image_filename.get()
        self.img = cv2.imread(self.fileName.name)
        self.origin_img = []
        self.rows, self.cols = self.img.shape[:2]

        # BGR trans to RGB, show img use RGB
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                blue, green, red = self.img[y, x]
                self.img[y, x] = (red, green, blue)

        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def recover_img(self):
        self.img = self.origin_img.pop() if len(self.origin_img) > 0 else self.img
        Label(self.root, text=u'已變動次數：'+str(len(self.origin_img))).grid(row=2, columnspan=6)

    def show_img(self):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        plt.xticks([]), plt.yticks([])
        self.ax2 = self.fig.add_subplot(122)
        plt.xticks([]), plt.yticks([])
        self.ax1.imshow(self.origin_img[-1])
        self.ax2.imshow(self.img)
        self.fig.show()

    def openFile(self):
        self.fileName = tkFileDialog.askopenfile(parent=self.root, initialdir='./', title='Select Image', filetypes=[('image files', '.png'), ('image files', '.jpg')])
        self.image_filename["text"] = u'Image 檔名：{}'.format(self.fileName.name)

    def init_window(self):
        self.root.title(u'Image Processing')
        self.image_filename = Label(self.root, text=u'Image 檔名：')
        self.image_filename.grid(row=0, column=0, columnspan=5, sticky=W)
        Button(self.root, text = 'Select Image', command=self.openFile).grid(row=0, column=5, columnspan=3)
        comfirm_button = Button(self.root, text=u'讀取 Image', command=self.load_img)
        comfirm_button.grid(row=1, columnspan=8)

        Button(self.root, text=u'裁切', command=self.corp_image).grid(row=2, column=0, columnspan=2)
        Button(self.root, text=u'二值化', command=self.binary_image).grid(row=2, column=2, columnspan=2)
        Button(self.root, text=u'轉成灰階', command=self.RGB_to_Gray).grid(row=2, column=4, columnspan=2)
        Button(self.root, text=u'侵蝕擴張法', command=self.erode_dilate).grid(row=2, column=6, columnspan=2)
        # image_nagetives_button = Button(self.root, text=u'轉成負片', command=self.image_nagetives)
        # image_nagetives_button.grid(row=3, column=2, columnspan=3)
        dark_idensity_transform_button = Button(self.root, text=u'亮度調暗', command=self.dark_image)
        dark_idensity_transform_button.grid(row=4, column=0, columnspan=3)
        bright_idensity_transform_button = Button(self.root, text=u'亮度調亮', command=self.bright_image)
        bright_idensity_transform_button.grid(row=4, column=1, columnspan=3)
        red_color_transform_button = Button(self.root, text=u'增強紅色(顏色處理)', command=self.red_color_transform)
        red_color_transform_button.grid(row=5, column=0, columnspan=2)
        green_color_transform_button = Button(self.root, text=u'增強綠色(顏色處理)', command=self.green_color_transform)
        green_color_transform_button.grid(row=5, column=1, columnspan=2)
        blue_color_transform_button = Button(self.root, text=u'增強藍色(顏色處理)', command=self.blue_color_transform)
        blue_color_transform_button.grid(row=5, column=2, columnspan=2)
        histogram_equalization_button = Button(self.root, text=u'分布圖均勻化(增加對比)', command=self.histogram_equalization)
        histogram_equalization_button.grid(row=6, column=0, columnspan=1)
        peak_and_valley_filter_button = Button(self.root, text=u'波峰波谷濾波器(去除雜訊)', command=self.peak_and_valley_filter)
        peak_and_valley_filter_button.grid(row=6, column=1, columnspan=1)
        median_filter_button = Button(self.root, text=u'中值濾波器(去除雜訊)', command=self.median_filter)
        median_filter_button.grid(row=6, column=2, columnspan=1)
        rotate_image_button = Button(self.root, text=u'向右旋轉 90 度', command=self.rotate_image)
        rotate_image_button.grid(row=6, column=3, columnspan=1)
        
        recover_button = Button(self.root, text=u'回到上一步', command=self.recover_img)
        recover_button.grid(row=7, columnspan=6)

        self.root.grid()
        self.root.mainloop()

if __name__ == "__main__":
    ImgP = ImageProcessing()