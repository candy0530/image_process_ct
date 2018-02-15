# -*- coding: utf-8 -*-
from tkinter import filedialog
from tkinter import *

import copy
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
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
        self.img = self.img[self.rows * 33 / 100:self.rows * 52 / 100, ]
        self.rows, self.cols = self.img.shape[:2]
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def corp_blood_vessel(self):

        # def corp_blood_vessel_fast(self):
        self.origin_img.append(copy.deepcopy(self.img))
        image, contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # length_contours = [len(x) for x in contours]
        # new_contours = [contours[length_contours.index(max(length_contours))]]
        # for x in range(0, self.rows, -1):
        #     array_contour = [y for y in new_contours[0] if y[0, 0] == x]
        #     array_y = [y[0, 1] for y in array_contour]
        #     for y in range(0, min(array_y)):
        #         self.img[y]

        new_contours = []
        for x in contours:
            if len(x) > 1000:
                # if abs(max(x[:,0,0])-min(x[:,0,0])-(max(x[:,0,1])-min(x[:,0,1])))>400:
                new_contours.append(x)
        # self.img = copy.deepcopy(self.origin_img[0])
        self.img = np.zeros_like(self.img)  # Create mask where white is what we want, black otherwise
        # cv2.drawContours(mask, new_contours, -1, 255, 2) # Draw filled contour in mask
        cv2.drawContours(self.img, new_contours, -1, 255, -1)  # Draw filled contour in mask
        self.save_img(self.img, '4-find_contours_over_1000_point.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        tmp_img = copy.deepcopy(self.img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for x in range(10):
            tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_ERODE, kernel)
            # cv2.imshow('image', tmp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        for x in range(10):
            tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_DILATE, kernel)
            # cv2.imshow('image', tmp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        tmp_img = cv2.bitwise_not(tmp_img)
        self.save_img(tmp_img, '5-after_erode_9_times_and_dilate_9_times.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        # cv2.imshow('image', tmp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.img = cv2.bitwise_and(self.img, tmp_img)
        self.save_img(self.img, '6-delete_not_vessels_part.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        # self.show_img()
        # self.img = cv2.bitwise_and(self.img, self.origin_img[0])

        self.dilate_erode()
        self.save_img(self.img, '6.5-recover_some_hole .png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')

        image, contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        length_contours = [len(x) for x in contours]
        new_contours = [contours[length_contours.index(max(length_contours))]]
        # min_y = min(np.array(new_contours[0])[:, :, 1])        
        # max_y = max(np.array(new_contours[0])[:, :, 1])

        # print(min_y, max_y)
        self.img = np.zeros_like(self.img)  # Create mask where white is what we want, black otherwise
        # cv2.drawContours(mask, new_contours, -1, 255, 2) # Draw filled contour in mask
        cv2.drawContours(self.img, new_contours, -1, 255, -1)  # Draw filled contour in mask
        self.save_img(self.img, '7-refind_contours.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        # self.show_img()
        self.img = cv2.bitwise_and(self.img, self.origin_img[0])
        self.save_img(self.img, '8-refill_origin_pic.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def _get_contour_min_y(self, contours):
        contour = np.array(contours[0])
        return min(contour[:, 1])

    def _get_contour_max_y(self, contours):
        contour = np.array(contours[0])
        return max(contours[0][:, 1])

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
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def image_nagetives(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x] / 255.0
                gray = red * 0.299 + green * 0.587 + blue * 0.114
                I, H, S = self._RGB_to_IHS(gray, gray, gray)
                self.img[y, x] = self._IHS_to_RGB(1 - I, np.nan_to_num(H), S)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

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
                        biggest_R = max(origin_R[x + 1][y], origin_R[x + 1][y + 1], origin_R[x][y + 1])
                        biggest_G = max(origin_G[x + 1][y], origin_G[x + 1][y + 1], origin_G[x][y + 1])
                        biggest_B = max(origin_B[x + 1][y], origin_B[x + 1][y + 1], origin_B[x][y + 1])
                    elif y == self.cols - 1:
                        biggest_R = max(origin_R[x + 1][y - 1], origin_R[x + 1][y], origin_R[x][y - 1])
                        biggest_G = max(origin_G[x + 1][y - 1], origin_G[x + 1][y], origin_G[x][y - 1])
                        biggest_B = max(origin_B[x + 1][y - 1], origin_B[x + 1][y], origin_B[x][y - 1])
                    else:
                        biggest_R = max(origin_R[x + 1][y + 1], origin_R[x + 1][y], origin_R[x + 1][y - 1],
                                        origin_R[x][y + 1], origin_R[x][y - 1])
                        biggest_G = max(origin_G[x + 1][y + 1], origin_G[x + 1][y], origin_G[x + 1][y - 1],
                                        origin_G[x][y + 1], origin_G[x][y - 1])
                        biggest_B = max(origin_B[x + 1][y + 1], origin_B[x + 1][y], origin_B[x + 1][y - 1],
                                        origin_B[x][y + 1], origin_B[x][y - 1])
                elif x == self.rows - 1:
                    if y == 0:
                        biggest_R = max(origin_R[x - 1][y], origin_R[x - 1][y + 1], origin_R[x][y + 1])
                        biggest_G = max(origin_G[x - 1][y], origin_G[x - 1][y + 1], origin_G[x][y + 1])
                        biggest_B = max(origin_B[x - 1][y], origin_B[x - 1][y + 1], origin_B[x][y + 1])
                    elif y == self.cols - 1:
                        biggest_R = max(origin_R[x - 1][y - 1], origin_R[x - 1][y], origin_R[x][y - 1])
                        biggest_G = max(origin_G[x - 1][y - 1], origin_G[x - 1][y], origin_G[x][y - 1])
                        biggest_B = max(origin_B[x - 1][y - 1], origin_B[x - 1][y], origin_B[x][y - 1])
                    else:
                        biggest_R = max(origin_R[x - 1][y + 1], origin_R[x - 1][y], origin_R[x - 1][y - 1],
                                        origin_R[x][y + 1], origin_R[x][y - 1])
                        biggest_G = max(origin_G[x - 1][y + 1], origin_G[x - 1][y], origin_G[x - 1][y - 1],
                                        origin_G[x][y + 1], origin_G[x][y - 1])
                        biggest_B = max(origin_B[x - 1][y + 1], origin_B[x - 1][y], origin_B[x - 1][y - 1],
                                        origin_B[x][y + 1], origin_B[x][y - 1])
                else:
                    if y == 0:
                        biggest_R = max(origin_R[x + 1][y], origin_R[x + 1][y + 1], origin_R[x][y + 1],
                                        origin_R[x - 1][y], origin_R[x - 1][y + 1])
                        biggest_G = max(origin_G[x + 1][y], origin_G[x + 1][y + 1], origin_G[x][y + 1],
                                        origin_G[x - 1][y], origin_G[x - 1][y + 1])
                        biggest_B = max(origin_B[x + 1][y], origin_B[x + 1][y + 1], origin_B[x][y + 1],
                                        origin_B[x - 1][y], origin_B[x - 1][y + 1])
                    elif y == self.cols - 1:
                        biggest_R = max(origin_R[x + 1][y - 1], origin_R[x + 1][y], origin_R[x][y - 1],
                                        origin_R[x - 1][y - 1], origin_R[x - 1][y])
                        biggest_G = max(origin_G[x + 1][y - 1], origin_G[x + 1][y], origin_G[x][y - 1],
                                        origin_G[x - 1][y - 1], origin_G[x - 1][y])
                        biggest_B = max(origin_B[x + 1][y - 1], origin_B[x + 1][y], origin_B[x][y - 1],
                                        origin_B[x - 1][y - 1], origin_B[x - 1][y])
                    else:
                        biggest_R = max(origin_R[x + 1][y + 1], origin_R[x + 1][y], origin_R[x + 1][y - 1],
                                        origin_R[x][y + 1], origin_R[x][y - 1], origin_R[x - 1][y + 1],
                                        origin_R[x - 1][y], origin_R[x - 1][y - 1])
                        biggest_G = max(origin_G[x + 1][y + 1], origin_G[x + 1][y], origin_G[x + 1][y - 1],
                                        origin_G[x][y + 1], origin_G[x][y - 1], origin_G[x - 1][y + 1],
                                        origin_G[x - 1][y], origin_G[x - 1][y - 1])
                        biggest_B = max(origin_B[x + 1][y + 1], origin_B[x + 1][y], origin_B[x + 1][y - 1],
                                        origin_B[x][y + 1], origin_B[x][y - 1], origin_B[x - 1][y + 1],
                                        origin_B[x - 1][y], origin_B[x - 1][y - 1])

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
                        smallest_R = min(peak_filter_R[x + 1][y], peak_filter_R[x + 1][y + 1], peak_filter_R[x][y + 1])
                        smallest_G = min(peak_filter_G[x + 1][y], peak_filter_G[x + 1][y + 1], peak_filter_G[x][y + 1])
                        smallest_B = min(peak_filter_B[x + 1][y], peak_filter_B[x + 1][y + 1], peak_filter_B[x][y + 1])
                    elif y == self.cols - 1:
                        smallest_R = min(peak_filter_R[x + 1][y - 1], peak_filter_R[x + 1][y], peak_filter_R[x][y - 1])
                        smallest_G = min(peak_filter_G[x + 1][y - 1], peak_filter_G[x + 1][y], peak_filter_G[x][y - 1])
                        smallest_B = min(peak_filter_B[x + 1][y - 1], peak_filter_B[x + 1][y], peak_filter_B[x][y - 1])
                    else:
                        smallest_R = min(peak_filter_R[x + 1][y + 1], peak_filter_R[x + 1][y],
                                         peak_filter_R[x + 1][y - 1], peak_filter_R[x][y + 1], peak_filter_R[x][y - 1])
                        smallest_G = min(peak_filter_G[x + 1][y + 1], peak_filter_G[x + 1][y],
                                         peak_filter_G[x + 1][y - 1], peak_filter_G[x][y + 1], peak_filter_G[x][y - 1])
                        smallest_B = min(peak_filter_B[x + 1][y + 1], peak_filter_B[x + 1][y],
                                         peak_filter_B[x + 1][y - 1], peak_filter_B[x][y + 1], peak_filter_B[x][y - 1])
                elif x == self.rows - 1:
                    if y == 0:
                        smallest_R = min(peak_filter_R[x - 1][y], peak_filter_R[x - 1][y + 1], peak_filter_R[x][y + 1])
                        smallest_G = min(peak_filter_G[x - 1][y], peak_filter_G[x - 1][y + 1], peak_filter_G[x][y + 1])
                        smallest_B = min(peak_filter_B[x - 1][y], peak_filter_B[x - 1][y + 1], peak_filter_B[x][y + 1])
                    elif y == self.cols - 1:
                        smallest_R = min(peak_filter_R[x - 1][y - 1], peak_filter_R[x - 1][y], peak_filter_R[x][y - 1])
                        smallest_G = min(peak_filter_G[x - 1][y - 1], peak_filter_G[x - 1][y], peak_filter_G[x][y - 1])
                        smallest_B = min(peak_filter_B[x - 1][y - 1], peak_filter_B[x - 1][y], peak_filter_B[x][y - 1])
                    else:
                        smallest_R = min(peak_filter_R[x - 1][y + 1], peak_filter_R[x - 1][y],
                                         peak_filter_R[x - 1][y - 1], peak_filter_R[x][y + 1], peak_filter_R[x][y - 1])
                        smallest_G = min(peak_filter_G[x - 1][y + 1], peak_filter_G[x - 1][y],
                                         peak_filter_G[x - 1][y - 1], peak_filter_G[x][y + 1], peak_filter_G[x][y - 1])
                        smallest_B = min(peak_filter_B[x - 1][y + 1], peak_filter_B[x - 1][y],
                                         peak_filter_B[x - 1][y - 1], peak_filter_B[x][y + 1], peak_filter_B[x][y - 1])
                else:
                    if y == 0:
                        smallest_R = min(peak_filter_R[x + 1][y], peak_filter_R[x + 1][y + 1], peak_filter_R[x][y + 1],
                                         peak_filter_R[x - 1][y], peak_filter_R[x - 1][y + 1])
                        smallest_G = min(peak_filter_G[x + 1][y], peak_filter_G[x + 1][y + 1], peak_filter_G[x][y + 1],
                                         peak_filter_G[x - 1][y], peak_filter_G[x - 1][y + 1])
                        smallest_B = min(peak_filter_B[x + 1][y], peak_filter_B[x + 1][y + 1], peak_filter_B[x][y + 1],
                                         peak_filter_B[x - 1][y], peak_filter_B[x - 1][y + 1])
                    elif y == self.cols - 1:
                        smallest_R = min(peak_filter_R[x + 1][y - 1], peak_filter_R[x + 1][y], peak_filter_R[x][y - 1],
                                         peak_filter_R[x - 1][y - 1], peak_filter_R[x - 1][y])
                        smallest_G = min(peak_filter_G[x + 1][y - 1], peak_filter_G[x + 1][y], peak_filter_G[x][y - 1],
                                         peak_filter_G[x - 1][y - 1], peak_filter_G[x - 1][y])
                        smallest_B = min(peak_filter_B[x + 1][y - 1], peak_filter_B[x + 1][y], peak_filter_B[x][y - 1],
                                         peak_filter_B[x - 1][y - 1], peak_filter_B[x - 1][y])
                    else:
                        smallest_R = min(peak_filter_R[x + 1][y + 1], peak_filter_R[x + 1][y],
                                         peak_filter_R[x + 1][y - 1], peak_filter_R[x][y + 1], peak_filter_R[x][y - 1],
                                         peak_filter_R[x - 1][y + 1], peak_filter_R[x - 1][y],
                                         peak_filter_R[x - 1][y - 1])
                        smallest_G = min(peak_filter_G[x + 1][y + 1], peak_filter_G[x + 1][y],
                                         peak_filter_G[x + 1][y - 1], peak_filter_G[x][y + 1], peak_filter_G[x][y - 1],
                                         peak_filter_G[x - 1][y + 1], peak_filter_G[x - 1][y],
                                         peak_filter_G[x - 1][y - 1])
                        smallest_B = min(peak_filter_B[x + 1][y + 1], peak_filter_B[x + 1][y],
                                         peak_filter_B[x + 1][y - 1], peak_filter_B[x][y + 1], peak_filter_B[x][y - 1],
                                         peak_filter_B[x - 1][y + 1], peak_filter_B[x - 1][y],
                                         peak_filter_B[x - 1][y - 1])

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
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

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
                        sort_R = [origin_R[x + 1][y], origin_R[x + 1][y + 1], origin_R[x][y + 1]]
                        sort_G = [origin_G[x + 1][y], origin_G[x + 1][y + 1], origin_G[x][y + 1]]
                        sort_B = [origin_B[x + 1][y], origin_B[x + 1][y + 1], origin_B[x][y + 1]]
                    elif y == self.cols - 1:
                        sort_R = [origin_R[x + 1][y - 1], origin_R[x + 1][y], origin_R[x][y - 1]]
                        sort_G = [origin_G[x + 1][y - 1], origin_G[x + 1][y], origin_G[x][y - 1]]
                        sort_B = [origin_B[x + 1][y - 1], origin_B[x + 1][y], origin_B[x][y - 1]]
                    else:
                        sort_R = [origin_R[x + 1][y + 1], origin_R[x + 1][y], origin_R[x + 1][y - 1],
                                  origin_R[x][y + 1], origin_R[x][y - 1]]
                        sort_G = [origin_G[x + 1][y + 1], origin_G[x + 1][y], origin_G[x + 1][y - 1],
                                  origin_G[x][y + 1], origin_G[x][y - 1]]
                        sort_B = [origin_B[x + 1][y + 1], origin_B[x + 1][y], origin_B[x + 1][y - 1],
                                  origin_B[x][y + 1], origin_B[x][y - 1]]
                elif x == self.rows - 1:
                    if y == 0:
                        sort_R = [origin_R[x - 1][y], origin_R[x - 1][y + 1], origin_R[x][y + 1]]
                        sort_G = [origin_G[x - 1][y], origin_G[x - 1][y + 1], origin_G[x][y + 1]]
                        sort_B = [origin_B[x - 1][y], origin_B[x - 1][y + 1], origin_B[x][y + 1]]
                    elif y == self.cols - 1:
                        sort_R = [origin_R[x - 1][y - 1], origin_R[x - 1][y], origin_R[x][y - 1]]
                        sort_G = [origin_G[x - 1][y - 1], origin_G[x - 1][y], origin_G[x][y - 1]]
                        sort_B = [origin_B[x - 1][y - 1], origin_B[x - 1][y], origin_B[x][y - 1]]
                    else:
                        sort_R = [origin_R[x - 1][y + 1], origin_R[x - 1][y], origin_R[x - 1][y - 1],
                                  origin_R[x][y + 1], origin_R[x][y - 1]]
                        sort_G = [origin_G[x - 1][y + 1], origin_G[x - 1][y], origin_G[x - 1][y - 1],
                                  origin_G[x][y + 1], origin_G[x][y - 1]]
                        sort_B = [origin_B[x - 1][y + 1], origin_B[x - 1][y], origin_B[x - 1][y - 1],
                                  origin_B[x][y + 1], origin_B[x][y - 1]]
                else:
                    if y == 0:
                        sort_R = [origin_R[x + 1][y], origin_R[x + 1][y + 1], origin_R[x][y + 1], origin_R[x - 1][y],
                                  origin_R[x - 1][y + 1]]
                        sort_G = [origin_G[x + 1][y], origin_G[x + 1][y + 1], origin_G[x][y + 1], origin_G[x - 1][y],
                                  origin_G[x - 1][y + 1]]
                        sort_B = [origin_B[x + 1][y], origin_B[x + 1][y + 1], origin_B[x][y + 1], origin_B[x - 1][y],
                                  origin_B[x - 1][y + 1]]
                    elif y == self.cols - 1:
                        sort_R = [origin_R[x + 1][y - 1], origin_R[x + 1][y], origin_R[x][y - 1],
                                  origin_R[x - 1][y - 1], origin_R[x - 1][y]]
                        sort_G = [origin_G[x + 1][y - 1], origin_G[x + 1][y], origin_G[x][y - 1],
                                  origin_G[x - 1][y - 1], origin_G[x - 1][y]]
                        sort_B = [origin_B[x + 1][y - 1], origin_B[x + 1][y], origin_B[x][y - 1],
                                  origin_B[x - 1][y - 1], origin_B[x - 1][y]]
                    else:
                        sort_R = [origin_R[x + 1][y + 1], origin_R[x + 1][y], origin_R[x + 1][y - 1],
                                  origin_R[x][y + 1], origin_R[x][y - 1], origin_R[x - 1][y + 1], origin_R[x - 1][y],
                                  origin_R[x - 1][y - 1]]
                        sort_G = [origin_G[x + 1][y + 1], origin_G[x + 1][y], origin_G[x + 1][y - 1],
                                  origin_G[x][y + 1], origin_G[x][y - 1], origin_G[x - 1][y + 1], origin_G[x - 1][y],
                                  origin_G[x - 1][y - 1]]
                        sort_B = [origin_B[x + 1][y + 1], origin_B[x + 1][y], origin_B[x + 1][y - 1],
                                  origin_B[x][y + 1], origin_B[x][y - 1], origin_B[x - 1][y + 1], origin_B[x - 1][y],
                                  origin_B[x - 1][y - 1]]

                sort_R.sort()
                sort_G.sort()
                sort_B.sort()
                median_filter_R[x][y] = sort_R[len(sort_R) / 2]
                median_filter_G[x][y] = sort_G[len(sort_G) / 2]
                median_filter_B[x][y] = sort_B[len(sort_B) / 2]

        for x in xrange(self.rows):
            for y in xrange(self.cols):
                self.img[x, y] = (median_filter_R[x][y], median_filter_G[x][y], median_filter_B[x][y])
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def histogram_equalization(self):
        self.origin_img.append(copy.deepcopy(self.img))
        self.img = cv2.equalizeHist(self.img)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def bright_image(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x] / 255.0
                I, H, S = self._RGB_to_IHS(red, green, blue)
                self.img[y, x] = self._IHS_to_RGB(I * 1.5 if I * 1.5 < 1 else 1, np.nan_to_num(H), S)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def dark_image(self):
        self.origin_img.append(copy.deepcopy(self.img))
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                red, green, blue = self.img[y, x] / 255.0
                I, H, S = self._RGB_to_IHS(red, green, blue)
                self.img[y, x] = self._IHS_to_RGB(I * 0.5, np.nan_to_num(H), S)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def binary_image(self):
        if self.img.dtype != 'uint8':
            self.BGR_to_Gray()
        self.origin_img.append(copy.deepcopy(self.img))
        # self.img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        ret, self.img = cv2.threshold(self.img, 100, 255, cv2.THRESH_BINARY)
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # for x in xrange(self.cols):
        #     for y in xrange(self.rows):
        #         red, green, blue = self.img[y, x]/255.0
        #         I, H, S = self._RGB_to_IHS(red, green, blue)
        #         if I < 80.0/255.0:
        #             self.img[y, x] = (0, 0, 0)
        #         else:
        #             self.img[y, x] = (255, 255, 255)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def corp_dialog(self):
        self.origin_img.append(copy.deepcopy(self.img))
        origin = copy.deepcopy(self.img)
        # row1, row2 = self._cal_row_bright()
        # col1, col2 = self._cal_col_bright()
        # rows = self._cal_row_bright()
        # cols = self._cal_col_bright()
        rows = self._cal_row_dark()
        cols = self._cal_col_dark()
        # print(rows, col1, col2)
        alter_img = []
        counter_table = 0
        for x in rows:
            for y in cols:
                save_img = self.img[x[0]:(x[1] + 1), y[0]:(y[1] + 1)]
                self.save_img(img=save_img, file_name=('test_' + str(counter_table) + '.png'))
                counter_table += 1
        # for x in range(row1, row2):
        #     alter_img.append([])
        #     for y in range(col1, col2):
        #         alter_img[-1].append(self.img[x, y])
        self.img = np.array(alter_img, dtype=np.uint8)
        self.rows, self.cols = self.img.shape[:2]
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def _cal_row_bright(self):
        count_row = []
        for x in xrange(self.rows):
            count_row.append(0)
            for y in xrange(self.cols):
                if self.img[x, y][0] == 255:
                    count_row[-1] += 1
        # print(count_row)
        count_row = np.array(count_row)
        average = count_row.sum() / len(count_row)
        sigma = count_row.std()
        row_index = []
        count_index = 0
        for x in range(len(count_row)):
            if count_row[x] > average + 10 * sigma:
                if count_index == 2:
                    if (x - row_index[-1]) < 5:
                        row_index[-1] = x
                    else:
                        row_index.append(x)
                        count_index = 1
                    continue
                if count_index == 1 and (x - row_index[-1]) >= 5:
                    row_index.append(row_index[-1])
                    count_index = 0
                row_index.append(x)
                count_index += 1

        print(row_index)
        # row_index.pop()
        row_index.remove(row_index[0])
        row_index = np.array(row_index).reshape([len(row_index) / 2, 2])

        print(row_index)
        return row_index
        # count_row = []
        # for x in xrange(self.rows):
        #     count_row.append(0)
        #     for y in xrange(self.cols):
        #         if self.img[x, y][0] == 255:
        #             count_row[-1] += 1
        # import heapq
        # max_two_value = heapq.nlargest(2, count_row)
        # if max_two_value[0] == max_two_value[1]:
        #     max_1 = count_row.index(max_two_value[0], 0)
        #     max_2 = count_row.index(max_two_value[0], max_1+1)
        #     return max_1, max_2
        # return count_row.index(max_two_value[0]), count_row.index(max_two_value[1])

    def _cal_col_bright(self):
        count_col = []
        for x in xrange(self.cols):
            count_col.append(0)
            for y in xrange(self.rows):
                if self.img[y, x][0] == 255:
                    count_col[-1] += 1
        # print(count_col)
        count_col = np.array(count_col)
        average = count_col.sum() / len(count_col)
        sigma = count_col.std()
        col_index = []
        count_index = 0
        for x in range(len(count_col)):
            if count_col[x] > average + 10 * sigma:
                if count_index == 2:
                    if (x - col_index[-1]) < 5:
                        col_index[-1] = x
                    else:
                        col_index.append(x)
                        count_index = 1
                    continue
                if count_index == 1 and (x - col_index[-1]) >= 5:
                    col_index.append(col_index[-1])
                    count_index = 0
                col_index.append(x)
                count_index += 1

        print(col_index)
        # col_index.pop()
        col_index.remove(col_index[0])
        col_index = np.array(col_index).reshape([len(col_index) / 2, 2])

        print(col_index)
        return col_index
        # count_col = []
        # for x in xrange(self.cols):
        #     count_col.append(0)
        #     for y in xrange(self.rows):
        #         if self.img[y, x][0] == 255:
        #             count_col[-1] += 1
        # import heapq
        # max_two_value = heapq.nlargest(2, count_col)
        # if max_two_value[0] == max_two_value[1]:
        #     max_1 = count_col.index(max_two_value[0], 0)
        #     max_2 = count_col.index(max_two_value[0], max_1+1)
        #     return max_1, max_2
        # return count_col.index(max_two_value[0]), count_col.index(max_two_value[1])

    def _cal_row_dark(self):
        count_row = []
        for x in xrange(self.rows):
            count_row.append(0)
            for y in xrange(self.cols):
                if self.img[x, y][0] == 0:
                    count_row[-1] += 1
        # print(count_row)
        count_row = np.array(count_row)
        average = count_row.sum() / len(count_row)
        sigma = count_row.std()
        row_index = []
        count_index = 0
        for x in range(len(count_row)):
            if count_row[x] > average + 2 * sigma:
                if count_index == 2:
                    if (x - row_index[-1]) < 5:
                        row_index[-1] = x
                    else:
                        row_index.append(x)
                        count_index = 1
                    continue
                if count_index == 1 and (x - row_index[-1]) >= 5:
                    row_index.append(row_index[-1])
                    count_index = 0
                row_index.append(x)
                count_index += 1
        row_index.pop()
        row_index.remove(row_index[0])
        row_index = np.array(row_index).reshape([len(row_index) / 2, 2])

        print(row_index)
        return row_index
        # import heapq
        # max_two_value = heapq.nlargest(2, count_row)
        # if max_two_value[0] >= max_two_value[1]-5:
        #     max_1 = count_row.index(max_two_value[0], 0)
        #     max_2 = count_row.index(max_two_value[1], max_1+5)
        #     return max_1, max_2
        # return count_row.index(max_two_value[0]), count_row.index(max_two_value[1])

    def _cal_col_dark(self):
        count_col = []
        for x in xrange(self.cols):
            count_col.append(0)
            for y in xrange(self.rows):
                if self.img[y, x][0] == 0:
                    count_col[-1] += 1
        # print(count_col)
        count_col = np.array(count_col)
        average = count_col.sum() / len(count_col)
        sigma = count_col.std()
        col_index = []
        count_index = 0
        for x in range(len(count_col)):
            if count_col[x] > average + 3.5 * sigma:
                if count_index == 2:
                    if (x - col_index[-1]) < 5:
                        col_index[-1] = x
                    else:
                        col_index.append(x)
                        count_index = 1
                    continue
                if count_index == 1 and (x - col_index[-1]) >= 5:
                    col_index.append(col_index[-1])
                    count_index = 0
                col_index.append(x)
                count_index += 1
        col_index.pop()
        col_index.remove(col_index[0])
        col_index = np.array(col_index).reshape([len(col_index) / 2, 2])

        print(col_index)
        return col_index
        # import heapq
        # max_two_value = heapq.nlargest(2, count_col)
        # if max_two_value[0] >= max_two_value[1]-5:
        #     max_1 = count_col.index(max_two_value[0], 0)
        #     max_2 = count_col.index(max_two_value[1], max_1+5)
        #     return max_1, max_2
        # return count_col.index(max_two_value[0]), count_col.index(max_two_value[1])

    def erode_dilate(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def erode(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_ERODE, kernel)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def dilate(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_DILATE, kernel)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def dilate_erode(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def _RGB_to_IHS(self, R, G, B):
        # Return (I, H, S)
        I = (R + G + B) / 3.0
        S = 1.0 - min(R, G, B) / I if I > 0 else 1.0
        H = np.arccos((R - G + R - B) / 2.0 / np.sqrt((R - G) * (R - G) + (R - B) * (G - B))) / np.pi * 180 if np.sqrt(
            (R - G) * (R - G) + (R - B) * (G - B)) > 0.0 else np.arccos(0) / np.pi * 180
        if B > G:
            H = 360 - H
        return I, H, S

    def _IHS_to_RGB(self, intensity, hue, saturation):
        # Return (R, G, B)
        if hue >= 0 and hue <= 120:
            b = (1 - saturation) / 3.0
            r = (1 + (saturation * np.cos(hue / 180 * np.pi) / np.cos((60 - hue) / 180 * np.pi))) / 3.0
            g = 1.0 - (r + b)
        elif hue > 120 and hue <= 240:
            hue = hue - 120
            r = (1 - saturation) / 3.0
            g = (1 + (saturation * np.cos(hue / 180 * np.pi) / np.cos((60 - hue) / 180 * np.pi))) / 3.0
            b = 1.0 - (r + g)
        elif hue > 240 and hue <= 360:
            hue = hue - 240
            g = (1 - saturation) / 3.0
            b = (1 + (saturation * np.cos(hue / 180 * np.pi) / np.cos((60 - hue) / 180 * np.pi))) / 3.0
            r = 1.0 - (g + b)
        else:
            print(hue)
        return (3 * intensity * r * 255 if 3 * intensity * r < 1 else 255,
                3 * intensity * g * 255 // 1 if 3 * intensity * g < 1 else 255,
                3 * intensity * b * 255 // 1 if 3 * intensity * b < 1 else 255)

    def BGR_to_Gray(self):
        self.origin_img.append(copy.deepcopy(self.img))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.show_img()
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def corp_blood_vessel_one_step(self):
        self.load_img()
        self.save_img(self.img, '1-original.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        self.binary_image()
        self.save_img(self.img, '2-after_binary.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        self.erode_dilate()
        self.save_img(self.img, '3-after_erode_and_dilate_once.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        self.corp_blood_vessel()

    def corp_table_one_step(self):
        self.load_img()
        self.corp_image()
        self.binary_image()
        self.corp_dialog()

    def save_img(self, img, file_name='test1.png', file_root='/Users/candy/Workspace/Paper_CT/'):
        cv2.imwrite(file_root + file_name, img)

    def load_img(self):
        print(self.fileName)
        if self.fileName == '' or self.fileName == None:
            self.img = cv2.imread('/Users/candy/Workspace/Paper_CT/3.jpg', 0)
        else:
            self.img = cv2.imread(self.fileName, 0)
        self.origin_img = []
        self.rows, self.cols = self.img.shape[:2]

        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def recover_img(self):
        self.img = self.origin_img.pop() if len(self.origin_img) > 0 else self.img
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def show_img(self):
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_img_old(self):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        plt.xticks([]), plt.yticks([])
        self.ax2 = self.fig.add_subplot(122)
        plt.xticks([]), plt.yticks([])
        self.ax1.imshow(self.origin_img[-1])
        self.ax2.imshow(self.img)
        self.fig.show()

    def openFile(self):
        self.fileName = filedialog.askopenfile(parent=self.root, initialdir='./', title='Select Image',
                                                 filetypes=[('image files', '.png'), ('image files', '.jpg')])
        self.fileName = self.fileName.name
        self.image_filename["text"] = u'Image 檔名：{}'.format(self.fileName)

    def init_window(self):
        self.root.title(u'Image Processing')
        self.image_filename = Label(self.root, text=u'Image 檔名：')
        self.image_filename.grid(row=0, column=0, columnspan=5, sticky=W)
        Button(self.root, text='Select Image', command=self.openFile).grid(row=0, column=5, columnspan=3)

        Button(self.root, text=u'讀取 Image', command=self.load_img).grid(row=1, columnspan=8)

        Button(self.root, text=u'裁切', command=self.corp_image).grid(row=3, column=0, columnspan=2)
        Button(self.root, text=u'二值化', command=self.binary_image).grid(row=3, column=2, columnspan=2)
        Button(self.root, text=u'轉成灰階', command=self.BGR_to_Gray).grid(row=3, column=4, columnspan=2)
        # Button(self.root, text=u'侵蝕 法', command=self.erode).grid(row=3, column=6, columnspan=2)
        # Button(self.root, text=u'侵蝕擴張法', command=self.erode_dilate).grid(row=3, column=6, columnspan=2)
        # Button(self.root, text=u'轉成負片', command=self.image_nagetives).grid(row=3, column=2, columnspan=3)

        Button(self.root, text=u'裁切對話框', command=self.corp_dialog).grid(row=4, column=0, columnspan=3)
        Button(self.root, text=u'亮度調暗', command=self.dark_image).grid(row=4, column=3, columnspan=3)
        Button(self.root, text=u'亮度調亮', command=self.bright_image).grid(row=4, column=6, columnspan=2)

        Button(self.root, text=u'侵蝕 法', command=self.erode).grid(row=5, column=0, columnspan=2)
        Button(self.root, text=u'擴張 法', command=self.dilate).grid(row=5, column=2, columnspan=2)
        Button(self.root, text=u'侵蝕擴張 法', command=self.erode_dilate).grid(row=5, column=4, columnspan=2)

        # Button(self.root, text=u'增強紅色(顏色處理)', command=self.red_color_transform).grid(row=5, column=0, columnspan=3)
        # Button(self.root, text=u'增強綠色(顏色處理)', command=self.green_color_transform).grid(row=5, column=3, columnspan=3)
        # Button(self.root, text=u'增強藍色(顏色處理)', command=self.blue_color_transform).grid(row=5, column=6, columnspan=2)

        Button(self.root, text=u'分布圖均勻化(增加對比)', command=self.histogram_equalization).grid(row=6, column=0, columnspan=2)
        Button(self.root, text=u'波峰波谷濾波器(去除雜訊)', command=self.peak_and_valley_filter).grid(row=6, column=2,
                                                                                           columnspan=2)
        Button(self.root, text=u'中值濾波器(去除雜訊)', command=self.median_filter).grid(row=6, column=4, columnspan=2)
        Button(self.root, text=u'向右旋轉 90 度', command=self.rotate_image).grid(row=6, column=6, columnspan=2)

        Button(self.root, text=u'框出血管', command=self.corp_blood_vessel).grid(row=7, column=0, columnspan=2)
        Button(self.root, text=u'一鍵擷取血管', command=self.corp_blood_vessel_one_step).grid(row=7, column=2, columnspan=2)
        Button(self.root, text=u'一鍵擷取表格', command=self.corp_table_one_step).grid(row=7, column=4, columnspan=2)
        Button(self.root, text=u'儲存結果', command=lambda: self.save_img(self.img)).grid(row=7, column=6, columnspan=2)

        Button(self.root, text=u'回到上一步', command=self.recover_img).grid(row=8, columnspan=8)

        self.root.grid()
        self.root.mainloop()


if __name__ == "__main__":
    ImgP = ImageProcessing()
