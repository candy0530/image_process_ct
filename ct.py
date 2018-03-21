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
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def corp_blood_vessel(self):

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
        # cv2.drawContours(self.img, contours, -1, 255, -1) # Draw filled contour in mask
        cv2.drawContours(self.img, new_contours, -1, 255, -1)  # Draw filled contour in mask
        self.save_img(self.img, '4-find_contours_over_1000_point.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        tmp_img = copy.deepcopy(self.img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for x in range(10):
            tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_ERODE, kernel)
        for x in range(10):
            tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_DILATE, kernel)
        tmp_img = cv2.bitwise_not(tmp_img)
        self.save_img(tmp_img, '5-after_erode_9_times_and_dilate_9_times.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')

        self.img = cv2.bitwise_and(self.img, tmp_img)
        self.save_img(self.img, '6-delete_not_vessels_part.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')


        self.dilate_erode()
        self.save_img(self.img, '6.5-recover_some_hole .png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')

        image, contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = []
        max_width_height = 0
        for index in range(len(contours)):
            if len(contours[index]) < 100:
                continue
            height = self._get_contour_height(contours, index)
            width = self._get_contour_width(contours, index)
            if height > max_width_height:
                new_contours = [contours[index]]
                max_width_height = height

        # length_contours = [len(x) for x in contours]
        # new_contours = [contours[length_contours.index(max(length_contours))]]
        # min_y = min(np.array(new_contours[0])[:, :, 1])        
        # max_y = max(np.array(new_contours[0])[:, :, 1])

        # print(min_y, max_y)
        self.img = np.zeros_like(self.img)  # Create mask where white is what we want, black otherwise
        # cv2.drawContours(mask, new_contours, -1, 255, 2) # Draw filled contour in mask
        cv2.drawContours(self.img, new_contours, -1, 255, -1)  # Draw filled contour in mask
        self.save_img(self.img, '7-refind_contours.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        self.img = cv2.bitwise_and(self.img, self.origin_img[0])
        self.save_img(self.img, '8-refill_origin_pic.png',
                      '/Users/candy/Workspace/Paper_CT/output/' + self.fileName.split('/')[-1].split('.')[0] + '/')
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def _get_contour_height(self, contours, index=0):
        contour = np.array(contours[index])
        print(contour)
        return max(contour[:, :, 1])-min(contour[:, :, 1])

    def _get_contour_width(self, contours, index=0):
        contour = np.array(contours[index])
        return max(contour[:, :, 0])-min(contour[:, :, 0])

    def histogram_equalization(self):
        self.origin_img.append(copy.deepcopy(self.img))
        self.img = cv2.equalizeHist(self.img)
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def binary_image(self):
        if self.img.dtype != 'uint8':
            self.BGR_to_Gray()
        self.origin_img.append(copy.deepcopy(self.img))
        # self.img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        ret, self.img = cv2.threshold(self.img, 95, 255, cv2.THRESH_BINARY)
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def erode_dilate(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def erode(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_ERODE, kernel)
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def dilate(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_DILATE, kernel)
        Label(self.root, text=u'已變動次數：' + str(len(self.origin_img))).grid(row=2, columnspan=8)

    def dilate_erode(self):
        self.origin_img.append(copy.deepcopy(self.img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
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
        self.refind_blood_vessel()


    def refind_blood_vessel(self):
        output_file_path = '/Users/candy/Workspace/Paper_CT/output'
        file_index = self.fileName.split('/')[-1].split('.')[0]
        img = cv2.imread('{}/{}/4-find_contours_over_1000_point.png'.format(output_file_path, file_index), 0)

        rows, cols = img.shape[:2]

        last_width = 0
        vessels_pos = []
        vessels_width = 60

        img_find_row = cv2.imread('{}/{}/7-refind_contours.png'.format(output_file_path, file_index), 0)
        row_index = rows
        break_flag = 0
        stop_flag = 0
        stop_row = 0
        for row in range(rows)[::-1]:
            for col in range(cols):
                if img_find_row[row, col] != 0:
                    row_index = row
                    break_flag = 1
                    break
                img[row, col] = 0
            if break_flag == 1:
                break
            vessels_pos.append([0, 0])

        verticle_flag = 0
        for row in range(row_index + 1)[::-1]:
            start_x1 = 0
            start_x2 = 0
            this_width = 0
            break_flag = 0

            if verticle_flag == 0:
                for col in range(cols):
                    if img[row, col] != 0 and break_flag == 0:
                        break_flag = 1
                        start_x1 = col
                        continue
                    elif img[row, col] == 0 and break_flag == 1:
                        if vessels_pos[-1][0] == 0 or (vessels_pos[-1][1] + vessels_width / 2 > col > vessels_pos[-1][
                            1] - vessels_width / 2 and start_x1 < vessels_pos[-1][0] + vessels_width / 2) or (
                                vessels_pos[-1][0] + vessels_width / 2 > start_x1 > vessels_pos[-1][
                            0] - vessels_width / 2 and col > vessels_pos[-1][1] - vessels_width / 2):
                            start_x2 = col
                            break_flag = 2
                            verticle_flag = 1
                            continue
                        else:
                            for x in range(start_x1, col):
                                img[row, x] = 0
                            start_x1 = -1
                            break_flag = 0
                            continue
                    elif break_flag == 2:
                        img[row, col] = 0

                this_width = start_x2 - start_x1

                if abs(last_width - this_width) > vessels_width:
                    if abs(vessels_pos[-1][0] - start_x1) <= abs(vessels_pos[-1][1] - start_x2):
                        for col in range(start_x1 + last_width, start_x2 + 1):
                            img[row, col] = 0
                            start_x2 = start_x1 + last_width
                    else:
                        for col in range(start_x1, start_x2 - last_width + 1):
                            img[row, col] = 0
                            start_x1 = start_x2 - last_width
                    if stop_flag == 0:
                        stop_row = row
                    stop_flag = 1
                    this_width = 0
                if this_width != 0:
                    stop_row = 0
                    stop_flag = 0
                    last_width = this_width
                # vessels_pos.append([start_x1, start_x2])
            else:
                min_x1 = None
                min_x2 = 0
                for col in range(vessels_pos[-1][0], vessels_pos[-1][1] + 1):
                    if img[row, col] != 0:
                        if min_x1 is None:
                            min_x1 = col
                        if col > min_x2:
                            min_x2 = col
                if min_x1 is None:
                    stop_row = row
                    for col in range(cols):
                        img[row, col] = 0
                    vessels_pos.append([0, 0])
                    continue
                start_x1 = min_x1
                start_x2 = min_x2
                if min_x1 == vessels_pos[-1][0]:
                    for col in range(min_x1 - vessels_width, min_x1)[::-1]:
                        if img[row, col] == 0:
                            start_x1 = col + 1
                            break
                if min_x2 == vessels_pos[-1][1]:
                    for col in range(min_x2, min_x2 + vessels_width):
                        if img[row, col] == 0:
                            start_x2 = col - 1
                            break
                for col in range(0, start_x1):
                    img[row, col] = 0
                for col in range(start_x2 + 1, cols):
                    img[row, col] = 0
            vessels_pos.append([start_x1, start_x2])

        for row in range(stop_row, rows)[::-1]:
            if vessels_pos[-row][1] - vessels_pos[-row][0] > 100:
                stop_row = row
                break

        for row in range(stop_row):
            for col in range(cols):
                img[row, col] = 0


        # last_img = cv2.imread('{}/{}/00-last_result.png'.format(output_file_path, file_index), 0)
        origin_img = cv2.imread('{}/{}/1-original.png'.format(output_file_path, file_index), 0)
        img = cv2.bitwise_and(img, origin_img)
        cv2.imwrite('{}/{}/01-last_result.png'.format(output_file_path, file_index), img)

        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y:y + h, x:x + w]

        cv2.imwrite('{}/{}/02-last_result.png'.format(output_file_path, file_index), img)

        rows, cols = img.shape[:2]
        zero_counter = 0
        for row in range(rows):
            for col in range(cols):
                if img[row, col] != 0:
                    zero_counter += 1
        print(sum(img.flatten()) // zero_counter)

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

    def openFile(self):
        self.fileName = filedialog.askopenfile(parent=self.root, initialdir='./wanfang/user1/', title='Select Image',
                                                 filetypes=[('image files', '.png'), ('image files', '.jpg')])
        self.fileName = self.fileName.name
        self.image_filename["text"] = u'Image 檔名：{}'.format(self.fileName)
        self.corp_blood_vessel_one_step()

    def init_window(self):
        self.root.title(u'Image Processing')
        self.image_filename = Label(self.root, text=u'Image 檔名：')
        self.image_filename.grid(row=0, column=0, columnspan=5, sticky=W)
        Button(self.root, text='Select Image', command=self.openFile).grid(row=0, column=5, columnspan=3)

        Button(self.root, text=u'讀取 Image', command=self.load_img).grid(row=1, columnspan=8)

        Button(self.root, text=u'框出血管', command=self.corp_blood_vessel).grid(row=3, column=0, columnspan=2)
        Button(self.root, text=u'一鍵擷取血管', command=self.corp_blood_vessel_one_step).grid(row=3, column=2, columnspan=2)

        Button(self.root, text=u'侵蝕 法', command=self.erode).grid(row=5, column=0, columnspan=2)
        Button(self.root, text=u'擴張 法', command=self.dilate).grid(row=5, column=2, columnspan=2)
        Button(self.root, text=u'侵蝕擴張 法', command=self.erode_dilate).grid(row=5, column=4, columnspan=2)

        Button(self.root, text=u'二值化', command=self.binary_image).grid(row=6, column=0, columnspan=3)
        Button(self.root, text=u'分布圖均勻化(增加對比)', command=self.histogram_equalization).grid(row=6, column=3, columnspan=3)

        Button(self.root, text=u'回到上一步', command=self.recover_img).grid(row=7, column=0, columnspan=3)
        Button(self.root, text=u'儲存結果', command=lambda: self.save_img(self.img)).grid(row=7, column=3, columnspan=3)

        self.root.grid()
        self.root.mainloop()


if __name__ == "__main__":
    ImgP = ImageProcessing()
