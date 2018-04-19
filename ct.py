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
        self.origin_img = None
        self.img = None
        self.cols = 0
        self.rows = 0
        self.file_index = 0
        self.output_path = '/Users/candy/Workspace/Paper_CT/output/'
        self.init_window()

    def _corp_blood_vessel(self):
        image, contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        new_contours = []
        for x in contours:
            if len(x) > 1000:
                new_contours.append(x)
        self.img = np.zeros_like(self.img)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(self.img, new_contours, -1, 255, -1)  # Draw filled contour in mask
        self.save_img(self.img, '4-find_contours_over_1000_point.png', self.output_path)
        tmp_img = copy.deepcopy(self.img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for x in range(10):
            tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_ERODE, kernel)
        for x in range(10):
            tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_DILATE, kernel)
        tmp_img = cv2.bitwise_not(tmp_img)
        self.save_img(tmp_img, '5-after_erode_10_times_and_dilate_10_times.png', self.output_path)

        self.img = cv2.bitwise_and(self.img, tmp_img)
        self.save_img(self.img, '6-delete_not_vessels_part.png', self.output_path)

        self.dilate_erode()
        self.save_img(self.img, '6.5-recover_some_hole .png', self.output_path)
        image, contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = []
        max_width_height = 0
        for index in range(len(contours)):
            if len(contours[index]) < 100:
                continue
            height = self._get_contour_height(contours, index)
            if height > max_width_height:
                new_contours = [contours[index]]
                max_width_height = height

        self.img = np.zeros_like(self.img)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(self.img, new_contours, -1, 255, -1)  # Draw filled contour in mask
        self.save_img(self.img, '7-refind_contours.png', self.output_path)
        self.img = cv2.bitwise_and(self.img, self.origin_img)

    def _get_contour_height(self, contours, index=0):
        contour = np.array(contours[index])
        print(contour)
        return max(contour[:, :, 1])-min(contour[:, :, 1])
        self.save_img(self.img, '8-refill_origin_pic.png', self.output_path)

    def histogram_equalization(self):
        self.img = cv2.equalizeHist(self.img)

    def _binary_image(self):
        ret, self.img = cv2.threshold(self.img, 95, 255, cv2.THRESH_BINARY)

    def erode_dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)

    def erode(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_ERODE, kernel)

    def dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_DILATE, kernel)

    def dilate_erode(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)

    def find_blood_vessel(self):
        self.save_img(self.img, '1-original.png', self.output_path)
        self._binary_image()
        self.save_img(self.img, '2-after_binary.png', self.output_path)
        self.erode_dilate()
        self.save_img(self.img, '3-after_erode_and_dilate_once.png', self.output_path)
        self._corp_blood_vessel()
        self._refind_blood_vessel()

    def thinning(self):
        self.img = cv2.imread('{}/01-last_result.png'.format(self.output_path), 0)
        ret, img = cv2.threshold(self.img, 90, 255, 0)
        import thinning
        self.img = thinning.guo_hall_thinning(img)
        self.save_img(self.img, '03-skeletonize_result.png', self.output_path)

    def find_branches(self):
        self.img = cv2.imread('{}/03-skeletonize_result.png'.format(self.output_path), 0)
        self.output_img = cv2.imread('{}/01-last_result.png'.format(self.output_path), 0)

        rows, cols = self.img.shape[:2]
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if self.img[row, col] == 0:
                    continue
                src = self.img[row - 1:row + 2, col - 1:col + 2]
                kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype='uint8')
                result = cv2.bitwise_and(src, kernel)
                if sum(result.flatten()) >= 3:
                    y = row
                    x = col
                    cv2.rectangle(self.output_img, (x - 10, y - 10), (x + 10, y + 10), 255, -1)
        self.save_img(self.output_img, '02-find_branches_result.png', self.output_path)

    def mark_biggest_angle(self):
        self.img = cv2.imread('{}/03-skeletonize_result.png'.format(self.output_path), 0)

        line_array = []

        for row in range(self.rows)[::-1]:
            first_col = -1
            last_col = -1
            for col in range(self.cols):
                if self.img[row, col] > 0:
                    first_col = col
                    break
            for col in range(self.cols)[::-1]:
                if self.img[row, col] > 0:
                    last_col = col
                    break
            if first_col == last_col == -1:
                line_array.append(-1)
                continue
            line_array.append((first_col + last_col) // 2)
        line_array = line_array[::-1]

        space = self.rows // 10
        angle_array = []
        black_counter = 0
        for row in range(self.rows - space * 2):
            if min(line_array[row:row + space * 2 + 1]) == -1:
                angle_array.append(200)
                continue
            point_A = np.array([row, line_array[row]])
            point_B = np.array([row + space, line_array[row + space]])
            point_C = np.array([row + space * 2, line_array[row + space * 2]])
            x = point_A - point_B
            y = point_C - point_B
            Lx = np.sqrt(x.dot(x))
            Ly = np.sqrt(y.dot(y))
            cos_angle = x.dot(y) / (Lx * Ly)
            angle = np.arccos(cos_angle)
            angle2 = abs(angle * 360 / 2 / np.pi)
            angle_array.append(angle2)
        y1 = angle_array.index(min(angle_array)) + space
        x1 = line_array[y1]

        origin_img = cv2.imread('{}/01-last_result.png'.format(self.output_path), 0)
        cv2.rectangle(origin_img, (x1 - 10, y1 - 10), (x1 + 10, y1 + 10), 255, -1)
        y2 = angle_array.index(min(angle_array)) + space * 2
        x2 = line_array[y2]
        cv2.rectangle(origin_img, (x2 - 10, y2 - 10), (x2 + 10, y2 + 10), 255, -1)
        y3 = angle_array.index(min(angle_array))
        x3 = line_array[y3]
        cv2.rectangle(origin_img, (x3 - 10, y3 - 10), (x3 + 10, y3 + 10), 255, -1)
        cv2.line(origin_img, (x1, y1), (x2, y2), 255, thickness=3)
        cv2.line(origin_img, (x1, y1), (x3, y3), 255, thickness=3)

        self.save_img(origin_img, '03-mark_result.png', self.output_path)

        plt.plot(range(len(angle_array)), angle_array, 'g', label='line 1', linewidth=1)
        plt.savefig('{}/03-angle_histogram.png'.format(self.output_path))
        plt.clf()
        self.combine_two_image()

    def _refind_blood_vessel(self):
        img = cv2.imread('{}/4-find_contours_over_1000_point.png'.format(self.output_path), 0)
        rows, cols = img.shape[:2]

        last_width = 0
        vessels_pos = []
        vessels_width = 60

        img_find_row = cv2.imread('{}/7-refind_contours.png'.format(self.output_path), 0)
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

        origin_img = cv2.imread('{}/1-original.png'.format(self.output_path), 0)
        img = cv2.bitwise_and(img, origin_img)
        cv2.imwrite('{}/01-last_result.png'.format(self.output_path), img)

        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y - 20:y + h + 20, x - 20:x + w + 20]

        cv2.imwrite('{}/02-last_result.png'.format(self.output_path), img)

        rows, cols = img.shape[:2]
        zero_counter = 0
        for row in range(rows):
            for col in range(cols):
                if img[row, col] != 0:
                    zero_counter += 1
        print(sum(img.flatten()) // zero_counter)

    def combine_two_image(self):
        imga = cv2.imread('{}/03-skeletonize_result.png'.format(self.output_path), 0)
        imgb = cv2.imread('{}/03-mark_result.png'.format(self.output_path), 0)
        ha, wa = imga.shape[:2]
        hb, wb = imgb.shape[:2]
        max_height = np.max([ha, hb])
        total_width = wa + wb
        new_img = np.zeros(shape=(max_height, total_width))
        new_img[:ha, :wa] = imga
        new_img[:hb, wa:wa + wb] = imgb
        self.save_img(new_img, 'combine.png', self.output_path)
        return new_img

    def save_img(self, img, file_name='test1.png', file_root='/Users/candy/Workspace/Paper_CT/output'):
        cv2.imwrite('{}/{}'.format(file_root, file_name), img)

    def load_img(self):
        if self.fileName == '' or self.fileName is None:
            self.origin_img = cv2.imread('/Users/candy/Workspace/Paper_CT/3.jpg', 0)
            self.img = cv2.imread('/Users/candy/Workspace/Paper_CT/3.jpg', 0)
        else:
            self.origin_img = cv2.imread(self.fileName, 0)
            self.img = cv2.imread(self.fileName, 0)
        self.rows, self.cols = self.img.shape[:2]

    def show_img(self):
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def open_file(self):
        self.fileName = filedialog.askopenfile(parent=self.root, initialdir='./wanfang/user1', title='Select Image',
                                               filetypes=[('image files', '.png'), ('image files', '.jpg')])
        self.fileName = self.fileName.name
        self.image_filename["text"] = u'Image 檔名：{}'.format(self.fileName)
        self.origin_img = cv2.imread(self.fileName, 0)
        self.img = cv2.imread(self.fileName, 0)
        self.rows, self.cols = self.img.shape[:2]
        self.file_index = self.fileName.split('/')[-1].split('.')[0]
        self.output_path = '/Users/candy/Workspace/Paper_CT/output/{}'.format(self.file_index)

    def init_window(self):
        self.root.title(u'Image Processing')
        self.image_filename = Label(self.root, text=u'Image 檔名：')
        self.image_filename.grid(row=0, column=0, columnspan=5, sticky=W)
        Button(self.root, text='Select Image', command=self.open_file).grid(row=0, column=5, columnspan=3)

        Button(self.root, text=u'重新讀取 Image', command=self.load_img).grid(row=1, columnspan=8)

        Button(self.root, text=u'1-擷取血管', command=self.find_blood_vessel).grid(row=3, column=2, columnspan=2)

        Button(self.root, text=u'2-細線化', command=self.thinning).grid(row=4, column=0, columnspan=2)
        Button(self.root, text=u'3-找分支', command=self.find_branches).grid(row=4, column=2, columnspan=2)
        Button(self.root, text=u'3-找最彎的角度', command=self.mark_biggest_angle).grid(row=4, column=4, columnspan=2)

        Button(self.root, text=u'侵蝕 法', command=self.erode).grid(row=5, column=0, columnspan=2)
        Button(self.root, text=u'擴張 法', command=self.dilate).grid(row=5, column=2, columnspan=2)
        Button(self.root, text=u'侵蝕擴張 法', command=self.erode_dilate).grid(row=5, column=4, columnspan=2)

        Button(self.root, text=u'分布圖均勻化(增加對比)', command=self.histogram_equalization).grid(row=6, column=3, columnspan=3)

        Button(self.root, text=u'儲存結果', command=lambda: self.save_img(self.img)).grid(row=7, column=3, columnspan=3)

        self.root.grid()
        self.root.mainloop()


if __name__ == "__main__":
    ImgP = ImageProcessing()
