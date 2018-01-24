import cv2

image_index = '15'
file_path = '.'
img = cv2.imread('{}/origin_data/{}-02-last_result.png'.format(file_path, image_index), 0)

rows, cols = img.shape[:2]
data_size = 15
data_number = 0
# row_array = range(22, 46)
# col_array = range(63, 77)

for row in range(rows-data_size+1):
    # yes_row_flag = 0
    # for x in range(data_size):
    #     if row + x in row_array:
    #         yes_row_flag = 1
    #         break
    for col in range(cols-data_size+1):
        # yes_flag = 0
        new_img = img[row:row + data_size, col:col + data_size]
        all_black_flag = 0
        for x in range(15):
            for y in range(15):
                if new_img[x, y] != 0:
                    all_black_flag = 1
                    break
        if all_black_flag == 0:
            continue
        # if yes_row_flag == 1:
        #     for y in range(data_size):
        #         if col + y in col_array:
        #             yes_flag = 1
        #             break
        # if yes_flag == 1:
        cv2.imwrite('{}/testing_data_{}/{}x{}.png'.format(file_path, image_index, row, col), new_img)
        # else:s
        #     cv2.imwrite('{}/{}x{}/no/{}x{}.png'.format(file_path, data_size, data_size, row, col), new_img)
        data_number += 1


def corp_vessels():
    corp_file_path = '/Users/candy/Workspace/Paper_CT/output/' + image_index
    image = cv2.imread(corp_file_path + '/01-last_result.png', 0)
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    print(x, y, w, h)
    image = image[y:y + h, x:x + w]

    cv2.imwrite(corp_file_path + '/02-last_result.png', image)
