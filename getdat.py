import os
import cv2
import numpy as np

# return path of all image in folder
def get_filepaths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths

#convert image into binary matrix
def convert_image(path):
    img = cv2.imread(path,0)

    row, col = img.shape

    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32)

    kernel = kernel / 9

    result = cv2.filter2D(img, -1, kernel)

    demo = np.array(result,np.float32)

    for i in range(row):
        for j in range(col):
            demo[i][j] /= 255
            if demo[i][j] > 0.5:
                img[i][j] = 1
            else:
                img[i][j] = 0

    return img

#convert data to training set
def image_to_matrix(lis_dir):
    lis = np.array([])
    lis_test = np.array([])

    print len(lis_dir)

    for i in range(5000):
        if i < 5000*3/4:
            matrix = convert_image(lis_dir[i])
            matrix = matrix.reshape((1,400))
            lis = np.append(lis, matrix)
        else:
            matrix = convert_image(lis_dir[i])
            matrix = matrix.reshape((1,400))
            lis_test = np.append(lis_test, matrix)
    x = 5000*3/4
    y = 5000 - 5000*3/4
    return lis,  x, lis_test, y

def getdata():
    training = []
    test = []

    lis = get_filepaths('/home/thangnx/code/triangle_competition/train/triangle')

    X1, len1, X1_test, len_test1 = image_to_matrix(lis)

    lis = get_filepaths('/home/thangnx/code/triangle_competition/train/non-triangle')
    X2, len2, X2_test, len_test2 = image_to_matrix(lis)

    X = np.append(X1, X2)
    X = X.reshape((len1 + len2, 400))

    tria = [[1],[0]]
    tria = np.array(tria)

    non_tria = [[0],[1]]
    non_tria = np.array(non_tria)

    for i in range(len1):
        temp = X[i]
        temp = np.array(temp).reshape((400,1))
        _temp = (temp, tria)
        training.append(_temp)

    for i in xrange(len1, len1+len2):
        temp = X[i]
        temp = np.array(temp).reshape((400,1))
        _temp = (temp, non_tria)
        training.append(_temp)

    X_test = np.append(X1_test, X2_test)
    X_test = X_test.reshape((len_test1 + len_test2, 400))

    for i in range(len_test1):
        temp = X_test[i]
        temp = np.array(temp).reshape((400,1))
        _temp = (temp, tria)
        test.append(_temp)

    for i in xrange(len_test1, len_test1+len_test2):
        temp = X_test[i]
        temp = np.array(temp).reshape((400,1))
        _temp = (temp, non_tria)
        test.append(_temp)

    print len(training)
    print len(test)

    return training, test
