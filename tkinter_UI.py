import codecs
from tkinter import *
import cv2
from PIL import Image,ImageTk

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
import math
import time
from threading import Thread
from pygame import mixer
import time



# 世界坐标系(UVW)：填写3D参考点
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]

# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]



EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3


MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3


HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3


COUNTER = 0
TOTAL = 0

mCOUNTER = 0
mTOTAL = 0

hCOUNTER = 0
hTOTAL = 0


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'dancheng_model/resnet_dancheng_model_train_fk.h5')

# 分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


global_pitch = 0
global_yaw = 0
global_roll = 0

EAR = ''
MAR = ''

def get_head_pose(shape):  # 头部姿态估计
    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    global global_pitch, global_yaw, global_roll
    global_pitch, global_yaw, global_roll =  pitch, yaw, roll
    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    return reprojectdst, euler_angle  # 投影误差，欧拉角



def construct_dict(file_path):
    word_freq = {}
    with open(file_path, "rb") as f:
        for line in f:
            info = line.split()
            word = info[0]
            frequency = info[1]
            word_freq[word] = frequency

    return word_freq

def computeTextureWeights(fin, sigma, sharpness):
    # print(fin)
    # fin = fin / 255.0

    dt0_v = np.diff(fin, 1, 0)  # 垂直差分
    dt0_v = np.concatenate((dt0_v, fin[:1, :] - fin[-1:, :]), axis=0)  # 第0行减去最后一行

    dt0_h = np.diff(fin, 1, 1)  # 水平差分
    dt0_h = np.concatenate((dt0_h, fin[:, :1] - fin[:, -1:]), axis=1)  # 第0列减去最后一列

    gauker_h = cv2.filter2D(dt0_h, -1, np.ones((1, sigma)), borderType=cv2.BORDER_CONSTANT)
    gauker_v = cv2.filter2D(dt0_v, -1, np.ones((sigma, 1)), borderType=cv2.BORDER_CONSTANT)
    # cv2这个filter2D（镜像翻转）与MATLAB的filter2（补0）不同

    W_h = 1.0 / (abs(gauker_h) * abs(dt0_h) + sharpness)
    W_v = 1.0 / (abs(gauker_v) * abs(dt0_v) + sharpness)

    return W_h, W_v


def convertCol(tmp):  # 按照列转成列。[[1, 2, 3], [4, 5, 6], [7, 8, 9]] # 转成[147258369].T(竖着)
    return np.reshape(tmp.T, (tmp.shape[0] * tmp.shape[1], 1))


def solveLinearEquation(IN, wx, wy, lambd):
    print('IN', IN.shape)
    r, c, ch = IN.shape[0], IN.shape[1], 1
    k = r * c
    dx = -lambd * convertCol(wx)  # 按列转成一列
    dy = -lambd * convertCol(wy)
    tempx = np.concatenate((wx[:, -1:], wx[:, 0:-1]), 1)  # 最后一列插入到第一列前面
    tempy = np.concatenate((wy[-1:, :], wy[0:-1, :]), 0)  # 最后一行插入到第一行前面
    dxa = -lambd * convertCol(tempx)
    dya = -lambd * convertCol(tempy)
    tempx = np.concatenate((wx[:, -1:], np.zeros((r, c - 1))), 1)  # 取wx最后一列放在第一列，其他为0
    tempy = np.concatenate((wy[-1:, :], np.zeros((r - 1, c))), 0)  # 取wy最后一行放在第一行，其他为0
    dxd1 = -lambd * convertCol(tempx)
    dyd1 = -lambd * convertCol(tempy)
    wx[:, -1:] = 0  # 最后一列置为0
    wy[-1:, :] = 0  # 最后一行置为0
    dxd2 = -lambd * convertCol(wx)
    dyd2 = -lambd * convertCol(wy)

    Ax = spdiags(np.concatenate((dxd1, dxd2), 1).T, np.array([-k + r, -r]), k, k)
    Ay = spdiags(np.concatenate((dyd1, dyd2), 1).T, np.array([-r + 1, -1]), k, k)
    # spdiags，与MATLAB不同，scipy是根据行来构造sp，而matlab是根据列来构造sp

    D = 1 - (dx + dy + dxa + dya)
    A = (Ax + Ay) + (Ax + Ay).T + spdiags(D.T, np.array([0]), k, k)

    A = A / 1000.0  # 需修改

    matCol = convertCol(IN)
    print('spsolve开始', str(datetime.now()))
    OUT = spsolve(A, matCol, permc_spec="MMD_AT_PLUS_A")
    print('spsolve结束', str(datetime.now()))
    OUT = OUT / 1000
    OUT = np.reshape(OUT, (c, r)).T
    return OUT


def tsmooth(I, lambd=0.5, sigma=5, sharpness=0.001):
    # print(I.shape)
    wx, wy = computeTextureWeights(I, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lambd)
    return S


def rgb2gm(I):
    print('I', I.shape)
    # I = np.maximum(I, 0.0)
    if I.shape[2] and I.shape[2] == 3:
        I = np.power(np.multiply(np.multiply(I[:, :, 0], I[:, :, 1]), I[:, :, 2]), (1.0 / 3))
    return I


def YisBad(Y, isBad):  # 此处需要修改得更高效
    return Y[isBad >= 1]
    # Z = []
    # [rows, cols] = Y.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         if isBad[i, j] >= 122:
    #             Z.append(Y[i, j])
    # return np.array([Z]).T


def applyK(I, k, a=-0.3293, b=1.1258):
    # print(type(I))
    if not type(I) == 'numpy.ndarray':
        I = np.array(I)
    # print(type(I))
    beta = np.exp((1 - (k ** a)) * b)
    gamma = (k ** a)
    BTF = np.power(I, gamma) * beta
    # try:
    #    BTF = (I ** gamma) * beta
    # except:
    #    print('gamma:', gamma, '---beta:', beta)
    #    BTF = I
    return BTF


def maxEntropyEnhance(I, isBad, mink=1, maxk=10):
    # Y = rgb2gm(np.real(np.maximum(imresize(I, (50, 50), interp='bicubic') / 255.0, 0)))
    Y = imresize(I, (50, 50), interp='bicubic') / 255.0
    Y = rgb2gm(Y)
    # bicubic较为接近
    # Y = rgb2gm(np.real(np.maximum(cv2.resize(I, (50, 50), interpolation=cv2.INTER_LANCZOS4  ), 0)))
    # INTER_AREA 较为接近
    # import matplotlib.pyplot as plt
    # plt.imshow(Y, cmap='gray');plt.show()

    print('isBad', isBad.shape)
    isBad = imresize(isBad.astype(int), (50, 50), interp='nearest')
    print('isBad', isBad.shape)

    # plt.imshow(isBad, cmap='gray');plt.show()

    # 取出isBad为真的Y的值，形成一个列向量Y
    # Y = YisBad(Y, isBad)  # 此处需要修改得更高效
    Y = Y[isBad >= 1]

    # Y = sorted(Y)

    print('-entropy(Y)', -entropy(Y))

    def f(k):
        return -entropy(applyK(Y, k))

    # opt_k = mink
    # k = mink
    # minF = f(k)
    # while k<= maxk:
    #     k+=0.0001
    #     if f(k)<minF:
    #         minF = f(k)
    #         opt_k = k
    opt_k = fminbound(f, mink, maxk)
    print('opt_k:', opt_k)
    # opt_k = 5.363584
    # opt_k = 0.499993757705
    # optk有问题，fminbound正确，f正确，推测Y不一样导致不对
    print('opt_k:', opt_k)
    J = applyK(I, opt_k) - 0.01
    return J


def HDR2dark(I, t_our, W):  # 过亮的地方变暗
    W = 1 - W
    I3 = I * W
    isBad = t_our > 0.8
    J3 = maxEntropyEnhance(I, isBad, 0.1, 0.5)  # 求k和曝光图
    J3 = J3 * (1 - W)  # 曝光图*权重
    fused = I3 + J3  # 增强图
    return I


def oneHDR(I, mu=0.5, a=-0.3293, b=1.1258):
    # mu照度图T的指数，数值越大，增强程度越大
    I = I / 255.0
    t_b = I[:, :, 0]  # t_b表示三通道图转成灰度图（灰度值为RGB中的最大值）,亮度矩阵L
    for i in range(I.shape[2] - 1):  # 防止输入图片非三通道
        t_b = np.maximum(t_b, I[:, :, i + 1])
    # t_b2 = cv2.resize(t_b, (0, 0), fx=0.5, fy=0.5)
    print('t_b', t_b.shape)
    # t_b2 = misc.imresize(t_b, (ceil(t_b.shape[0] / 2), ceil(t_b.shape[1] / 2)),interp='bicubic')
    # print('t_b2', t_b2.shape)
    # t_b2 = t_b / 255.0

    t_b2 = imresize(t_b, (256, 256), interp='bicubic', mode='F')  # / 255
    t_our = tsmooth(t_b2)  # 求解照度图T（灰度图）
    print('t_our前', t_our.shape)
    t_our = imresize(t_our, t_b.shape, interp='bicubic', mode='F')  # / 255
    print('t_our后', t_our.shape)

    # W: Weight Matrix 与 I2
    # 照度图L（灰度图） ->  照度图L（RGB图）：灰度值重复3次赋给RGB
    # size为(I, 3) ， 防止与原图尺寸有偏差
    t = np.ndarray(I.shape)
    for ii in range(I.shape[2]):
        t[:, :, ii] = t_our
    print('t', t.shape)

    W = t ** mu  # 原图的权重。三维矩阵

    cv2.imwrite(filepath + 'W.jpg', W * 255)
    cv2.imwrite(filepath + '1-W.jpg', (1 - W) * 255)
    cv2.imwrite(filepath + 't.jpg', t * 255)
    cv2.imwrite(filepath + '1-t.jpg', (1 - t) * 255)

    print('W', W.shape)
    # 变暗
    # isBad = t_our > 0.8  # 是高光照的像素点
    # I = maxEntropyEnhance(I, isBad)  # 求k和曝光图
    # 变暗
    I2 = I * W  # 原图*权重

    # 曝光率->k ->J
    isBad = t_our < 0.5  # 是低光照的像素点
    J = maxEntropyEnhance(I, isBad)  # 求k和曝光图
    J2 = J * (1 - W)  # 曝光图*权重
    fused = I2 + J2  # 增强图

    # 存储中间结果
    cv2.imwrite(filepath + 'I2.jpg', I2 * 255.0)
    cv2.imwrite(filepath + 'J2.jpg', J2 * 255.0)

    # 变暗
    # fused = HDR2dark(fused, t_our, W)

    return fused
    # return res

# 图像增强，效果不错，人脸光照充足情况下没必要使用
def test():
    inputImg = cv2.imread(filepath + 'input.jpg')
    outputImg = oneHDR(inputImg)
    # outputImg = outputImg * 255.0
    cv2.imwrite(filepath + 'out.jpg', outputImg * 255)
    print("HDR完成，已保存到本地")

    print('程序结束', str(datetime.now()))

    cv2.imshow('inputImg', inputImg)
    cv2.imshow('outputImg', outputImg)
    # print(inputImg.dtype,outputImg.dtype)
    # outputImg = outputImg.astype(int)
    # print(inputImg.dtype, outputImg.dtype)
    # compare = np.concatenate((inputImg,outputImg),axis=1)
    # cv2.imshow('compare', compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_cn_words_dict(file_path):
    cn_words_dict = ""
    with open(file_path, "rb") as f:
        for word in f:
            cn_words_dict += word.strip().decode("utf-8")
    return cn_words_dict


def edits1(phrase, cn_words_dict):
    splits = [(phrase[:i], phrase[i:]) for i in range(len(phrase) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in cn_words_dict]
    inserts = [L + c + R for L, R in splits for c in cn_words_dict]
    return set(deletes + transposes + replaces + inserts)


def known(phrases): return set(phrase for phrase in phrases if phrase.encode("utf-8") in phrase_freq)


def get_candidates(error_phrase):
    candidates_1st_order = []
    candidates_2nd_order = []
    candidates_3nd_order = []

    p = Pinyin()
    error_pinyin = p.get_pinyin(error_phrase)
    re.sub("-", "/", error_pinyin)
    cn_words_dict = load_cn_words_dict("HW10/Autochecker4Chinese-master/cn_dict.txt")
    candidate_phrases = list(known(edits1(error_phrase, cn_words_dict)))

    for candidate_phrase in candidate_phrases:
        # candidate_pinyin = pinyin.get(candidate_phrase, format="strip", delimiter="/").encode("utf-8")
        candidate_pinyin = p.get_pinyin(candidate_phrase)
        re.sub("-", "/", candidate_pinyin)
        if candidate_pinyin == error_pinyin:
            candidates_1st_order.append(candidate_phrase)
        elif candidate_pinyin.split("/")[0] == error_pinyin.split("/")[0]:
            candidates_2nd_order.append(candidate_phrase)
        else:
            candidates_3nd_order.append(candidate_phrase)

    return candidates_1st_order, candidates_2nd_order, candidates_3nd_order


def find_max(c1_order):
    maxo = ''
    maxi = 0
    for i in range(0, len(c1_order)):
        if c1_order[i].encode("utf-8") in phrase_freq:
            freq = int(phrase_freq.get(c1_order[i].encode('utf-8')))
            if freq > maxi:
                maxi = freq
                maxo = c1_order[i]
    return maxo


def auto_correct(error_phrase):
    c1_order, c2_order, c3_order = get_candidates(error_phrase)
    if c1_order:
        return find_max(c1_order)
    elif c2_order:
        return find_max(c2_order)
    else:
        return find_max(c3_order)


def auto_correct_sentence(error_sentence, verbose=True):
    jieba_cut = jieba.cut(error_sentence, cut_all=False)
    seg_list = "\t".join(jieba_cut).split("\t")

    correct_sentence = ""

    for phrase in seg_list:

        correct_phrase = phrase
        # check if item is a punctuation
        if phrase not in PUNCTUATION_LIST:
            # check if the phrase in our dict, if not then it is a misspelled phrase
            if phrase.encode('utf-8') not in phrase_freq.keys():
                correct_phrase = auto_correct(phrase)
                if verbose:
                    print(phrase, correct_phrase)

        correct_sentence += correct_phrase

    return correct_sentence



def test_case(err_sent_1):
    print('===============')
    correct_sent = auto_correct_sentence(err_sent_1)
    t1 = "original sentence:" + err_sent_1 + "\n==>\n" + "corrected sentence:" + correct_sent
    print(t1)


def create_inverted_index(fenci_txt, param):
    pass


def charge_spimi():
    fenci_txt = "fenci.txt"
    with codecs.open(fenci_txt, 'w', encoding="UTF-8-SIG") as f:
        path = "data\\page"
        files = os.listdir(path)
        s = []
        for file in files:
            if not os.path.isdir(file):
                print('>>>处理', file)
                for line in open(path + "\\" + file, encoding='utf-8', errors='ignore').readlines():
                    # 去标点
                    line = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", line)
                    # 分词
                    seg_list = jieba.cut(line, cut_all=True)
                    # 写入199801_new.txt
                    f.write(" ".join(seg_list) + "\n")
                    # 建立倒排索引
                create_inverted_index(fenci_txt, re.sub(r'\D', "", file))

    with codecs.open("my_index.txt", 'w', encoding="UTF-8-SIG") as i:
        for key in index.keys():
            i.write(key + str(index[key]) + "\n")




def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):  # 嘴部
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


def dete_tired(frame):
    frame = imutils.resize(frame, width=660)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    mar = ''
    ear = ''
    global COUNTER, TOTAL, mCOUNTER, mTOTAL, hCOUNTER, hTOTAL

    # 循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)

        # 将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)

        # 提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]

        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        '''
            分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
        '''
        # 循环，满足条件的，眨眼次数+1
        if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.2
            COUNTER += 1

        else:
            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
            if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 阈值：3
                TOTAL += 1
            # 重置眼帧计数器
            COUNTER = 0

        global EAR, MAR
        EAR = "{:.2f}".format(ear)
        MAR = "{:.2f}".format(mar)

        # 进行画图操作，同时使用cv2.putText将眨眼次数进行显示
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        '''
            计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
        '''
        if mar > MAR_THRESH:  # 张嘴阈值0.5
            mCOUNTER += 1
            cv2.putText(frame, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 如果连续3次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0
        cv2.putText(frame, "COUNTER: {}".format(mCOUNTER), (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        """
        瞌睡点头
        """
        # 获取头部姿态
        reprojectdst, euler_angle = get_head_pose(shape)

        har = euler_angle[0, 0]  # 取pitch旋转角度
        if har > HAR_THRESH:  # 点头阈值0.3
            hCOUNTER += 1
        else:
            # 如果连续3次都小于阈值，则表示瞌睡点头一次
            if hCOUNTER >= NOD_AR_CONSEC_FRAMES:  # 阈值：3
                hTOTAL += 1
            # 重置点头帧计数器
            hCOUNTER = 0

        # 绘制正方体12轴
        # for start, end in line_pairs:
        #     cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
        # 显示角度结果
        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0), thickness=2)  # GREEN
        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 0), thickness=2)  # BLUE
        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), thickness=2)  # RED

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # print('嘴巴实时长宽比:{:.2f} '.format(mar) + "\t是否张嘴：" + str([False, True][mar > MAR_THRESH]))
    # print('眼睛实时长宽比:{:.2f} '.format(ear) + "\t是否眨眼：" + str([False, True][COUNTER >= 1]))

    # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头15次

    #if TOTAL >= 20 or mTOTAL >= 10 or hTOTAL >= 15:
    if TOTAL >= 4 or mTOTAL >= 4 or hTOTAL >= 4:
        cv2.putText(frame, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        mixer.init()
        mixer.music.load('d:/pljs.mp3')
        mixer.music.play(5, 0.0)
        time.sleep(2)
        mixer.music.stop()
    return frame


def take_snapshot():
    # video_loop()
    global TOTAL, mTOTAL, hTOTAL
    TOTAL = 0
    mTOTAL = 0
    hTOTAL = 0

def video_loop():
    success, img = camera.read()  # 从摄像头读取照片
    if success:
        detect_result = dete_tired(img)
        cv2image = cv2.cvtColor(detect_result, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)#将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)

        ####################################updata########################
        global hCOUNTER, mCOUNTER, TOTAL, mTOTAL, hTOTAL
        global global_pitch, global_yaw, global_roll
        global EAR, MAR

        root.update()  # 不断更新
        root.after(10)
        Label(root, text='打盹时间:'+str(hCOUNTER), font=("黑体", 14), fg="red", width=12, height=2).place(x=10, y=570,
                                                                                       anchor='nw')
        Label(root, text='打哈欠时间:' + str(mCOUNTER), font=("黑体", 14), fg="red", width=12, height=2).place(x=140, y=570,
                                                                                                anchor='nw')

        Label(root, text='眨眼次数:' + str(TOTAL), font=("黑体", 14), fg="red", width=12, height=2).place(x=270, y=570,
                                                                                                        anchor='nw')
        Label(root, text='哈欠次数:' + str(mTOTAL), font=("黑体", 14), fg="red", width=12, height=2).place(x=400, y=570,
                                                                                                       anchor='nw')
        Label(root, text='钓鱼次数:' + str(hTOTAL), font=("黑体", 14), fg="red", width=12, height=2).place(x=530, y=570,
                                                                                                       anchor='nw')

        Label(root, text='嘴部面积 : ' + str(MAR), font=("黑体", 14), fg="red", width=20, height=2).place(x=130, y=610, anchor='nw')
        Label(root, text='眼部面积 : ' + str(EAR), font=("黑体", 14), fg="red", width=20, height=2).place(x=320, y=610, anchor='nw')
        # Label(root, text='头部yaw:' + str(global_yaw), font=("黑体", 14), fg="red", width=12, height=2).place(x=140, y=600, anchor='nw')
        # Label(root, text='头部横滚角:' + str(global_roll), font=("黑体", 14), fg="red", width=12, height=2).place(x=270, y=600, anchor='nw')

        root.after(1, video_loop)



if __name__=='__main__':
    camera = cv2.VideoCapture(0)  # 摄像头

    root = Tk()
    root.title("opencv + tkinter")
    # root.protocol('WM_DELETE_WINDOW', detector)

    panel = Label(root)  # initialize image panel
    panel.pack(padx=10, pady=10)
    root.config(cursor="arrow")

    btn = Button(root, text="疲劳提醒解锁!", command=take_snapshot)
    btn.pack(fill="both", expand=True, padx=10, pady=10)

    # strs = '测试文字'
    Label(root, text=' ', font=("黑体", 14), fg="red", width=12, height=2).pack(fill="both", expand=True, padx=10, pady=20)



    video_loop()

    root.mainloop()
    # 当一切都完成后，关闭摄像头并释放所占资源
    camera.release()
    cv2.destroyAllWindows()