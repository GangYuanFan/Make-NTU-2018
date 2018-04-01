"""
@ Real-Time face recognition with CNN structure
this code can recognize Jerry or not.

@ CNN->MaxPooling -> CNN->MaxPooling -> CNN->MaxPooling -> CNN->MaxPooling -> FC1 -> FC2
Loss: cross-entropy with softmax
Optimizer: using Adam-OP
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import pandas as pd
import math
import imutils
from imutils import face_utils
import dlib

n_class = 2
batch_size = 1
lr = 5e-4
lr_decay = 0.1
checkpoint_dir = 'D:/DeepLearning/face/train_landmark_vgg/'
train_data_path = 'D:/DeepLearning/face/train_landmark_vgg/'
train_label_path = 'D:/DeepLearning/face/train_landmark_vgg/'
test_data_path = 'D:/DeepLearning/face/test_landmark_vgg/'
test_label_path = 'D:/DeepLearning/face/test_landmark_vgg/'
face_cascPath = 'D:/DeepLearning/face/FaceDetect-master/haarcascade_frontalface_default.xml'  #face detect model
trainset_size = 13600
testset_size = 6800
train_iter = 500
report_freq = 50
resize = 0.8
num_feat = 68
KeepProb = 0.125  # for dropout used
# CAM_FLAG = 0
detector = dlib.get_frontal_face_detector()
fname = 'D:/DeepLearning/face/Face-LandMark_with_Dlib/facial-landmarks/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(fname)
l_eye_pct = 0.33
r_eye_pct = 0.66
eyes_level_pct = 0.4

class Neural_Network:
    def __init__(self):
        self.train_indices = np.random.permutation(trainset_size//num_feat)  # generate random training-data index
        self.test_indices = np.random.permutation(testset_size//num_feat)  # generate random testing-data index
        with tf.name_scope('inputs'):
            self.xs1 = tf.placeholder(tf.float32, [None, num_feat])  # strength input
            self.xs2 = tf.placeholder(tf.float32, [None, num_feat])  # angle input
            # with tf.name_scope('batch_normalization'):
            #     self.xs1_tmp = self.batch_norm(self.xs1, num_feat)
            #     self.xs2_tmp = self.batch_norm(self.xs2, num_feat)
        with tf.name_scope('y_label'):
            self.ys = tf.placeholder(tf.float32, [None, n_class])
        with tf.name_scope('hyper_parameters'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.learning_rate = tf.placeholder(tf.float32)
            self.state_flag = tf.placeholder(tf.int8)
        with tf.variable_scope('nn_parameters'):
            self.nn_param()
        with tf.variable_scope('DNN'):
            self.DNN()
        with tf.name_scope('loss'):
            self.compute_cost()
        with tf.name_scope('train_optimizer'):
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.y)
        with tf.name_scope('accuracy'):
            self.compute_accu()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.classifier = ['Jerry', 'Stranger']
        # faceCascade = cv2.CascadeClassifier(cascPath)
        print('start testing...')
        print('=========================================================')
        saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
        # saver.restore(sess, checkpoint_dir + 'model.ckpt-%d' %train_iter)
        saver.restore(self.sess, checkpoint_dir + 'model.ckpt-550')
        # self.cap = cv2.VideoCapture(cam_flag)
        self.face_flag = 251

    def read_data(self, dataset=1):
        """
        reading data from dir
        with random
        :param dataset:1 for trainning, 0 for testing
        :return strength, angle, label
        """
        if dataset == 1:
            tmp_output_txt = os.path.join(train_label_path, 'train_face_label.txt')
            tmp_output_txt_writer = open(tmp_output_txt, 'r')
            Label = tmp_output_txt_writer.read()
            df = pd.read_csv(train_data_path + 'trainset.csv')
            strength_tmp = np.array(df['strength'])
            angle_tmp = np.array(df['angle'])
            strength = []
            angle = []
            label = []
            for _ in range(trainset_size):
                if _ % num_feat == 0:
                    strength_tmp2 = []
                    angle_tmp2 = []
                    for __ in range(num_feat):
                        strength_tmp2.append(strength_tmp[__ + _])
                        angle_tmp2.append(angle_tmp[__ + _])
                    strength.append(strength_tmp2)
                    angle.append(angle_tmp2)
                    label.append([int(Label[_ * (n_class + 1)]), int(Label[_ * (n_class + 1) + 1])])
                    del strength_tmp2
                    del angle_tmp2
            del strength_tmp
            del angle_tmp
            strength = np.array(strength)
            angle = np.array(angle)
            strength2 = np.zeros_like(strength)
            angle2 = np.zeros_like(angle)
            label2 = np.zeros_like(label)
            for i in range(trainset_size//num_feat):
                counter = self.train_indices[i]
                strength2[i] = strength[counter]
                angle2[i] = angle[counter]
                label2[i] = label[counter]
            tmp_output_txt_writer.flush()
            tmp_output_txt_writer.close()
            del Label
            del label
            del angle
            del strength
            return strength2, angle2 / 180, label2
        elif dataset == 0:
            x = []
            tmp_output_txt = os.path.join(test_data_path, 'test_face_label.txt')
            tmp_output_txt_writer = open(tmp_output_txt, 'r')
            Label = tmp_output_txt_writer.read()
            df = pd.read_csv(test_data_path + 'testset.csv')
            strength_tmp = np.array(df['strength'])
            angle_tmp = np.array(df['angle'])
            strength = []
            angle = []
            label = []
            for _ in range(testset_size):
                if _ % num_feat == 0:
                    strength_tmp2 = []
                    angle_tmp2 = []
                    for __ in range(num_feat):
                        strength_tmp2.append(strength_tmp[__ + _])
                        angle_tmp2.append(angle_tmp[__ + _])
                    strength.append(strength_tmp2)
                    angle.append(angle_tmp2)
                    label.append([int(Label[_ * (n_class + 1)]), int(Label[_ * (n_class + 1) + 1])])
                    del strength_tmp2
                    del angle_tmp2
            del strength_tmp
            del angle_tmp
            strength = np.array(strength)
            angle = np.array(angle)
            strength2 = np.zeros_like(strength)
            angle2 = np.zeros_like(angle)
            label2 = np.zeros_like(label)
            for i in range(testset_size // num_feat):
                counter = self.test_indices[i]
                strength2[i] = strength[counter]
                angle2[i] = angle[counter]
                label2[i] = label[counter]
            tmp_output_txt_writer.flush()
            tmp_output_txt_writer.close()
            del Label
            del label
            del angle
            del strength
            return strength2, angle2 / 180, label2
        else:
            print('argument error')

    with tf.device('/device:GPU:0'):
        def weight_variable(self, shape):
            initial = tf.truncated_normal(shape=shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(self, shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def batch_norm(self, bn_in, out_size):
            # Batch Normalize
            mean, var = tf.nn.moments(
                bn_in,
                axes=[0],  # the dimension you wanna normalize, here [0] for batch
                # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            return tf.nn.batch_normalization(bn_in, mean, var, shift, scale, epsilon)

    def nn_param(self):
        self.s1_param = 224
        self.a1_param = 224
        self.s2_param = 448
        self.a2_param = 448
        self.layer3_param = 136
        self.layer4_param = n_class

    def DNN(self):
        ## dnn layer 1##
        with tf.name_scope('DNN_layer_1'):
            with tf.name_scope('DNN_layer_1_strength'):
                Ws1 = self.weight_variable([num_feat, self.s1_param])
                bs1 = self.bias_variable([self.s1_param])
                out_s1 = tf.nn.relu(tf.matmul(self.xs1, Ws1) + bs1, name='relu_s1') # shape = [1, 224]
            with tf.name_scope('DNN_layer_1_angle'):
                Wa1 = self.weight_variable([num_feat, self.a1_param])
                ba1 = self.bias_variable([self.a1_param])
                out_a1 = tf.nn.tanh(tf.matmul(self.xs2, Wa1) + ba1, name='tanh_a1')  # shape = [1, 224]

        ## dnn layer 2 ##
        with tf.name_scope('DNN_layer_2'):
            with tf.name_scope('DNN_layer_2_strength'):
                Ws2 = self.weight_variable([self.s1_param, self.s2_param])
                bs2 = self.bias_variable([self.s2_param])
                out_s2 = tf.nn.relu(tf.matmul(out_s1, Ws2) + bs2, name='relu_s2') # shape = [224, 448]
            with tf.name_scope('DNN_layer_2_angle'):
                Wa2 = self.weight_variable([self.a1_param, self.a2_param])
                ba2 = self.bias_variable([self.a2_param])
                out_a2 = tf.nn.relu(tf.matmul(out_a1, Wa2) + ba2, name='relu_a2')  # shape = [224, 448]

        # batch normalization ##
        # with tf.name_scope('bn_two_inputs'):
        #     out_s2 = self.batch_norm(out_s2, self.s2_param)
        #     out_a2 = self.batch_norm(out_a2, self.a2_param)

        ## stack two inputs ##
        with tf.name_scope('stack_two_inputs'):
            stack = tf.stack([out_s2, out_a2], axis=1)  # shape = [448, 2]
            stack = tf.reshape(stack, [1, -1])  # shape = [896]

        ## dnn layer 3 ##
        with tf.name_scope('DNN_layer_3'):
            W3 = self.weight_variable([self.a2_param * 2, self.layer3_param])
            b3 = self.bias_variable([self.layer3_param])
            out_3 = tf.nn.relu(tf.matmul(stack, W3) + b3, name='relu_3')  # shape = [136]
            with tf.name_scope('Dropout'):
                if self.state_flag == 1:
                    out_3_drop = tf.nn.dropout(out_3, self.keep_prob)
                else:
                    out_3_drop = out_3

        ## dnn layer 4 ##
        with tf.name_scope('DNN_layer_4'):
            W4 = self.weight_variable([self.layer3_param, self.layer4_param])
            b4 = self.bias_variable([self.layer4_param])
            self.y = tf.matmul(out_3_drop, W4) + b4  # shape = [n_class,]

        # the error between prediction and real data
        # loss = tf.reduce_mean(tf.pow(y - ys, 2))

    def compute_cost(self):
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.ys))  # loss
        self.loss = tf.reduce_mean(tf.pow(self.y - self.ys, 2))
    def compute_accu(self):
        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.ys, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self):  ### create tensorflow structure end ###
        saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
        with tf.device('/device:GPU:0'):
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
                # writer = tf.train.SummaryWriter('logs/', sess.graph)
            else:
                init = tf.global_variables_initializer()
                # writer = tf.summary.FileWriter("logs/", sess.graph)
            print('reading data from %s...' % (train_data_path))
            train_X_strength, train_X_angle, train_Y = self.read_data(dataset=1)  # train set
            vali_X_strength, vali_X_angle, vali_Y = self.read_data(dataset=0)  # validation set
            print('=========================================================')
            sess.run(init)
            print('start training...')
            print('=========================================================')
            Num_train_batch = trainset_size // num_feat // batch_size
            Num_vali_batch = testset_size // num_feat // batch_size
            step_list = []
            time_list = []
            train_accu_list = []
            vali_accu_list = []
            train_one_iter_accu = 0
            vali_one_iter_accu = 0
            for step in range(train_iter):  # start training
                train_indices = np.random.permutation(trainset_size // num_feat)
                vali_indices = np.random.permutation(testset_size // num_feat)
                for _ in range(Num_train_batch):
                    offset = _ * batch_size
                    train_x_strength = np.array(train_X_strength[train_indices[offset:offset + batch_size]], np.float32)
                    train_x_angle = np.array(train_X_angle[train_indices[offset:offset + batch_size]], np.float32)
                    train_y = np.array(train_Y[train_indices[offset:offset + batch_size]], np.float32)
                    if step < 0.6 * train_iter:
                        sess.run(self.train_op, feed_dict={self.xs1: train_x_strength, self.xs2: train_x_angle, self.ys: train_y, self.learning_rate: lr, self.keep_prob: KeepProb, self.state_flag: 1})
                    else:
                        sess.run(self.train_op,
                                 feed_dict={self.xs1: train_x_strength, self.xs2: train_x_angle, self.ys: train_y, self.learning_rate: lr * lr_decay, self.keep_prob: KeepProb, self.state_flag: 1})
                    if (step + 1) % report_freq == 0 or step == 0:
                        train_accu = sess.run(self.accuracy, feed_dict={self.xs1: train_x_strength, self.xs2: train_x_angle, self.ys: train_y, self.state_flag: 0})
                        if _ == (Num_train_batch - 1):
                            train_accu_tmp = (train_one_iter_accu + train_accu) / Num_train_batch
                            print('iter:%d' % (step + 1), 'train_accuracy:', '%2.5f' % train_accu_tmp)
                            train_accu_list.append(train_accu_tmp)
                            train_one_iter_accu = 0
                        else:
                            train_one_iter_accu = train_one_iter_accu + train_accu

                for _ in range(Num_vali_batch):
                    if (step + 1) % report_freq == 0 or step == 0:
                        offset = _ * batch_size
                        vali_x_strength = np.array(vali_X_strength[vali_indices[offset:offset + batch_size]], np.float32)
                        vali_x_angle = np.array(vali_X_angle[vali_indices[offset:offset + batch_size]], np.float32)
                        vali_y = np.array(vali_Y[vali_indices[offset:offset + batch_size]], np.float32)
                        vali_accu = sess.run(self.accuracy,
                                             feed_dict={self.xs1: vali_x_strength, self.xs2: vali_x_angle, self.ys: vali_y, self.state_flag: 0})
                        if _ == (Num_vali_batch - 1):
                            vali_accu_tmp = (vali_one_iter_accu + vali_accu) / Num_vali_batch
                            print('iter:%d' % (step + 1), 'vali_accuracy:',
                                  '%2.5f' % vali_accu_tmp)
                            step_list.append(step + 1)
                            vali_accu_list.append(vali_accu_tmp)
                            vali_one_iter_accu = 0
                        else:
                            vali_one_iter_accu = vali_one_iter_accu + vali_accu

                if (step + 1) % report_freq == 0 or step == 0:
                    saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step + 1)
                    time_list.append(time.asctime(time.localtime(time.time())))
                    print('every %d-steps time:' % report_freq, '%s' % time_list[-1])
                    df = pd.DataFrame(data={'time': time_list, 'step': step_list, 'train_accuracy': train_accu_list,
                                            'validation_accuracy': vali_accu_list})
                    df.to_csv(checkpoint_dir + 'accuracy.csv')
                    print('=========================================================')

    def test(self, ret, img):
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        # classifier = ['Jerry', 'Stranger']
        # # faceCascade = cv2.CascadeClassifier(cascPath)
        # print('start testing...')
        # print('=========================================================')
        # saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
        # # saver.restore(sess, checkpoint_dir + 'model.ckpt-%d' %train_iter)
        # saver.restore(sess, checkpoint_dir + 'model.ckpt-550')
        # self.cap = cv2.VideoCapture(CAM_FLAG)
        # while 1:
        # ret, img = self.cap.read()
        self.face_flag = 251
        if ret:
            # img = cv2.resize(img, (int(img.shape[1] * resize), int(img.shape[0] * resize)), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                landmark, shape, (x, y, w, h) = get_landmarks(gray)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # for _ in range(len(landmark)):
                #     if _ < 28:
                #         if _ % 4 == 0:
                #             cv2.line(img, (int(landmark[_]+x), int(landmark[_ + 1]+y)), (int(landmark[_ + 4]+x), int(landmark[_ + 5]+y)), (0, 255, 0), 2)
                strength = []
                angle = []
                for _ in range(len(landmark)):
                    # if _ % 4 == 0:
                    #     print('x = %d' % landmark[_])
                    # elif _ % 4 == 1:
                    #     print('y = %d' % landmark[_])
                    if _ % 4 == 2:
                        # print('strength = %d' % landmark[_])
                        strength.append(landmark[_])
                    elif _ % 4 == 3:
                        # print('angle = %d' % landmark[_])
                        angle.append(landmark[_] / 180)
                maxval, idx = Maximum(strength)
                strength = np.array(strength / maxval, np.float32)
                angle = np.array(angle, np.float32)
                strength = np.reshape(strength, [1, -1])
                angle = np.reshape(angle, [1, -1])
                pred = self.sess.run(self.prediction, feed_dict={self.xs1: strength, self.xs2: angle, self.state_flag: 0})
                maxval, index = Maximum(pred[0])
                if index == 0:
                    self.face_flag = 252
                    cv2.putText(img, self.classifier[0], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
                else:
                    self.face_flag = 251
                    cv2.putText(img, self.classifier[1], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for (_x, _y) in shape:
                    cv2.circle(img, (_x, _y), 1, (0, 0, 255), -1)
            except:
                print('no faces!!!')
            return self.face_flag
                # cv2.imshow('Face Recognition', img)
                # waitkey = cv2.waitKey(1)
                # if waitkey == ord('q') or waitkey == ord('Q'):
                #     cv2.destroyAllWindows()
                #     break


def get_detect(gray):
    try:
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return shape, (x, y, w, h)
    except:
        print("detect None")


def get_landmarks(gray):
    try:
        shape, (x, y, w, h) = get_detect(gray)
        xlist = []
        ylist = []
        frame = gray[y:y+h, x:x+w]
        for (_x, _y) in shape:
            xlist.append(_x)
            ylist.append(_y)
        left_eyeX = np.array(xlist[36:41])
        left_eyeY = np.array(ylist[36:41])
        right_eyeX = np.array(xlist[42:47])
        right_eyeY = np.array(ylist[42:47])
        left_center = (int(np.mean(left_eyeX)), int(np.mean(left_eyeY)))
        right_center = (int(np.mean(right_eyeX)), int(np.mean(right_eyeY)))
        # eye_center = (int((left_center[0] + right_center[0]) / 2), int((left_center[1] + right_center[1]) / 2))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        objpoints = np.array([(left_center[0] - x, left_center[1] - y),
                              (right_center[0] - x, right_center[1] - y),
                              (int(xmean) - x, int(ymean) - y)],
                             dtype=np.float32)
        imgpoints = np.array([(int(w * l_eye_pct), int(h * eyes_level_pct)),
                              (int(w * r_eye_pct), int(h * eyes_level_pct)),
                              (int(w / 2), int(h / 2))],
                             dtype=np.float32)
        M = cv2.getAffineTransform(objpoints, imgpoints)
        warped_image = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        objpoints = np.array([(int(warped_image.shape[1] * 0.1), int(warped_image.shape[0] * 0.33)),
                              (int(warped_image.shape[1] * 0.9), int(warped_image.shape[0] * 0.33)),
                              (int(warped_image.shape[1] * 0.9), int(warped_image.shape[0] * 0.66)),
                              (int(warped_image.shape[1] * 0.1), int(warped_image.shape[0] * 0.66))],
                             dtype=np.float32)
        imgpoints = np.array([(0, 0),
                              (int(warped_image.shape[1]), 0),
                              (int(warped_image.shape[1]), int(warped_image.shape[0])),
                              (0, int(warped_image.shape[0]))],
                             dtype=np.float32)
        M = cv2.getPerspectiveTransform(objpoints, imgpoints)
        warped_image = cv2.warpPerspective(warped_image, M, (warped_image.shape[1], warped_image.shape[0]))
        shape1, (x0, y0, w0, h0) = get_detect(warped_image)
        xlist = []
        ylist = []
        for (_x, _y) in shape1:  # 0~27 are face shape
            # cv2.circle(image, (_x, _y), 1, (0, 0, 255), -1)
            xlist.append(_x)
            ylist.append(_y)
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(_x - xmean) for _x in xlist]
        ycentral = [(_y - ymean) for _y in ylist]
        # cv2.circle(image, (int(xmean), int(ymean)), 2, (255, 255, 255), 0)

        landmarks_vectorised = []
        # landmarks_vectorised (x, y, length(point2central), angle)
        for _x, _y, _w, _z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(_w)
            landmarks_vectorised.append(_z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((_z, _w))
            dist = np.linalg.norm(coornp - meannp)  # find norm of vector
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(_y, _x) * 360) / (2 * math.pi))
        # for _ in range(len(landmarks_vectorised)):
        #   if _ % 4 == 0:
        #       cv2.line(image, (int(landmarks_vectorised[_]), int(landmarks_vectorised[_+1])), (int(xmean), int(ymean)), (0, 255, 0), 1)
        return landmarks_vectorised, shape, (x, y, w, h)
    except:
        print("no landmarks")


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# def HOG(img):
#     """
#     Histogram of oriented gradient
#     :param img: gray-scale image
#     :return: HOG image
#     """
#     img2 = np.zeros_like(img)
#     cell_size = (8, 8)  # h x w in pixels
#     block_size = (2, 2)  # h x w in cells
#     nbins = 9  # number of orientation bins
#
#     # winSize is the size of the image cropped to an multiple of the cell size
#     hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
#                                       img.shape[0] // cell_size[0] * cell_size[0]),
#                             _blockSize=(block_size[1] * cell_size[1],
#                                         block_size[0] * cell_size[0]),
#                             _blockStride=(cell_size[1], cell_size[0]),
#                             _cellSize=(cell_size[1], cell_size[0]),
#                             _nbins=nbins)
#
#     n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
#     hog_feats = hog.compute(img) \
#         .reshape(n_cells[1] - block_size[1] + 1,
#                  n_cells[0] - block_size[0] + 1,
#                  block_size[0], block_size[1], nbins) \
#         .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
#     # hog_feats now contains the gradient amplitudes for each direction,
#     # for each cell of its group for each group. Indexing is by rows then columns.
#
#     gradients = np.zeros((n_cells[0], n_cells[1], nbins))
#
#     # count cells (border cells appear less often across overlapping groups)
#     cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
#
#     for off_y in range(block_size[0]):
#         for off_x in range(block_size[1]):
#             gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
#             off_x:n_cells[1] - block_size[1] + off_x + 1] += \
#                 hog_feats[:, :, off_y, off_x, :]
#             cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
#             off_x:n_cells[1] - block_size[1] + off_x + 1] += 1
#
#     # Average gradients
#     gradients /= cell_count
#     for off_y in range(gradients.shape[0]):
#         for off_x in range(gradients.shape[1]):
#             drawX = off_x * cell_size[1]
#             drawY = off_y * cell_size[0]
#             mx = drawX + cell_size[1] / 2
#             my = drawY + cell_size[0] / 2
#             # cv2.rectangle(img2, (drawX, drawY), (drawX + cell_size[1], drawY + cell_size[0]), (255, 255, 255), 1)
#             for bins in range(nbins):
#                 currentGradStrength = gradients[off_y][off_x][bins]
#                 if currentGradStrength == 0:
#                     continue
#                 currRad = bins * (math.pi / nbins) + (math.pi / nbins) / 2
#                 dirVecX = math.cos(currRad)
#                 dirVecY = math.sin(currRad)
#                 maxVecLen = float(cell_size[0] / 2.)
#                 scale = 2.5
#                 x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale
#                 y1 = my - dirVecY * currentGradStrength * maxVecLen * scale
#                 x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale
#                 y2 = my + dirVecY * currentGradStrength * maxVecLen * scale
#                 cv2.line(img2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
#     return img2


def Maximum(input_array):
    """find max in 1D-array"""
    max_val = input_array[0]
    idx = 0
    for i in range(len(input_array)):
        if input_array[i] > max_val:
            max_val = input_array[i]
            idx = i
    return max_val, idx

if __name__ == '__main__':
    nn = Neural_Network()
    # nn.train()
    # time.sleep(5)
    nn.test()
