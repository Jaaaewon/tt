import numpy as np
import tensorflow as tf
import csv
import os
import cv2
import re
import tempfile
import shutil
from tensorflow.contrib.layers import batch_norm, flatten
import matplotlib.pyplot as plt


class saveCsvF:
    def __init__(self, data, tdata):
        self.data = {
            'path': [],
            'label': []
        }
        self.tdata = {
            'path': [],
            'label': []
        }
        self.data['path'] = data['path']
        self.data['label'] = data['label']
        self.tdata['path'] = tdata['path']
        self.tdata['label'] = tdata['label']

    def saveCsv(self, csvPath='./save.csv', tcsvPath='./tsave.csv'):
        try:
            f = open(csvPath, 'w', encoding='euc_kr', newline='')
            wr = csv.writer(f)
            for i, _ in enumerate(self.data['path']):
                wr.writerow([i + 1, self.data['path'][i], self.data['label'][i]])
            f.close()
        except:
            print("No File")
        try:
            f = open(tcsvPath, 'w', encoding='euc_kr', newline='')
            wr = csv.writer(f)
            for i, _ in enumerate(self.tdata['path']):
                wr.writerow([i + 1, self.tdata['path'][i], self.tdata['label'][i]])
            f.close()
        except:
            print("No File")


def prepareData(datasetFolder='./image-data/trainset', testsetFolder='./image-data/testset', maxAug=150):
    data = {
        'path': [],
        'label': []
    }
    tdata = {
        'path': [],
        'label': []
    }
    for img in os.listdir(datasetFolder):
        if os.path.isfile(datasetFolder + "/" + img):
            data['path'].append(datasetFolder + "/" + img)
            data['label'].append(img[0:1])
            # print(img)
            # imgs.append((img))
    for img in os.listdir(testsetFolder):
        if os.path.isfile(testsetFolder + "/" + img):
            tdata['path'].append(testsetFolder + "/" + img)
            tdata['label'].append(img[0:1])
            # print(img)
            # imgs.append((img))
    return data, tdata

training_epochs = 50  # 전체 데이터셋을 train하는 횟수
num_models = 1  # 모델 개수
batch_size = 100  # 메모리에 올리는 batch data의 개수
y_size = 14  # y클래스 개수
#learning_rate = 0.007
epsilon = 1e-8
dropout_rate = 0.3
growth_rate = 12  # feature map의 channel 개수
nb_block = 2  # dense block + transition layer 개수

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def conv_layer(self, input, filter, kernel_size, strides=1, padding='SAME'):
        return tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel_size, padding=padding, strides=strides,
                                activation=tf.nn.relu)

    def batch_normalization(self, input, training):
        return tf.layers.batch_normalization(inputs=input, center=True, scale=True, training=training)

    def drop_out(self, input, rate, training):
        return tf.layers.dropout(inputs=input, rate=rate, training=training)

    def relu(self, input):
        return tf.nn.relu(input)

    def linear(self, input):
        return tf.layers.dense(inputs=input, units=y_size, name='linear')

    def max_pooling(self, input, pool_size, padding='VALID', strides=2):
        return tf.layers.max_pooling2d(inputs=input, pool_size=pool_size, strides=strides, padding=padding)

    def average_pooling(self, input, pool_size=[2, 2], strides=2, padding='VALID'):
        return tf.layers.average_pooling2d(inputs=input, pool_size=pool_size, strides=strides, padding=padding)

    def concatenation(self, layers):
        return tf.concat(layers, axis=3)

    def bottleneck_layer(self, input):
        x = self.batch_normalization(input=input, training=self.training)
        x = self.relu(x)
        x = self.conv_layer(input=x, filter=4 * growth_rate, kernel_size=1)

        x = self.batch_normalization(input=x, training=self.training)
        x = self.relu(x)
        x = self.conv_layer(x, filter=growth_rate, kernel_size=3)
        x = self.drop_out(x, rate=dropout_rate, training=self.training)

        return x

    def transition_layer(self, input):
        x = self.batch_normalization(input=input, training=self.training)
        x = self.relu(x)
        channel = x.shape[-1]
        x = self.conv_layer(input=x, filter=channel, kernel_size=1)
        x = self.drop_out(input=x, rate=dropout_rate, training=self.training)
        x = self.average_pooling(input=x)

        return x

    def dense_block(self, input, nb_layers):
        layers_concat = list()
        layers_concat.append(input)

        x = self.bottleneck_layer(input)
        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = self.concatenation(layers_concat)
            x = self.bottleneck_layer(x)
            layers_concat.append(x)

        x = self.concatenation(layers_concat)

        return x

    def global_average_pooling(self, input, stride=1):
        width = np.shape(input)[1]
        height = np.shape(input)[2]
        pool_size = [width, height]
        return tf.layers.average_pooling2d(inputs=input, pool_size=pool_size, strides=stride)
        # The stride value does not matter

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])

            self.Y = tf.placeholder(tf.int32, [None, 1])
            self.Y_one = tf.one_hot(self.Y, 14)
            self.Y_one = tf.reshape(self.Y_one, [-1, 14])
            self.training = tf.placeholder(tf.bool)

            x = self.conv_layer(input=X_img, filter=2 * growth_rate, kernel_size=7, strides=2)
            x = self.max_pooling(input=x, pool_size=3, strides=2)

            for i in range(nb_block):
                x = self.dense_block(input=x, nb_layers=4)
                x = self.transition_layer(input=x)

            x = self.dense_block(input=x, nb_layers=32)

            x = self.batch_normalization(input=x, training=self.training)
            x = self.relu(x)
            x = self.global_average_pooling(x)
            x = flatten(x)

            self.logits = self.linear(x)

            global_step = tf.Variable(0, trainable=False, name='global_step')
            starter_learning_rate = 0.01
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       100000, 0.96, staircase=True)
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y_one))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=global_step)
            is_correct = tf.equal(tf.math.argmax(self.logits, 1), tf.math.argmax(self.Y_one, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            self.merged = tf.summary.merge_all()

    def get_pred(self, x_data, training=False):
        return self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: x_data, self.training: training})

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

    def logi(self, x_data, training=False):
        return self.sess.run(self.logits , feed_dict={self.X: x_data, self.training: training})

    def saveM(self):
        if self.sess:
            saver = tf.train.Saver()
            saver.save(self.sess, './save/saved.cpkt')
            print("저장")

    def restoreM(self):
        if self.sess:
            saver = tf.train.Saver()
            saver.restore(self.sess, "./save/saved.cpkt")
            print("불러오기")


def imreadEX(image_path):
    if re.compile('[^ㄱ-ㅣ가-힣]+').sub('', image_path):
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        if not img is None:
            return img
        else:
            file_tmp = tempfile.NamedTemporaryFile().name
            shutil.copy(image_path, file_tmp)
            image_path = file_tmp
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img


def loadImg(csvPath='./save.csv', tcsvPath='./tsave.csv'):
    data = {
        'path': [],
        'label': []
    }
    trainData = {
        'img': [],
        'label': []
    }
    testData = {
        'img': [],
        'label': []
    }
    with open(csvPath, 'rt') as f:
        reader = csv.reader(f, delimiter=',')

        for csvData in reader:
            data['path'].append(csvData[1])
            data['label'].append(csvData[2])
        trainData['label'] = data['label']
        for i, _ in enumerate(data['path']):
            readImg = imreadEX(data['path'][i])
            # a = cv2.imread(data['path'][i],cv2.IMREAD_COLOR)
            readImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2GRAY)
            # print(readImg.shape)
            reshapeImg = cv2.resize(readImg, (28, 28), interpolation=cv2.INTER_AREA)
            reshapeImg = np.reshape(reshapeImg, 784).astype(float)
            print(np.max(reshapeImg))

            reshapeImg /= np.max(reshapeImg)
            print(np.max(reshapeImg))
            exit(0)
            trainData['img'].append(reshapeImg)
    data = {
        'path': [],
        'label': []
    }
    with open(tcsvPath, 'rt') as f:
        reader = csv.reader(f, delimiter=',')

        for csvData in reader:
            data['path'].append(csvData[1])
            data['label'].append(csvData[2])
        testData['label'] = data['label']
        for i, _ in enumerate(data['path']):
            readImg = imreadEX(data['path'][i])
            # a = cv2.imread(data['path'][i],cv2.IMREAD_COLOR)
            readImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2GRAY)
            # print(readImg.shape)
            reshapeImg = cv2.resize(readImg, (28, 28), interpolation=cv2.INTER_AREA)
            reshapeImg = np.reshape(reshapeImg, 784).astype(float)
            reshapeImg /= np.max(reshapeImg)
            testData['img'].append(reshapeImg)

    return trainData, testData


'''
def dataTrain(trainData,testData):
    if not trainData:
        print("없는데?")
        return
    print("있는데?")
    sess = tf.Session()
    models = []
    num_models = 7
    for m in range(num_models):
        models.append(Model(sess, "model" + str(m)))

    sess.run(tf.global_variables_initializer())
    print('Learning Started')


    for epoch in range(training_epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(len(trainData['img']) / batch_size)

        # for i in range(total_batch):
            #batch_xs, batch_ys = trainData['img'][](batch_size)

        batch_xs = trainData['img']
        batch_ys = trainData['label']
        tbatch_xs = testData['img']
        tbatch_ys = testData['label']
        # string 을 Y로 줄 수가 없네?
        for i,_ in enumerate(batch_ys):
            if batch_ys[i] == '가':
                batch_ys[i] = 0
            if batch_ys[i] == '나':
                batch_ys[i] = 1
            if batch_ys[i] == '다':
                batch_ys[i] = 2
        batch_ys = np.reshape(batch_ys, [len(batch_ys), 1])
        for m_index, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_index] += c
            #avg_cost_list[m_index] += c / total_batch

        # print('\r%d' % (i / total_batch * 100) + '%', end='')

        print('\rEpoch: ', '%04d' % (epoch + 1), 'cost = ', avg_cost_list)
    #for i, _ in enumerate(batch_ys):
        #if batch_ys[i] == 1:
        #    batch_ys[i] = '가'
        #if batch_ys[i] == 2:
        #    batch_ys[i] = '나'
        #if batch_ys[i] == 3:
        #    batch_ys[i] = '다'   
    best_m =[]
    print('Learning Finished')

    for i, _ in enumerate(tbatch_ys):
        if tbatch_ys[i] == '가':
            tbatch_ys[i] = 0
        if tbatch_ys[i] == '나':
            tbatch_ys[i] = 1
        if tbatch_ys[i] == '다':
            tbatch_ys[i] = 2
    tbatch_ys = np.reshape(tbatch_ys, [len(tbatch_ys), 1])
    test_size = len(tbatch_ys)
    predictions = np.zeros(test_size*3).reshape(test_size,3)
    for m_index, m in enumerate(models):
        best_m.append(m.get_accuracy(tbatch_xs, tbatch_ys))
        print("Accuracy: ", best_m[m_index])
        p = m.predict(tbatch_xs)
        predictions += p
    ensemble_correct_prediction = tf.equal(
        tf.argmax(predictions, 1), tf.argmax(tbatch_ys, 1))
    ensemble_accuracy = tf.reduce_mean(
        tf.cast(ensemble_correct_prediction, tf.float32))
    print('Ensemble accuracy:', sess.run(ensemble_accuracy))

    #best_model = m[np.argmax(best_m)]
    #saver = tf.train.Saver()
    #saver.save(best_model.sess , "./saved.cpkt")
'''


def loadTrained(sess):
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "./save/saved.cpkt")


def noEnsem(trainData, testData):
    if not trainData:
        print("없는데?")
        return
    print("있는데?")
    sess = tf.Session()
    m = Model(sess, 'first')
    sess.run(tf.global_variables_initializer())
    batch_xs = trainData['img']
    batch_ys = trainData['label']
    tbatch_xs = testData['img']
    tbatch_ys = testData['label']

    for i, _ in enumerate(batch_ys):
        if batch_ys[i] == '가':
            batch_ys[i] = 0
        if batch_ys[i] == '나':
            batch_ys[i] = 1
        if batch_ys[i] == '다':
            batch_ys[i] = 2
        if batch_ys[i] == '라':
            batch_ys[i] = 3
        if batch_ys[i] == '마':
            batch_ys[i] = 4
        if batch_ys[i] == '바':
            batch_ys[i] = 5
        if batch_ys[i] == '사':
            batch_ys[i] = 6
        if batch_ys[i] == '아':
            batch_ys[i] = 7
        if batch_ys[i] == '자':
            batch_ys[i] = 8
        if batch_ys[i] == '차':
            batch_ys[i] = 9
        if batch_ys[i] == '카':
            batch_ys[i] = 10
        if batch_ys[i] == '타':
            batch_ys[i] = 11
        if batch_ys[i] == '파':
            batch_ys[i] = 12
        if batch_ys[i] == '하':
            batch_ys[i] = 13

    for i, _ in enumerate(tbatch_ys):
        if tbatch_ys[i] == '가':
            tbatch_ys[i] = 0
        if tbatch_ys[i] == '나':
            tbatch_ys[i] = 1
        if tbatch_ys[i] == '다':
            tbatch_ys[i] = 2
        if tbatch_ys[i] == '라':
            tbatch_ys[i] = 3
        if tbatch_ys[i] == '마':
            tbatch_ys[i] = 4
        if tbatch_ys[i] == '바':
            tbatch_ys[i] = 5
        if tbatch_ys[i] == '사':
            tbatch_ys[i] = 6
        if tbatch_ys[i] == '아':
            tbatch_ys[i] = 7
        if tbatch_ys[i] == '자':
            tbatch_ys[i] = 8
        if tbatch_ys[i] == '차':
            tbatch_ys[i] = 9
        if tbatch_ys[i] == '카':
            tbatch_ys[i] = 10
        if tbatch_ys[i] == '타':
            tbatch_ys[i] = 11
        if tbatch_ys[i] == '파':
            tbatch_ys[i] = 12
        if tbatch_ys[i] == '하':
            tbatch_ys[i] = 13
    #print(tbatch_ys)
    #print(tf.one_hot(tbatch_ys,depth=14).eval(session=sess))
    '''
    b = tf.one_hot(tbatch_ys,depth=14).eval(session=sess)
    a = tf.one_hot(batch_ys,depth=14).eval(session=sess)'''
    tbatch_ys = np.reshape(tbatch_ys, [len(tbatch_ys), 1])
    batch_ys = np.reshape(batch_ys, [len(batch_ys), 1])


    writer = tf.summary.FileWriter("./logs/hello_tf_180115-1")
    writer.add_graph(sess.graph)  # Show the graph

    print('Learning Started')
    for epoch in range(training_epochs):
        c, _ = m.train(batch_xs, batch_ys)
        #s, _, _ = m.train(batch_xs, batch_ys)
        print('\rEpoch: ', '%04d' % (epoch + 1), 'cost = ', c)
        #writer.add_summary(s, global_step=epoch)

    #test_size = len(tbatch_ys)
    #predictions = np.zeros(test_size * 3).reshape(test_size, 3)
    print("Accuracy: ", m.get_accuracy(tbatch_xs, tbatch_ys))

    m.saveM()
    print('Learning Finished')
    '''
    imgPaths = ['./imagesave/라.png', './imagesave/사_297.jpeg', './imagesave/KakaoTalk_20190812_142617889.jpg']
    a = testImg(imgPaths)
    pre = m.get_pred(a)
    print(pre)
    for i in pre:
        if i == 0:
            print("가")
        if i == 1:
            print("나")
        if i == 2:
            print("다")
        if i == 3:
            print("라")
        if i == 4:
            print("마")
        if i == 5:
            print("바")
        if i == 6:
            print("사")
        if i == 7:
            print("아")
        if i == 8:
            print("자")
        if i == 9:
            print("차")
        if i == 10:
            print("카")
        if i == 11:
            print("타")
        if i == 12:
            print("파")
        if i == 13:
            print("하")'''


def testImg(imgPaths):
    imgs = []
    for imgPath in imgPaths:
        readImg = imreadEX(imgPath)
        # a = cv2.imread(data['path'][i],cv2.IMREAD_COLOR)
        readImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2GRAY)
        # print(readImg.shape)
        reshapeImg = cv2.resize(readImg, (28, 28), interpolation=cv2.INTER_AREA)
        reshapeImg = np.reshape(reshapeImg, 784).astype(float)
        reshapeImg /= np.max(reshapeImg)
        imgs.append(reshapeImg)
    return imgs


def savertest(testFolder='./test'):
    sess = tf.Session()
    m = Model(sess, 'first')
    sess.run(tf.global_variables_initializer())
    m.restoreM()
    datas = []
    for img in os.listdir(testFolder):
        if os.path.isfile(testFolder + "/" + img):
            datas.append(testFolder + "/" + img)
            print(datas)
        else:
            print('없네?')
            # print(img)
            # imgs.append((img))
    dropout_rate = 1.0
    a = testImg(datas)
    #print(a[0])
    pre = m.get_pred(a)
    lo = m.logi(a)
    print(pre)
    logits = []
    print(lo)
    for i in pre:
        if i == 0:
            logits.append("가")
        if i == 1:
            logits.append("나")
        if i == 2:
            logits.append("다")
        if i == 3:
            logits.append("라")
        if i == 4:
            logits.append("마")
        if i == 5:
            logits.append("바")
        if i == 6:
            logits.append("사")
        if i == 7:
            logits.append("아")
        if i == 8:
            logits.append("자")
        if i == 9:
            logits.append("차")
        if i == 10:
            logits.append("카")
        if i == 11:
            logits.append("타")
        if i == 12:
            logits.append("파")
        if i == 13:
            logits.append("하")
    print(logits)


def main():
    # 저장 없으면 이걸로

    #data, tdata = prepareData()
    #sf = saveCsvF(data, tdata)
    #sf.saveCsv()
    trainData, testData = loadImg()
    noEnsem(trainData,testData)

    # dataTrain(trainData, testData)

    # 저장된거 있으면 이걸로
    #savertest()

    # loadTrained()


if __name__ == '__main__':
    main()
