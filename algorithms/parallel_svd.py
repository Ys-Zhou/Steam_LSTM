import multiprocessing as mp
import math
import random
import time
from dbconnector import GetCursor


class SvdProc(mp.Process):
    def __init__(self, inTg, inLock, inSvdNowStep, inAvgNowStep, inSvdItemFeature, inAvgItemFeature, inDescentRate):
        super(SvdProc, self).__init__()
        self.tg = inTg
        self.train = {}
        self.userList = []
        self.itemList = []
        self.user_feature = {}
        self.item_feature = {}
        # Load server values
        self.lock = inLock
        self.svdNowStep = inSvdNowStep
        self.avgNowStep = inAvgNowStep
        self.svdItemFeature = inSvdItemFeature
        self.avgItemFeature = inAvgItemFeature
        self.descentRate = inDescentRate
        # Local attributes
        self.feature = 20
        self.processNum = 3

    def load_data(self):
        with GetCursor() as cur:
            query = 'SELECT DISTINCT userid FROM traindata_yep'
            cur.execute(query)
            res = list(zip(*cur.fetchall()))[0]
            self.userList = list(map(str, res))

            query = 'SELECT DISTINCT gameid FROM traindata_yep'
            cur.execute(query)
            res = list(zip(*cur.fetchall()))[0]
            self.itemList = list(map(str, res))

            query = 'SELECT userid, gameid, rating FROM traindata_yep WHERE tg = %d AND rating != 0' % self.tg
            cur.execute(query)
            for item in cur:
                self.train.setdefault(str(item[0]), {})
                self.train[str(item[0])][str(item[1])] = float(item[2])

    def initial_feature(self):
        random.seed(0)
        for i in self.userList:
            self.user_feature.setdefault(i, {})
            for j in range(1, self.feature + 1):
                self.user_feature[i].setdefault(j, random.uniform(0.1, 1))

        for i in self.itemList:
            self.item_feature.setdefault(i, {})
            for j in range(1, self.feature + 1):
                self.item_feature[i].setdefault(j, random.uniform(0.1, 1))

    def savetxt(self):
        file_name = 'user_feature_tg%d_yep.txt' % self.tg
        f = open(file_name, 'w+')
        for i in self.userList:
            f.write('%s' % i)
            for j in range(1, self.feature + 1):
                f.write('\t%f' % self.user_feature[i][j])
            f.write('\n')
        f.close()

    def run(self):
        self.load_data()
        print('Process-%d: load data success' % self.tg)

        self.initial_feature()
        print('Process-%d: initial user and item feature, respectly success' % self.tg)

        # SVD parameters
        gama = 0.02
        lamda = 0.1
        slowRate = 0.01
        stopAt = 0.001

        # Save rmse
        fileName = 'rmse_tg%d_yep.txt' % self.tg
        f = open(fileName, 'w+')

        step = 0
        preRmse = 1000000000.0
        nowRmse = 0.0

        while step < 200:  # max step
            rmse = 0.0
            n = 0
            for u in self.train.keys():
                for i in self.train[u].keys():
                    pui = 0
                    for k in range(1, self.feature + 1):
                        pui += self.user_feature[u][k] * self.item_feature[i][k]
                    eui = self.train[u][i] - pui
                    rmse += pow(eui, 2)
                    n += 1
                    for k in range(1, self.feature + 1):
                        self.user_feature[u][k] += gama * (
                                eui * self.item_feature[i][k] - lamda * self.user_feature[u][k])
                        self.item_feature[i][k] += gama * (
                                eui * self.user_feature[u][k] - lamda * self.item_feature[i][k])

            nowRmse = math.sqrt(rmse * 1.0 / n)
            dRate = 1 - nowRmse / preRmse
            f.write('%f\n' % nowRmse)
            print('Process-%d: step: %d      Rmse: %f      %f' % (self.tg, (step + 1), nowRmse, dRate))

            preRmse = nowRmse
            gama *= (1 - slowRate)
            step += 1

            while True:
                if self.lock.acquire():
                    if self.svdNowStep[self.tg - 1] < step:
                        self.descentRate[self.tg - 1] = dRate
                        self.svdItemFeature[self.tg - 1] = self.item_feature
                        self.svdNowStep[self.tg - 1] = step
                    if map(lambda x: x < stopAt, self.descentRate) == [True] * self.processNum:
                        self.lock.release()
                        f.close()
                        self.savetxt()
                        print('Process-%d: svd + stochastic gradient descent success' % self.tg)
                        return
                    if self.avgNowStep.value == step:
                        self.item_feature.update(self.avgItemFeature)
                        self.lock.release()
                        break
                    self.lock.release()
                    time.sleep(1)


class AvgProc(mp.Process):
    def __init__(self, inLock, inSvdNowStep, inAvgNowStep, inSvdItemFeature, inAvgItemFeature, inDescentRate):
        super(AvgProc, self).__init__()
        # Load server values
        self.lock = inLock
        self.svdNowStep = inSvdNowStep
        self.avgNowStep = inAvgNowStep
        self.svdItemFeature = inSvdItemFeature
        self.avgItemFeature = inAvgItemFeature
        self.descentRate = inDescentRate
        # Local attributes
        self.feature = 20
        self.processNum = 3
        self.stopAt = 0.001

    def run(self):
        with GetCursor() as cur:
            query = 'SELECT DISTINCT gameid FROM traindata_yep'
            cur.execute(query)
            res = list(zip(*cur.fetchall()))[0]
            itemList = list(map(str, res))

        while True:
            if self.lock.acquire():
                if list(self.svdNowStep) == [self.avgNowStep.value + 1] * self.processNum:
                    lSvdItemFeature = [None] * self.processNum
                    for t in range(self.processNum):
                        lSvdItemFeature[t] = self.svdItemFeature[t]
                    lAvgItemFeature = {}
                    for i in itemList:
                        lAvgItemFeature.setdefault(i, {})
                        for j in range(1, self.feature + 1):
                            value = 0.0
                            for t in range(self.processNum):
                                value += lSvdItemFeature[t][i][j]
                            lAvgItemFeature[i].setdefault(j, value / float(self.processNum))
                    self.avgItemFeature.update(lAvgItemFeature)
                    self.avgNowStep.value += 1
                    print('Process-avg: next step')
                if map(lambda x: x < self.stopAt, self.descentRate) == [True] * self.processNum:
                    self.lock.release()
                    return
                self.lock.release()
            time.sleep(1)


if __name__ == '__main__':
    # Main process values
    feature = 20
    processNum = 3

    # Process locker
    lock = mp.Lock()

    # Server Process
    manager = mp.Manager()
    svdNowStep = manager.list([None] * processNum)
    avgNowStep = manager.Value('i', 0)
    svdItemFeature = manager.list([None] * processNum)
    avgItemFeature = manager.dict()
    descentRate = manager.list([1.0] * processNum)

    # Start all processes
    svd1 = SvdProc(1, lock, svdNowStep, avgNowStep, svdItemFeature, avgItemFeature, descentRate)
    svd1.daemon = True
    svd1.start()

    svd2 = SvdProc(2, lock, svdNowStep, avgNowStep, svdItemFeature, avgItemFeature, descentRate)
    svd2.daemon = True
    svd2.start()

    svd3 = SvdProc(3, lock, svdNowStep, avgNowStep, svdItemFeature, avgItemFeature, descentRate)
    svd3.daemon = True
    svd3.start()

    avg = AvgProc(lock, svdNowStep, avgNowStep, svdItemFeature, avgItemFeature, descentRate)
    avg.daemon = True
    avg.start()

    svd1.join()
    svd2.join()
    svd3.join()
    avg.join()
