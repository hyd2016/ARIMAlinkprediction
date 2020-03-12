# coding=utf-8
import numpy as np
import networkx as nx
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from scipy import stats
import sys


class Graph:
    def __init__(self, nx_G, list_G, test_G, is_directed):
        self.G = nx_G
        self.is_directed = is_directed
        self.list_G = list_G
        self.test_G = test_G
        self.n = len(nx_G)
        self.edge_info = []

    def CN(self):
        Matrix_similarity = np.zeros((self.n, self.n))
        max_num = -1
        for i, x in enumerate(self.G.nodes()):
            for j, y in enumerate(self.G.nodes()):
                common_neighbors = len(list(nx.common_neighbors(self.G, x, y)))
                if common_neighbors > max_num:
                    max_num = common_neighbors
                Matrix_similarity[i][j] = common_neighbors

        return Matrix_similarity / max_num

    def arima_s(self):
        """
        计算arima矩阵St
        edges_series:矩阵边值序列
        edge_feature:边特征矩阵，大小为(n*n, 7)
        :return: None
        """
        n = len(self.G.nodes())
        St = np.zeros((n, n))

        edges_series = []
        index = 0
        for i, x in enumerate(self.G.nodes()):
            for j, y in enumerate(self.G.nodes()):
                self.edge_info.append((x, y))
                if (x, y) not in self.G.edges():
                    continue
                edge_series = []
                for g in self.list_G:
                    if (x, y) in g.edges():
                        edge_series.append(float(g[x][y]['weight']))
                    else:
                        edge_series.append(0.0)
                edges_series.append(edge_series)
                real_p, real_d, real_q = 1, 1, 1
                aic = sys.maxint
                # for p in range(1, 3):
                #     for d in range(0, 1):
                #         for q in range(1, 3):
                #             try:
                #                 model = ARIMA(edge_series, order=(p, d, q)).fit(disp=0)
                #                 aic_cur = model.aic
                #             except:
                #                 continue
                #             if aic_cur < aic:
                #                 aic, real_p, real_d, real_q = aic_cur, p, d, q
                try:
                    model = ARIMA(edge_series, order=(real_p, real_d, real_q)).fit(disp=0)
                    forecast, fcerr, conf_int = model.forecast(1)
                    St[i][j] = stats.norm.pdf(1, forecast, fcerr)
                except:
                    continue
        return St

    def predict(self, alpha):
        S = np.zeros((self.n, self.n))
        Ss = self.CN()
        St = self.arima_s()
        Ss = Ss / np.sum(Ss)
        St = St / np.sum(St)
        ms = np.flatnonzero(Ss).min()
        mt = np.flatnonzero(St).min()
        for i in range(self.n):
            for j in range(self.n):
                S[i][j] = (Ss[i][j] + ms / alpha) * (St[i][j] + mt / alpha)
        fpr, tpr, thresholds = metrics.roc_curve(np.array(nx.adjacency_matrix(self.test_G).todense()).flatten(),
                                                 S.flatten(), pos_label=1)
        plt.plot(fpr, tpr, marker='o')
        plt.show()
        auc_score = roc_auc_score(np.array(nx.adjacency_matrix(self.test_G).todense()).flatten(), S.flatten())
        print auc_score
        return
