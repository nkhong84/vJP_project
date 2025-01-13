import pickle
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import random
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import r2_score

from collections import defaultdict


class Jump:
    
    def __int__(self,data,fps,dw,dh):
        self.sklt = np.array(data)
        self.fps = fps
        self.dw = dw
        self.dh = dh


    def point_order(self, v):
        x1,x2,x3,x4 = v["x1"],v["x2"],v["x3"],v["x4"]
        y1,y2,y3,y4 = v["y1"],v["y2"],v["y3"],v["y4"]

        points = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]

        sum_np = np.array(points).sum(axis=-1)

        left_top = points[np.argmin(sum_np)]
        right_bottom = points[np.argmax(sum_np)]

        points = [p for p in points if p != left_top and p != right_bottom]

        if points[0][0] < points[1][0]:
            right_top = points[1]
            left_bottom = points[0]
        else:
            right_top = points[0]
            left_bottom = points[1]

        return left_top,left_bottom,right_bottom,right_top
    

    def distance(self,p1,p2):
        return ((((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))**0.5)

    def distance_x(self,p1,p2):
        return abs(p2[0]-p1[0])

    def distance_y(self,p1,p2):
    #     return abs(p2[1]-p1[1])
        return p2[1]-p1[1]
    
    def line_intersection(self,line1,line2):
        xdiff = (line1[0][0]-line1[1][0],line2[0][0]-line2[1][0])
        ydiff = (line1[0][1]-line1[1][1],line2[0][1]-line2[1][1])

        def det(a,b):
            return a[0]*b[1] - a[1]*b[0]

        div = det(xdiff,ydiff)
        if div == 0:
            raise Exception("lines do not intersect")

        d = (det(*line1),det(*line2))
        x = det(d,xdiff) / div
        y = det(d,ydiff) / div

        return int(x),int(y)
    def midpoint(self,p1,p2):
        return int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2)
    def cal_cob(self,v,i):
        ps = {"x1":val[i,0,5,0],"y1":val[i,0,5,1],"x2":val[i,0,6,0],"y2":val[i,0,6,1],"x3":val[i,0,11,0],"y3":val[i,0,11,1],"x4":val[i,0,12,0],"y4":val[i,0,12,1]}
        lt,lb,rb,rt = self.point_order(ps)
        cross_point = sefl.line_intersection([lt,rb],[lb,rt])
        hip_mid_point = self.midpoint(lb,rb)

        x,y = self.midpoint(cross_point,hip_mid_point)

        return x,y