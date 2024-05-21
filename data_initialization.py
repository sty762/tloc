import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import math

from configurations.config import *

lonStep_1m = 0.0000105
latStep_1m = 0.0000090201

class RoadGrid:
    def __init__(self, label, grid_size):
        length = grid_size*latStep_1m
        width = grid_size*lonStep_1m
        self.length = length
        self.width = width
        def orginal_plot(label):
            tr = np.max(label,axis=0)
            tr[0]+=25*lonStep_1m
            tr[1]+=25*latStep_1m
            # plot(label[:,0], label[:,1], 'b,')
            bl = np.min(label,axis=0)
            bl[0]-=25*lonStep_1m
            bl[1]-=25*latStep_1m

            # width = (tr[1]-bl[1])/100
            # wnum =int(np.ceil((tr[1]-bl[1])/length))
            # for j in range(wnum):
                # hlines(y = bl[1]+length*j, xmin = bl[0], xmax = tr[0], color = 'red')

            # lnum = int(np.ceil((tr[0]-bl[0])/width))
            # for j in range(lnum):
                # vlines(x = bl[0]+width*j, ymin = bl[1], ymax = tr[1], color = 'red')
            return bl[0], tr[0], bl[1], tr[1]
        

        xl,xr,yb,yt = orginal_plot(label)
        self.xl = xl
        self.xr = xr
        self.yb = yb
        self.yt = yt
        gridSet = set()
        grid_dict = {}
        for pos in label:
            lon = pos[0]
            lat = pos[1]

            m = int((lon-xl)/width)
            n = int((lat-yb)/length)
            if (m,n) not in grid_dict:
                grid_dict[(m,n)] = []
            grid_dict[(m,n)].append((lon, lat))
            gridSet.add((m,n))
        # print len(gridSet)
        gridlist = list(gridSet)

        grid_center = [tuple(np.mean(np.array(lonlat_list),axis=0)) for (i,j), lonlat_list in grid_dict.items()]

        # for gs in gridSet:
            # xlon = xl+gs[0]*width
            # ylat = yb+gs[1]*length
            # bar(xlon,length,width,ylat,color='#7ED321')
        self.gridlist = gridlist

        self.grids = [(xl+i[0]*width,yb + i[1]*length) for i in grid_dict.keys()]
        self.grid_center=grid_center
        self.n_grid = len(self.grid_center)

    def transform(self, label, sparse=True):
        def one_hot(idx, n):
            a = [0] * n
            a[idx] = 1
            return a
        grid_pos = [self.gridlist.index((int((i[0]-self.xl)/self.width),int((i[1]-self.yb)/self.length))) for i in label]
        if sparse:
            grid_pos = np.array([one_hot(x, len(self.gridlist)) for x in grid_pos], dtype=np.int32)
        return grid_pos

col_name_new = [
    #'MRTime',
    'RNCID_1',
    'CellID_1',
    'AsuLevel_1',
    # 'Dbm_1',
    'SignalLevel_1',
    'RNCID_2',
    'CellID_2',
    'AsuLevel_2',
    # 'Dbm_2',
    'SignalLevel_2',
    'RNCID_3',
    'CellID_3',
    'AsuLevel_3',
    # 'Dbm_3',
    'SignalLevel_3',
    'RNCID_4',
    'CellID_4',
    'AsuLevel_4',
    # 'Dbm_4',
    'SignalLevel_4',
    'RNCID_5',
    'CellID_5',
    'AsuLevel_5',
    # 'Dbm_5',
    'SignalLevel_5',
    'RNCID_6',
    'CellID_6',
    'AsuLevel_6',
    # 'Dbm_6',
    'SignalLevel_6',
    'RNCID_7',
    'CellID_7',
    'AsuLevel_7',
    # 'Dbm_7',
    'SignalLevel_7',
    #'RSSI_6',
]

def make_rf_dataset(data, eng_para):
    for i in range(1, 8):
        data = data.merge(eng_para, left_on=['RNCID_%d' % i, 'CellID_%d' % i], right_on=['RNCID_1','CellID_1'], how='left', suffixes=('', '%d' % i))
        temp=data['CellID_%d'% i].tolist()
        new = list()
        for item in temp:
            if math.isnan(item):
                new.append(0)
            elif int(item)<=0:
                new.append(0)
            else:
                new.append(item)
        data['CellID_%d' % i]=new
    data = data.fillna(-999.)
    #print data.columns
    
    # feature = data[col_name_new+['MRTime','TrajID','Lon','Lat','Lon2','Lat2','Lon3','Lat3','Lon4','Lat4',
    #                             'Lon5','Lat5','Lon6','Lat6','Longitude', 'Latitude']]
    feature = data[col_name_new+['MRTime','Lon','Lat','Lon2','Lat2','Lon3','Lat3','Lon4','Lat4',
                                'Lon5','Lat5','Lon6','Lat6','Longitude', 'Latitude']]
    feature['re_lon'] = feature['Longitude'] - feature['Lon']
    feature['re_lat'] = feature['Latitude'] - feature['Lat']
    
    rg = RoadGrid(feature[['re_lon', 're_lat']].values, 20)
    feature['re_ID'] = rg.transform(feature[['re_lon', 're_lat']].values, False)
   
    label = data[['Longitude', 'Latitude']]

    return feature, label, rg

def merge_2g_engpara():
    eng_para = pd.read_csv(folder_tloc_data + "/BS_ALL.csv", encoding='gbk')
    eng_para = eng_para[['RNCID_1', 'CellID_1', 'Lon','Lat']]
    eng_para = eng_para.drop_duplicates()

    return eng_para

eng_para = merge_2g_engpara()

def initialize_data_TLoc(data_path,batch_size, prior_r, real_r, dev, warm_start_flag):
    data_2g = pd.read_csv(data_path)
    data_2g = data_2g.drop_duplicates(col_name_new)
    train, label, rg = make_rf_dataset(data_2g, eng_para)
    data_2g = train

    train.groupby(['RNCID_1', 'CellID_1'])

    tr_feature, te_feature, tr_label, te_label = train_test_split(train, label, test_size=0.2, random_state=30,shuffle=False)
    tr_feature, val_feature, tr_label, val_label = train_test_split(tr_feature, tr_label, test_size=0.1, random_state=30,shuffle=False)

    # test_target = torch.from_numpy(te_label).to(dev)
    # train_target = torch.from_numpy(tr_label).to(dev)
    # val_target = torch.from_numpy(val_label).to(dev)

    # only test loader used in encoder test
    train_loader = None
    val_loader = None
    test_loader = None

    # return train_loader, val_loader, test_loader,tr_feature, train_target, val_feature, val_target, te_feature, test_target
    return train_loader, val_loader, test_loader,tr_feature, tr_label, val_feature, val_label, te_feature, te_label

def gen_state_data(data, prior_r, real_r, warm_start_flag):
    first_index = data.index[0]
    data["vLon"] = pd.Series(index=range(first_index,first_index+len(data)))
    data["vLat"] = pd.Series(index=range(first_index,first_index+len(data)))

    data["vLon"].loc[first_index] = 0
    data["vLat"].loc[first_index] = 0
    for i in range(first_index+1,first_index+len(data)):
        # data.iloc[i]["vLon"] = data["Longitude",i] - data["Longitude",i-1]
        # data.iloc[i]["vLat"] = data["Latitude",i] - data["Latitude",i-1]
        data["vLon"].loc[i] = data["Longitude"].loc[i] - data["Longitude"].loc[i-1]
        data["vLat"].loc[i] = data["Latitude"].loc[i] - data["Latitude"].loc[i-1]