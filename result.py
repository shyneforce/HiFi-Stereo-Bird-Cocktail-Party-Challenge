#! /usr/bin/env python3

from byol_a.models import AudioNTT2020
import matplotlib.pyplot as plt
import csv
import wave
import struct
from byol_a.common import (os, sys, np, Path, random, torch, nn, DataLoader,
     get_logger, load_yaml_config, seed_everything, get_timestamp)

from utils.torch_mlp_clf import MLP
from evaluate import get_pre,calc_norm_datastats
device = torch.device('cpu')



import seaborn as sns
#import umap
import matplotlib.pyplot as plt
import torch
import numpy as np
#import hdbscan

from sklearn.metrics import *

def cluster(feature):
    clusterable_embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(feature)

    clusterer = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=140)

    clusterer.fit(clusterable_embedding)
    print("聚类数:",clusterer.labels_.max()+1)

    score_sil=silhouette_score(clusterable_embedding,clusterer.labels_)

    print("轮廓系数:",score_sil)
    palette = sns.color_palette()

    plt.figure(dpi=300)
    plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],c=clusterer.labels_,  s=0.1, cmap='Spectral')
    plt.show()


def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window


def pretune(co, data):
    # 实现对信号预加重，co为预加重系数，data为加重对象,一维数组.
    size_data = len(data)
    ad_signal = np.zeros(size_data)
    ad_signal[0] = data[0]
    # print(size_data)
    for i in range(1, size_data, 1):
        ad_signal[i] = data[i] - co * data[i - 1]  # 取矩阵中的数值用方括号
    return ad_signal

def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧,且去掉过短的信号
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    winfunc窗函数winfunc = signal.hamming(nw)
    '''
    signal_length=len(signal) #信号总长度
    
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=0
        return None, nf
    else: #否则，计算帧的总长度
        nf=int(np.floor((1.0*signal_length-nw+inc)/inc))
        whole_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
        pro_signal=signal[0: whole_length]#截去后的信号记为pro_signal
        indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
        frames=pro_signal[indices] #得到帧信号
        win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
        print("enframe finished")
        return frames*win,nf   #返回帧信号矩阵

def dsilence(data, alfa, samplerate):

    # 去除data的静音区，alfa为能量门限值
    edata = (data[0,:]+data[1,:])/2
    nordata = edata / abs(edata).max()  # 对语音进行归一化

    # frame_length分帧长度,hop_length帧偏移
    frame_length = int(50 * samplerate / 1000)  # 50ms帧长
    hop_length = frame_length
    winfunc = choose_windows('Hanning', frame_length)
    frames, nf = enframe(nordata, frame_length, hop_length, winfunc)
    if nf != 1:
        frames = frames.T

        # 要以分割得到的帧数作为row
        row = frames.shape[1]  # 帧数
        col = frames.shape[0]  # 帧长

        print('帧数', frames.shape)
        Energy = np.zeros((1, row))

        # 短时能量函数
        for i in range(0, row):
            Energy[0, i] = np.sum(abs(frames[:, i] * frames[:, i]), 0)  # 不同分帧函数这里要换

        Ave_Energy = Energy.sum() / row
        Delete = np.zeros((1, row))

        # Delete(i)=1 表示第i帧为清音帧

        for i in range(0, row):
            if Energy[0, i] < Ave_Energy * alfa:
                Delete[0, i] = 1

        # 保存去静音的数据
        ds_data1 = np.zeros((frame_length * int(row - Delete.sum())))
        ds_data2 = np.zeros((frame_length * int(row - Delete.sum())))
        ds_data = np.zeros((2,(frame_length * int(row - Delete.sum()))))

        begin = 0
        for i in range(0, row - 1):
            if Delete[0, i] == 0:
                for j in range(0, frame_length, 1):
                    ds_data1[begin * frame_length + j] = data[0,i * hop_length + j]
                    ds_data2[begin * frame_length + j] = data[1,i * hop_length + j]
                begin = begin + 1
        ds_data[0,:] = ds_data1
        ds_data[1, :] = ds_data2
        return ds_data


def audioread(file_path):
    # Load audio file at its native sampling rate
    f = wave.open(file_path, 'rb')

    params = f.getparams()
    nchannels, sampwidth, samplerate, nframes = params[:4]  # nframes就是点数

    # print("read audio dimension", nchannels, sampwidth, samplerate, nframes)
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = np.reshape(waveData, [nframes, nchannels]).T

    f.close()
    return waveData, samplerate


def writewav(audiodata, samplerate, audiopath):
    outData = audiodata  # 待写入wav的数据，这里仍然取waveData数据
    print("write audio size", outData.shape)
    print("max value", outData.max())
    outwave = wave.open(audiopath, 'wb')  # 定义存储路径以及文件名
    nchannels = 1
    sampwidth = 2  # 和数据存储的位数有关
    fs = samplerate
    data_size = len(outData)
    framerate = int(fs)
    nframes = data_size
    print("write nframes", nframes)

    comptype = "NONE"
    compname = "not compressed"
    outwave.setparams((nchannels, sampwidth, framerate, nframes,
                       comptype, compname))
    for v in outData:
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))  # outData:16位转化为二进制，-32767~32767，注意不要溢出

    outwave.close()


def re_wav(byolweight_file,mlpweight_file, unit_sec, audiofilespath, frame_time=1000, overlap_rate=0.5):  # 保存剪切的音频'E:/基金/fastchirplet-master/fastchirplet-master/audio/'

    cfg = load_yaml_config('config.yaml')
    cfg.unit_sec = unit_sec

    model = AudioNTT2020(n_mels=cfg.n_mels, d=cfg.feature_d)
    model.load_weight(byolweight_file, device)
    # embeddings
    model = model.to(device)
    model.eval()

    mlpmodel = MLP(2048, hidden_sizes=((128,)), output_size=35)
    mlpmodel.load_state_dict(torch.load(mlpweight_file), strict = True)
    mlpmodel = mlpmodel.to(device)
    mlpmodel.eval()


    # 音频文件路径，帧长时间单位，帧的重叠率稍微大些
    # filename = os.path.basename(audiofilespath)#get filename
    # print("class name:",filename)
    directory = os.fsencode(audiofilespath)
    #a_pre= []
    pre=[]
    w_pre = []
    for file in os.listdir(directory):
        wavfilename = os.fsdecode(file)
        if wavfilename.endswith(".wav"):
            print("wav name:", wavfilename)
            file_path = audiofilespath + '/' + wavfilename
            #file_path = file_path.replace('/', '\\')
            #print("wav path:", file_path)
            # Load audio file at its native sampling rate
            data, sr = audioread(file_path)
            splitdata = np.array(np.split(data,4,1))
            #print("slice read audio size", splitdata.shape, sr)
            # frame_time=500 #改变分割的长度，单位ms

            for index in range(0,4):
                frame_length = int(sr * frame_time / 1000)
                hop_length = int((1 - overlap_rate) * frame_length)
                eachdata = splitdata[index, :, :]
                print("which part",index)
                # 去除静音
                ds_data = dsilence(eachdata, 0.2, sr)  # 0.5是无声阈值，越高留下越少

                winfunc = choose_windows('Hanning', frame_length)
                frames1, nf1 = enframe(ds_data[0,:], frame_length, hop_length, winfunc)  # (帧数，帧长)
                frames2, nf2 = enframe(ds_data[1, :], frame_length, hop_length, winfunc)  # (帧数，帧长)

                if nf1 != 0:
                    list_frames1 = frames1.tolist()
                    list_frames2 = frames2.tolist()
                    norm_stats1 = calc_norm_datastats(cfg, list_frames1, n_stats=10000)
                    norm_stats2 = calc_norm_datastats(cfg, list_frames2, n_stats=10000)

                    pre1 = get_pre(cfg, list_frames1, model, mlpmodel,norm_stats1)

                    pre2 = get_pre(cfg, list_frames2, model, mlpmodel,norm_stats2)
                    pre1_tensor = torch.from_numpy(pre1)
                    pre2_tensor = torch.from_numpy(pre2)

                    pro_pre1 = torch.nn.functional.softmax(pre1_tensor, dim=1)
                    pro_pre2 = torch.nn.functional.softmax(pre2_tensor, dim=1)
                    #print("prediction2", pro_pre2)
                    pre = (np.array(pro_pre1) + np.array(pro_pre2))/2
                    #each_pre = np.max(pre, axis=0)
                    w_pre.append(pre)
                    print("prediction", pre)

                

        #a_pre[wavfilename] = w_pre

        print("prediction of all audio:", len(w_pre))
    #cluster(w_pre)


    with open('pre_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in w_pre:
            writer.writerow(row)


if __name__ == '__main__':
    audio_path = 'your path of challenge dataset'
    mlpweight_file= 'your path of trained classifier weights'
    weightpath = 'your path of trained byol-a weights'
    unit_sec = 0.95
    re_wav(weightpath,mlpweight_file,unit_sec, audio_path)