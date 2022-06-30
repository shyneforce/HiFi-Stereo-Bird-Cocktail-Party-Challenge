 # -*- coding: utf-8 -*-
import csv
import numpy as np
import re


with open('preresult.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    labels = ['name', '0-5', '6-10', '11-15', '16-20']
    with open('prob_result.csv', 'w', newline='') as probf:
        writer = csv.DictWriter(probf, fieldnames=labels)
        writer.writeheader()

        listp = {}
        listb = {}
        arrayp = list(range(512))
        arrayb = list(range(512))
        index = 0
        for e in reader:

            prob = e[1].replace('array(','').replace('dtype=float32)','').replace('\n ','')
            prob1 = prob.split(',     , ',3)
            prob1[0]=prob1[0].replace('[[','[')
            prob1[0] = re.sub('[[]','',prob1[0])
            prob1[0] = re.sub('[]]', '',prob1[0])
            #prob1[0] = re.sub('[,]', '', prob1[0])
            prob1[1] = re.sub('[[]', '',prob1[1])
            prob1[1] = re.sub('[]]', '',prob1[1])
            #prob1[1] = re.sub('[,]', '', prob1[1])
            prob1[2] = re.sub('[[]', '',prob1[2])
            prob1[2] = re.sub('[]]', '',prob1[2])
            #prob1[2] = re.sub('[,]', '', prob1[2])
            prob1[3] = prob1[3].replace(',     ]','')
            prob1[3] = re.sub('[[]', '',prob1[3])
            prob1[3] = re.sub('[]]', '',prob1[3])
            #prob1[3] = re.sub('[,]', '', prob1[3])
            a = np.zeros((4, 35))
            b = np.zeros((4, 35))
            for i in range(0, 4):
                a[i] = prob1[i].split(',')
                b[i] = np.array(a[i])
            d1 = np.zeros((4,35))
            #d = np.zeros((4, 35))
            for j in range(0,35):
                for i in range(0,4):
                    d1[i][j] =  float(b[i][j])
                    #d[i][j] = float(b[i][j])


            listp['name']=e[0]
            listp['0-5']=d1[0]
            listp['6-10']=d1[1]
            listp['11-15'] = d1[2]
            listp['16-20'] = d1[3]
            writer.writerow(listp)

with open('preresult.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    labels = ['name', '0-5', '6-10', '11-15', '16-20']
    with open('bin_result.csv', 'w', newline='') as probb:
        writer = csv.DictWriter(probb, fieldnames=labels)
        writer.writeheader()

        listp = {}
        listb = {}
        arrayp = list(range(512))
        arrayb = list(range(512))
        for e in reader:

            prob = e[1].replace('array(','').replace('dtype=float32)','').replace('\n ','')
            prob1 = prob.split(',     , ',3)
            prob1[0]=prob1[0].replace('[[','[')
            prob1[0] = re.sub('[[]','',prob1[0])
            prob1[0] = re.sub('[]]', '',prob1[0])
            #prob1[0] = re.sub('[,]', '', prob1[0])
            prob1[1] = re.sub('[[]', '',prob1[1])
            prob1[1] = re.sub('[]]', '',prob1[1])
            #prob1[1] = re.sub('[,]', '', prob1[1])
            prob1[2] = re.sub('[[]', '',prob1[2])
            prob1[2] = re.sub('[]]', '',prob1[2])
            #prob1[2] = re.sub('[,]', '', prob1[2])
            prob1[3] = prob1[3].replace(',     ]','')
            prob1[3] = re.sub('[[]', '',prob1[3])
            prob1[3] = re.sub('[]]', '',prob1[3])
            #prob1[3] = re.sub('[,]', '', prob1[3])
            a = np.zeros((4, 35))
            b = np.zeros((4, 35))
            for i in range(0, 4):
                a[i] = prob1[i].split(',')
                b[i] = np.array(a[i])
            #d1 = np.zeros((4,35))
            d = np.zeros((4, 35))
            for j in range(0,35):
                for i in range(0,4):
                    d[i][j] = float(b[i][j])


            for i in range(0,4):
                maxv = 0.7 * d[i].max()
                for j in range(0,35):
                    if d[i][j] > maxv:
                        d[i][j]=1
                    else:
                        d[i][j] = 0
            listb['name'] = e[0]
            listb['0-5'] = d[0]
            listb['6-10'] = d[1]
            listb['11-15'] = d[2]
            listb['16-20'] = d[3]
            writer.writerow(listb)





