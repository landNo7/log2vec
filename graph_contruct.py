#-*- coding : utf-8-*-
# coding:unicode_escape
import numpy as np
import networkx as nx
import math
import csv
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import datetime
from construct_rule import *


def get_node_from_data(dir_path, csvname):
    print("Get node from data : ")
    vertex_list = []
    csvname += ".csv"
    with open(os.path.join(dir_path, csvname), 'r', encoding='unicode_escape') as file:
        print("...%s.csv..." % (csvname))
        read = csv.reader(file)
        next(read)
        num = 0
        anomaly_list = list()
        for i in tqdm(read):
            if num == 0:
                last_read = i
            else:
                for j,item in enumerate(i):
                    if item == '':
                        i[j] = last_read[j]
            vertex_id = i[0]
            start_time = time.mktime(time.strptime(i[8],'%Y/%m/%d %H:%M'))
            end_time = time.mktime(time.strptime(i[9],'%Y/%m/%d %H:%M'))
            # vertex = { 'vertex_type': 'attack',
            #             'eventip': vertex_id,
            #             'assetip': i[2],
            #             'country': i[1],
            #             'label': i[3],
            #             'attack_nums': i[4],
            #             'level': i[5],
            #             'effect': i[6],
            #             'disposition': i[7],
            #             'start_time': start_time,
            #             'end_time': end_time
            #             }
            vertex = { 'vertex_type': 'attack',
                        'eventip': vertex_id,
                        'assetip': i[2],
                        'label': i[3],
                        'start_time': start_time
                        }
            vertex_list.append(vertex)
            if num < 1000 and vertex_id not in anomaly_list:
                anomaly_list.append(vertex_id)
            num += 1
            last_read = i
    sorted_vertex_list = sorted(vertex_list, key=lambda e: (e.__getitem__('eventip'), e.__getitem__('start_time')))
    with open(os.path.join(dir_path, "anomaly_list.txt"), 'w+') as f:
        for eventip in anomaly_list:
            f.write(eventip+'\\n')
    return sorted_vertex_list


def get_delta_days(timestamp1, timestamp2):
    x = datetime.datetime.fromtimestamp(timestamp1) - datetime.datetime.fromtimestamp(timestamp2)
    return x.days


def get_days_from_dataset(sorted_vertex_list):
    end_time = 0
    st_time = 9999999999
    for vertex in sorted_vertex_list:
        if vertex['start_time'] > end_time:
            end_time = vertex['start_time']
        if vertex['start_time'] < st_time:
            st_time = vertex['start_time']

    print("Data delta days : ", get_delta_days(end_time, st_time)) 
    return get_delta_days(end_time, st_time) + 2


def split_node_by_day(sorted_vertex_list, day_delta):
    # 1000条数据大概4天
    st_time = 9999999999
    for vertex in sorted_vertex_list:
        if vertex['start_time'] < st_time:
            st_time = vertex['start_time']

    daily_sequences_list = [None] * day_delta
    print("...split node by day...")
    for vertex in tqdm(sorted_vertex_list):
        # Day of the vertex, and actual day should be increased by 1
        day_of_vertex = get_delta_days(vertex['start_time'], st_time) - 1
        # print(day_of_vertex)
        # If the sequence graph not exists, create it
        if not daily_sequences_list[day_of_vertex]:
            # multiGraph 无向图 可以让两个节点之间有多个边，为啥要用这个graph..
            daily_sequences_list[day_of_vertex] = nx.MultiGraph()
        # daily_sequences_list[day_of_vertex].add_node(vertex['eventip'], type=vertex['vertex_type'],
        #                                                     assetip=vertex['assetip'], country=vertex['country'], label=vertex['label'],
        #                                                     level=vertex['level'], effect=vertex['effect'], disposition=vertex['disposition'],
        #                                                     start_time=vertex['start_time'], end_time=vertex['end_time'])
        
        daily_sequences_list[day_of_vertex].add_node(vertex['eventip'], type=vertex['vertex_type'],
                                                            assetip=vertex['assetip'], label=vertex['label'],
                                                            start_time=vertex['start_time'])
    return daily_sequences_list


st_time = time.time()
version = "r_part"
csvname = "attack"
sorted_vertex_list = get_node_from_data("./data/", csvname)
# print(sorted_vertex_list)
day_delta = get_days_from_dataset(sorted_vertex_list)
daily_sequences_list = split_node_by_day(sorted_vertex_list, day_delta)
# print(daily_sequences_list)
daily_sequences_list = rule_1(daily_sequences_list)
daily_sequences_list, H_tuple_list, A_tuple_list = rule_23(daily_sequences_list, day_delta)
graph = rule_456(daily_sequences_list, H_tuple_list, A_tuple_list, day_delta)

nx.write_edgelist(graph, "./data/{csvname}_graph_edge_list".format(csvname=csvname))
nx.write_gpickle(graph, "./data/{csvname}_graph.gpickle".format(csvname=csvname))

print("Graph save done")
print("Time cost : ", time.time() - st_time) 