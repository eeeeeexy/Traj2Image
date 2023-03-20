import pickle
from typing import Tuple
from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL.Image import Image
import mpl_toolkits.axes_grid1 as axes_grid1
from collections import defaultdict, Counter
from math import sin, asin, cos, radians, fabs, sqrt
import os, shutil
import sys
sys.setrecursionlimit(10000000)
import csv
import cv2, copy


EARTH_RADIUS = 6371  # 地球平均半径，6371km

def hav(theta):
    s = sin(theta / 2)
    return s * s

def get_distance_hav(location_a, location_b):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(location_a[0])
    lat1 = radians(location_b[0])
    lng0 = radians(location_a[1])
    lng1 = radians(location_b[1])

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance

def broken_info():
    broken_list = []
    all_list = []
    for trip in trip_motion_all_user_with_label_train:

        features = trip[0]
        lat = trip[0][-2] 
        lon = trip[0][-1]
        delta_time = trip[0][0]
        speed = trip[0][1]
        bearing = trip[0][2]
        acc = trip[0][3]

        lat_len_A = get_distance_hav([max(lat), min(lon)], [min(lat), min(lon)]) * 1000 / pixel_size
        lon_len_A = get_distance_hav([min(lat), max(lon)], [min(lat), min(lon)]) * 1000 / pixel_size
        # print(f'lat len: {lat_len_A}')
        # print(f'lon len: {lon_len_A}')

        dis_list = []
        for i in range(len(trip[0][0]) - 1):
            temp_list = []
            dis = get_distance_hav([trip[0][-2][i], trip[0][-1][i]], [trip[0][-2][i+1], trip[0][-1][i+1]]) * 1000
            if dis > sqrt((lat_len_A) ** 2 + (lon_len_A) ** 2):
                temp_list = [i, dis, trip[0][0][i], trip[0][1][i], trip[0][2][i], trip[0][3][i], trip[0][4][i], trip[0][-2][i], trip[0][-1][i]]
                all_list.append(temp_list)
                
                # print(f'{i} {dis} delta time: {trip[0][0][i]} speed: {trip[0][1][i]} bearing: {trip[0][2][i]} acc: {trip[0][3][i]} lat: {trip[0][-2][i]} lon: {trip[0][-1][i]}')
                dis_list.append(dis)

        if len(dis_list) != 0:
            all_list.append([len(lat)])
            all_list.append(' ')

        broken_list.append(len(set(dis_list)))

    with open('broken.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'distance', 'delta time', 'speed', 'bearing', 'acc', 'bearing_rate', 'lat', 'lon'])
        writer.writerows(all_list)
# broken_info()

def info_record():

    print('Descriptive statistics for labeled test lat',  pd.Series(unconti).describe(percentiles=[0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 0.65, 
                                                                                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]))
    # calculate the speed distribution of each category
    class_dict_segs={}
    for segs in trip_motion_all_user_with_label_train:
        if segs[1] not in class_dict_segs:
            class_dict_segs[segs[1]] = []
            for speed_i in segs[0][1]:
                class_dict_segs[segs[1]].append(speed_i)
        else:
            for speed_i in segs[0][1]:
                class_dict_segs[segs[1]].append(speed_i)

    for i in class_dict_segs.keys():
        print(f'class: {i}')
        print('Descriptive statistics for labeled test lat',  pd.Series(class_dict_segs[i]).describe(percentiles=[0.05, 0.1, 0.2, 0.5, 0.6, 
                            0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1]).round(4))

    # calculate the same points of each category
    class_dict_segs={}
    for segs in trip_motion_all_user_with_label_train:
        if segs[1] not in class_dict_segs:
            class_dict_segs[segs[1]] = []
            for i in range(len(segs[0][1])):
                class_dict_segs[segs[1]].append((segs[0][-2][i], segs[0][-1][i]))
        else:
            for i in range(len(segs[0][1])):
                class_dict_segs[segs[1]].append((segs[0][-2][i], segs[0][-1][i]))

    coor_dict = {}
    for i in class_dict_segs.keys():
        print(f'class: {i}')
        coor_dict[i] = dict(Counter(class_dict_segs[i]))
        list_i = [value for key, value in coor_dict[i].items() if value > 1]

        print('Descriptive statistics for labeled test lat',  pd.Series(list_i).describe(percentiles=[0.05, 0.1, 0.2, 0.5, 0.6, 
                            0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1]).round(4))


# std for all features

# delta_time: same
# speed / mode_speed
# bearing/ 180
# acc / mode_speed
# bearing_rate / 180


def feature_maxvalue():
    feature_max = {}
    speed = []
    bearing = []
    acc =[]
    bearing_rate = []
    # [delta_time, speed, bearing, acc, bearing_rate, x_coord, y_coord]

    for trip in trip_motion_all_user_with_label_train:
        for sp in trip[0][1]:
            speed.append(sp)
        for be in trip[0][2]:
            bearing.append(be)
        for ac in trip[0][3]:
            acc.append(ac)
        for ba in trip[0][4]:
            bearing_rate.append(ba)

    for trip in trip_motion_all_user_with_label_test:
        for sp in trip[0][1]:
            speed.append(sp)
        for be in trip[0][2]:
            bearing.append(be)
        for ac in trip[0][3]:
            acc.append(ac)
        for ba in trip[0][4]:
            bearing_rate.append(ba)

    feature_max['speed'] = max(speed)
    feature_max['bearing'] = max(bearing)
    feature_max['acc'] = max(acc)
    feature_max['bearing_rate'] = max(bearing_rate)

    return feature_max


def normalize(trip_motion_all_user_with_label_train):
    trip_motion_all_user_with_label_train_new = []
    for trip in trip_motion_all_user_with_label_train:
        
        feature = trip[0]
        mode = trip[1]
        delta_time = [i/len(feature[0]) for i in range(len(feature[0]))]
        speed = [i/feature_max['speed'] for i in feature[1]]
        bearing = [i/feature_max['bearing'] for i in feature[2]]
        acc = [i/feature_max['acc'] for i in feature[3]]
        bearing_rate = [i/feature_max['bearing_rate'] for i in feature[4]]

        lat = feature[5]
        lon = feature[6]
        new_feature = [delta_time, speed, bearing, acc, bearing_rate, lat, lon]
        trip_motion_all_user_with_label_train_new.append((new_feature, mode))
    
    return trip_motion_all_user_with_label_train_new


def traj_to_image_shift(traj_data, max_class_all_gap, min_class_all_gap, shift, fixed, expand):

    default_pixel = 0

    TITS_image_1feature = []
    TITS_image_2feature = []
    TITS_image_3feature = []
    TITS_image_3feature2 = []
    TITS_image_4feature = []
    TITS_image_5feature = []
    TITS_image_6feature = []
    label_list = []
    
    for index, trip in enumerate(traj_data):

        shape_array = np.zeros((pixel_size, pixel_size))
        count_array = np.zeros((pixel_size, pixel_size))
        speed_array_final = np.zeros((pixel_size, pixel_size))
        bearing_array_final = np.zeros((pixel_size, pixel_size))
        bearing_rate_array_final = np.zeros((pixel_size, pixel_size))
        deltatime_array_final = np.zeros((pixel_size, pixel_size))
        acc_array_final = np.zeros((pixel_size, pixel_size))

        # speed
        speed_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                speed_array[i][j] = []
        # bearing
        bearing_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                bearing_array[i][j] = []
        # bearing rate
        bearing_rate_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                bearing_rate_array[i][j] = []
        # delta time
        delta_time_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                delta_time_array[i][j] = []
        # acc
        acc_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                acc_array[i][j] = []
        
        # [delta_time, speed, bearing, acc, x_coord, y_coord]
        lat = trip[0][-2]
        lon = trip[0][-1]
        delta_time = trip[0][0]
        speed = trip[0][1]
        bearing = trip[0][2]
        acc = trip[0][3]
        bearing_rate = trip[0][4]

        start_point = (min(lat), min(lon))

        lat_id = []
        lon_id = []
        speed_id = []
        bearing_id = []
        bearing_rate_id = []
        deltatime_id = []
        acc_id = []
        time_count = 0

        image_center_point = (floor(pixel_size/2)-1, floor(pixel_size/2)-1)

        # print(f'class gap: {class_all_gap[trip[1]][0], class_all_gap[trip[1]][1]}')
        # print(f'traj gap: {(max(lat) - min(lat)), (max(lon) - min(lon))}')

        if fixed == True:
            max_lat = (max(lat) - min(lat))
            max_lon = (max(lon) - min(lon))
            class_max_lat = max_class_all_gap[trip[1]][0]
            class_max_lon = max_class_all_gap[trip[1]][1]
            class_min_lat = min_class_all_gap[trip[1]][0]
            class_min_lon = min_class_all_gap[trip[1]][1]

            if class_min_lat < max_lat < class_max_lat and class_min_lon < max_lon < class_max_lon:
                all_lat = class_max_lat
                all_lon = class_max_lon
            else:
                all_lat = max_lat
                all_lon = max_lon
                
            sub_lat = all_lat/pixel_size * 1.05
            sub_lon = all_lon/pixel_size * 1.05
            if sub_lat == 0 or sub_lon == 0:
                continue

            for point in zip(lat, lon):
                if int((point[0] - start_point[0])/sub_lat) <= pixel_size - 1 and int((point[1] - start_point[1])/sub_lon) <= pixel_size - 1:
                    lat_id.append(int((point[0] - start_point[0])/sub_lat))
                    lon_id.append(int((point[1] - start_point[1])/sub_lon))
                    speed_id.append(speed[lat.index(point[0])])
                    bearing_id.append(bearing[lat.index(point[0])])
                    bearing_rate_id.append(bearing_rate[lat.index(point[0])])
                    deltatime_id.append(delta_time[lat.index(point[0])])
                    acc_id.append(acc[lat.index(point[0])])
                    time_count += 1
        
        elif fixed == False:
            

            all_lat = max(lat) - min(lat)
            all_lon = max(lon) - min(lon)
            sub_lat = (all_lat / pixel_size) * 1.05
            sub_lon = (all_lon / pixel_size) * 1.05
            if sub_lat == 0 or sub_lon == 0:
                continue
            for point in zip(lat, lon):
                if int((point[0] - start_point[0])/sub_lat) <= pixel_size - 1 and int((point[1] - start_point[1])/sub_lon) <= pixel_size - 1:
                    lat_id.append(int((point[0] - start_point[0])/sub_lat))
                    lon_id.append(int((point[1] - start_point[1])/sub_lon))
                    speed_id.append(speed[lat.index(point[0])])
                    bearing_id.append(bearing[lat.index(point[0])])
                    bearing_rate_id.append(bearing_rate[lat.index(point[0])])
                    deltatime_id.append(delta_time[lat.index(point[0])])
                    acc_id.append(acc[lat.index(point[0])])
                    time_count += 1

        print(index, len(lat_id), len(lon_id), len(speed_id), len(bearing_id), len(deltatime_id), len(acc_id), len(bearing_rate_id))

        lat_lon_id = list(zip(lon_id, lat_id, speed_id, bearing_id, deltatime_id, acc_id, bearing_rate_id))

        if len(lat_lon_id) == 0:
            continue

        for id in lat_lon_id:
            shape_array[id[0]][id[1]] = 1
        
        nonzero_array = np.nonzero(shape_array)
        traj_center_point = (int(nonzero_array[0].sum()/len(nonzero_array[0])), int(nonzero_array[1].sum()/len(nonzero_array[1])))
        shape_array = np.zeros((pixel_size, pixel_size))

        if shift == False:
            for id in lat_lon_id:
                shape_array[id[0]][id[1]] = 1
                count_array[id[0]][id[1]] += 1
                speed_array[id[0]][id[1]].append(id[2])
                bearing_array[id[0]][id[1]].append(id[3])
                delta_time_array[id[0]][id[1]].append(id[4])
                acc_array[id[0]][id[1]].append(id[5])
                bearing_rate_array[id[0]][id[1]].append(id[6])
        
        elif shift == True:
            lat_shift = image_center_point[0] - traj_center_point[0]
            lon_shift = image_center_point[1] - traj_center_point[1]

            if max(nonzero_array[0]) < pixel_size/2 and max(nonzero_array[1]) < pixel_size/2:
                for id in lat_lon_id:
                    shape_array[id[0]+lat_shift][id[1]+lon_shift] = 1
                    count_array[id[0]+lat_shift][id[1]+lon_shift] += 1
                    speed_array[id[0]+lat_shift][id[1]+lon_shift].append(id[2])
                    bearing_array[id[0]+lat_shift][id[1]+lon_shift].append(id[3])
                    delta_time_array[id[0]+lat_shift][id[1]+lon_shift].append(id[4])
                    acc_array[id[0]+lat_shift][id[1]+lon_shift].append(id[5])
                    bearing_rate_array[id[0]+lat_shift][id[1]+lon_shift].append(id[6])

            elif max(nonzero_array[0]) < pixel_size/2 and max(nonzero_array[1]) > pixel_size/2:
                for id in lat_lon_id:
                    shape_array[id[0]+lat_shift][id[1]] = 1
                    count_array[id[0]+lat_shift][id[1]] += 1
                    speed_array[id[0]+lat_shift][id[1]].append(id[2])
                    bearing_array[id[0]+lat_shift][id[1]].append(id[3])
                    delta_time_array[id[0]+lat_shift][id[1]].append(id[4])
                    acc_array[id[0]+lat_shift][id[1]].append(id[5])
                    bearing_rate_array[id[0]+lat_shift][id[1]].append(id[6])

            elif max(nonzero_array[0]) > pixel_size/2 and max(nonzero_array[1]) < pixel_size/2:
                for id in lat_lon_id:
                    shape_array[id[0]][id[1]+lon_shift] = 1
                    count_array[id[0]][id[1]+lon_shift] += 1
                    speed_array[id[0]][id[1]+lon_shift].append(id[2])
                    bearing_array[id[0]][id[1]+lon_shift].append(id[3])
                    delta_time_array[id[0]][id[1]+lon_shift].append(id[4])
                    acc_array[id[0]][id[1]+lon_shift].append(id[5])
                    bearing_rate_array[id[0]][id[1]+lon_shift].append(id[6])

            else:
                for id in lat_lon_id:
                    shape_array[id[0]][id[1]] = 1
                    count_array[id[0]][id[1]] += 1
                    speed_array[id[0]][id[1]].append(id[2])
                    bearing_array[id[0]][id[1]].append(id[3])
                    delta_time_array[id[0]][id[1]].append(id[4])
                    acc_array[id[0]][id[1]].append(id[5])
                    bearing_rate_array[id[0]][id[1]].append(id[6])
                    
        # print('-->final normalized image')
        for i in range(pixel_size):
            for j in range(pixel_size):
                if len(speed_array[i][j]) == 0:
                    count_array[i][j] = default_pixel
                    speed_array_final[i][j] = default_pixel
                    bearing_array_final[i][j] = default_pixel
                    deltatime_array_final[i][j] = default_pixel
                    acc_array_final[i][j] = default_pixel
                    bearing_rate_array_final[i][j] = default_pixel
                    # print(i, j, acc_array[i][j], acc_array_final[i][j])
                else:
                    # count_array[i][j] = np.array(final_array[i][j]).mean()
                    speed_array_final[i][j] = np.array(speed_array[i][j]).mean()
                    bearing_array_final[i][j] = np.array(bearing_array[i][j]).mean()
                    deltatime_array_final[i][j] = np.array(delta_time_array[i][j]).mean()
                    acc_array_final[i][j] = np.array(acc_array[i][j]).mean()
                    bearing_rate_array_final[i][j] = np.array(bearing_rate_array[i][j]).mean()
                    # print(i, j, acc_array_final[i][j], speed_array_final[i][j], time_array_final[i][j])
        

        count_array = count_array /count_array.max() * expand
        # speed_array_final = (speed_array_final - speed_array_final.mean())/speed_array_final.std()
        # bearing_array_final = (bearing_array_final - bearing_array_final.mean())/bearing_array_final.std()
        # deltatime_array_final = (deltatime_array_final - deltatime_array_final.mean())/deltatime_array_final.std()
        # acc_array_final = (acc_array_final - acc_array_final.mean())/acc_array_final.std()
        # bearing_rate_array_final = (bearing_rate_array_final - bearing_rate_array_final.mean())/bearing_rate_array_final.std()

        # check nan
        if np.isnan(count_array).any() == True or np.isnan(speed_array_final).any() == True or np.isnan(deltatime_array_final).any() == True or np.isnan(bearing_array_final).any() == True:
            continue
        
        TITS_image_1feature.append([speed_array_final])
        TITS_image_2feature.append([speed_array_final, count_array])
        TITS_image_3feature.append([speed_array_final, count_array, acc_array_final])
        TITS_image_3feature2.append([speed_array_final, count_array, bearing_array_final])
        TITS_image_4feature.append([speed_array_final, count_array, acc_array_final, bearing_array_final])
        TITS_image_5feature.append([speed_array_final, count_array, acc_array_final, bearing_array_final, bearing_rate_array_final])
        TITS_image_6feature.append([speed_array_final, count_array, acc_array_final, bearing_array_final, bearing_rate_array_final, deltatime_array_final])      
        
        label_list.append(trip[1])
    
    return TITS_image_1feature, TITS_image_2feature, TITS_image_3feature, TITS_image_3feature2, TITS_image_4feature, TITS_image_5feature, TITS_image_6feature, label_list

def traj_to_image_shift_scale(traj_data, shift):

    default_pixel = 0
    count = 0

    TITS_image_1feature = []
    TITS_image_2feature = []
    TITS_image_3feature = []
    TITS_image_3feature2 = []
    TITS_image_4feature = []
    TITS_image_5feature = []
    TITS_image_6feature = []
    label_list = []
    scalesize_list = []
    
    for index, trip in enumerate(traj_data):

        shape_array = np.zeros((pixel_size, pixel_size))
        count_array_final = np.zeros((pixel_size, pixel_size))
        count_array = np.zeros((pixel_size, pixel_size))
        speed_array_final = np.zeros((pixel_size, pixel_size))
        bearing_array_final = np.zeros((pixel_size, pixel_size))
        bearing_rate_array_final = np.zeros((pixel_size, pixel_size))
        deltatime_array_final = np.zeros((pixel_size, pixel_size))
        acc_array_final = np.zeros((pixel_size, pixel_size))

        # speed
        speed_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                speed_array[i][j] = []
        # bearing
        bearing_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                bearing_array[i][j] = []
        # bearing rate
        bearing_rate_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                bearing_rate_array[i][j] = []
        # delta time
        delta_time_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                delta_time_array[i][j] = []
        # acc
        acc_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                acc_array[i][j] = []
        
        # [delta_time, speed, bearing, acc, x_coord, y_coord]
        lat = trip[0][-2]
        lon = trip[0][-1]
        delta_time = trip[0][0]
        speed = trip[0][1]
        bearing = trip[0][2]
        acc = trip[0][3]
        bearing_rate = trip[0][4]

        start_point = (min(lat), min(lon))

        lat_id = []
        lon_id = []
        speed_id = []
        bearing_id = []
        bearing_rate_id = []
        deltatime_id = []
        acc_id = []
        time_count = 0

        image_center_point = (floor(pixel_size/2)-1, floor(pixel_size/2)-1)

        # print(f'class gap: {class_all_gap[trip[1]][0], class_all_gap[trip[1]][1]}')
        # print(f'traj gap: {(max(lat) - min(lat)), (max(lon) - min(lon))}')

        all_lat = max(lat) - min(lat)
        all_lon = max(lon) - min(lon)
        sub_lat = (all_lat / pixel_size) * 1.05
        sub_lon = (all_lon / pixel_size) * 1.05
        if sub_lat == 0 or sub_lon == 0:
            continue
        for point in zip(lat, lon):
            if int((point[0] - start_point[0])/sub_lat) <= pixel_size - 1 and int((point[1] - start_point[1])/sub_lon) <= pixel_size - 1:
                lat_id.append(int((point[0] - start_point[0])/sub_lat))
                lon_id.append(int((point[1] - start_point[1])/sub_lon))
                speed_id.append(speed[lat.index(point[0])])
                bearing_id.append(bearing[lat.index(point[0])])
                bearing_rate_id.append(bearing_rate[lat.index(point[0])])
                deltatime_id.append(delta_time[lat.index(point[0])])
                acc_id.append(acc[lat.index(point[0])])
                time_count += 1

        print(index, len(lat_id), len(lon_id), len(speed_id), len(bearing_id), len(deltatime_id), len(acc_id), len(bearing_rate_id))

        lat_lon_id = list(zip(lon_id, lat_id, speed_id, bearing_id, deltatime_id, acc_id, bearing_rate_id))

        if len(lat_lon_id) == 0:
            continue

        for id in lat_lon_id:
            shape_array[id[0]][id[1]] = 1
        
        nonzero_array = np.nonzero(shape_array)
        traj_center_point = (int(nonzero_array[0].sum()/len(nonzero_array[0])), int(nonzero_array[1].sum()/len(nonzero_array[1])))
        shape_array = np.zeros((pixel_size, pixel_size))
    
        for id in lat_lon_id:
            shape_array[id[0]][id[1]] = 1
            count_array[id[0]][id[1]] += 1
            speed_array[id[0]][id[1]].append(id[2])
            bearing_array[id[0]][id[1]].append(id[3])
            delta_time_array[id[0]][id[1]].append(id[4])
            acc_array[id[0]][id[1]].append(id[5])
            bearing_rate_array[id[0]][id[1]].append(id[6])
        
        # print('-->final normalized image')
        for i in range(pixel_size):
            for j in range(pixel_size):
                if len(speed_array[i][j]) == 0:
                    count_array[i][j] = default_pixel
                    speed_array_final[i][j] = default_pixel
                    bearing_array_final[i][j] = default_pixel
                    deltatime_array_final[i][j] = default_pixel
                    acc_array_final[i][j] = default_pixel
                    bearing_rate_array_final[i][j] = default_pixel
                    # print(i, j, acc_array[i][j], acc_array_final[i][j])
                else:
                    speed_array_final[i][j] = np.array(speed_array[i][j]).mean()
                    bearing_array_final[i][j] = np.array(bearing_array[i][j]).mean()
                    deltatime_array_final[i][j] = np.array(delta_time_array[i][j]).mean()
                    acc_array_final[i][j] = np.array(acc_array[i][j]).mean()
                    bearing_rate_array_final[i][j] = np.array(bearing_rate_array[i][j]).mean()
                    # print(i, j, acc_array_final[i][j], speed_array_final[i][j], time_array_final[i][j])
        count_array = count_array /count_array.max()

        # # initial image
        # image_count = 0
        # fig = plt.figure()
        # fig.set_facecolor('white')
        # grid_shift = axes_grid1.AxesGrid(
        #     fig, 111, nrows_ncols=(1, 6), axes_pad = 0.5, cbar_location = "right",
        #     cbar_mode="each", cbar_size="15%", cbar_pad="5%",) 
        # im0 = grid_shift[image_count].imshow(speed_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[0].colorbar(im0)
        # im1 = grid_shift[image_count+1].imshow(bearing_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[1].colorbar(im1)
        # im2 = grid_shift[image_count+2].imshow(acc_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[2].colorbar(im2)
        # im3 = grid_shift[image_count+3].imshow(deltatime_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[3].colorbar(im3)
        # im4 = grid_shift[image_count+4].imshow(count_array, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[4].colorbar(im4)
        # im5 = grid_shift[image_count+5].imshow(bearing_rate_array_final, cmap='jet', interpolation='ne   arest')
        # grid_shift.cbar_axes[5].colorbar(im5)
        # img_path = 'init_image_class_fixed_pixel%d_%d.png'%(pixel_size, count)
        # print(img_path)
        # fig.savefig(img_path)

        global_ = {0: 0.003798, 1: 0.003862}
        self_lat = (max(lat) - min(lat))
        self_lon = (max(lon) - min(lon))
        # way 1 
        # lat_scale = global_[0] / self_lat
        # lon_scale = global_[1] / self_lon
        # way 2
        lat_scale = self_lat / global_[0]
        lon_scale = self_lon / global_[1]
        scale_list = [lat_scale, lon_scale]
        scale_size = min(scale_list)
        index_ = scale_list.index(scale_size)
        all_lat = self_lat * scale_size
        all_lon = self_lon * scale_size

        scalesize_list.append(scale_size)

        if (all_lon/global_[1])*pixel_size > pixel_size - 0.5:
            expand = pixel_size
        else:
            expand = round((all_lon/global_[1])*pixel_size + 0.5)

        lat_size = 0
        lon_size = 0
        
        if index_ == 0:
            shape_array_scale = cv2.resize(np.array(shape_array.astype(float)), (pixel_size, expand))
            speed_array_scale = cv2.resize(np.array(speed_array_final), (pixel_size, expand))
            bearing_array_scale = cv2.resize(np.array(bearing_array_final), (pixel_size, expand))
            acc_array_scale = cv2.resize(np.array(bearing_array_final), (pixel_size, expand))
            delta_time_array_scale = cv2.resize(np.array(deltatime_array_final), (pixel_size, expand))
            count_array_scale = cv2.resize(np.array(count_array.astype(float)), (pixel_size, expand))
            bearing_rate_array_scale = cv2.resize(np.array(bearing_rate_array_final), (pixel_size, expand))
            lat_size = pixel_size
            lon_size = expand
        elif index_ == 1:
            shape_array_scale = cv2.resize(np.array(shape_array.astype(float)), (expand, pixel_size))
            speed_array_scale = cv2.resize(np.array(speed_array_final), (expand, pixel_size))
            bearing_array_scale = cv2.resize(np.array(bearing_array_final), (expand, pixel_size))
            acc_array_scale = cv2.resize(np.array(bearing_array_final), (expand, pixel_size))
            delta_time_array_scale = cv2.resize(np.array(deltatime_array_final), (expand, pixel_size))
            count_array_scale = cv2.resize(np.array(count_array.astype(float)), (expand, pixel_size))
            bearing_rate_array_scale = cv2.resize(np.array(bearing_rate_array_final), (expand, pixel_size))
            lat_size = expand
            lon_size = pixel_size
        
        # # resize the image
        # im0 = grid_shift[image_count].imshow(speed_array_scale, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[0].colorbar(im0)
        # im1 = grid_shift[image_count+1].imshow(bearing_array_scale, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[1].colorbar(im1)
        # im2 = grid_shift[image_count+2].imshow(acc_array_scale, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[2].colorbar(im2)
        # im3 = grid_shift[image_count+3].imshow(delta_time_array_scale, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[3].colorbar(im3)
        # im4 = grid_shift[image_count+4].imshow(count_array_scale, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[4].colorbar(im4)
        # im5 = grid_shift[image_count+5].imshow(bearing_rate_array_scale, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[5].colorbar(im5)
        # img_path = 'scale_image_class_fixed_pixel%d_%d.png'%(pixel_size, count)
        # print(img_path)
        # fig.savefig(img_path)
        
        shape_array = np.zeros((pixel_size, pixel_size))
        count_array = np.zeros((pixel_size, pixel_size))
        speed_array = np.zeros((pixel_size, pixel_size))
        bearing_array = np.zeros((pixel_size, pixel_size))
        delta_time_array = np.zeros((pixel_size, pixel_size))
        acc_array = np.zeros((pixel_size, pixel_size))
        bearing_rate_array = np.zeros((pixel_size, pixel_size))

        shape_array[:lat_size][:lon_size] = shape_array_scale[:lat_size][:lon_size]
        speed_array[:lat_size][:lon_size] = speed_array_scale[:lat_size][:lon_size]
        bearing_array[:lat_size][:lon_size] = bearing_array_scale[:lat_size][:lon_size]
        acc_array[:lat_size][:lon_size] = acc_array_scale[:lat_size][:lon_size]
        delta_time_array[:lat_size][:lon_size] = delta_time_array_scale[:lat_size][:lon_size]
        count_array[:lat_size][:lon_size] = count_array_scale[:lat_size][:lon_size]
        bearing_rate_array[:lat_size][:lon_size] = bearing_rate_array_scale[:lat_size][:lon_size]

        # # resize image and paste the image to the original image
        # im0 = grid_shift[image_count].imshow(speed_array, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[0].colorbar(im0)
        # im1 = grid_shift[image_count+1].imshow(bearing_array, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[1].colorbar(im1)
        # im2 = grid_shift[image_count+2].imshow(acc_array, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[2].colorbar(im2)
        # im3 = grid_shift[image_count+3].imshow(delta_time_array, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[3].colorbar(im3)
        # im4 = grid_shift[image_count+4].imshow(count_array, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[4].colorbar(im4)
        # im5 = grid_shift[image_count+5].imshow(bearing_rate_array, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[5].colorbar(im5)
        # img_path = 'image_class_fixed_pixel%d_%d.png'%(pixel_size, count)
        # print(img_path)
        # fig.savefig(img_path)
      

        # print('-->image array')

        shape_array_final = np.zeros((pixel_size, pixel_size))
        count_array_final = np.zeros((pixel_size, pixel_size))
        speed_array_final = np.zeros((pixel_size, pixel_size))
        bearing_array_final = np.zeros((pixel_size, pixel_size))
        delta_time_array_final = np.zeros((pixel_size, pixel_size))
        acc_array_final = np.zeros((pixel_size, pixel_size))
        bearing_rate_array_final = np.zeros((pixel_size, pixel_size))
        

        if shift == False:
            for i in range(pixel_size):
                for j in range(pixel_size):
                    shape_array_final[i][j] = shape_array[i][j]
                    count_array_final[i][j] = count_array[i][j]
                    speed_array_final[i][j] = speed_array[i][j]
                    bearing_array_final[i][j] = bearing_array[i][j]
                    delta_time_array_final[i][j] = delta_time_array[i][j]
                    acc_array_final[i][j] = acc_array[i][j]
                    bearing_rate_array_final[i][j] = bearing_rate_array[i][j]
        
        elif shift == True:
            lat_shift = image_center_point[0] - traj_center_point[0]
            lon_shift = image_center_point[1] - traj_center_point[1]

            if max(nonzero_array[0]) < pixel_size/2 and max(nonzero_array[1]) < pixel_size/2:
                for i in range(pixel_size):
                    for j in range(pixel_size):
                        shape_array_final[i+lat_shift][j+lon_shift] = shape_array[i][j]
                        count_array_final[i+lat_shift][j+lon_shift] = count_array[i][j]
                        speed_array_final[i+lat_shift][j+lon_shift] = speed_array[i][j]
                        bearing_array_final[i+lat_shift][j+lon_shift] = bearing_array[i][j]
                        delta_time_array_final[i+lat_shift][j+lon_shift] = delta_time_array[i][j]
                        acc_array_final[i+lat_shift][j+lon_shift] = acc_array[i][j]
                        bearing_rate_array_final[i+lat_shift][j+lon_shift] = bearing_rate_array[i][j]

            elif max(nonzero_array[0]) < pixel_size/2 and max(nonzero_array[1]) > pixel_size/2:
                for i in range(pixel_size):
                    for j in range(pixel_size):
                        shape_array_final[i+lat_shift][j] = shape_array[i][j]
                        count_array_final[i+lat_shift][j] = count_array[i][j]
                        speed_array_final[i+lat_shift][j] = speed_array[i][j]
                        bearing_array_final[i+lat_shift][j] = bearing_array[i][j]
                        delta_time_array_final[i+lat_shift][j] = delta_time_array[i][j]
                        acc_array_final[i+lat_shift][j] = acc_array[i][j]
                        bearing_rate_array_final[i+lat_shift][j] = bearing_rate_array[i][j]

            elif max(nonzero_array[0]) > pixel_size/2 and max(nonzero_array[1]) < pixel_size/2:
                for i in range(pixel_size):
                    for j in range(pixel_size):
                        shape_array_final[i][j+lon_shift] = shape_array[i][j]
                        count_array_final[i][j+lon_shift] = count_array[i][j]
                        speed_array_final[i][j+lon_shift] = speed_array[i][j]
                        bearing_array_final[i][j+lon_shift] = bearing_array[i][j]
                        delta_time_array_final[i][j+lon_shift] = delta_time_array[i][j]
                        acc_array_final[i][j+lon_shift] = acc_array[i][j]
                        bearing_rate_array_final[i][j+lon_shift] = bearing_rate_array[i][j]

            else:
                for i in range(pixel_size):
                    for j in range(pixel_size):
                        shape_array_final[i][j] = shape_array[i][j]
                        count_array_final[i][j] = count_array[i][j]
                        speed_array_final[i][j] = speed_array[i][j]
                        bearing_array_final[i][j] = bearing_array[i][j]
                        delta_time_array_final[i][j] = delta_time_array[i][j]
                        acc_array_final[i][j] = acc_array[i][j]
                        bearing_rate_array_final[i][j] = bearing_rate_array[i][j]

        # # final image ->> after resize and shift
        # im0 = grid_shift[image_count].imshow(speed_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[0].colorbar(im0)
        # im1 = grid_shift[image_count+1].imshow(bearing_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[1].colorbar(im1)
        # im2 = grid_shift[image_count+2].imshow(acc_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[2].colorbar(im2)
        # im3 = grid_shift[image_count+3].imshow(delta_time_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[3].colorbar(im3)
        # im4 = grid_shift[image_count+4].imshow(count_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[4].colorbar(im4)
        # im5 = grid_shift[image_count+5].imshow(bearing_rate_array_final, cmap='jet', interpolation='nearest')
        # grid_shift.cbar_axes[5].colorbar(im5)
        # img_path = 'shift_image_class_fixed_pixel%d_%d.png'%(pixel_size, count)
        # fig.savefig(img_path)
        # count += 1

        # check nan
        if np.isnan(count_array).any() == True or np.isnan(speed_array_final).any() == True or np.isnan(deltatime_array_final).any() == True or np.isnan(bearing_array_final).any() == True:
            continue
        
        TITS_image_1feature.append([speed_array_final])
        TITS_image_2feature.append([speed_array_final, count_array])
        TITS_image_3feature.append([speed_array_final, count_array, acc_array_final])
        TITS_image_3feature2.append([speed_array_final, count_array, bearing_array_final])
        TITS_image_4feature.append([speed_array_final, count_array, acc_array_final, bearing_array_final])
        TITS_image_5feature.append([speed_array_final, count_array, acc_array_final, bearing_array_final, bearing_rate_array_final])
        TITS_image_6feature.append([speed_array_final, count_array, acc_array_final, bearing_array_final, bearing_rate_array_final, deltatime_array_final])      
        
        label_list.append(trip[1])
    
    return TITS_image_1feature, TITS_image_2feature, TITS_image_3feature, TITS_image_3feature2, TITS_image_4feature, TITS_image_5feature, TITS_image_6feature, label_list, scalesize_list



other_features = True
# normalize = True
interplotion = False
pixel_size = 40
print('pixel size:', pixel_size)

seg_size = 600
trip_min = 20
num_class = 6
pd.set_option('float_format', lambda x: '%.6f' % x)

# Mode_Index = {0: "walk",  1: "bike",  2: "car", 3: "taxi", 4: "bus", 5: "subway", 6: "train"}
Mode_Index = {0: "walk",  1: "bike",  2: "car", 2: "taxi", 3: "bus", 4: "subway", 5: "train"}
speed_list = {0: 5, 1: 10, 2: 30, 3: 25, 4: 20, 5: 80}


# --> trips
# filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/Trajectory_Label_Geolife_sorted_%dclass_limit_train&test_degree180_segsize%d_triptime%d.pickle'%(num_class, seg_size, trip_min)

filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/Trajectory_Label_Geolife_sorted_6class_limit_train&test_degree180_segsize600_triptime20.pickle'

with open(filename, 'rb') as f:
    trip_motion_all_user_with_label_train, trip_motion_all_user_with_label_test = pickle.load(f)
# [delta_time, speed, bearing, acc, bearing_rate, x_coord, y_coord]

class_trip = {}
for i in range(len(trip_motion_all_user_with_label_train)):
    if trip_motion_all_user_with_label_train[i][1] not in class_trip:
        class_trip[trip_motion_all_user_with_label_train[i][1]] = []
        class_trip[trip_motion_all_user_with_label_train[i][1]].append(trip_motion_all_user_with_label_train[i])
    else:
        class_trip[trip_motion_all_user_with_label_train[i][1]].append(trip_motion_all_user_with_label_train[i])

for i in range(len(trip_motion_all_user_with_label_test)):
    if trip_motion_all_user_with_label_test[i][1] not in class_trip:
        class_trip[trip_motion_all_user_with_label_test[i][1]] = []
        class_trip[trip_motion_all_user_with_label_test[i][1]].append(trip_motion_all_user_with_label_test[i])
    else:
        class_trip[trip_motion_all_user_with_label_test[i][1]].append(trip_motion_all_user_with_label_test[i])

print([len(class_trip[k]) for k in class_trip])

latlon_gap = None
if latlon_gap == True:
    max_class_all_gap = {}
    min_class_all_gap = {}
    all_lat_gap = []
    all_lon_gap = []
    for class_id in class_trip.keys():
        temp = 0
        k = -1
        for i in range(len(class_trip[class_id])):
            if max(class_trip[class_id][i][0][-2]) - min(class_trip[class_id][i][0][-2]) > temp:
                temp = max(class_trip[class_id][i][0][-2]) - min(class_trip[class_id][i][0][-2])
                k = i

        lat_gap = [(max(trip[0][-2]) - min(trip[0][-2])) % 360 for trip in class_trip[class_id]]
        lon_gap = [(max(trip[0][-1]) - min(trip[0][-1])) % 360 for trip in class_trip[class_id]]

        all_lat_gap += lat_gap
        all_lon_gap += lon_gap

        max_lat_gap_list = list(pd.Series(lat_gap).describe(percentiles=[0.05, 0.1, 0.2, 0.9, 1]))
        max_lon_gap_list = list(pd.Series(lon_gap).describe(percentiles=[0.05, 0.1, 0.2, 0.9, 1]))
        min_lat_gap_list = list(pd.Series(lat_gap).describe(percentiles=[0.05, 0.1, 1]))
        min_lon_gap_list = list(pd.Series(lon_gap).describe(percentiles=[0.05, 0.1, 1]))

        # get 30%-80% data
        max_class_all_gap[class_id] = [max_lat_gap_list[-3], max_lon_gap_list[-3]]
        min_class_all_gap[class_id] = [min_lat_gap_list[-3], min_lon_gap_list[-3]]

    # print(f'max class_all_gap: {max_class_all_gap}; min class all gap: {min_class_all_gap}')

    # lat gap for all trips
    print('Descriptive statistics for labeled all trip lat',  pd.Series(all_lat_gap).describe(percentiles=[0.05, 0.1, 0.2, 0.5, 0.6, 
                                                                                            0.7, 0.8, 0.9, 1]))
    print('Descriptive statistics for labeled all trip lon',  pd.Series(all_lon_gap).describe(percentiles=[0.05, 0.1, 0.2, 0.5, 0.6, 
                                                                                            0.7, 0.8, 0.9, 1]))


feature_max = feature_maxvalue()
trip_motion_all_user_with_label_train = normalize(trip_motion_all_user_with_label_train)
trip_motion_all_user_with_label_test = normalize(trip_motion_all_user_with_label_test)

# train data
train_TITS_image_1feature, train_TITS_image_2feature, train_TITS_image_3feature, \
train_TITS_image_3feature2, train_TITS_image_4feature, train_TITS_image_5feature, \
train_TITS_image_6feature, train_data_label, train_data_scale = \
        traj_to_image_shift_scale(traj_data=trip_motion_all_user_with_label_train, shift=True)
## test data
test_TITS_image_1feature, test_TITS_image_2feature, test_TITS_image_3feature, \
test_TITS_image_3feature2, test_TITS_image_4feature, test_TITS_image_5feature, \
test_TITS_image_6feature, test_data_label, test_data_scale = \
        traj_to_image_shift_scale(traj_data=trip_motion_all_user_with_label_test, shift=True)


# print(len(train_TITS_image_5feature), len(train_data_label))
print(len(test_TITS_image_5feature), len(test_data_label))

# # # # save the file
filename_TITS_123456 = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/trips_traj2image_trip_shift_rescale_%dclass_pixelsize%d_back0_180bearing_unfixed_TITS.pickle'%(num_class, pixel_size)
with open(filename_TITS_123456, 'wb') as f:
    pickle.dump([train_TITS_image_1feature, train_TITS_image_2feature, train_TITS_image_3feature, 
                 train_TITS_image_3feature2, train_TITS_image_4feature, train_TITS_image_5feature, 
                 train_TITS_image_6feature, train_data_label, train_data_scale,
                 test_TITS_image_1feature, test_TITS_image_2feature, test_TITS_image_3feature, 
                 test_TITS_image_3feature2, test_TITS_image_4feature, test_TITS_image_5feature, 
                 test_TITS_image_6feature, test_data_label, test_data_scale], f)

with open(filename_TITS_123456, 'rb') as f:
    dataset = pickle.load(f)

k = 0
train_TITSdata_1feature = dataset[0]
train_TITSdata_2feature = dataset[1]
train_TITSdata_3feature = dataset[2]
train_TITSdata_3feature2 = dataset[3]
train_TITSdata_4feature = dataset[4]
train_TITSdata_5feature = dataset[5]
train_TITSdata_6feature = dataset[6]
train_data_label = dataset[7]
train_data_scale = dataset[8]

# test_TITSdata_1feature = dataset[8]
# test_TITSdata_2feature = dataset[9]
# test_TITSdata_3feature = dataset[10]
# test_TITSdata_3feature2 = dataset[11]
# test_TITSdata_4feature = dataset[12]
# test_TITSdata_5feature = dataset[13]
# test_TITSdata_6feature = dataset[14]
# test_data_label = dataset[15]

test_TITSdata_1feature = dataset[9]
test_TITSdata_2feature = dataset[10]
test_TITSdata_3feature = dataset[11]
test_TITSdata_3feature2 = dataset[12]
test_TITSdata_4feature = dataset[13]
test_TITSdata_5feature = dataset[14]
test_TITSdata_6feature = dataset[15]
test_data_label = dataset[16]
test_data_scale = dataset[17]

from random import sample

def resample_traj(TITSdata_nfeature, data_label, random, sample_size):
    
    path = 'cls_image_triptime%d_segsize%d'%(trip_min, seg_size)
    isExists = os.path.exists(path)
    if isExists:
        shutil.rmtree(path)
    os.makedirs(path)

    class_dict_train = {}

    for i in range(len(data_label)):
        if data_label[i] not in class_dict_train:
            class_dict_train[data_label[i]] = []
            class_dict_train[data_label[i]].append(TITSdata_nfeature[i])
        else:
            class_dict_train[data_label[i]].append(TITSdata_nfeature[i])
       
    image_count = 0
    fig = plt.figure()
    fig.set_facecolor('white')
    grid_shift = axes_grid1.AxesGrid(
        fig, 111, nrows_ncols=(1, 6), axes_pad = 0.5, cbar_location = "right",
        cbar_mode="each", cbar_size="15%", cbar_pad="5%",)
    # fig = plt.figure()

    count = 0
    
    for i in class_dict_train.keys():
        sample_img = []
        if random == True:
            sample_img = sample(class_dict_train[i], sample_size)
        elif random == False:
            sample_img = class_dict_train[i][:sample_size]
                
        for img in sample_img: 

            speed_array_final = img[0]  
            count_final = img[1]     
            acc_final = img[2]
            bearing_array_final = img[3]
            bearing_rate_array_final = img[4]
            # bearing_rate_array_final = img[5]

            # speed_array_final, count_array, acc_array_final, bearing_array_final, bearing_rate_array_final, deltatime_array_final 
            im0 = grid_shift[image_count].imshow(speed_array_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[0].colorbar(im0)

            im1 = grid_shift[image_count+1].imshow(count_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[1].colorbar(im1)   

            im2 = grid_shift[image_count+2].imshow(acc_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[2].colorbar(im2)

            im3 = grid_shift[image_count+3].imshow(bearing_array_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[3].colorbar(im3)    

            im4 = grid_shift[image_count+4].imshow(bearing_rate_array_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[4].colorbar(im4)

            # im5 = grid_shift[image_count+5].imshow(deltatime_array_final, cmap='jet', interpolation='nearest')
            # grid_shift.cbar_axes[5].colorbar(im5)

            img_path = 'image_class_fixed_pixel%d_%s_%d.png'%(pixel_size, str(Mode_Index[i]), count)
            print(img_path)
            fig.savefig(img_path)
            plt.title('fig.%d'%count)
            count += 1

resample_traj(test_TITSdata_5feature, test_data_label, random=False, sample_size=5)


