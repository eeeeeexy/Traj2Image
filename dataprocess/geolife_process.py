
from cmath import isnan
import numpy as np
import pickle
#from geopy.distance import vincenty
from geopy.distance import geodesic
import math
import time
import random
import pandas as pd
import pdb


# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.

#current = time.clock()
min_threshold = 20
max_threshold = 248
min_distance = 150
min_time = 60

prefix = '/datafiles'
DATASET='/Geolife' # &MTL
mode='interpolatedLinear_5s'
interpolation= False

trip_min = 20
seg_size = 600
num_class = 6

filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/Trajectory_Label_Geolife_%dclass_dataset.pickle'%num_class

with open(filename, 'rb') as f:
    trajectory_all_user_with_label_initial, trajectory_all_user_wo_label = pickle.load(f)


trajectory_all_user_with_label_Final = []
for trajectory in trajectory_all_user_with_label_initial:
    i = 0
    trajectory_new = []
    while i < len(trajectory) - 1:
        if len(trajectory_new) == 0:
            trajectory_new.append(trajectory[i])
            i += 1
        
        if trajectory[i] != trajectory_new[-1] and trajectory[i][2] != trajectory_new[-1][2]:
            trajectory_new.append(trajectory[i])
            i += 1
        else:
            # print(f'diff1: {trajectory[i][2]}, diff2: {trajectory_new[-1][2]}')
            # print(f'diff1: {trajectory[i]}, diff2: {trajectory_new[-1]}')
            i += 1
    
    if (len(trajectory_new)) > 0:
        trajectory_all_user_with_label_Final.append(trajectory_new) 


def interpolate_mid_pt(pt1, pt2, ratio):
    if 'interpolatedNAN' in mode:
        new_pt = [np.NAN,np.NAN,np.NAN,np.NAN]
    elif 'interpolatedLinear' in mode:
        new_pt = [i*ratio+j*(1-ratio) for i, j in zip(pt1, pt2)]
    else:
        raise NotImplementedError
    return new_pt


def labeled_gps_to_trip(trajectory_one_user, trip_time):
    """
    This function divides total labeled-GPS trajectory of one user into some trips, when either the travel time between
    two consecutive GPS points exceeds the "trip time" or the mode changes.
    Also, remove the erroneous GPS points that their travel time is non-positive.
    :param trajectory_one_user: A sequence of a users' all GPS points.
    :param trip_time: the maximum time for dividing a GPS sequence into trips.
    :return: a user's  trips
    """
    # global total_cnt, diff_cnt, diff_list
    trip_time *= 60
    trip = []
    all_trip_one_user = []
    i = 0
    # i = len(trajectory_one_user) - 10
    # print(trajectory_one_user[i+1][2],len(trajectory_one_user))
    while i < len(trajectory_one_user) - 1: 

        delta_time = (trajectory_one_user[i+1][2] - trajectory_one_user[i][2])  # the unit is the second
        mode_not_change = (trajectory_one_user[i+1][3] == trajectory_one_user[i][3])
        
        if 0 < delta_time <= trip_time and mode_not_change:
            # trajectory_one_user[i].append(1)
            trip.append(trajectory_one_user[i])
            i += 1

        elif delta_time > trip_time or not mode_not_change:
            # trajectory_one_user[i].append(1)
            trip.append(trajectory_one_user[i])
            # all_trip_one_user.append(trip)
            if len(trip) > 1:
                all_trip_one_user.append(trip)
            trip = []
            i += 1
            
        elif delta_time <= 0:
            trajectory_one_user.remove(trajectory_one_user[i + 1])
    
    # print(f'num trips: {len(all_trip_one_user)}', [len(trip) for trip in all_trip_one_user])
            
    return all_trip_one_user


print('gps -> trip')
# trip_all_user_with_label = [labeled_gps_to_trip(trajectory, trip_min) for trajectory in trajectory_all_user_with_label_Final]
# num_trips = [len(labeled_gps_to_trip(trajectory, trip_min)) for trajectory in trajectory_all_user_with_label_Final]


filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/trips_temp_%dclass_trip%d.pickle'%(num_class, trip_min)
# with open(filename, 'wb') as f:
#     pickle.dump([trip_all_user_with_label], f)

with open(filename, 'rb') as f:
    trip_all_user_with_label = pickle.load(f)[0]

print(f'before insert trips: {np.array([len(trips) for trips in trip_all_user_with_label]).sum()}')

def insert_pts(trips_one_user, insert_size):

    inserted_trips_one_user = []

    for trip_traj in trips_one_user:
        inserted_trip = []
        inserted_trip.append(trip_traj[0])
        for i in range(len(trip_traj) - 1):

            delta_time = (trip_traj[i+1][2] - trip_traj[i][2])  # second
            
            if delta_time <= insert_size and delta_time != 0:
                inserted_trip.append(trip_traj[i+1])

            elif delta_time > insert_size:
                
                num_insert = round(delta_time / insert_size)               
                lat_span = (trip_traj[i+1][0] - trip_traj[i][0])/num_insert
                lon_span = (trip_traj[i+1][1] - trip_traj[i][1])/num_insert
                start = inserted_trip[-1]

                for s in range(num_insert - 1):
                    # lat, lon, second, mode
                    inserted_trip.append([start[0] + (s+1)*lat_span, start[1] + (s+1)*lon_span, start[2]+1 + s, start[-1]])

                inserted_trip.append(trip_traj[i+1])

        for i in range(len(inserted_trip) - 1):
            if (inserted_trip[i+1][-2] - inserted_trip[i][-2]) > 1:
                print(f'{i}: {inserted_trip[i]}')
                print(f'{i+1}: {inserted_trip[i+1]}')
        
        inserted_trips_one_user.append(inserted_trip)
    
    return inserted_trips_one_user
    
# trip_all_user_with_interpolation = [insert_pts(trips_one_user, insert_size=1) for trips_one_user in trip_all_user_with_label]


def compute_delta_time(p1, p2):
    """
    :param p2: trajectory_one_user[i + 1]
    :param p1: trajectory_one_user[i]
    :return:
    """
    return (p2[2] - p1[2])


def compute_distance(p1, p2):
    # print(p1, p2)
    lat_long_1 = (p1[0], p1[1])
    lat_long_2 = (p2[0], p2[1])
    if np.isnan(lat_long_1[0]) or np.isnan(lat_long_2[0]):
        return 0.
    try:
        if lat_long_1[0]==400.166666666667:
            lat_long_1=(40.166666666667, lat_long_1[1])
        if lat_long_2[0]==400.166666666667:
            lat_long_2=(40.166666666667, lat_long_2[1])
        return geodesic(lat_long_1, lat_long_2).meters
    except:
        pdb.set_trace()


def compute_speed(distance, delta_time, mode, speed_list):
    if distance/delta_time > speed_list[mode]:
        return speed_list[mode]
    else:
        return distance/delta_time


def compute_acceleration(speed1, speed2, delta_time):
    # if np.isnan(speed1) or np.isnan(speed2):
    #     return np.NAN
    return (speed2 - speed1) / delta_time


def compute_jerk(acc1, acc2, delta_time):
    if np.isnan(acc1) or np.isnan(acc2):
        return np.NAN
    return (acc2 - acc1) / delta_time


def compute_bearing(p1, p2):
    y = math.sin(math.radians(p2[1]) - math.radians(p1[1])) * math.cos(math.radians(p2[0]))
    x = math.cos(math.radians(p1[0])) * math.sin(math.radians(p2[0])) - \
        math.sin(math.radians(p1[0])) * math.cos(math.radians(p2[0])) \
        * math.cos(math.radians(p2[1]) - math.radians(p1[1]))
    # Convert radian from -pi to pi to [0, 360] degree
    return (math.atan2(y, x) * 180. / math.pi + 360) % 180


def compute_bearing_rate(bearing1, bearing2):
    return abs(bearing1 - bearing2)


def compute_trip_motion_features(all_trip_one_user):
    """
    This function computes the motion features for every trip (i.e., a sequence of GPS points).
    There are four types of motion features: speed, acceleration, jerk, and bearing rate.
    :param trip: a sequence of GPS points
    :param data_type: is it related to a 'labeled' and 'unlabeled' data set.
    :return: A list with four sub-lists, where every sub-list is a motion feature.
    """
    all_trip_motion_features_one_user = []
    speed_list = {0: 5, 1: 10, 2: 30, 3: 25, 4: 20, 5: 80}
    for trip in all_trip_one_user:

        # print(f'trip: {len(trip)}')

        if len(trip) < 3:
            continue

        mode = trip[0][3]
        x_coord=[]
        y_coord=[]
        delta_time = []
        speed = []
        bearing = []
        bearing_rate = []
        acc = []

        # the first and second points: {0, 1}
        delta_time_0 = compute_delta_time(trip[0], trip[1])
        distance_0 = compute_distance(trip[0], trip[1])
        speed0 = compute_speed(distance_0, delta_time_0, mode, speed_list)  
        delta_time_1 = compute_delta_time(trip[1], trip[2])
        distance_1 = compute_distance(trip[1], trip[2])
        speed1 = compute_speed(distance_1, delta_time_1, mode, speed_list)
        acc0 = compute_acceleration(speed0, speed1, delta_time_0)

        delta_time.append(delta_time_0)
        speed.append(speed0)
        bearing.append(compute_bearing(trip[0], trip[1]))
        
        x_coord.append(trip[0][0])
        y_coord.append(trip[0][1])

        delta_time.append(delta_time_1)
        speed.append(speed1)
        bearing.append(compute_bearing(trip[1], trip[2]))
        acc.append(acc0)
        bearing_rate.append(compute_bearing_rate(compute_bearing(trip[0], trip[1]), compute_bearing(trip[1], trip[2])))

        x_coord.append(trip[1][0])
        y_coord.append(trip[1][1])

        for i in range(len(trip) - 3):
            x_coord.append(trip[i+2][0])
            y_coord.append(trip[i+2][1])
            
            delta_time_1 = compute_delta_time(trip[i + 1], trip[i + 2])
            delta_time_2 = compute_delta_time(trip[i + 2], trip[i + 3])

            distance_2 = compute_distance(trip[i + 2], trip[i + 3])
            speed2 = compute_speed(distance_2, delta_time_2, mode, speed_list)
            acc2 = compute_acceleration(speed1, speed2, delta_time_1)

            delta_time.append(delta_time_2)
            speed.append(speed2)
            bearing.append(compute_bearing(trip[i + 2], trip[i + 3]))
            acc.append(acc2)
            bearing_rate.append(compute_bearing_rate(compute_bearing(trip[i + 1], trip[i + 2]), compute_bearing(trip[i + 2], trip[i + 3])))
            
            delta_time_1 = delta_time_2
            distance_0 = distance_1
            distance_1 = distance_2
            speed0 = speed1
            speed1 = speed2
        
        delta_time.append(delta_time[-1])
        speed.append(speed[-1])
        bearing.append(bearing[-1])
        acc.append(acc[-1])
        acc.append(acc[-1])
        bearing_rate.append(bearing_rate[-1])
        bearing_rate.append(bearing_rate[-1])

        x_coord.append(trip[-1][0])
        y_coord.append(trip[-1][1])

        trip_motion_features = [delta_time, speed, bearing, acc, bearing_rate, x_coord, y_coord]
        # print(f'acc: {len(acc)}; bearing rate: {len(bearing_rate)}')
        
        all_trip_motion_features_one_user.append((trip_motion_features, mode))

    return all_trip_motion_features_one_user


print('trip -> all feature')
# trip_motion_all_user_with_label = [compute_trip_motion_features(trip) for trip in trip_all_user_with_interpolation]


filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/trips_motion_features_%dclass_xy_limit_traintest_trip%d.pickle'%(num_class, trip_min)
# with open(filename, 'wb') as f:
#     pickle.dump([trip_motion_all_user_with_label], f)
with open(filename, 'rb') as f:
    trip_motion_all_user_with_label = pickle.load(f)[0]


for trip_oneuser in trip_motion_all_user_with_label:
    for trip in trip_oneuser:
        for i in range(len(trip[0][0])-1):
            if trip[0][0][i] > 1:
                print(f'{i} {trip[0][0][i]}')


print(len(trip_motion_all_user_with_label))


# Mode_Index = {"walk": 0, "bike": 1, "car": 2, "taxi": 3, "bus": 4, "subway": 5, "train": 6}
Mode_Index = {"walk": 0, "bike": 1, "car": 2, "bus": 3, "subway": 4, "train": 5}

class_dict={}
for users in trip_motion_all_user_with_label:
    for trip in users:
        if trip[1] not in class_dict:
            class_dict[trip[1]] = 1
        else:
            class_dict[trip[1]] += 1
print(class_dict) # {0: 4937, 1: 2667, 2: 1830, 3: 1162, 4: 4311, 5: 941, 6: 1858} & total= 17706 trips


class_dict_trips={}
for users in trip_motion_all_user_with_label:
    for trip in users:
        if trip[1] not in class_dict_trips:
            class_dict_trips[trip[1]] = []
            class_dict_trips[trip[1]].append(trip)
        else:
            class_dict_trips[trip[1]].append(trip)
print(len(class_dict_trips))


def trip_to_seg(all_trips, seg_size):

    all_segments = []

    for user_id, user_trips in enumerate(all_trips):

        for trip_id, trip in enumerate(user_trips):

            i = 0     
            mode = trip[1]
            segment = [[] for i in range(len(trip[0]))]
            feature = trip[0]
        
            if len(feature[0]) <= seg_size and len(feature[0]) >= 50:
                all_segments.append(trip)
                # print('short:', len(all_segments))

            else:
                while i < len(feature[0]) - 1:
                    if len(segment[0]) <= seg_size:
                        # [delta_time, speed, bearing, acc, bearing_rate, x_coord, y_coord]
                        segment[0].append(feature[0][i])
                        segment[1].append(feature[1][i])
                        segment[2].append(feature[2][i])
                        segment[3].append(feature[3][i])
                        segment[4].append(feature[4][i])
                        segment[5].append(feature[5][i])
                        segment[6].append(feature[6][i])
                    i += 1
                    if len(segment[0]) == seg_size:
                        all_segments.append((segment, mode))
                        segment = [[] for i in range(len(trip[0]))]
                
                if len(segment[0]) > 50:
                    all_segments.append((segment, mode))

        # print(user_id, len(all_segments))

    return all_segments

print(f'trips -> segments (size: {seg_size})')

# segments_all_trips = trip_to_seg(trip_motion_all_user_with_label, seg_size)
# print(f'segments: {len(segments_all_trips)}')

filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/segments_features_%dclass_limit_segments%d_triptime%d.pickle'%(num_class, seg_size, trip_min)
# with open(filename, 'wb') as f:
#     pickle.dump([segments_all_trips], f)
with open(filename, 'rb') as f:
    segments_all_trips = pickle.load(f)[0]
     

seg_len = [len(seg[0][0]) for seg in segments_all_trips]
print('Descriptive statistics for labeled test lat',  pd.Series(seg_len).describe(percentiles=[0.01, 0.02, 0.05, 0.15, 0.5, 0.6, 0.65, 
                                                                                            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]))

# Mode_Index = {"walk": 0, "bike": 1, "car": 2, "taxi": 3, "bus": 4, "subway": 5, "train": 6}
Mode_Index = {"walk": 0, "bike": 1, "car": 2, "bus": 3, "subway": 4, "train": 5}
class_dict={}
for segs in segments_all_trips:
    if segs[1] not in class_dict:
        class_dict[segs[1]] = 1
    else:
        class_dict[segs[1]] += 1
print(class_dict) # {0: 4937, 1: 2667, 2: 1830, 3: 1162, 4: 4311, 5: 941, 6: 1858} & total= 17706 trips


class_dict_segs={}
for segs in segments_all_trips:
    if segs[1] not in class_dict_segs:
        class_dict_segs[segs[1]] = []
        class_dict_segs[segs[1]].append(segs)
    else:
        class_dict_segs[segs[1]].append(segs)
print(len(class_dict_segs))

# import pdb; pdb.set_trace()

trajectory_test_user_with_label = []
trajectory_train_user_with_label = []
for cls_seg in class_dict_segs.keys():
    for i, segment in enumerate(class_dict_segs[cls_seg]):
        if len(segment)<=1:
            continue
        if i%5==0:
            trajectory_test_user_with_label.append(segment)
        else:
            trajectory_train_user_with_label.append(segment)
print(f'train dataset: {len(trajectory_train_user_with_label)}, test dataset: {len(trajectory_test_user_with_label)}')

for seg_id, seg in enumerate(trajectory_train_user_with_label):
    print(len(seg[0]), len(seg[0][0]))
    break

with open('/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/Trajectory_Label_Geolife_sorted_%dclass_limit_train&test_degree180_segsize%d_triptime%d.pickle'%(num_class, seg_size, trip_min), 'wb') as f:
    pickle.dump([trajectory_train_user_with_label, trajectory_test_user_with_label], f)