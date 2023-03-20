
   
import pickle
from datetime import datetime
import os
import time
#start_time = time.clock()
import pdb
import numpy as np


def days_date(time_str):
    date_format = "%Y/%m/%d %H:%M:%S"
    current = datetime.strptime(time_str, date_format)
    date_format = "%Y/%m/%d"
    bench = datetime.strptime('1899/12/30', date_format)
    no_days = current - bench
    delta_time_seconds = no_days.days * 24 * 3600 + current.hour * 3600 + current.minute * 60 + current.second
    return delta_time_seconds

def second_data(day, time):
    date_format = "%Y-%m-%d"
    current = datetime.strptime(day, date_format)
    date_format = "%Y/%m/%d"
    bench = datetime.strptime('1899/12/30', date_format)
    time_format = "%H:%M:%S"
    extra_time = datetime.strptime(time, time_format)
    no_days = current - bench
    delta_time_seconds = no_days.days * 24 * 3600 + extra_time.hour * 3600 + extra_time.minute * 60 + extra_time.second
    return delta_time_seconds


# Change Mode Name to Mode index
Mode_Index = {"walk": 0,  "bike": 1,  "car": 2, "taxi": 2, "bus": 3, "subway": 4, "train": 5, 
                "railway": 9, "motocycle": 9, "boat": 9, "airplane": 9, "other": 9, "run": 9}

# Ground modes are the modes that we use in this paper.
Ground_Mode = ['walk', 'bike', 'bus', 'car', 'taxi', 'subway', 'train']

geolife_dir = '/home/xieyuan/Transportation-mode/Data/Geolife_Trajectories_1.3-Raw-All/Data/'
users_folder = os.listdir(geolife_dir)
trajectory_all_user_wo_label = []
trajectory_all_user_with_label = []
label_all_user = []
cnt=0
for folder in users_folder:
    count = 0
    if len(os.listdir(geolife_dir + folder)) == 1:
        trajectory_dir = geolife_dir + folder + '/Trajectory/'
        user_trajectories = os.listdir(trajectory_dir)
        #sort all the .plt files in trajectory_dir
        user_trajectories = list(map(lambda sorted_plt : str(sorted_plt) + '.plt', sorted(list(map(lambda x: int(x.rstrip('.plt')), 
        user_trajectories)))))
        trajectory_one_user = []
        for plt in user_trajectories:
            cnt+=1
            with open(trajectory_dir + plt, 'r', newline='', encoding='utf-8') as f:
                GPS_logs = filter(lambda x: len(x.split(',')) == 7, f)
                GPS_logs_split = map(lambda x: x.rstrip('\r\n').split(','), GPS_logs)
                for row in GPS_logs_split:
                    trajectory_one_user.append([float(row[0]), float(row[1]), second_data(row[-2], row[-1])])

        trajectory_all_user_wo_label.append(trajectory_one_user)

    elif len(os.listdir(geolife_dir + folder)) == 2:
        trajectory_dir = geolife_dir + folder + '/Trajectory/'
        user_trajectories = os.listdir(trajectory_dir)
        #sort all the .plt files in trajectory_dir
        user_trajectories = list(map(lambda sorted_plt : str(sorted_plt) + '.plt', sorted(list(map(lambda x: int(x.rstrip('.plt')), 
        user_trajectories)))))
        trajectory_one_user = []
        for plt in user_trajectories:
            cnt+=1
            with open(trajectory_dir + plt, 'r', newline='', encoding='utf-8') as f:
                GPS_logs = filter(lambda x: len(x.split(',')) == 7, f)
                GPS_logs_split = map(lambda x: x.rstrip('\r\n').split(','), GPS_logs)
                for row in GPS_logs_split:
                    trajectory_one_user.append([float(row[0]), float(row[1]), second_data(row[-2], row[-1])])
        trajectory_all_user_with_label.append(trajectory_one_user)

        label_dir = geolife_dir + folder + '/labels.txt'
        with open(label_dir, 'r', newline='', encoding='utf-8') as f:
            print(label_dir)
            label = list(map(lambda x: x.rstrip('\r\n').split('\t'), f))
            label_filter = list(filter(lambda x: len(x) == 3 and x[2] in Ground_Mode, label))
            label_one_user = []
            for row in label_filter:
                label_one_user.append([days_date(row[0]), days_date(row[1]), Mode_Index[row[2]]])
        label_all_user.append(label_one_user)
print(f'users with label: {len(trajectory_all_user_with_label)}; users without labels: {len( trajectory_all_user_wo_label)}; total trajs: {cnt}') # 69, 113


trajectory_all_user_with_label_Final = []  # Only contain users' trajectories that have labels
for index, user in enumerate(label_all_user):

    trajectory_user = trajectory_all_user_with_label[index] #(lat, long, date)
    classes = {0: [], 1: [], 2: [], 3: [], 4:[], 5:[]}
    start_index = 0
    for row in user: # (start, end, mode)
        if not trajectory_user:
            break

        start = row[0]
        end = row[1]
        mode = row[2]

        # print(f'traj user: {trajectory_user[0]}; {len(trajectory_user)}')

        if trajectory_user[0][2] > end or trajectory_user[-1][2] < start:
            continue

        for index1, trajectory in enumerate(trajectory_user):
            if start <= trajectory[2] <= end:
                start_index += index1
                trajectory_user = trajectory_user[index1 + 1:]
                break

        if trajectory_user[-1][2] < end:
            end_index = start_index + 1 + len(trajectory_user)
            classes[mode].extend(list(range(start_index, end_index)))
            break
        else:
            for index2, trajectory in enumerate(trajectory_user):
                if trajectory[2] > end:
                    end_index = start_index + 1 + index2
                    trajectory_user = trajectory_user[index2 + 1:]
                    classes[mode].extend(list(range(start_index, end_index)))
                    start_index = end_index + 1
                    break

    # make valid pts (lat,long,date) -> (lat,long,date,mode)
    for k, v in classes.items():
        # print(index, k, len(trajectory_all_user_with_label[index]), len(v))
        for value in v:
            try:
                trajectory_all_user_with_label[index][value].append(k)
            except:
                #print(index, k, len(trajectory_all_user_with_label[index]), len(v))
                continue
                pdb.set_trace()
        
    #labeled_trajectory:
    labeled_trajectory = list(filter(lambda x: len(x) == 4, trajectory_all_user_with_label[index]))
    trajectory_all_user_with_label_Final.append(labeled_trajectory)
    # unlabel trajs
    unlabeled_trajectory = list(filter(lambda x: len(x) == 3, trajectory_all_user_with_label[index]))
    trajectory_all_user_wo_label.append(unlabeled_trajectory)

print(len(trajectory_all_user_with_label_Final), len(trajectory_all_user_wo_label)) # 69, 182


# Save Trajectory_Array and Label_Array for all users
with open('/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/Trajectory_Label_Geolife_6class_dataset.pickle', 'wb') as f:
    pickle.dump([trajectory_all_user_with_label_Final, trajectory_all_user_wo_label], f)


