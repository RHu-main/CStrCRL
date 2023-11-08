import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory(data, joint, save_file_path,ax_list, version='2d'):
    if version == '2d':
        # plt.clf()

        ske_traj = np.squeeze(data[:, joint, :])
        example_plot_self(ax_list[0], ske_traj, 0)
        example_plot_self(ax_list[1], ske_traj, 1)
        example_plot_self(ax_list[2], ske_traj, 2)
        #         ax1.plot(range(data.shape[0]),ske_traj[:,0])
        #         ax2.plot(range(data.shape[0]),ske_traj[:,1])
        #         ax3.plot(range(data.shape),ske_traj[:,2])
        plt.tight_layout()
        plt.title('Motion Trajectory for Joints Self', fontsize=24)


    elif version == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ske_traj = np.squeeze(data[:, joint, :])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ske_traj = np.squeeze(data[:, 1, :])
        ax.plot(ske_traj[:, 0], ske_traj[:, 1], ske_traj[:, 2], marker='o', linestyle='-', color='b')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title('3D Motion Trajectory')
        # ax.legend()

        # Display the 3D plot


def example_plot_self(ax, data, axis):
    ax.plot(range(data.shape[0]), data[:, axis])
    ax.set_xlabel('Frames', fontsize=16)
    ax.set_ylabel(f'relative location {axis}', fontsize=16)

def read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = []
    bodymat['njoints'] = njoints
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount == 0:
            continue
            # skip the empty frame
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)

            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame, joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame, joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame, joint] = jointinfo[5:7]
    # prune the abundant bodys
    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
    return bodymat
def motion_trajectory_self(ske_data,max_body=4):
    new_ske_motion = np.zeros((ske_data['skel_body0'].shape[0],25,3,max_body))
    for body in range(max_body):
        if 'skel_body{}'.format(body) in ske_data.keys():
            ske_array = ske_data['skel_body{}'.format(body)]
            for i in range(ske_array.shape[1]):
                for j in range(ske_array.shape[0]-1):
                    new_ske_motion[j,i,:,body]=np.subtract(ske_array[j+1,i,:], ske_array[0,i,:])
        else:
            continue
    return new_ske_motion

def motion_trajectory_joints(ske_data,max_body=4):
    new_ske_motion = np.zeros((ske_data['skel_body0'].shape[0],25,25,3,max_body))
    for body in range(max_body):
        if 'skel_body{}'.format(body) in ske_data.keys():
            ske_array = ske_data['skel_body{}'.format(body)]
            for i in range(25):
                for j in range(25):
                    new_ske_motion[:,i,j,:,body] = ske_array[:,j,:]-ske_array[:,i,:]
        else:
            continue
    return new_ske_motion
def read_ske_files(root_path,
                   format_str='.skeleton',save_path = r'./', missing_file_path = '/disk/XhWorks/h2/dataset/NTU_RGB+D_120/NTU_RGBD120_samples_with_missing_skeletons.txt'):

    if missing_file_path != None:
        with open(missing_file_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    dir_list = os.listdir(root_path)
    dir_list.sort()
    for (idx, ske_name) in enumerate(dir_list):
        if ske_name in ignored_samples:
            continue
        else:
            #     stat_path = osp.join(save_path, 'statistics')
            #     skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
            #     skes_name = np.loadtxt(skes_name_file, dtype=str)
            #     num_files = skes_name.size
            #     print('Found %d available skeleton files.' % num_files)
            #     for (idx,ske_name) in enumerate(skes_name):
            ske_file = osp.join(root_path, ske_name)
            bodies_data = read_skeleton(ske_file)
            ske_joints_self_motion = motion_trajectory_self(bodies_data)
            ske_joints_joints_motion = motion_trajectory_joints(bodies_data)
            action_class = int(ske_name[ske_name.find('A') + 1:ske_name.find('A') + 4])
            camera_class = int(ske_name[ske_name.find('C') + 1:ske_name.find('C') + 4])
            print(ske_name.split('.')[0])
            save_file_path = os.path.join(save_path, str(action_class), str(camera_class), ske_name.split('.')[0])
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)
            body_parts = {'down': [0, 12, 13, 14, 15, 16, 17, 18, 19], 'left_arm': [4, 5, 6], 'left_hand': [7, 21, 22],
                          'right_arm': [8, 9, 10], 'right_hand': [11, 23, 24], 'head_body': [3, 2, 1, 20]}
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            for i in range(25):  # whole body joints
                plot_trajectory(ske_joints_self_motion, i, save_file_path,[ax1,ax2,ax3])
            plt.savefig(save_file_path + f'2dversion_whole.png')

            for k, v in body_parts.items():
                save_file_bd_path = osp.join(save_file_path, str(k))
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
                for j in v:
                    plot_trajectory(ske_joints_self_motion, j, save_file_bd_path,[ax1,ax2,ax3])
                plt.savefig(save_file_path + f'/2dversion_part_{k}.png')
            # file name -class-
        #  -camera-
        #  -skeleton-
        #  -body part-

if __name__ == '__main__':
    ske_path = '/disk/XhWorks/h2/dataset/NTU_RGB+D_120/nturgbd_skeletons_s018_to_s032/'
    save_path = '/disk/XhWorks/h2/dataset/NTU_RGB+D_120/NTURGB_raw_ske_visualization/'
    read_ske_files(ske_path,save_path=save_path)