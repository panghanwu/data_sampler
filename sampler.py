from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import os
import cv2
import csv


class VideoSampler():
    def __init__(self):
        self.sample_period = 1
        self.period_scale = 's'
        self.output_dir = 'sampled_frames/'
        self.output_format = 'jpg'
        """
        period_scale: 's', 'm', 'h'
        output_shape: 'same', '1x1', '3x4', '16x9' 
        output_dir: '/' at the end
        """
        
    
    def __create_dir(self, dir_path):
        try:
            os.mkdir(dir_path)
        except:
            pass
        
    
    def __show_progress(self, progress, total, title='Sampling'):
        progress += 1
        print('\r[{}]:[{:<30s}] {:5.1f}%'.format(title, '█'*int(progress*30/total), 
                                                 progress/total*100), end='')
        if progress == total:
            print()
        return progress
    
    
    def __time_scale(self, scale):
        assert self.period_scale in ['s', 'm', 'h']
        if self.period_scale == 's':
            return 1
        elif self.period_scale == 'm':
            return 60
        elif self.period_scale == 'h':
            return 3600
            
    
    def load_video(self, path):
        self.vid = cv2.VideoCapture(path)
        
        self.width = int(self.vid.get(3))
        self.height = int(self.vid.get(4))
        self.fps = self.vid.get(5)
        self.total_frames = int(self.vid.get(7))
        
        name = os.path.split(path)[-1]
        self.name = os.path.splitext(name)[0]
        
        print('Load video "{}" successfully.'.format(self.name))
        print('Totally {} frames, w = {}, h = {}, FPS = {}.'.format(self.total_frames,
                                                                    self.width, 
                                                                    self.height, 
                                                                    self.fps))
    
    
    def sample_by_detector(self, detector, if_no_object='random', save_bbox=True):
        """
        The `detector` is a function.
        It should be inputted a numpy array image and return objection with the following format:
        [[xmin,ymin,xmax,ymax,label], [xmin,ymin,xmax,ymax,label], ...]
        """
        assert if_no_object in [None, 'random']
        self.__create_dir(self.output_dir)  # create folder
        
        group_size = int(self.sample_period * self.__time_scale(self.period_scale) * self.fps)
        sample_size = self.total_frames // int(group_size)
        assert sample_size > 0
        
        bbox_list = []
        idx = 0
        for g in range(sample_size):
            # create sample group index and shuffle
            sample_idx = np.arange(0, group_size) + g*group_size
            np.random.shuffle(sample_idx)
            n = 0
            while True:
                f_no = sample_idx[n]
                self.vid.set(1, f_no)
                success, frame = self.vid.read()
                # detect frame
                if success:
                    d_frame = frame.copy()
                    bbox = detector(d_frame)
                n += 1
                
                # (1) save the frame if there is an object
                # (2) operation when no object in this group
                if bbox != None:
                    name = self.output_dir + self.name + '_{:0>6d}.'.format(idx) + self.output_format
                    cv2.imwrite(name, frame)
                    if save_bbox:
                        for b in bbox:
                            line = [os.path.split(name)[-1]] + b
                            bbox_list.append(line)
                    break
                    
                elif n == group_size:
                    if if_no_object == 'random':
                        name = self.output_dir + self.name + '_{:0>6d}.'.format(idx) + self.output_format
                        cv2.imwrite(name, frame)
                        break
                    else:
                        break
            idx = self.__show_progress(idx, sample_size)
            
        if save_bbox:
            fields = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'image']
            name = self.output_dir + self.name + '_anno.csv'
            with open(name, 'w') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(fields)
                csv_writer.writerows(bbox_list)
                
        print('Done sampling: totally {} frames.'.format(sample_size))
    
    
    def sample_by_kmeans(self, pixel_stride=10, step=20):
        self.__create_dir(self.output_dir)  # create folder
        group_size = int(self.sample_period * self.__time_scale(self.period_scale) * self.fps)
        cluster_n = self.total_frames // int(group_size)
        vectors = []
        
        self.vid.set(1, 0)
        # pixel sampling
        total = self.total_frames
        prgrs = 0
        while self.vid.isOpened():
            success, frame = self.vid.read()
            if success:
                # sampling with stride
                img_sample = frame[::pixel_stride, ::pixel_stride, :]
                # reshape to 1D
                img_sample = img_sample.reshape(-1)
                vectors.append(img_sample)
                prgrs = self.__show_progress(prgrs, total, title='Initializing')
            else:
                break
        vectors = np.array(vectors)
        estimator = KMeans(n_clusters=cluster_n)
        estimator.fit(vectors)
        centers = estimator.cluster_centers_
        # get index of the nearest center frames
        sample_idx, _ = pairwise_distances_argmin_min(centers, vectors)
        sample_idx = np.sort(sample_idx)
        
        idx = 0
        for f_no in sample_idx:
            self.vid.set(1, f_no)
            _, frame = self.vid.read()
            name = self.output_dir + self.name + '_{:0>6d}.'.format(idx) + self.output_format
            cv2.imwrite(name, frame)
            idx = self.__show_progress(idx, len(sample_idx))
        print('Done sampling: totally {} frames.'.format(len(sample_idx)))
        
    
    def sample_by_uniform(self, mode='random'):
        assert mode in ['random', 'center']
        self.__create_dir(self.output_dir)  # create folder
        prgrs = 0
        
        group_size = int(self.sample_period * self.__time_scale(self.period_scale) * self.fps)
        sample_size = self.total_frames // int(group_size)
        assert sample_size > 0

        # generate index list of frames to sample
        if mode == 'random':
            sample_idx = [int(np.random.randint(group_size)+(group_size*i)) for i in range(sample_size)]
        elif mode == 'center':
            sample_idx = [int(group_size*(i+0.5)) for i in range(sample_size)]
        
        total = len(sample_idx)
        idx = 0

        # save frames in sample list
        for f_no in sample_idx:
            self.vid.set(1, f_no)
            _, frame = self.vid.read()
            name = self.output_dir + self.name + '_{:0>6d}.'.format(idx) + self.output_format
            cv2.imwrite(name, frame)
            idx = self.__show_progress(idx, total)
        print('Done sampling: totally {} frames.'.format(total))
    
    
    def sample_all_frames(self):
        idx = 0
        self.vid.set(1, 0)
        while self.vid.isOpened():
            success, frame = self.vid.read()
            if success:
                name = self.output_dir + self.name + '_{:0>6d}.'.format(idx) + self.output_format
                cv2.imwrite(name, frame)
                idx = self.__show_progress(idx, self.total_frames)
            else:
                break
        print('Done sampling: totally {} frames.'.format(self.total_frames))