from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import csv
import cv2
import os
import shutil



class Sampler:
    def __init__(self):
        self.output_dir = 'samples/'
        self.output_shape = None
        """
        `output_shape` will ratate (random wise) and crop (random center) frames into a ratio. 
        It should be (width, height) packed by tuple or list. Set "None" will turn off it.
        """
    
    
    def __create_dir(self, dir_path):     
        if dir_path != '':
            if dir_path[-1] != '/':
                dir_path += '/'
                self.output_dir = dir_path
            try:
                os.mkdir(dir_path)
            except:
                pass

                
    def __show_progress(self, progress, total, title='Sampling'):
        progress += 1
        print(
            '[{}]:[{:<30s}] {:5.1f}%'.format(
                title, 
                '█'*int(progress*30 / total), 
                progress / total*100
                ), 
            end='\r'
        )
        
        if progress == total:
            print()
        return progress
    
    
    def cropping_image(self, image, ratio:[tuple, list, None]):
        if ratio != None:
            w = image.shape[1]
            h = image.shape[0]
            do_cropping = int(w/ratio[0]) != int(h/ratio[1])
            if do_cropping:
                cut_w = h/ratio[1] - w/ratio[0] < 0
                if cut_w:
                    size = int(h*ratio[0] / ratio[1])
                    offset = np.random.randint(w - size + 1)
                    image = image[:, offset:offset+size]
                else:
                    size = int(w*ratio[1] / ratio[0])
                    offset = np.random.randint(h - size + 1)
                    image = image[offset:offset+size, :]
        return image    

    

class VideoSampler(Sampler):
    def __init__(self):
        super(VideoSampler, self).__init__()
        self.output_format = 'jpg'
        self.sample_period = 1
        self.period_scale = 's' 
        """
        period_scale: 's', 'm', 'h'
        """
    
    
    def __time_scale(self, scale):
        try:
            assert self.period_scale in ['s', 'm', 'h']
        except:
            raise AssertionError('The period scale should be "s": second, "m": minute, or "h": hour.')
            
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
    
    
    def sample_all_frames(self):
        self._Sampler__create_dir(self.output_dir)
        idx = 0
        self.vid.set(1, 0)  # restart video
        while self.vid.isOpened():
            success, frame = self.vid.read()
            if success:
                frame = self.cropping_image(frame, self.output_shape)
                name = self.output_dir + self.name + '_{:0>6d}.'.format(idx) + self.output_format
                cv2.imwrite(name, frame)
                idx = self._Sampler__show_progress(idx, self.total_frames)
            else:
                break
        print('Done sampling: totally {} frames.'.format(self.total_frames))
        return self.total_frames
    
    
    def sample_by_uniform(self, mode='random'):
        assert mode in ['random', 'center']
        self._Sampler__create_dir(self.output_dir)  # create folder
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
            frame = self.cropping_image(frame, self.output_shape)
            name = self.output_dir + self.name + '_{:0>6d}.'.format(idx) + self.output_format
            cv2.imwrite(name, frame)
            idx = self._Sampler__show_progress(idx, total)
        print('Done sampling: totally {} frames.'.format(total))

        return total    
    

    def sample_by_kmeans(self, pixel_stride=10):
        self._Sampler__create_dir(self.output_dir)  # create folder
        group_size = int(self.sample_period * self.__time_scale(self.period_scale) * self.fps)
        cluster_n = self.total_frames // int(group_size)
        vectors = []
        
        self.vid.set(1, 0) # restart video
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
                prgrs = self._Sampler__show_progress(prgrs, total, title='Initializing')
            else:
                break
                
        # cluster
        print('Clustering initializing...')
        vectors = np.array(vectors)
        estimator = KMeans(n_clusters=cluster_n, n_init=1, verbose=1)
        estimator.fit(vectors)
        centers = estimator.cluster_centers_
        # get index of the nearest center frames
        sample_idx, _ = pairwise_distances_argmin_min(centers, vectors)
        
        prgrs = 0
        for f_no in sample_idx:
            self.vid.set(1, f_no)
            _, frame = self.vid.read()
            frame = self.cropping_image(frame, self.output_shape)
            name = self.output_dir + self.name + '_{:0>6d}.'.format(f_no) + self.output_format
            cv2.imwrite(name, frame)
            prgrs = self._Sampler__show_progress(prgrs, len(sample_idx))
        print('Done sampling: totally {} frames.'.format(len(sample_idx)))
        return len(sample_idx)
    
    
    def sample_by_detector(self, detector, skip_ratio:float=1, save_bbox:bool=True):
        """
        detector: is a function. It should be inputted a numpy array image and return objection with the following format:
        [[xmin,ymin,xmax,ymax,label], [xmin,ymin,xmax,ymax,label], ...]

        skip_ratio: skip sampling after detecting "skip_ratio" of total frames in an interval to avoid too long sampling time.
        save_bbox: save bounding-box data as csv-file.
        """
        assert 0. < skip_ratio <= 1.
        self._Sampler__create_dir(self.output_dir)  # create folder
        
        group_size = int(self.sample_period * self.__time_scale(self.period_scale) * self.fps)
        sample_size = self.total_frames // group_size
        assert sample_size > 0
        
        bbox_list = []
        prgrs = 0
        total = 0
        for g in range(sample_size):
            # create sample group index and shuffle
            sample_idx = np.arange(0, group_size) + g*group_size  
            np.random.choice(sample_idx, size=int(skip_ratio*group_size), replace=False)
            
            # do sampling in this group
            for f_no in sample_idx:
                self.vid.set(1, f_no)
                success, frame = self.vid.read()
                # detect frame
                if success:
                    frame = self.cropping_image(frame, self.output_shape)
                    d_frame = frame.copy()
                    bbox = detector(d_frame)
                
                    # (1) save the frame if there is an object
                    if bbox != []:
                        name = self.output_dir + self.name + '_{:0>6d}.'.format(f_no) + self.output_format
                        cv2.imwrite(name, frame)
                        total += 1
                        if save_bbox:
                            for b in bbox:
                                line = [os.path.split(name)[-1]] + b
                                bbox_list.append(line)
                        break

            prgrs = self._Sampler__show_progress(prgrs, sample_size)
            
        if save_bbox:
            fields = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'image']
            name = self.output_dir + self.name + '_anno.csv'
            with open(name, 'w') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(fields)
                csv_writer.writerows(bbox_list)
                
        print('Done sampling: totally {} frames.'.format(total))
        return total 
    
    
    
class ImageSampler(Sampler):
    def __init__(self):
        super(ImageSampler, self).__init__()
        self.image_path_list = []
        self.sample_ratio = 0.3
        self.copy_image = True
    
    
    def __save_samples(self, sample_list):
        # copy or move images
        if self.copy_image:
            for img in sample_list: 
                dst = self.output_dir + os.path.split(img)[-1]
                shutil.copyfile(img, dst)
        else:
            for img in sample_list: 
                dst = self.output_dir + os.path.split(img)[-1]
                shutil.move(img, dst)
                
    
    def sample_by_uniform(self, mode='random'):
        assert mode in ['random', 'center']
        self._Sampler__create_dir(self.output_dir)  # create folder
        prgrs = 0
        
        assert self.sample_ratio > 0
        sample_size = int(len(self.image_path_list)*self.sample_ratio)

        
        # get sample list
        if mode == 'random':
            sample_list = list(np.random.choice(self.image_path_list, size=sample_size, replace=False))
        elif mode == 'center':
            sample_list = self.image_path_list[::int(1/self.sample_ratio)]
        
        self.__save_samples(sample_list)

        print('Done sampling: totally {} images.'.format(len(sample_list)))
        return len(sample_list)
    
    
    def sample_by_kmeans(self, pixel_stride=10):
        self._Sampler__create_dir(self.output_dir)  # create folder
        cluster_n = int(len(self.image_path_list)*self.sample_ratio)
        vectors = []
        
        # pixel sampling
        total = len(self.image_path_list)
        prgrs = 0
        for img_path in self.image_path_list:
            img =cv2.imread(img_path)
            # sampling with stride
            img_sample = img[::pixel_stride, ::pixel_stride, :]
            # reshape to 1D
            img_sample = img_sample.reshape(-1)
            vectors.append(img_sample)
            prgrs = self._Sampler__show_progress(prgrs, total, title='Initializing')
        
        # cluster
        print('Clustering initializing...')
        vectors = np.array(vectors)
        estimator = KMeans(n_clusters=cluster_n, n_init=1, verbose=1)
        estimator.fit(vectors)
        centers = estimator.cluster_centers_
        # get index of the nearest center frames
        sample_idx, _ = pairwise_distances_argmin_min(centers, vectors)
        sample_idx = np.sort(sample_idx)
        
        sample_list = [self.image_path_list[i] for i in sample_idx]
            
        self.__save_samples(sample_list)
        print('Done sampling: totally {} images.'.format(len(sample_list)))
        return len(sample_list)
    
    
    def sample_by_detector(self, detector, skip_ratio:float=1, save_bbox:bool=True):
        pass
