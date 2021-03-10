from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
import numpy as np
import csv
import cv2
import os
import shutil


# TODO random
# TODO tqdm


class Sampler:
    """Parent Class of other sampler.

    Attributes:
        n_samples   : How many samples will be extracted.
        output_dir  : Output directory path.

    """
    def __init__(self):        
        self.n_samples = 1
        self.output_dir = 'outputs/'
        

    def create_dir(self, dir_path):     
        if dir_path != '':
            if dir_path[-1] != '/':
                dir_path += '/'
                self.output_dir = dir_path
            try:
                os.mkdir(dir_path)
            except:
                pass

    
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
    """A sampler for video.

    Attributes:
        n_samples    : How many samples will be extracted.
        output_dir   : Output directory path.
        output_shape : A tuple expressed as (width, height). Image Sample will 
                       be cropped randomly according shape ratio.
        output_format: Format of output frames.
        output_shape : A tuple expressed as (width, height). Image Sample will 
                       be cropped randomly according shape ratio.
    
        After use "VideoSampler.load()" load a video, the information can be getten by:
            total_frames, width, height, fps.

    """
    def __init__(self):
        super(VideoSampler, self).__init__()
        self.output_format = 'jpg'
        self.output_shape = None
    
    
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

    
    def load(self, path, show_info=True):
        """Load video and refresh video infomation.
        This function should be conducted every time when the target video changed.
        """

        self.vid = cv2.VideoCapture(path)
        
        self.width = int(self.vid.get(3))
        self.height = int(self.vid.get(4))
        self.fps = self.vid.get(5)
        self.total_frames = int(self.vid.get(7))
        
        name = os.path.basename(path)
        self.name = os.path.splitext(name)[0]
        
        if show_info:
            print(f'Load video "{self.name}" successfully.')
            print(f'Totally {self.total_frames} frames, w = {self.width}, h = {self.height}, FPS = {self.fps}.')
    
    
    def sample_all_frames(self, title=None):
        """Extract all frames from the target video.
        Returns: Total frames sampled.
        """
        self.create_dir(self.output_dir)
        self.vid.set(1, 0)  # restart video
        
        with tqdm(total=self.total_frames, desc=title, unit='f') as pbar:  # progress bar
            while self.vid.isOpened():
                success, frame = self.vid.read()
                if success:

                    if self.output_shape is not None:  # crop image
                        frame = self.cropping_image(frame, self.output_shape)

                    f_no = int(self.vid.get(0))
                    name = f'{self.output_dir}{self.name}_{f_no:0>6d}.{self.output_format}'
                    cv2.imwrite(name, frame)  # save image
                    pbar.update(1)  # update progress bar
                else:
                    break

        print(f'Totally {len(sample_idx)} frames sampling completed.')

        return self.total_frames
    
    
    def sample_by_uniform(self, mode='random', title=None):
        assert mode in ['random', 'center']
        self.create_dir(self.output_dir)  # create folder
        
        # create n_sample sample indexes by mode
        sample_idx = np.arange(self.total_frames)
        if mode == 'random':
            sample_idx = np.random.choice(sample_idx, self.n_samples, replace=False)
        elif mode == 'center':
            interval = self.total_frames // self.n_samples
            sample_idx = sample_idx[::interval]
        sample_idx = tqdm(sample_idx, desc=title, unit='f')

        # save frames in sample list
        for f_no in sample_idx:
            self.vid.set(1, f_no)
            _, frame = self.vid.read()

            if self.output_shape is not None:  # crop image
                frame = self.cropping_image(frame, self.output_shape)

            name = f'{self.output_dir}{self.name}_{f_no:0>6d}.{self.output_format}'
            cv2.imwrite(name, frame)

        print(f'Totally {len(sample_idx)} frames sampling completed.')

        return len(sample_idx)
    

    def sample_by_kmeans(self, pixel_stride=10):
        self.create_dir(self.output_dir)  # create folder

        self.vid.set(1, 0) # restart video
        # pixel sampling
        vectors = []
        with tqdm(total=self.total_frames, desc='Initializing', unit='f') as pbar:  # progress bar
            while self.vid.isOpened():
                success, frame = self.vid.read()
                if success:
                    # sampling with stride
                    img_sample = frame[::pixel_stride, ::pixel_stride, :]
                    # reshape to 1D
                    img_sample = img_sample.reshape(-1)
                    vectors.append(img_sample)
                    pbar.update(1)  # update progress bar
                else:
                    break
                
        # cluster
        print('Clustering initializing...')
        vectors = np.array(vectors)
        estimator = KMeans(n_clusters=self.n_samples, n_init=1, verbose=1)
        estimator.fit(vectors)
        centers = estimator.cluster_centers_
        # get index of the nearest center frames
        sample_idx, _ = pairwise_distances_argmin_min(centers, vectors)
        
        with tqdm(total=len(sample_idx), desc='Sampling', unit='f') as pbar:  # progress bar
            for f_no in sample_idx:
                self.vid.set(1, f_no)
                _, frame = self.vid.read()

                if self.output_shape is not None:  # crop image
                    frame = self.cropping_image(frame, self.output_shape)

                name = f'{self.output_dir}{self.name}_{f_no:0>6d}.{self.output_format}'
                cv2.imwrite(name, frame)
                pbar.update(1)  # update progress bar

        print(f'Totally {len(sample_idx)} frames sampling completed.')

        return len(sample_idx)
    
    
    def sample_by_detector(self, detector, save_bbox:bool=True, title=None, fields:list=None):
        """Sampling frames if the detector get objects.
        
        Args:
            detector : A function. Its input is numpy array, and returns list for example:
                       [label, xmin, ymin, xmax, ymax]
                       It needs to notice that the channels shold be (B, G, R) because of OpenCV.
            mode     : Assert in "random" and "uniform".
            save_bbox: save bounding-box data as csv-file.

        Returns: None
        Raises: None
        """
        
        self.create_dir(self.output_dir)  # create folder
        
        # create n_sample random sample indexes
        sample_idx = np.arange(self.total_frames)
        sample_idx = np.random.choice(sample_idx, self.n_samples, replace=False)
        sample_idx = tqdm(sample_idx, desc=title, unit='f')

        if save_bbox:
            bbox_list = []  # bbox container
        
        counter = 0
        # sample if detects objects
        for f_no in sample_idx:
            self.vid.set(1, f_no)
            _, frame = self.vid.read()

            if self.output_shape is not None:  # crop image
                    frame = self.cropping_image(frame, self.output_shape)

            # detect frame
            bbox = detector(frame)
            
            # (1) save the frame if there is an object
            if bbox != []:
                name = f'{self.output_dir}{self.name}_{f_no:0>6d}.{self.output_format}'
                cv2.imwrite(name, frame)
                if save_bbox:
                    for b in bbox:
                        line = [os.path.basename(name)] + b
                        bbox_list.append(line)
                counter += 1
            
        if save_bbox:
            name = f'{self.output_dir}{self.name}_anno.csv'
            with open(name, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                if fields is not None:
                    csv_writer.writerow(fields)
                csv_writer.writerows(bbox_list)
                
        print(f'Totally {counter} frames sampling completed.')

        return counter
    
    
    
class ImageSampler(Sampler):
    def __init__(self):
        super(ImageSampler, self).__init__()
        self.image_path_list = []
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
        self.create_dir(self.output_dir)  # create folder
        # get sample list
        if mode == 'random':
            sample_list = np.random.choice(self.image_path_list, self.n_samples, replace=False)
        elif mode == 'center':
            interval = len(self.image_path_list) // self.n_samples
            sample_list = self.image_path_list[::interval]
        
        self.__save_samples(sample_list)

        print(f'Totally {len(sample_list)} images sampling completed.')

        return len(sample_list)
    
    
    def sample_by_kmeans(self, pixel_stride=10):
        self.create_dir(self.output_dir)  # create folder
        vectors = []

        # pixel sampling
        with tqdm(total=len(self.image_path_list), desc='Initializing', unit='f') as pbar:
            for img_path in self.image_path_list:
                img =cv2.imread(img_path)
                # sampling with stride
                img_sample = img[::pixel_stride, ::pixel_stride, :]
                # reshape to 1D
                img_sample = img_sample.reshape(-1)
                vectors.append(img_sample)
                pbar.update(1)  # update progress bar
        
        # cluster
        print('Clustering initializing...')
        vectors = np.array(vectors)
        estimator = KMeans(n_clusters=self.n_samples, n_init=1, verbose=1)
        estimator.fit(vectors)
        centers = estimator.cluster_centers_
        # get index of the nearest center frames
        sample_idx, _ = pairwise_distances_argmin_min(centers, vectors)
        sample_idx = np.sort(sample_idx)
        
        sample_list = [self.image_path_list[i] for i in sample_idx]
            
        self.__save_samples(sample_list)

        print(f'Totally {len(sample_list)} images sampling completed.')

        return len(sample_list)
    
    
    def sample_by_detector(self, detector, save_bbox:bool=True, title=None, fields:list=None):
        self.create_dir(self.output_dir)  # create folder
        
        # create n_sample random sample indexes
        raw_list = np.random.choice(self.image_path_list, self.n_samples, replace=False)
        raw_list = tqdm(raw_list, desc=title, unit='f')

        if save_bbox:
            bbox_list = []  # bbox container
        
        counter = 0
        sample_list = []
        # sample if detects objects
        for img_path in raw_list:
            img = cv2.imread(img_path)
            # detect frame
            bbox = detector(img)
            
            # (1) save the frame if there is an object
            if bbox != []:
                sample_list.append(img_path)
                if save_bbox:
                    for b in bbox:
                        bbox_list.append(b)
                counter += 1
        
        self.__save_samples(sample_list)  # save samples

        if save_bbox:
            name = f'{self.output_dir}anno.csv'
            with open(name, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                if fields in not None:
                    csv_writer.writerows(fields)
                csv_writer.writerows(bbox_list)
                
        print(f'Totally {counter} frames sampling completed.')

        return counter


    def __len__(self):
        return len(self.image_path_list)