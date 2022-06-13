import os
import torch
import numpy as np
import h5py

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
from torch import nn
from torch.utils.data import Dataset

class TriggerWords(Dataset):
    def __init__(self, dataset_file: str, dataset_size: int=1000):
        """If the specified 'dataset_file' is found then data is 
        loaded from the said file. However, if the file doesn't 
        exist, then the a new dataset of size 'dataset_size' is 
        created and then stored in the file named 'dataset_file'.

        dataset_file: the name of the file which contains the data
        dataset_size: the size of the dataset to be created
        """
        super(TriggerWords, self).__init__()

        pos, neg, bg = self.__load_data()
        self.audio_utils = AudioUtils(
            pos_clips=pos, neg_clips=neg, bg_clips=bg)
        
        self.__X = list()
        self.__Y = list()
        
        self.T_y = 1375

        self.data_file = f'../data/trigger_words/{dataset_file}'

        assert dataset_size > 0, \
            f"Invalid dataset size, must be positive: {dataset_size}"
        
        self.__build_dataset(dataset_size=dataset_size)
        
    def __load_data(self):
        """
        Returns a triplet (positives, negatives, backgrounds), with 
        each element being a list of corresponding type of audio-segments.
        """    
        def __load_directory(dir: str):
            wav_files = list()
            for f in os.listdir(dir.format("")):
                if f.endswith("wav"):
                    file = AudioSegment.from_wav(dir.format(f))
                    wav_files.append(file)

            return wav_files

        return __load_directory(dir='../data/trigger_words/positives/{0}'), \
            __load_directory(dir='../data/trigger_words/negatives/{0}'), \
            __load_directory(dir='../data/trigger_words/backgrounds/{0}')

    def __build_dataset(self, dataset_size: int):
        if os.path.exists(self.data_file):
            print(f"Data file found at {self.data_file}! Loading data...")
            with h5py.File(self.data_file, 'r') as f:
                self.__X = torch.Tensor(f['x'])
                self.__Y = torch.Tensor(f['y'])
        else:
            print(f"Data file not found at {self.data_file}. Building one... [size={dataset_size}]")
            for j in tqdm(np.arange(dataset_size)):
                spec_gram, seg_history, bg_step_len = self.audio_utils.generate_example()
                self.__X.append(torch.Tensor(spec_gram))

                y = torch.zeros(size=(self.T_y,))
                for s_start, s_end in seg_history['pos']:
                    s_idx = np.int32(s_end * self.T_y / np.float32(bg_step_len))
                    y[s_idx+1:s_idx+51] = 1
                self.__Y.append(y)
            
            self.__X = torch.stack(tensors=self.__X, dim=0)
            self.__Y = torch.stack(tensors=self.__Y, dim=0)

            with h5py.File(name=self.data_file, mode='w') as f:
                f['x'] = self.__X
                f['y'] = self.__Y

    def __len__(self):
        assert len(self.__X) == len(self.__Y), \
            f"Lengths of input [={len(self.__X)}] and label [={len(self.__Y)}] datasets don't match"

        return len(self.__Y)
    
    def __getitem__(self, idx: int):
        return self.__X[idx,:,:], self.__Y[idx,:]


class AudioUtils():
    def __init__(self, pos_clips: list, neg_clips: list, bg_clips: list) -> None:
        super(AudioUtils, self).__init__()

        self.pos_clips = pos_clips
        self.neg_clips = neg_clips
        self.bg_clips = bg_clips

        self.rng = np.random.default_rng(846382)
    
    @staticmethod
    def check_segment_overlap(segment: tuple, seg_history: list):
        """Returns True if the segment overlaps with any of the other segments. 
        """

        s_start, s_end = segment
        status = False
        for o_start, o_end in seg_history:
            status = (s_start <= o_start and o_start <= s_end) \
                or (s_start <= o_end and o_end <= s_end)
            if status: break
        
        return status

    def get_random_time_segment(self, total_len: int, seg_len: int):
        """Returns a random time-interval of the format [start, end]
        """

        seg_start = self.rng.integers(low=0, high=total_len - seg_len, 
            endpoint=False)
        seg_end = seg_start + seg_len - 1

        assert seg_end < total_len, \
            f"Segment-end [={seg_end}] time exceeds total clip-length [={total_len}]."
        
        return (seg_start, seg_end)

    def overlay_audio_clip(self, seg_clip: AudioSegment, 
            bg_clip: AudioSegment, seg_history: list):
        """Overlays the specified segment-clip over the specified 
        background-clip, at a randomly choosen segment of time such 
        that it doesn't overlap with the specified seg_history.
        """
        clip_seg = None
        is_overlap = True

        debug_count = 0 # to prevent potential infinite-loops
        while is_overlap:
            clip_seg = self.get_random_time_segment(
                total_len=len(bg_clip), seg_len=len(seg_clip))
            is_overlap = AudioUtils.check_segment_overlap(
                    segment=clip_seg, seg_history=seg_history)
            debug_count += 1

            if debug_count > 100:
                clip_seg = None
                break

        if clip_seg:
            bg_clip = bg_clip.overlay(seg=seg_clip, position=clip_seg[0])

        return bg_clip, clip_seg

    def generate_example(self):
        seg_history = {
            'pos': list(),
            'neg': list()
        }

        i = self.rng.integers(low=0, high=len(self.bg_clips))
        bg_clip = self.bg_clips[i]

        # make the background quieter
        bg_clip = bg_clip # - 20

        pos_count = self.rng.integers(low=0, high=4, endpoint=True)
        for i in self.rng.integers(low=len(self.pos_clips), size=pos_count):
            bg_clip, clip_seg = self.overlay_audio_clip(
                seg_clip=self.pos_clips[i], bg_clip=bg_clip, 
                seg_history=seg_history['pos'])
            
            if clip_seg:
                seg_history['pos'].append(clip_seg)
        
        neg_count = self.rng.integers(low=0, high=2, endpoint=True)
        for i in self.rng.integers(low=len(self.neg_clips), size=neg_count):
            bg_clip, clip_seg = self.overlay_audio_clip(
                seg_clip=self.neg_clips[i], bg_clip=bg_clip,
                seg_history=[*seg_history['pos'], *seg_history['neg']])
            seg_history['neg'].append(clip_seg)
        
        # standardize the volume of the audio clip
        bg_clip = bg_clip.apply_gain(-20 - bg_clip.dBFS)
        
        # TODO: use a spectrogram function outside matplotlibs'
        bg_clip.export('clip.wav', format='wav')
        spectrogram = self.graph_spectrogram(wav_file='clip.wav')

        return spectrogram, seg_history, len(bg_clip)
    
    # Calculate and plot spectrogram for a wav audio file
    def graph_spectrogram(self, wav_file):
        rate, data = wavfile.read(wav_file)

        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        
        return pxx
