import numpy as np
import random
import torch

from torch import nn
from torch.utils.data import Dataset
from babel.dates import format_date
from faker import Faker
from tqdm import tqdm

FORMATS = ['short', 'medium', 'long', 
    'd MMM YYY', 'dd MMM YYY', 'd MMM, YYY',
    'd MMMM, YYY', 'dd, MMM YYY', 'dd.MM.YY',
    'd MM YY', 'MMMM d YYY', 'MMMM d, YYY'] \
        + ['full'] * 10 + ['d MMMM YYY'] * 2

class FakeDates(Dataset):
    def __init__(self, dataset_size: int) -> None:
        super(FakeDates, self).__init__()

        assert dataset_size > 0, f'Invalid dataset-size: {dataset_size}'
        self.__data = None

        # shape -> (m,T_x,n_x); V_x is size of inputs' vocabulary
        self.__X = None

        # shape -> (m,T_y)
        self.__Y = None

        self.char_unk = '<unk>'
        self.char_pad = '<pad>'

        self.hvocab_idx = dict()
        self.mvocab_idx = dict()

        T_x, T_y = self.__generate_random_dates(dataset_size=dataset_size)
        self.__preprocess_data(T_x, T_y)
        
        print(f"Created a dataset of size: {self.__len__()}")

    def __generate_random_dates(self, dataset_size: int):
        self.__data = list()
        T_y = 10 # since, the ISO-date has the format YYYY-MM-DD -> 10 chars long
        T_x = -1

        human_vocab = {self.char_unk, self.char_pad}
        machine_vocab = set()

        faker = Faker()
        for i in tqdm(np.arange(start=0, stop=dataset_size, step=1)):
            dt = faker.date_object()
            h = format_date(date=dt, 
                    format=random.choice(FORMATS), locale='en_US')\
                        .lower().replace(",", "")
            m = dt.isoformat() # defines T_y's length

            T_x = len(h) if len(h) > T_x else T_x

            self.__data.append((h, m))
            
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

        # sorted() ensures consistend chr-idx mapping across multiple runs
        # because the formation of one-hot vectors must be consistent over 
        # multiple runs of the model
        self.hvocab_idx = {chr:idx for idx, chr in enumerate(sorted(human_vocab))}
        self.mvocab_idx = {chr:idx for idx, chr in enumerate(sorted(machine_vocab))}
        
        return T_x, T_y

    def __preprocess_data(self, T_x: int, T_y: int):
        X_tensors = list()
        Y_tensors = list()

        for h, m in self.__data:
            h_idx = [self.hvocab_idx.get(ch, self.hvocab_idx[self.char_unk]) 
                for ch in h]
            
            # padding the input
            if len(h_idx) < T_x:
                h_idx += [self.hvocab_idx[self.char_pad]] * (T_x - len(h_idx))
            elif len(h_idx) > T_x:
                h_idx = h_idx[:T_x]
            
            m_idx = [self.mvocab_idx.get(ch) for ch in m]
            assert len(m_idx) == T_y, \
                f"Encoded output shorted than expected: {len(m_idx)} != {T_y}"

            X_tensors.append(nn.functional.one_hot(
                input=torch.LongTensor(h_idx), num_classes=len(self.hvocab_idx)))
            Y_tensors.append(torch.LongTensor(m_idx))
        
        self.__X = torch.stack(tensors=X_tensors, dim=0).float()
        self.__Y = torch.stack(tensors=Y_tensors, dim=0)

        assert len(self.__X) == len(self.__Y), \
            f"Dataset Corrupt: X.len[={len(self.__X)}] != Y.len[={len(self.__Y)}]"

    def __len__(self):
        return len(self.__X)
    
    def __getitem__(self, idx: int):
        if idx >= self.__len__():
            raise ValueError(
                f"Index [={idx}] is greater than dataset-size [={self.__len__()}].")
        
        return self.__X[idx, :, :], self.__Y[idx, :]
