import torch
import csv
from torch.utils.data import Dataset

class EmojifyText(Dataset):
    def __init__(self, is_train: bool, word_to_index: dict, 
            seq_length: int=float('inf'), suffix_pad: str=' ') -> None:
        """Loads dataset for training/testing a model for emojification of text.

        is_train: if true loads the training-set, otherwise loads the test-set.
        word_to_index: a mapping from words to their corresponding 
            word-embeddings (i.e. the row number) in the embeddings 
            matrix/layer.
        seq_length: the length of the input sequence. If float('inf'), then 
            no padding is performed. Sequence length must be specified 
            if the batch-size is > 1, otherwise stacking as performed by 
            the DataLoader might lead to an error.

            If None, then the seq_length is computed as the maximum of all 
            the sequence lenghts from the dataset.

            If the number of words in a sentence is greater than the 
            specified sequence length, then the sentence is truncated 
            down to the sequence length.
        suffix_pad: the string to be padded with, and must be the string 
            whose word-embedding is the zero vector.
        """

        super(EmojifyText, self).__init__()

        self.is_train = is_train
        self.word_to_index = word_to_index

        assert (seq_length is None) or (seq_length > 0), \
            f"Sequence length must be non-negative (> 0), but got {seq_length}"

        self.seq_length = seq_length
        self.suffix_pad = suffix_pad

        self.train_sets = [
            "../data/emojify-train-1.csv",
            "../data/emojify-train-2.csv"]
        self.test_sets = [
            "../data/emojify-test.csv"]
        
        self.data = list()
        if self.is_train:
            self.__load_train()
        else:
            self.__load_test()
        
        print(f"[is_train: {is_train}] Sequence-length: {self.seq_length}")

    def __add_data(self, phrase: str, label: str):
        max_len = len(phrase.strip().split(" "))
        if (self.seq_length is None) or (self.seq_length < max_len):
            self.seq_length = max_len

        self.data.append(
            (phrase.lower(), int(label)))

    def __load_train(self):
        data_dict = dict()
        for fp in self.train_sets:
            data_dict.update(self.__load_csv(file_path=fp))
        
        for phrase, label in data_dict.items():
            self.__add_data(phrase=phrase.strip(), label=label)

    def __load_test(self):
        data_dict = dict()
        for fp in self.test_sets:
            data_dict.update(self.__load_csv(file_path=fp))

        for phrase, label in data_dict.items():
            self.__add_data(phrase=phrase.strip("\t").strip(), label=label)

    def __load_csv(self, file_path: str) -> None:
        data = dict()
        with open(file=file_path, mode='r') as f:
            reader = csv.reader(f)

            for row in reader:
                phrase, label = row[0], row[1]
                data[phrase] = label

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        phrase, label = self.data[index]
        indices = list()        
        for w in phrase.strip().split(" "):
            idx = self.word_to_index.get(w, -1)
            if idx == -1:
                raise ValueError(
                    f"No word-embedding found for word '{w}'")
            indices.append(idx)

        if self.seq_length < float('inf'):
            if len(indices) < self.seq_length:
                indices += [self.word_to_index[self.suffix_pad]]\
                        * (self.seq_length - len(indices))
            elif len(indices) > self.seq_length:
                indices = indices[:self.seq_length]

        return torch.LongTensor(indices), label
