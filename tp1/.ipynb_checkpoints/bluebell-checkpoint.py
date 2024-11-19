import pandas as pd
import numpy as np
from PIL import Image

class Bluebell():
    def __init__(self, root_dir, mode='train', split = 0):
        self.index = 0
        self.root_dir = root_dir
        self.mode = mode
        split = np.clip(split, 0, 4) if mode != 'test' else 0
        self.split = split

        self.frame = pd.read_csv('{}/labels.csv'.format(root_dir), header=None, names=['img', 'cls'])
        begin = split * 300 if mode != 'test' else 0
        if mode == 'train':
            self.frame = self.frame.drop(np.arange(begin, begin+300))
            self.frame = self.frame.drop(np.arange(1500, 1800))
        elif mode == 'val':
            if begin > 0:
                self.frame = self.frame.drop(np.arange(0, begin))
            self.frame = self.frame.drop(np.arange(begin+300, 1800))
        elif mode == 'test':
            self.frame = self.frame.drop(np.arange(0, 1500))
        else:
            raise ValueError('invalid mode: "train", "val" or "test" only')

        self.frame['cls_id'] = self.frame['cls'].astype('category').cat.codes.astype(int)
        self.frame = self.frame.set_index(pd.Index(np.arange(len(self.frame))))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.frame):
            raise IndexError
        filename = self.frame['img'][idx]
        label = self.frame['cls_id'][idx]
        img = Image.open('{}/img/{}'.format(self.root_dir, filename))
        return np.array(img), label
    


