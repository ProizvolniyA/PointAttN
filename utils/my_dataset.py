import torch
import torch.utils.data as data
import h5py
import numpy as np

class CustomH5Dataset(data.Dataset):
    def __init__(self, h5_path):
        # Открываем файл
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Предполагаем, что данные уже загружены в память (если датасет огромный, логику нужно менять)
        self.partials = self.h5_file['partial'][:] 
        self.gts = self.h5_file['gt'][:]
        
        # Закрываем файл после чтения в память
        self.h5_file.close()

        print(f"Loaded {len(self.partials)} samples from {h5_path}")

    def __getitem__(self, index):
        partial = self.partials[index]
        gt = self.gts[index]

        # Приводим к типу float32 (обязательно для PyTorch)
        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(gt).float()

        # PointAttN обычно ожидает формат (N, 3). 
        # Если ваши данные (3, N), используйте .transpose(0, 1)
        
        # Возвращаем кортеж: (вход, выход, имя/индекс)
        # Третий аргумент часто нужен для логирования, можно вернуть индекс
        return partial, gt, index 

    def __len__(self):
        return self.partials.shape[0]
