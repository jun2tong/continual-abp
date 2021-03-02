import cv2

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, file_path, label_map, transform=None):
        self.file_path = file_path
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        file_name = self.file_path[idx]
        label = self.label_map[file_name.split("/")[-2]]

        image = cv2.imread(file_name)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label
