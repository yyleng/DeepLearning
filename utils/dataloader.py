import torchvision
from torchvision import transforms
import torch
from enum import Enum
from typing import Optional


class DatasetName(Enum):
    FASHION_MNIST = 1


def _get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4


def _get_dataset_handler(dataset_name: DatasetName):
    """Return the dataset handler according to the dataset name."""
    name2class = {DatasetName.FASHION_MNIST: torchvision.datasets.FashionMNIST}
    if dataset_name in name2class:
        return name2class[dataset_name]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_offical_data(
    batch_size: int,
    dataset: DatasetName,
    data_root: str = "../data",
    resize: Optional[int] = None,
) -> tuple:
    """Download the dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    handler = _get_dataset_handler(dataset)
    train_data = handler(root=data_root, train=True, transform=trans, download=True)
    test_data = handler(root=data_root, train=False, transform=trans, download=True)
    return (
        torch.utils.data.DataLoader(
            train_data, batch_size, shuffle=True, num_workers=_get_dataloader_workers()
        ),
        torch.utils.data.DataLoader(
            test_data, batch_size, shuffle=False, num_workers=_get_dataloader_workers()
        ),
    )


def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    # 正态分布，均值为0，标准差为1, shape=(num_examples, w.shape[0])
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, torch.reshape(y, (-1, 1))


def load_array(data_arrays: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Construct a PyTorch data iterator."""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)
