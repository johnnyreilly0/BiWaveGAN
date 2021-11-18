import torch
import utils.data_utils


def test_load_wav_shape():
    x = utils.data_utils.load_wav("data/usv1.wav", 250000)
    assert x.shape == torch.Size([1, 32216])


def test_dataset_len():
    dataset = utils.data_utils.WAVDataset("data", sample_rate=250000, slice_len=250000)
    assert len(dataset) == 10


def test_dataset_getitem():
    dataset = utils.data_utils.WAVDataset("data", sample_rate=250000, slice_len=32768)
    for i in range(len(dataset)):
        assert dataset[i].shape == torch.Size([1, 32768])