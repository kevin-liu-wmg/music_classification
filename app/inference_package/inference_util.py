import torch


def pad_split_input(x, batch_size, n_segs, segment_length, overlap_length, signal_rate):
    """
    Pad and split audio frames into short frames
    Args:
      x: (batch_size, audio_length)
      n_segs: target number of short segmentations
      segment_length: target length for segmentation
      overlap_length: target length for overlapping between two short frames
      signal_rate: audio signal rate
    Outputs:
     output: (batch_size * n_segs, segment_length * signal_rate)
    """
    total_length = int(((n_segs - 1) * (segment_length - overlap_length) + segment_length) * signal_rate)
    if x.shape[1] < total_length:
        x = torch.cat((x.float(), torch.zeros(batch_size, total_length - x.shape[1])), axis = 1)
    res = torch.zeros((batch_size * n_segs, int(segment_length * signal_rate)))
    for i in range(batch_size):
        # print(i)
        for j in range(n_segs):
            # print(j)
            start_time = (segment_length - overlap_length) * j
            end_time = start_time + segment_length
            # print(i* batch_size + j)
            res[i * n_segs + j, :] = x[i, int(start_time * signal_rate): int(end_time * signal_rate)]
    return res


def audio_sample_pad(x, n_clips, sample_length):
    res = torch.zeros((n_clips, sample_length))
    pad_size = x.shape[1] - (n_clips - 1) * sample_length
    for i in range(n_clips - 1):
        res[i, :] = x[0, i * sample_length: (i + 1) * sample_length]
    res[(n_clips - 1), 0:pad_size] = x[0, (n_clips - 1) * sample_length:(pad_size + (n_clips - 1) * sample_length)]
    return res


def move_data_to_device(x, device):
    """
    Convert data into torch tensor and move it to cpu or gpu device
    Args:
        x: input data
        device: 'cpu' or 'gpu' device

    Returns: x in the device

    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)
