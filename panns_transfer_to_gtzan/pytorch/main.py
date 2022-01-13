import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import time
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils import data


from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
                    hop_size, window, pad_mode, center, ref, amin, top_db, lb_to_idx)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging,
                       StatisticsContainer, Mixup)
from data_generator import GTZANDataset, collate_fn

from models import Transfer_Cnn14, Cnn14, CNN
from evaluate import Evaluator
from torch.utils.tensorboard import SummaryWriter

def train(args):
    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.epoch
    audio_augment = args.audio_augment
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8


    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False
    if model_type == "CNN":
        writer = SummaryWriter(log_dir=f"runs/basic_CNN")
    else:
        writer = SummaryWriter(log_dir=f"runs/model={model_type}, pretrained={pretrain}, freeze_base={freeze_base}, audio_augment={audio_augment}")

    train_hdf5_path = os.path.join(workspace, 'features', 'train.h5')
    valid_hdf5_path = os.path.join(workspace, 'features', 'valid.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
                                   model_type, 'pretrain={}'.format(pretrain),
                                   'loss_type={}'.format(loss_type),
                                   'augmentation={}'.format(augmentation),
                                   'batch_size={}'.format(batch_size),
                                   'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename,
                                   model_type, 'pretrain={}'.format(pretrain),
                                   'loss_type={}'.format(loss_type),
                                   'augmentation={}'.format(augmentation),
                                   'batch_size={}'.format(batch_size),
                                   'freeze_base={}'.format(freeze_base),
                                   'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, model_type,
                            'pretrain={}'.format(pretrain),
                            'loss_type={}'.format(loss_type),
                            'augmentation={}'.format(augmentation),
                            'batch_size={}'.format(batch_size),
                            'freeze_base={}'.format(freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Model
    if model_type == "CNN":
        model = CNN(num_channels=16, sample_rate=sample_rate, n_fft=1024,
                    f_min=fmin, f_max=fmax, num_mels=128, num_classes=10)
    else:
        Model = eval(model_type)
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                      classes_num, freeze_base)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info(
            'Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)


    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    train_data = GTZANDataset(hdf5_path=train_hdf5_path, is_augment=audio_augment, lb_to_ix= lb_to_idx)
    train_loader = data.DataLoader(dataset=train_data,
                                  batch_size=batch_size*2 if 'mixup' in augmentation else batch_size,
                                  shuffle=True,
                                  drop_last=True if 'mixup' in augmentation else False,
                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    # Data generator
    valid_data = GTZANDataset(hdf5_path=valid_hdf5_path, is_augment=False, lb_to_ix= lb_to_idx)
    validate_loader = data.DataLoader(dataset=valid_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0., amsgrad=True)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Evaluator
    evaluator = Evaluator(model=model)
    # Train on mini batches
    best_acc = 0
    for epoch in range(num_epochs):
        print(f'epoch {epoch} starts')
        losses = []
        model.train()
        train_bgn_time = time.time()
        for batch_data_dict in train_loader:
            if 'mixup' in augmentation:
                batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                    len(batch_data_dict['waveform']))

                for key in batch_data_dict.keys():
                    batch_data_dict[key] = move_data_to_device(
                        batch_data_dict[key],
                        device)

                batch_output_dict = model(batch_data_dict['waveform'],
                                          batch_data_dict['mixup_lambda'])
                """{'clipwise_output': (batch_size, classes_num), ...}"""

                batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                                                        batch_data_dict[
                                                            'mixup_lambda'])}
                """{'target': (batch_size, classes_num)}"""
            else:

                for key in batch_data_dict.keys():
                    batch_data_dict[key] = move_data_to_device(
                        batch_data_dict[key],
                        device)

                batch_output_dict = model(batch_data_dict['waveform'], None)
                """{'clipwise_output': (batch_size, classes_num), ...}"""

                batch_target_dict = {'target': batch_data_dict['target']}
                """{'target': (batch_size, classes_num)}"""

            # loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Evaluate
        logging.info('------------------------------------')
        logging.info('Epoch: {}'.format(epoch))

        train_fin_time = time.time()

        statistics = evaluator.evaluate(validate_loader)
        logging.info(
            'Validate accuracy: {:.3f}'.format(statistics['accuracy']))
        print(f'Epoch: {epoch}, Validate accuracy: {statistics["accuracy"]}')
        statistics_container.append(epoch, statistics, 'validate')
        statistics_container.dump()

        writer.add_scalar("Loss/train", np.mean(losses), epoch)
        writer.add_scalar("Accuracy",
                          statistics['accuracy'],
                          epoch)

        train_time = train_fin_time - train_bgn_time
        validate_time = time.time() - train_fin_time

        logging.info(
            'Train time: {:.3f} s, validate time: {:.3f} s'
            ''.format(train_time, validate_time))

        # Save model
        if statistics['accuracy'] > best_acc:
            print(f'Saving the best model at Epoch :{epoch}')
            best_acc = statistics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model': model.module.state_dict()}
            checkpoint_path = os.path.join(
                checkpoints_dir, 'BestAcc.pth')
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True,
                              help='Directory of your workspace.')
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true',
                              default=False)
    parser_train.add_argument('--audio_augment', action='store_true',
                              default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str,
                              choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')