"""
Train a RawCTCNet model with CTC loss and end-of-epoch CTCBeamDecoder logging.
"""
# numerical libs:
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchnet as tnt
from torchnet.engine import Engine
# ctc modules:
from warpctc_pytorch import CTCLoss
from ctcdecode import CTCBeamDecoder
# custom modules/datasets:
from models.raw_ctcnet import RawCTCNet
from modules.sequence_decoders import argmax_decode, labels2strings
from utils.data import SeqTensorDataset, sequence_collate_fn, concat_labels
from utils.parser import parse_dataset_paths
# etc.:
from tqdm import tqdm
import argparse
import itertools
import os
import sys


def main(cfg, cuda=torch.cuda.is_available()):
    ### flush cfg to output log file:
    tqdm.write(str(cfg), file=cfg['logfile'])
    tqdm.write('-' * 80)

    ### define function that returns a data loader:
    def get_iterator(mode='train'):
        # choose between train/valid data based on `mode`:
        if mode == 'train':
            datasets = cfg['train_data_paths']
            pin_memory_flag = cuda
            num_workers_setting = 4
        if mode == 'valid':
            datasets = cfg['valid_data_paths']
            pin_memory_flag = False
            num_workers_setting = 1

        # form a (possibly concatenated) dataset:
        ds = SeqTensorDataset(torch.load(datasets[0][0]), torch.load(datasets[0][1]), torch.load(datasets[0][2]))
        for dataset in datasets[1:]:
            ds += SeqTensorDataset(torch.load(dataset[0]), torch.load(dataset[1]), torch.load(dataset[2]))

        # return a loader that iterates over the dataset of choice; pagelock the memory location if GPU detected:
        return DataLoader(ds, batch_size=cfg['batch_size'],
                          shuffle=True,
                          num_workers=num_workers_setting,
                          collate_fn=sequence_collate_fn,
                          pin_memory=pin_memory_flag)

    ### build RawCTCNet model:
    in_dim = 1
    layers = [ (256,256,d,3) for d in [1,2,4,8,16,32,64] ] * cfg['num_stacks']
    num_labels = 5
    out_dim = 512
    network = RawCTCNet(in_dim, num_labels, layers, out_dim, input_kw=1, input_dil=1,
                        positions=True, softmax=False, causal=False)
    print("Constructed network.")
    if cuda:
        print("CUDA detected; placed network on GPU.")
        network.cuda()
    if cfg['model'] is not None:
        print("Loading model file...")
        try:
            network.load_state_dict(torch.load(cfg['model']))
        except:
            print("ERR: could not restore model. Check model datatype/dimensions.")

    ### build CTCLoss and model evaluation function:
    ctc_loss_fn = CTCLoss()
    print("Constructed CTC loss function.")
    maybe_gpu = lambda tsr, has_cuda: tsr if not has_cuda else tsr.cuda()
    def model_loss(sample):
        # unpack inputs and wrap as `torch.autograd.Variable`s:
        signals_, signal_lengths_, sequences_, sequence_lengths_ = sample
        signals = Variable(maybe_gpu(signals_.permute(0,2,1), cuda)) # BxTxD => BxDxT
        signal_lengths = Variable(signal_lengths_)
        sequences = Variable(concat_labels(sequences_, sequence_lengths_))
        sequence_lengths = Variable(sequence_lengths_)
        # compute predicted labels:
        transcriptions = network(signals).permute(2,0,1) # Permute: BxDxT => TxBxD
        # compute CTC loss and return:
        loss = ctc_loss_fn(transcriptions, sequences.int(), signal_lengths.int(), sequence_lengths.int())
        return loss, transcriptions

    ### build optimizer:
    opt = optim.Adam(network.parameters(), lr=0.0001)
    print("Constructed Adam optimizer.")

    ### build beam search decoder:
    beam_labels = [' ', 'A', 'G', 'C', 'T']
    beam_blank_id = 0
    beam_decoder = CTCBeamDecoder(beam_labels, beam_width=100, blank_id=beam_blank_id, num_processes=4)
    print("Constructed CTC beam search decoder.")

    ### build engine, meters, and hooks:
    engine = Engine()
    loss_meter = tnt.meter.MovingAverageValueMeter(windowsize=5)
    print("Constructed engine. Running training loop...")

    #-- hook: reset all meters
    def reset_all_meters():
        loss_meter.reset()

    #-- hook: don't do anything for now when obtaining a data sample
    def on_sample(state):
        pass

    #-- hook: don't do anything on gradient update for now
    def on_update(state):
        pass

    #-- hook: update loggers at each forward pass; only happens in test-mode
    def on_forward(state):
        loss_meter.add(state['loss'].data[0])
        if (state['t'] % cfg['print_every'] == 0):
            tqdm.write("Step: {0} | Loss: {1}".format(state['t'], state['loss'].data[0]), file=cfg['logfile'])

    #-- hook: reset all meters at the start of the epoch
    def on_start_epoch(state):
        reset_all_meters()
        network.train() # set to training mode for batch norm
        state['iterator'] = tqdm(state['iterator'])
    
    #-- hook: perform validation and beam-search-decoding at end of each epoch:
    def on_end_epoch(state):
        network.eval() # set to validation mode for batch-norm
        # 10 steps of validation; average the loss:
        val_losses = []
        base_seqs = []
        val_data_iterator = get_iterator('valid')
        for k,val_sample in enumerate(val_data_iterator):
            if k > 10: break
            val_loss, transcriptions = model_loss(val_sample)
            val_losses.append(val_loss.data[0])
            sequences = val_sample[2]
            base_seqs.append((sequences, transcriptions))
        avg_val_loss = np.mean(val_losses)
        tqdm.write("EPOCH {0} | Avg. Val Loss: {1}".format(state['epoch'], avg_val_loss), file=cfg['logfile'])
        
        # beam search decoding:
        _nt_dict_ = { 0: ' ', 1: 'A', 2: 'G', 3: 'C', 4: 'T' }
        def convert_to_string(toks, voc, num):
            return ''.join([voc[t] for t in toks[0:num]])
        for true_seqs, transcriptions in base_seqs:
            true_nts = labels2strings(true_seqs, lookup=_nt_dict_)
            logits = F.softmax(transcriptions.permute(1,0,2), dim=2) # (TxBxD => BxTxD)
            amax_nts = labels2strings(argmax_decode(logits), lookup=_nt_dict_)
            beam_result, beam_scores, beam_times, beam_lengths = beam_decoder.decode(logits.data)
            pred_nts = [ convert_to_string(beam_result[k][0], _nt_dict_, beam_lengths[k][0]) for k in range(len(beam_result)) ]
            for i in range(min(len(true_nts), len(pred_nts))):
                tqdm.write("True Seq: {0}".format(true_nts[i]), file=cfg['logfile'])
                tqdm.write("Beam Seq: {0}".format(pred_nts[i]), file=cfg['logfile'])
                tqdm.write("Amax Seq: {0}".format(amax_nts[i]), file=cfg['logfile'])
                tqdm.write("- " * 40, file=cfg['logfile'])

        # save model:
        try:
            mdl_dtype = "cuda" if cuda else "cpu"
            mdl_path = os.path.join(cfg['save_dir'], "ctc_encoder.{0}.{1}.pth".format(state['epoch'], mdl_dtype))
            torch.save(network.state_dict(), mdl_path)
            tqdm.write("Saved model.", file=cfg['logfile'])
        except:
            tqdm.write("Unable to serialize models. Moving on...", file=cfg['logfile'])
        
        # reset all meters for next epoch:
        reset_all_meters()


    ### engine setup & training:
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(model_loss, get_iterator('train'), maxepoch=cfg['max_epochs'], optimizer=opt)


if __name__ == '__main__':
    # read config and run main():
    parser = argparse.ArgumentParser(description="Train an encoder-decoder model on realistically-modelled data.")
    parser.add_argument("--max_epochs", dest='max_epochs', default=100, type=int,
                        help="Number of epochs. [100]")
    parser.add_argument("--print_every", dest='print_every', default=25, type=int,
                        help="Log the loss to stdout every N steps. [25]")
    parser.add_argument("--batch_size", dest='batch_size', default=1, type=int,
                        help="Number of sequences per batch [1]")
    parser.add_argument("--num_stacks", dest='num_stacks', default=3, type=int,
                        help="Number of repeated residual stacks. [3]")
    parser.add_argument("--save_dir", dest="save_dir", default="./",
                        help="Path to save models at end of each epoch. [cwd]")
    parser.add_argument("--logfile", dest="logfile", default=sys.stdout,
                        help="File to log progress and validation results. [STDOUT]")
    parser.add_argument("--model", dest="model", default=None,
                        help="Continue training from a previously-saved model [None]")
    parser.add_argument('--train_data', required=True, dest='train_data', type=str,
                        help="Paths to training dataset(s), in comma-semicolon list format:"
                        "'idx0.pth,sig0.pth,seq0.pth[;idx1.pth,sig1.pth,seq1.pth;...]'")
    parser.add_argument('--valid_data', required=True, dest='valid_data', type=str,
                        help="Paths to validation dataset(s), in comma-semicolon list format:"
                        "'idx0.pth,sig0.pth,seq0.pth[;idx1.pth,sig1.pth,seq1.pth;...]'")
    args = parser.parse_args()
    assert os.path.exists(args.save_dir), "save directory does not exist"
    assert ((args.logfile is sys.stdout) or os.path.exists(args.logfile)), "log file does not exist"
    assert (args.max_epochs < 300), "max_epochs too high --- concatenate your datasets"
    if args.model is not None:
        assert os.path.exists(args.model), "Model file does not exist"
    train_datasets = parse_dataset_paths(args.train_data)
    valid_datasets = parse_dataset_paths(args.valid_data)
    log_fp = open(args.logfile, 'w') if not (args.logfile is sys.stdout) else args.logfile
    cfg = {
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'num_stacks': args.num_stacks,
        'print_every': args.print_every,
        'save_dir': args.save_dir,
        'logfile': log_fp,
        'model': args.model,
        'train_data_paths': train_datasets,
        'valid_data_paths': valid_datasets
    }
    try:
        main(cfg)
    except KeyboardInterrupt:
        print("Interrupted training from keyboard.")
    log_fp.close()
