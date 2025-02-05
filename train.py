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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchnet as tnt
from utils.engine import Engine
# ctc modules:
from warpctc_pytorch import CTCLoss
from ctcdecode import CTCBeamDecoder
# custom modules/datasets:
from models.raw_ctcnet import RawCTCNet
from modules.sequence_decoders import argmax_decode, labels2strings
from utils.data import SeqTensorDataset, sequence_collate_fn, concat_labels, mask_padding
from utils.parser import parse_dataset_paths
from utils.align import ssw
# etc.:
from tqdm import tqdm
import argparse
import itertools
import os
import sys
import traceback


def main(cfg, cuda_avail=torch.cuda.is_available()):
    ### flush cfg to output log file:
    tqdm.write(str(cfg), file=cfg['logfile'])
    tqdm.write('-' * 80)

    ### define function that returns a data loader:
    def get_iterator(mode='train'):
        # choose between train/valid data based on `mode`:
        if mode == 'train':
            datasets = cfg['train_data_paths']
            pin_memory_flag = (cuda_avail and cfg['cuda'])
            num_workers_setting = 4
        if mode == 'valid':
            datasets = cfg['valid_data_paths']
            pin_memory_flag = False
            num_workers_setting = 1

        # form a (possibly concatenated) dataset:
        ds = SeqTensorDataset(torch.load(datasets[0][0]), torch.load(datasets[0][1]),
                              torch.load(datasets[0][2]), torch.load(datasets[0][3]))
        for dataset in datasets[1:]:
            ds += SeqTensorDataset(torch.load(dataset[0]), torch.load(dataset[1]),
                                   torch.load(dataset[2]), torch.load(dataset[3]))

        # return a loader that iterates over the dataset of choice; pagelock the memory location if GPU detected:
        return DataLoader(ds, 
                          batch_size=cfg['batch_size'],
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
                        positions=True, softmax=False, causal=False, batch_norm=True)
    print("Constructed network.")
    if (cuda_avail and cfg['cuda']):
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

    #--- this function performs the gradient descent in synchronous batched mode:
    def batch_model_loss(sample):
        # unpack inputs and wrap as `torch.autograd.Variable`s:
        signals_, signal_lengths_, sequences_, sequence_lengths_ = sample
        signals = Variable(maybe_gpu(signals_.permute(0,2,1), (cuda_avail and cfg['cuda']))) # BxTxD => BxDxT
        signal_lengths = Variable(signal_lengths_)
        sequences = Variable(concat_labels(sequences_, sequence_lengths_))
        sequence_lengths = Variable(sequence_lengths_)
        # compute predicted labels:
        transcriptions = network(signals).permute(2,0,1) # Permute: BxDxT => TxBxD
        # compute CTC loss and return:
        loss = ctc_loss_fn(transcriptions, sequences.int(), signal_lengths.int(), sequence_lengths.int())
        loss.backward()
        return loss, transcriptions

    #--- for evaluation-mode, batch-parallel:
    def batch_model_eval(sample):
        # unpack inputs and wrap as `torch.autograd.Variable`s:
        signals_, signal_lengths_, sequences_, sequence_lengths_ = sample
        signals = Variable(maybe_gpu(signals_.permute(0,2,1), (cuda_avail and cfg['cuda'])), volatile=True) # BxTxD => BxDxT
        signal_lengths = Variable(signal_lengths_, volatile=True)
        sequences = Variable(concat_labels(sequences_, sequence_lengths_), volatile=True)
        sequence_lengths = Variable(sequence_lengths_, volatile=True)
        # compute predicted labels:
        transcriptions = network(signals).permute(2,0,1) # Permute: BxDxT => TxBxD
        # compute CTC loss and return:
        loss = ctc_loss_fn(transcriptions, sequences.int(), signal_lengths.int(), sequence_lengths.int())
        return loss, transcriptions
        
    #--- asynchronous gradient accumulation mode
    # compute target seqs/losses sequentially over each example, average gradients
    def async_model_loss(sample):
        # unpack inputs, optionally place on CUDA:
        signals_, signal_lengths_, sequences_, sequence_lengths_ = sample
        signals = maybe_gpu(signals_.permute(0,2,1), (cuda_avail and cfg['cuda'])) # BxTxD => BxDxT

        # sequential compute over the batch:
        total_loss = 0.0
        transcriptions_list = []
        bsz = signals.size(0)
        for k in range(bsz):
            # fetch k-th input from batched sample and wrap as Variable:
            sig_k_scalar = signal_lengths_[k]
            seq_k_scalar = sequence_lengths_[k]
            sig_k_length = Variable(torch.IntTensor([sig_k_scalar]))
            seq_k_length = Variable(torch.IntTensor([seq_k_scalar]))
            signal_k = Variable(signals[k,:,:sig_k_scalar].unsqueeze(0))
            sequence_k = Variable(sequences_[k,:seq_k_scalar].unsqueeze(0))

            # compute transcription output:
            trans_k = network(signal_k).permute(2,0,1) # Permute: 1xDxT => Tx1xD

            # compute normalized CTC loss and accumulate gradient:
            loss = ctc_loss_fn(trans_k, sequence_k.int(), sig_k_length.int(), seq_k_length.int())
            loss.backward()
            total_loss += loss
            transcriptions_list.append(trans_k)

        # combine transcriptions back into a batch and return:
        max_length = max([t.size(0) for t in transcriptions_list])
        transcriptions = Variable(torch.zeros(max_length, bsz, num_labels))
        for j,tr in enumerate(transcriptions_list):
            transcriptions[0:tr.size(0),j,:] = tr[:,0,:]
        return total_loss, transcriptions

    #--- asynchronous gradient accumulation mode
    # compute target seqs/losses sequentially over each example, average gradients
    def async_model_eval(sample):
        # unpack inputs, optionally place on CUDA:
        signals_, signal_lengths_, sequences_, sequence_lengths_ = sample
        signals = maybe_gpu(signals_.permute(0,2,1), (cuda_avail and cfg['cuda'])) # BxTxD => BxDxT

        # sequential compute over the batch:
        total_loss = 0.0
        transcriptions_list = []
        bsz = signals.size(0)
        for k in range(bsz):
            # fetch k-th input from batched sample and wrap as Variable:
            sig_k_scalar = signal_lengths_[k]
            seq_k_scalar = sequence_lengths_[k]
            sig_k_length = Variable(torch.IntTensor([sig_k_scalar]), volatile=True)
            seq_k_length = Variable(torch.IntTensor([seq_k_scalar]), volatile=True)
            signal_k = Variable(signals[k,:,:sig_k_scalar].unsqueeze(0), volatile=True)
            sequence_k = Variable(sequences_[k,:seq_k_scalar].unsqueeze(0), volatile=True)

            # compute transcription output:
            trans_k = network(signal_k).permute(2,0,1) # Permute: 1xDxT => Tx1xD

            # compute normalized CTC loss and accumulate gradient:
            loss = ctc_loss_fn(trans_k, sequence_k.int(), sig_k_length.int(), seq_k_length.int())
            total_loss += loss
            transcriptions_list.append(trans_k)

        # combine transcriptions back into a batch and return:
        max_length = max([t.size(0) for t in transcriptions_list])
        transcriptions = Variable(torch.zeros(max_length, bsz, num_labels), volatile=True)
        for j,tr in enumerate(transcriptions_list):
            transcriptions[0:tr.size(0),j,:] = tr[:,0,:]
        return total_loss, transcriptions

    #--- choose appropriate model loss/eval functions depending on command line argument:
    model_loss = async_model_loss if cfg['async'] else batch_model_loss
    model_eval = async_model_eval if cfg['async'] else batch_model_eval

    ### build optimizer and LR scheduler:
    if (cfg['optim'] == 'adamax'):
        opt = optim.Adamax(network.parameters(), lr=cfg['lr'])
    elif (cfg['optim'] == 'adam'):
        opt = optim.Adam(network.parameters(), lr=cfg['lr'])
    else:
        raise Exception("Optimizer not recognized!")
    sched = ReduceLROnPlateau(opt, mode='min', patience=5)
    print("Constructed {} optimizer.".format(cfg['optim']))

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

    #-- hook: update loggers at each forward pass
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
        # K steps of validation; average the loss:
        val_losses = []
        base_seqs = []
        val_data_iterator = get_iterator('valid')
        for k,val_sample in enumerate(val_data_iterator):
            if k > cfg['num_valid_steps']: break
            val_loss, transcriptions = model_eval(val_sample)
            val_losses.append(val_loss.data[0])
            sequences = val_sample[2]
            # mask out the padding & permute (TxBxD => BxTxD):
            scores = mask_padding(transcriptions.permute(1,0,2), val_sample[1], fill_logit_idx=0)
            logits = F.softmax(scores, dim=2)
            base_seqs.append((sequences, logits))
        avg_val_loss = np.mean(val_losses)
        # log to both logfile and stdout:
        tqdm.write("EPOCH {0} | Avg. Val Loss: {1}".format(state['epoch'], avg_val_loss), file=cfg['logfile'])
        print("EPOCH {0} | Avg. Val Loss: {1}".format(state['epoch'], avg_val_loss))

        # send average val. loss to learning rate scheduler:
        sched.step(avg_val_loss)
        
        # beam search decoding:
        # (wrapped in try-excepts to prevent a thrown error from aborting training)
        _nt_dict_ = { 0: ' ', 1: 'A', 2: 'G', 3: 'C', 4: 'T' }
        def convert_to_string(toks, voc, num):
            try:
                nt = ''.join([voc[t] for t in toks[0:num]])
            except:
                nt = ''
            return nt
        for true_seqs, logits in base_seqs:
            try:
                true_nts = labels2strings(true_seqs, lookup=_nt_dict_)
                amax_nts = labels2strings(argmax_decode(logits), lookup=_nt_dict_)
                beam_result, beam_scores, beam_times, beam_lengths = beam_decoder.decode(logits.data)
                pred_nts = [ convert_to_string(beam_result[k][0], _nt_dict_, beam_lengths[k][0]) for k in range(len(beam_result)) ]
                for i in range(min(len(true_nts), len(pred_nts))):
                    tqdm.write("True Seq: {0}".format(true_nts[i]), file=cfg['logfile'])
                    tqdm.write("Beam Seq: {0}".format(pred_nts[i]), file=cfg['logfile'])
                    tqdm.write("Amax Seq: {0}".format(amax_nts[i]), file=cfg['logfile'])
                    tqdm.write(("- " * 10 + "Local Beam Alignment" + " -" * 10), file=cfg['logfile'])
                    tqdm.write(ssw(true_nts[i], pred_nts[i]), file=cfg['logfile'])
                    tqdm.write("= " * 40, file=cfg['logfile'])
            except:
                tqdm.write("(WARN: Could not parse batch; skipping...)", file=cfg['logfile'])
                continue

        # save model:
        try:
            mdl_dtype = "cuda" if (cuda_avail and cfg['cuda']) else "cpu"
            mdl_path = os.path.join(cfg['save_dir'], "ctc_encoder.{0}.{1}.pth".format(state['epoch'], mdl_dtype))
            torch.save(network.state_dict(), mdl_path)
            tqdm.write("Saved model.", file=cfg['logfile'])
        except:
            print("Unable to serialize model; Moving on. Traceback:")
            traceback.print_exc()
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
    parser.add_argument("--num_valid_steps", dest='num_valid_steps', default=10, type=int,
                        help="Number of validation steps to take. [10]")
    parser.add_argument("--batch_size", dest='batch_size', default=1, type=int,
                        help="Number of sequences per batch. [1]")
    parser.add_argument("--async_grads", dest='async_grads', choices=('on','off'), default='off',
                        help="Use averaged async grad updates [False]")
    parser.add_argument("--lr", dest='lr', default=0.0002, type=float,
                        help="Learning rate for SGD optimizer. [0.0002]")
    parser.add_argument("--num_stacks", dest='num_stacks', default=3, type=int,
                        help="Number of repeated residual stacks. [3]")
    parser.add_argument("--save_dir", dest="save_dir", default="./",
                        help="Path to save models at end of each epoch. [cwd]")
    parser.add_argument("--logfile", dest="logfile", default=sys.stdout,
                        help="File to log progress and validation results. [STDOUT]")
    parser.add_argument("--model", dest="model", default=None,
                        help="Continue training from a previously-saved model [None]")
    parser.add_argument("--optim", dest="optim", choices=('adam', 'adamax'), default='adamax',
                        help="Choice of optimizer: either 'adam' or 'adamax' [adamax]")
    parser.add_argument("--cuda", dest="cuda", choices=('on','off'), default='on',
                        help="Use CUDA if available; does nothing on CPU-only. [on]")
    parser.add_argument('--train_data', required=True, dest='train_data', type=str,
                        help="Paths to training dataset(s), in comma-semicolon list format:"
                        "'sig_idx0.pth,sig0.pth,seq_idx0.pth,seq0.pth[;sig_idx1.pth,sig1.pth,seq_idx0.pth,seq1.pth;...]'")
    parser.add_argument('--valid_data', required=True, dest='valid_data', type=str,
                        help="Paths to validation dataset(s), in comma-semicolon list format:"
                        "'sig_idx0.pth,sig0.pth,seq_idx0.pth,seq0.pth[;sig_idx1.pth,sig1.pth,seq_idx0.pth,seq1.pth;...]''")
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
        'async': True if (args.async_grads == 'on') else False,
        'lr': args.lr,
        'num_valid_steps': args.num_valid_steps,
        'num_stacks': args.num_stacks,
        'print_every': args.print_every,
        'save_dir': args.save_dir,
        'logfile': log_fp,
        'model': args.model,
        'optim': args.optim,
        'cuda': (True if (args.cuda == 'on') else False),
        'train_data_paths': train_datasets,
        'valid_data_paths': valid_datasets
    }
    try:
        main(cfg)
    except KeyboardInterrupt:
        print("Interrupted training from keyboard.")
    log_fp.close()
