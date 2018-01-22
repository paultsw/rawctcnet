"""
Model evaluation script. Run this on a model and a test dataset.
"""
# torch libs:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchnet as tnt
from torchnet.engine import Engine
# ctc libs:
from warpctc_pytorch import CTCLoss
from ctcdecode import CTCBeamDecoder
# custom models/datasets:
from models.raw_ctcnet import RawCTCNet
from modules.sequence_decoders import argmax_decode, labels2strings
from utils.data import SeqTensorDataset, sequence_collate_fn, concat_labels, mask_padding
from utils.parser import parse_dataset_paths
# etc:
from tqdm import tqdm
import numpy as np
import argparse
import os
import sys


def main(cfg, cuda=torch.cuda.is_available()):
    ### flush cfg to output log file:
    tqdm.write(str(cfg), file=cfg['logfile'])
    tqdm.write('-' * 80, file=cfg['logfile'])

    ### define dataloader factory:
    def get_iterator():
        # set up dataloader config:
        datasets = cfg['data_paths']
        pin_mem = cuda
        nworkers = cfg['num_workers']
        
        # (possibly) concatenate datasets together:
        ds = SeqTensorDataset(torch.load(datasets[0][0]), torch.load(datasets[0][1]), torch.load(datasets[0][2]))
        for dataset in datasets[1:]:
            ds += SeqTensorDataset(torch.load(dataset[0]), torch.load(dataset[1]), torch.load(dataset[2]))
        
        # return a dataloader iterating over datasets; pagelock memory location if GPU detected:
        return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=nworkers, collate_fn=sequence_collate_fn, pin_memory=pin_mem)

    ### build RawCTCNet model:
    in_dim = 1
    layers = [ (256,256,d,3) for d in [1,2,4,8,16,32,64] ] * 3
    num_labels = 5
    out_dim = 512
    network = RawCTCNet(in_dim, num_labels, layers, out_dim, input_kw=1, input_dil=1,
                        positions=True, softmax=False, causal=False, batch_norm=True)
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
    
    ### build CTC loss function and model evaluation function:
    ctc_loss_fn = CTCLoss()
    print("Constructed CTC loss function.")
    maybe_gpu = lambda tsr, has_cuda: tsr if not has_cuda else tsr.cuda()
    def model_loss(sample):
        # unpack inputs and wrap in Variables:
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

    ### build beam search decoder:
    beam_labels = [ ' ', 'A', 'G', 'C', 'T' ]
    beam_blank_id = 0
    beam_decoder = CTCBeamDecoder(beam_labels, beam_width=100, blank_id=beam_blank_id, num_processes=cfg['num_workers'])
    print("Constructed CTC beam search decoder.")

    ### build engine, meters, and hooks:
    engine = Engine()
    
    # Wrap a tqdm meter around the losses:
    def on_start(state):
        network.eval()
        state['iterator'] = tqdm(state['iterator'])

    # (Currently don't do anything w/r/t the sample.)
    def on_sample(state):
        pass

    # occasionally log the loss value and perform beam search decoding:
    def on_forward(state):
        if (state['t'] % cfg['print_every'] == 0):
            # log the ctc loss:
            tqdm.write("Step {0} | Loss: {1}".format(state['t'], state['loss'].data[0], file=cfg['logfile']))
            # beam search decoding:
            _, logit_lengths_t, seq_t, seq_lengths_t = state['sample']
            scores = mask_padding(state['output'].permute(1,0,2), logit_lengths_t, fill_logit_idx=0)
            logits = F.softmax(scores, dim=2)
            _nt_dict_ = {0: ' ', 1: 'A', 2: 'G', 3: 'C', 4: 'T' }
            convert_to_string = lambda toks,voc,num : ''.join([ voc[t] for t in toks[0:num] ])
            true_nts = labels2strings(seq_t, lookup=_nt_dict_)
            amax_nts = labels2strings(argmax_decode(logits), lookup=_nt_dict_)
            beam_result, beam_scores, beam_times, beam_lengths = beam_decoder.decode(logits.data)
            pred_nts = [ convert_to_string(beam_result[k][0], _nt_dict_, beam_lengths[k][0]) for k in range(len(beam_result)) ] #[???]
            for i in range(min(len(true_nts), len(pred_nts))):
                tqdm.write("True Seq: {0}".format(true_nts[i]), file=cfg['logfile'])
                tqdm.write("Beam Seq: {0}".format(pred_nts[i]), file=cfg['logfile'])
                tqdm.write("Amax Seq: {0}".format(amax_nts[i]), file=cfg['logfile'])
                tqdm.write("- " * 40, file=cfg['logfile'])

    # (Currently don't do anything at end of epoch.)
    def on_end(state):
        pass

    print("Constructed engine. Running validation loop...")

    ### run validation loop:
    engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end'] = on_end
    engine.test(model_loss, get_iterator())


if __name__ == '__main__':
    # ingest command line arguments:
    parser = argparse.ArgumentParser(description="Run RawCTCNet on a validation dataset.")
    parser.add_argument("--batch_size", dest='batch_size', default=16, type=int,
                        help="Number of sequences per batch. [16]")
    parser.add_argument("--print_every", dest='print_every', default=1000, type=int,
                        help="Log the loss/basecalls to stdout every N steps. [1000]")
    parser.add_argument("--logfile", dest='logfile', default=sys.stdout,
                        help="File to log validation results. [STDOUT]")
    parser.add_argument("--workers", dest='workers', default=1, type=int,
                        help="Number of processes for dataloading/decoding. [1]")
    parser.add_argument("--model", dest='model', default=None,
                        help="Path to saved model file. [None/Random initialization]")
    parser.add_argument("--dataset", dest='dataset', required=True, type=str,
                        help="Path(s) to validation datasets, in comma-semicolon format.")
    args = parser.parse_args()
    # parse arguments and run sanity checks:
    assert ((args.logfile is sys.stdout) or os.path.exists(args.logfile)), "log file does not exist"
    datasets = parse_dataset_paths(args.dataset)
    log_fp = open(arg.logfile, 'w') if not (args.logfile is sys.stdout) else args.logfile
    cfg = { 
        'batch_size': args.batch_size,
        'print_every': args.print_every,
        'logfile': log_fp,
        'num_workers': args.workers,
        'model': args.model,
        'data_paths': datasets
    }
    try:
        main(cfg)
    except KeyboardInterrupt:
        print("Interrupted validation from keyboard.")
    finally:
        log_fp.close()
