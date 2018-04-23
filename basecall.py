"""
Basecalling script. Given a saved model file and signal indices/data, print out basecalled
sequences one at a time (no batching).
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
from utils.data import SignalDataset, concat_labels, mask_padding
# etc:
from tqdm import tqdm
import numpy as np
import argparse
import os
import sys


def main(cfg, cuda=torch.cuda.is_available()):
    ### flush cfg to output log file:
    print(('-' * 37) + 'Config' + ('-' * 37), file=sys.stderr)
    for k,v in cfg.items():
        print(str(k) + ':' + str(v), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    ### define dataloader factory:
    def get_iterator():
        # set up dataloader config:
        datasets = cfg['data_paths']
        pin_mem = cuda
        nworkers = cfg['num_workers']
        
        # construct signal dataset:
        if cfg['dtype'] == 'npy':
            ds = SignalDataset(torch.from_numpy(np.load(datasets[0])).int(), torch.from_numpy(np.load(datasets[1])).float())
        else:
            ds = SignalDataset(torch.load(datasets[0]), torch.load(datasets[1]))
        
        # return a dataloader iterating over datasets; pagelock memory location if GPU detected:
        return DataLoader(ds, batch_size=1, shuffle=False,
                          sampler=torch.utils.data.sampler.SequentialSampler(ds),
                          num_workers=nworkers, pin_memory=pin_mem)

    ### build RawCTCNet model:
    in_dim = 1
    layers = [ (256,256,d,3) for d in [1,2,4,8,16,32,64] ] * 3
    num_labels = 5
    out_dim = 512
    network = RawCTCNet(in_dim, num_labels, layers, out_dim, input_kw=1, input_dil=1,
                        positions=True, softmax=False, causal=False, batch_norm=True)
    print("Constructed network.", file=sys.stderr)
    if cuda:
        print("CUDA detected; placed network on GPU.", file=sys.stderr)
        network.cuda()
    if cfg['model'] is not None:
        print("Loading model file...", file=sys.stderr)
        try:
            network.load_state_dict(torch.load(cfg['model']))
            print("...Model file loaded.", file=sys.stderr)
        except:
            print("ERR: could not restore model. Check model datatype/dimensions.", file=sys.stderr)
    
    ### build basecaller function:
    maybe_gpu = lambda tsr, has_cuda: tsr if not has_cuda else tsr.cuda()
    def basecall(sample):
        # unpack input sample and wrap in a Variable:
        signals = Variable(maybe_gpu(sample.permute(0,2,1), cuda), volatile=True) # BxTxD => BxDxT
        # compute predicted labels:
        transcriptions = network(signals).permute(2,0,1) # Permute: BxDxT => TxBxD
        return (0.0, transcriptions)

    ### build beam search decoder:
    if (cfg['decoder'] == 'beam'):
        beam_labels = [ ' ', 'A', 'G', 'C', 'T' ]
        beam_blank_id = 0
        beam_decoder = CTCBeamDecoder(beam_labels, beam_width=100, blank_id=beam_blank_id, num_processes=cfg['num_workers'])

    ### build engine, meters, and hooks:
    engine = Engine()
    
    # Wrap a tqdm meter around the losses:
    def on_start(state):
        network.eval() # (set to eval mode for batch-norm)
        print("Basecalling to STDOUT. (This could take a while.)", file=sys.stderr)
        state['iterator'] = tqdm(state['iterator'], file=sys.stderr)

    # (Currently don't do anything w/r/t the sample.)
    def on_sample(state):
        pass

    # decode outputs:
    def on_forward(state):
        logits = F.softmax(state['output'].permute(1,0,2), dim=2)
        # ctc-beam decoding: return best hypothesis
        if (cfg['decoder'] == 'beam'):
            _nt_dict_ = {0: ' ', 1: 'A', 2: 'G', 3: 'C', 4: 'T' }
            def convert_to_string(toks, voc, num):
                try:
                    nt = ''.join([ voc[t] for t in toks[0:num] ])
                except:
                    nt = ''
                return nt
            try:
                beam_result, beam_scores, beam_times, beam_lengths = beam_decoder.decode(logits.data)
                pred_nts = [ convert_to_string(beam_result[k][0], _nt_dict_, beam_lengths[k][0]) for k in range(len(beam_result)) ]
                print(pred_nts[0])
            except:
                print("(WARN: Could not parse batch; skipping...)")
        # arg-max decoding:
        else:
            try:
                _nt_dict_ = {0: '', 1: 'A', 2: 'G', 3: 'C', 4: 'T' }
                amax_nts = labels2strings(argmax_decode(logits), lookup=_nt_dict_)
                print(amax_nts[0])
            except:
                print("(WARN: Could not parse batch; skipping...)")

    # (Currently don't do anything at end of epoch.)
    def on_end(state):
        pass

    print("Constructed engine. Running basecaller loop...", file=sys.stderr)

    ### run validation loop:
    engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end'] = on_end
    engine.test(basecall, get_iterator())


if __name__ == '__main__':
    # ingest command line arguments:
    parser = argparse.ArgumentParser(description="Run RawCTCNet on a validation dataset.")
    parser.add_argument("--dtype", dest="dtype", choices=('pth','npy'), default='pth',
                        help="Datatype of input tensors; either PyTorch or NumPy. [pth]")
    parser.add_argument("--workers", dest='workers', default=1, type=int,
                        help="Number of processes for dataloading/decoding. [1]")
    parser.add_argument("--model", dest='model', default=None,
                        help="Path to saved model file. [None/Random initialization]")
    parser.add_argument("--decoder", dest="decoder", choices=('beam','argmax'), default='beam',
                        help="Type of output decoder to use (CTC-Beam or ArgMax) [beam]")
    parser.add_argument("--dataset", dest='dataset', required=True, type=str,
                        help="Path(s) to comma-separated signal dataset.")
    args = parser.parse_args()
    # parse arguments and run sanity checks:
    try:
        datasets = args.dataset.strip().split(",")
        assert (len(datasets) == 2)
        assert os.path.exists(datasets[0])
        assert os.path.exists(datasets[1])
    except Exception as e:
        print("ERR: signals data does not exist! Check `--dataset` argument", file=sys.stderr)
        raise(e)
    cfg = { 
        'dtype': args.dtype,
        'num_workers': args.workers,
        'model': args.model,
        'decoder': args.decoder,
        'data_paths': datasets
    }
    try:
        main(cfg)
    except KeyboardInterrupt:
        print("Interrupted validation from keyboard.", file=sys.stderr)
