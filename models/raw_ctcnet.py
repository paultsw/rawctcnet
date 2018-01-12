"""
A classifier network module that maps from raw signal.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch.autograd import Variable

from modules.block import ResidualBlock
from modules.conv_ops import reshape_in, reshape_out

class RawCTCNet(nn.Module):
    """
    Specification of a classifier network.
    """
    def __init__(self, in_dim, num_labels, layers, out_dim, input_kw=2, input_dil=1,
                 positions=False, softmax=False, causal=False):
        """
        Constructor for WaveNetClassifier.

        Args:
        * num_features: python int; the number of channels in the featurized input sequence.
        * feature_kwidth: python int; the kernel width of the featurization layer.
        * num_labels: the number of labels in the softmax distribution at the output.
        * layers: list of (non-causal) convolutional layers to stack. Each entry is of the form
          (in_channels, out_channels, kernel_size, dilation).
        * out_dim: the final dimension of the output before the dimensionality reduction to logits over labels.
        * input_kw: size of the internal kernel of the conv1d going from input to conv-stack.
        * input_dil: dilation for conv block going from input to conv-stack.
        * positions: if True, use conv1x1 to mix position information to features.
        * softmax: if True, softmax the output layer before returning. If False, return un-normalized sequence.
        * causal: if True, use causal convolutions; otherwise use standard convolutions.
        """
        ### parent constructor
        super(RawCTCNet, self).__init__()

        ### attributes
        self.in_dim = in_dim
        self.num_labels = num_labels
        self.layers = layers
        self.num_layers = len(layers)
        self.out_dim = out_dim
        # input 1x1Conv layer:
        self.input_kw = input_kw
        self.input_dil = input_dil
        # position-mixing on/off:
        self.positions = positions
        # softmax on/off:
        self.softmax = softmax
        # causal vs. standard convolutions:
        self.causal = causal

        ### submodules
        # convolutional featurization layer:
        self.feature_layer = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0, dilation=1),
            nn.ELU(),
            nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0, dilation=1),
            nn.ELU())

        # position-mixing bilinear layer:
        if self.positions:
            self.positions_conv1x1 = nn.Sequential(
                nn.Conv1d(1, in_dim, kernel_size=1, padding=0, dilation=1),
                nn.Hardtanh())

        # input layer:
        self.input_block = ResidualBlock(in_dim, layers[0][0], input_kw, input_dil, causal=self.causal)
        self.input_skip_bottleneck = nn.Conv1d(layers[0][0], out_dim, kernel_size=1, padding=0, dilation=1)

        # stack of residual convolutions and their bottlenecks for skip connections:
        convolutions = []
        skip_conn_bottlenecks = []
        for (c_in,c_out,k,d) in layers:
            convolutions.append( ResidualBlock(c_in, c_out, k, d, causal=self.causal) )
            skip_conn_bottlenecks.append( nn.Conv1d(c_out, out_dim, kernel_size=1, padding=0, dilation=1) )
        self.convolutions = nn.ModuleList(convolutions)
        self.bottlenecks = nn.ModuleList(skip_conn_bottlenecks)

        # (1x1 Conv + ReLU + 1x1 Conv) stack, going from output dimension to logits over labels:
        self.output_block = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, dilation=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_dim, num_labels, kernel_size=1, dilation=1))

        ### sensible initializations for parameters:
        eps = 0.0001
        if self.positions:
            for p in self.positions_conv1x1.parameters():
                if len(p.size()) > 1:
                    nn_init.eye(p.view(p.size(0),p.size(1)))
                    p.data.add_(torch.randn(p.size()).mul_(eps))
                if len(p.size()) == 1: p.data.zero_().add_(torch.randn(p.size()).mul_(eps))
        for p in self.feature_layer.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_().add_(torch.randn(p.size()).mul_(eps))
        for p in self.input_block.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_().add_(torch.randn(p.size()).mul_(eps))
        for p in self.convolutions.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_().add_(torch.randn(p.size()).mul_(eps))
        for p in self.bottlenecks.parameters():
            if len(p.size()) > 1:
                nn_init.eye(p.view(p.size(0),p.size(1)))
                p.data.add_(torch.randn(p.size()).mul_(eps))
            if len(p.size()) == 1: p.data.zero_().add_(torch.randn(p.size()).mul_(eps))
        for p in self.output_block.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_().add_(torch.randn(p.size()).mul_(eps))


    def forward(self, seq):
        """
        Run the sequence classification stack on an input sequence.
        
        Args:
        * seq: a FloatTensor variable of shape (batch_size, 1, seq_length).
        
        Returns:
        * logit_seq: a FloatTensor variable of shape (batch_size, out_seq_dim, seq_length).
        """
        # initial featurization from raw:
        out = self.feature_layer(seq)

        # optionally mix position information:
        if self.positions:
            incr_ = torch.arange(0., out.size(2))
            if seq.is_cuda: incr_ = incr_.cuda()
            frame_positions = Variable(incr_)
            out = out + self.positions_conv1x1(frame_positions.unsqueeze(0).unsqueeze(1))
        
        # pass thru the input layer block:
        skips_sum = Variable(out.data.new(out.size(0), self.out_dim, out.size(2)).fill_(0.))
        out, skip = self.input_block(out)
        skips_sum = skips_sum + self.input_skip_bottleneck(skip)

        # run through convolutional stack (& accumulate skip connections thru bottlenecks):
        for l in range(self.num_layers):
            out, skip = self.convolutions[l](out)
            skips_sum = skips_sum + self.bottlenecks[l](skip)

        # run through output stack:
        logit_seq = self.output_block(skips_sum)

        if not self.softmax: return logit_seq
        
        reshaped_logit_seq, _axes = reshape_in(logit_seq)
        return reshape_out(F.softmax(reshaped_logit_seq), _axes)
