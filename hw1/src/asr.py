import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.util import init_weights, init_gate
from src.module import VGGExtractor, CNNExtractor, RNNLayer, ScaleDotAttention, LocationAwareAttention



class Transducer(nn.Module):

    def __init__(self, input_size, vocab_size, init_adadelta, ctc_weight, encoder, attention, decoder, emb_drop=0.0):
        super(Transducer, self).__init__()

        # Setup
        assert 0 <= ctc_weight <= 1
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1
        self.lm = None
        
        # Modules
        self.encoder = Encoder(input_size, **encoder)
        self.predictor = Predictor(vocab_size, False, 1024, 'LSTM', 1024, 2, 0.5)
        self.decoder = Decoder(2048, 512, vocab_size)
        
    
    def forward(self, audio_feature, feature_len, targets, targets_len):
        # print('one:', audio_feature.shape, feature_len.shape, targets.shape, targets_len.shape)
        # torch.Size([16, 665, 120]) torch.Size([16]) torch.Size([16, 92]) torch.Size([16])
        enc_state, enc_len = self.encoder(audio_feature, feature_len)
        # print('two:', enc_state.shape)
        # torch.Size([16, 166, 1024])
        
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)
        
        # print('three:', concat_targets.shape)
        # torch.Size([16, 93])
        # becase we pad 1 in the leftmost, so the len need to add 1
        pred_state, _ = self.predictor(concat_targets, targets_len.add(1))
        # print('four:', pred_state.shape)
        # torch.Size([16, 93, 1024])
        logits = self.decoder(enc_state, pred_state)
        
        return logits, targets.int(), enc_len.int(), targets_len.int()
    
    def decode(self, audio_feature, feature_len):
        
        def _decode(enc_state, lengths):
            token_list = []
            pred_state, hidden = self.predictor(zero_token, torch.ones([1]))
            for t in range(lengths):
                cnt = 0
                while True:
                    cnt += 1
                    if cnt > 1000:
                        break
                    logits = self.decoder(enc_state[t].view(1, 1, enc_state[t].shape[0]), pred_state)
                    logits = logits.view(-1)
                    out = F.softmax(logits, dim=0).detach()
                    pred = torch.argmax(out, dim=0)
                    pred = int(pred.item())
                    if pred == 0:
                        break
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if enc_state.is_cuda:
                        token = token.cuda()
                    
                    pred_state, hidden = self.predictor(token, torch.ones([1]), hidden=hidden)
                    
            return token_list
            
        batch_size = audio_feature.shape[0]
        enc_states, enc_len = self.encoder(audio_feature, feature_len)

        zero_token = torch.LongTensor([[0]])
        if enc_states.is_cuda:
            zero_token = zero_token.cuda()

        results = []
        for i in range(batch_size):
            decode_seq = _decode(enc_states[i], enc_len[i])
            results.append(decode_seq)
        
        return results

    
    def beam_decode(self, audio_feature, feature_len, beam_size, device):
        import math
        def _log_aplusb(a, b):
            return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))
        
        def _isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True
        
        def _beam_decode(enc_state, lengths):
            B = [Sequence(blank=0, pred_hidden=None)]
            for idx, x in enumerate(enc_state):
                x = x.view(1, 1, x.shape[0])
                B.sort(key=lambda x: len(x.token), reverse=True) # larger sequence first
                A = B
                B = []
                # try to calculate more alignment for approximation
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not _isprefix(A[i].token, A[j].token): continue
                            
                        # j is the prefix of i
                        pred_state, pred_hidden = self.predictor(torch.LongTensor([[A[i].token[-1]]]).to(device), pred_len, A[i].pred_hidden)
                        '''
                        e.g.
                         j = a b c d e 
                         i = a b 
                        get the index of c in i (the next token that j want to generate)
                        '''
                        idx = len(A[i].token)
                        logits = self.decoder(x, pred_state)
                        logits = logits.view(-1)
                        logp = F.log_softmax(logits, dim=0).cpu().numpy()                        
                        curlogp = A[i].logp + float(logp[A[j].token[idx]]) # cur prob is i = a b c
                        
                        # continue to calculate the prob of i = a b c "d e" 
                        for k in range(idx, len(A[j].token) - 1):
                            # we can use pred_state of j because it will be the same
                            # it only depend on token input sequence
                            logits = self.decoder(x, A[j].pred_state[k])
                            logits = logits.view(-1)
                            logp = F.log_softmax(logits, dim=0).cpu().numpy()         
                            curlogp += float(logp[A[j].token[k+1]])
                        
                        A[j].logp = _log_aplusb(A[j].logp, curlogp)

                while True:
                    y_hat = max(A, key=lambda x: x.logp)
                    A.remove(y_hat)
                    pred_state, pred_hidden = self.predictor(torch.LongTensor([[y_hat.token[-1]]]).to(device), pred_len, y_hat.pred_hidden)

                    logits = self.decoder(x, pred_state)
                    logits = logits.view(-1)
                    logp = F.log_softmax(logits, dim=0).cpu().numpy()
                    
                    # enumerate all next token and there prob
                    for k in range(self.vocab_size):
                        yk = Sequence(y_hat)
                        yk.logp += float(logp[k])
                        if k == 0:
                            B.append(yk)
                            continue
                        yk.pred_hidden = pred_hidden
                        yk.token.append(k)
                        yk.pred_state.append(pred_state)
                        A.append(yk)
                    
                    y_hat = max(A, key=lambda x: x.logp)
                    yb = max(B, key=lambda x: x.logp)
                    if len(B) > beam_size and yb.logp >= y_hat.logp:
                        break
                # we want to find most probable beam_size result currently end with null token
                sorted(B, key=lambda x: x.logp, reverse=True)
                B = B[:beam_size]
            return B[0].token
                        
        batch_size = audio_feature.shape[0]
        enc_states, enc_len = self.encoder(audio_feature, feature_len)
        zero_token = torch.LongTensor([[0]]).to(device)
        pred_len = torch.ones([1]).to(device)
        
        results = []
        for i in range(batch_size):
            decode_seq = _beam_decode(enc_states[i], enc_len[i])
            results.append(decode_seq)
        return results

class Sequence():
    def __init__(self, seq=None, pred_hidden=None, blank=0):
        if seq is None:
            self.pred_state = [] # predictions of language model
            self.token = [blank] # prediction label
            self.pred_hidden = pred_hidden
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.pred_state = seq.pred_state[:] # save for prefixsum
            self.token = seq.token[:]
            self.pred_hidden = seq.pred_hidden
            self.logp = seq.logp

class Decoder(nn.Module):

    def __init__(self, input_dim, inner_dim, vocab_size):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.vocab_size = vocab_size

        self.forward_layer = nn.Linear(input_dim, inner_dim)
        self.tanh = nn.Tanh()
        self.proj = nn.Linear(inner_dim, vocab_size)

    def forward(self, enc_state, dec_state):
        # print(enc_state.shape, dec_state.shape)
        enc_state = enc_state.unsqueeze(2)
        dec_state = dec_state.unsqueeze(1)
        # print(enc_state.shape, dec_state.shape)
        # torch.Size([16, 166, 1, 1024]) torch.Size([16, 1, 93, 1024])
        t = enc_state.size(1)
        u = dec_state.size(2)
        enc_state = enc_state.repeat([1, 1, u, 1])
        dec_state = dec_state.repeat([1, t, 1, 1])
        
        # print(enc_state.shape, dec_state.shape)
        # torch.Size([16, 166, 93, 1024]) torch.Size([16, 166, 93, 1024])
        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        # print(concat_state.shape)
        # torch.Size([16, 166, 93, 2048])
        
        outputs = self.forward_layer(concat_state)
        # print(outputs.shape)
        # torch.Size([16, 141, 84, 512])
        outputs = self.tanh(outputs)
        outputs = self.proj(outputs)
        # print(outputs.shape)
        # torch.Size([16, 141, 84, 45])
        return outputs


class Encoder(nn.Module):
    def __init__(self, input_size, prenet, module, bidirection, dim, dropout, layer_norm, proj, sample_rate, sample_style):
        super(Encoder, self).__init__()

        # Hyper-parameters checking
        self.vgg = prenet == 'vgg'
        self.cnn = prenet == 'cnn'
        self.sample_rate = 1
        assert len(sample_rate) == len(dropout), 'Number of layer mismatch'
        assert len(dropout) == len(dim), 'Number of layer mismatch'
        num_layers = len(dim)
        assert num_layers >= 1, 'Encoder should have at least 1 layer'

        # Construct model
        module_list = []
        input_dim = input_size

        # Prenet on audio feature
        if self.vgg:
            vgg_extractor = VGGExtractor(input_size)
            module_list.append(vgg_extractor)
            input_dim = vgg_extractor.out_dim
            self.sample_rate = self.sample_rate*4
        if self.cnn:
            cnn_extractor = CNNExtractor(input_size, out_dim=dim[0])
            module_list.append(cnn_extractor)
            input_dim = cnn_extractor.out_dim
            self.sample_rate = self.sample_rate*4

        # Recurrent encoder
        if module in ['LSTM', 'GRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l], bidirection, dropout[l], layer_norm[l],
                                            sample_rate[l], sample_style, proj[l]))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate*sample_rate[l]
        else:
            raise NotImplementedError

        # Build model
        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

    def forward(self, input_x, enc_len):
        for _, layer in enumerate(self.layers):
            input_x, enc_len = layer(input_x, enc_len)
        return input_x, enc_len


class Predictor(nn.Module):
    def __init__(self, vocab_size, emb_tying, emb_dim, module, dim, n_layers, dropout):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.emb_tying = emb_tying
        if emb_tying:
            assert emb_dim == dim, "Output dim of RNN should be identical to embedding if using weight tying."
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.rnn = getattr(nn, module.upper())(
            emb_dim, dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        
    
    def forward(self, x, lens, hidden=None):
        emb_x = self.dp1(self.emb(x))
        if not self.training:
            self.rnn.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb_x, lens, batch_first=True, enforce_sorted=False)
        # output: (seq_len, batch, hidden)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        
        return self.dp2(outputs), hidden
