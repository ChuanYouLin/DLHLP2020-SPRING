import copy
import torch
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from src.solver import BaseSolver
from src.asr import Transducer
from src.decode import BeamDecoder
from src.data import load_dataset
import os

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)

        # ToDo : support tr/eval on different corpus
        assert self.config['data']['corpus']['name'] == self.src_config['data']['corpus']['name']
        self.config['data']['corpus']['path'] = self.src_config['data']['corpus']['path']
        self.config['data']['corpus']['bucketing'] = False

        # The follow attribute should be identical to training config
        self.config['data']['audio'] = self.src_config['data']['audio']
        self.config['data']['text'] = self.src_config['data']['text']
        self.config['model'] = self.src_config['model']
        self.config['hparas'] = self.src_config['hparas']
        # Output file
        self.output_file = self.config['decode']['output_file']

        # Override batch size for beam decoding
        self.greedy = self.config['decode']['beam_size'] == 1
        if not self.greedy:
            self.config['data']['corpus']['batch_size'] = 1
        else:
            # ToDo : implement greedy
            raise NotImplementedError

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.dv_set, self.tt_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
            load_dataset(self.paras.njobs, self.paras.gpu,
                         self.paras.pin_memory, False, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model '''
        # Model
        init_adadelta = False
        self.model = Transducer(self.feat_dim, self.vocab_size, init_adadelta, **self.config['model']).to(self.device)
        
        print(self.paras.load)
        ckpt = torch.load(self.paras.load, 
            map_location=self.device if self.mode == 'train' else 'cpu'
        )
        self.model.encoder.load_state_dict(ckpt['encoder'])
        self.model.predictor.load_state_dict(ckpt['predictor'])
        self.model.decoder.load_state_dict(ckpt['decoder'])
        self.model.eval()
        
    
    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        name, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)

        return name, feat, feat_len, txt, txt_len
    
    def exec(self):
        ''' Testing End-to-end ASR system '''
        for s, ds in zip(['test'], [self.tt_set]):
            # Setup output
            # self.cur_output_path = self.output_file.format(s, 'output')
            f = open(self.output_file, 'w')
            
            beam_decode_func = partial(beam_decode, model=copy.deepcopy(
                    self.model), device=self.device, beam_size=self.config['decode']['beam_size'])
            results = Parallel(n_jobs=self.paras.njobs)(
                    delayed(beam_decode_func)(data) for data in tqdm(ds))
            
            if self.config['decode']['kaggle_format'] == True:
                f.write('id,answer\n')
            else:
                f.write('idx\thyp\ttruth\n')
            
            for name, hyp, truth in results:
                if self.config['decode']['kaggle_format'] == True:
                    f.write(name + ',' + self.tokenizer.decode(hyp) + '\n')
                else:
                    f.write('\t'.join([name, self.tokenizer.decode(hyp), self.tokenizer.decode(truth)]) + '\n')
            
        self.verbose('All done !')

def beam_decode(data, model, device, beam_size):
    # Fetch data : move data/model to device
    name, feat, feat_len, txt = data
    feat = feat.to(device)
    feat_len = feat_len.to(device)
    txt = txt.to(device)
    txt_len = torch.sum(txt != 0, dim=-1)
    model = model.to(device)
    # Decode
    with torch.no_grad():
        result = model.beam_decode(feat, feat_len, beam_size, device)
        
    return (name[0], result[0], txt[0].cpu().tolist())  # Note: bs == 1
