import torch
from src.solver import BaseSolver

from src.asr import ASR, Transducer
from src.optim import Optimizer
from src.data import load_dataset
from src.util import human_format, cal_er, feat_to_fig

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_wer = {'rnnt': 3.0}
        self.best_loss = {'rnnt': 9999999999999999}
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)

        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
            load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.curriculum > 0, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'
        #self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **self.config['model']).to(self.device)
        self.model = Transducer(self.feat_dim, self.vocab_size, init_adadelta, **self.config['model']).to(self.device)
        
        # load pre-trained model
        print('loading pre-trained predictor')
        self.model.predictor.load_state_dict(torch.load(
                '/home/eden.chien/DLHLP/DEV/ckpt/lm_dlhlp_sd0/best_ppx.pth', map_location=self.device if self.mode == 'train' else 'cpu')['model'])
        print('loading pre-trained encoder')
        self.model.encoder.load_state_dict(torch.load(
                '/home/eden.chien/DLHLP/RNNT/ckpt/asr_dlhlp_ctc05_layer5_sd0/best_ctc.pth', map_location=self.device if self.mode == 'train' else 'cpu')['encoder'])
        #self.verbose(self.model.create_msg())
        #model_paras = [{'params': self.model.parameters()}]

        # Losses
        # self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Note: zero_infinity=False is unstable?
        # self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)
        
        from warprnnt_pytorch import RNNTLoss
        self.rnntloss = RNNTLoss()
        # Plug-ins
        '''
        self.emb_fuse = False
        self.emb_reg = ('emb' in self.config) and (
            self.config['emb']['enable'])
        if self.emb_reg:
            from src.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb']).to(self.device)
            model_paras.append({'params': self.emb_decoder.parameters()})
            self.emb_fuse = self.emb_decoder.apply_fuse
            if self.emb_fuse:
                self.seq_loss = torch.nn.NLLLoss(ignore_index=0)
            self.verbose(self.emb_decoder.create_msg())
        '''
        # Optimizer
        # self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        # self.verbose(self.optimizer.create_msg())
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        # Enable AMP if needed
        self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        # self.load_ckpt()

    
    def save_checkpoint(self, f_name, metric, score, show_msg=True):
        '''' 
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            "encoder": self.model.encoder.state_dict(),
            "decoder": self.model.decoder.state_dict(), 
            "predictor": self.model.predictor.state_dict(),
            metric: score
        }
        # Additional modules to save
        # if self.amp:
        #    full_dict['amp'] = self.amp_lib.state_dict()
        # if self.emb_decoder is not None:
            # full_dict['emb_decoder'] = self.emb_decoder.state_dict()
        
        torch.save(full_dict, ckpt_path)
        if show_msg:
            self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".
                         format(human_format(self.step), metric, score, ckpt_path))
    
    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        ctc_loss, att_loss, emb_loss = None, None, None
        n_epochs = 0
        self.timer.set()
        while self.step < self.max_step:
            # Renew dataloader to enable random sampling
            if self.curriculum > 0 and n_epochs == self.curriculum:
                self.verbose(
                    'Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _ = \
                    load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                                 False, **self.config['data'])
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                # tf_rate = self.optimizer.pre_step(self.step)
                total_loss = 0

                # Fetch data
                feat, feat_len, txt, txt_len = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                
                logits, targets, enc_len, targets_len = self.model(feat, feat_len, txt, txt_len)
                rnnt_loss = self.rnntloss(logits, targets, enc_len, targets_len)
                total_loss += rnnt_loss
                '''
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model(feat, feat_len, max(txt_len), tf_rate=tf_rate,
                               teacher=txt, get_dec_state=self.emb_reg)
                # Plugins
                if self.emb_reg:
                    emb_loss, fuse_output = self.emb_decoder(
                        dec_state, att_output, label=txt)
                    total_loss += self.emb_decoder.weight*emb_loss

                # Compute all objectives
                if ctc_output is not None:
                    if self.paras.cudnn_ctc:
                        ctc_loss = self.ctc_loss(ctc_output.transpose(0, 1),
                                                 txt.to_sparse().values().to(device='cpu', dtype=torch.int32),
                                                 [ctc_output.shape[1]] *
                                                 len(ctc_output),
                                                 txt_len.cpu().tolist())
                    else:
                        ctc_loss = self.ctc_loss(ctc_output.transpose(
                            0, 1), txt, encode_len, txt_len)
                    total_loss += ctc_loss*self.model.ctc_weight
                if att_output is not None:
                    b, t, _ = att_output.shape
                    att_output = fuse_output if self.emb_fuse else att_output
                    att_loss = self.seq_loss(
                        att_output.view(b*t, -1), txt.view(-1))
                    total_loss += att_loss*(1-self.model.ctc_weight)
                '''
                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)
                self.step += 1

                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    # lr = self.optimizer.param_groups[0]["lr"]
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                                  .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
                    self.write_log(
                        'loss', {'rnnt_loss': rnnt_loss})
                    # self.write_log('emb_loss', {'tr': emb_loss})
                    '''
                    self.write_log('wer', {'tr_att': cal_er(self.tokenizer, att_output, txt),
                                           'tr_ctc': cal_er(self.tokenizer, ctc_output, txt, ctc=True)})
                    if self.emb_fuse:
                        if self.emb_decoder.fuse_learnable:
                            self.write_log('fuse_lambda', {
                                           'emb': self.emb_decoder.get_weight()})
                        self.write_log(
                            'fuse_temp', {'temp': self.emb_decoder.get_temp()})
                    '''
                    
                # Validation
                if (self.step == 1) or (self.step % self.valid_step == 0):
                    self.lr_scheduler.step(rnnt_loss)
                    if self.step > 15000: 
                        self.validate()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
        self.log.close()

    def validate(self):
        # Eval mode
        self.model.eval()
        dev_wer = {'rnnt': []}
        dev_loss = {'rnnt': []}

        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            
            '''
            # Forward model
            with torch.no_grad():
                logits, targets, enc_len, targets_len = self.model(feat, feat_len, txt, txt_len)
                rnnt_loss = self.rnntloss(logits, targets, enc_len, targets_len)
            
            dev_loss['rnnt'].append(rnnt_loss.item())
            '''

            with torch.no_grad():
                rnnt_output = self.model.decode(feat, feat_len)
            dev_wer['rnnt'].append(cal_er(self.tokenizer, rnnt_output, txt))
            
            # Show some example on tensorboard
            if i == len(self.dv_set)//2:
                for i in range(min(len(txt), self.DEV_N_EXAMPLE)):
                    if self.step == 1:
                        self.write_log('true_text{}'.format(
                            i), self.tokenizer.decode(txt[i].tolist()))
                    self.write_log('rnnt_text{}'.format(i), self.tokenizer.decode(
                            rnnt_output[i]))
        
        '''
        # Ckpt if performance improves
        for task in ['rnnt']:
            dev_loss[task] = sum(dev_loss[task])/len(dev_loss[task])
            if dev_loss[task] < self.best_loss[task]:
                self.best_loss[task] = dev_loss[task]
                self.save_checkpoint('best_{}.pth'.format(task), 'loss', dev_loss[task])
            self.write_log('loss', {'dv_'+task: dev_loss[task]})
        self.save_checkpoint('latest.pth', 'loss', dev_loss['rnnt'], show_msg=False)
        '''
        
        for task in ['rnnt']:
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            if dev_wer[task] < self.best_wer[task]:
                self.best_wer[task] = dev_wer[task]
                self.save_checkpoint('best_{}.pth'.format(task), 'wer', dev_wer[task])
            self.write_log('wer', {'dv_'+task: dev_wer[task]})
        self.save_checkpoint('latest.pth', 'wer', dev_wer['rnnt'], show_msg=False)
        # Resume training
        self.model.train()
