import pytorch_transformers as pytrf
import torch
import torch.nn as nn
import numpy as np

import os
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


XLNET_PRETRAINED = {
    'EN': {'layers': 5, 'heads': 12, 'vocab': 32000, 'file': 'xlnet_english.model', 'name': 'xlnet-base-cased',
           'd_model': 768, 'd_inner': 3072},
    'ZH': {'layers': 5, 'heads': 12, 'vocab': 32000, 'file': 'xlnet_chinese.model', 'name': 'xlnet-base-chinese',
           'd_model': 768, 'd_inner': 3072}
}


class XLNetTokenizer():
    def __init__(self, language='EN', pretrained_path='./'):
        super().__init__()

        self.language = language

        name = XLNET_PRETRAINED[self.language]['name'] + '-spiece.model'
        self.tokenizer = pytrf.XLNetTokenizer(os.path.join(pretrained_path, name))

    # TODO revisit threading
    def _multithreading(self, func, args, workers):
        begin_time = time.time()
        # print("Threading with {} workers".format(workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            res = executor.map(func, args)
        return list(res)

    def _multiprocessing(self, func, args, workers):
        begin_time = time.time()
        # print("Threading with {} workers".format(workers))
        p = mp.Pool(workers)
        res = p.map(func, args)
        p.close()
        p.join()
        return res

    def tokenize_text(self, text, max_str_len, max_tok_len=512):
        text = text[:max_str_len] if np.random.uniform() > 0.5 else text[-max_str_len:]
        text = self.tokenizer.tokenize(text)
        text = text[:max_tok_len] if np.random.uniform() > 0.5 else text[-max_tok_len:]
        return self._encode_tokens(text)

    def _encode_tokens(self, text):
        return self.tokenizer.convert_tokens_to_ids(text)

    def tokenize(self, data, max_str_len, max_tok_len=512, workers=4):
        token_fn = partial(self.tokenize_text, max_str_len=max_str_len, max_tok_len=max_tok_len)
        if workers > 1:
            res = self._multiprocessing(token_fn, data, workers)
        else:
            res = [token_fn(d) for d in data]
        return res


class NLPXLNetClassifier(nn.Module):
    def __init__(self, metadata, xlnet_metadata, pretrained=None, vocab_size=None, encoder_layers=1, attn_heads=2,
                 classifier_layers=1, classifier_units=768, finetuning=True):
        """
        metadata: dict
            metadata from AutoNLP dataset
        pretrained: str
            folder to load a pretrained model from for the given config
        vocab_size: int or None
            vocab size for embedding layer in XLNET model
            can be ignored if pretrained model is provided, else, it has to be given
        encoder_layers: int
            number of hidden encoder layers in XLNET model
        attn_heads: int
            number of attention heads in XLNET model
        classifier_layers: int
            number of linear layers in classifier
        classifier_units: int
            number of neurons in intermediate classifier layers
            only used when classifier_layers > 1
        """
        super().__init__()
        self.xlnet_metadata = xlnet_metadata
        self.language = metadata['language']
        self.out_dim = metadata['class_num'] if metadata['class_num'] > 2 else 1

        self.layers = encoder_layers if pretrained is None else self.xlnet_metadata[self.language]['layers']
        self.heads = attn_heads if pretrained is None else self.xlnet_metadata[self.language]['heads']
        self.vocab_size = vocab_size if pretrained is None else self.xlnet_metadata[self.language]['vocab']

        self.finetuning = finetuning

        # load XLNET
        if pretrained is not None:
            self._load_pretrained(pretrained)
        else:
            self.config = pytrf.XLNetConfig(self.vocab_size, n_layer=self.layers, n_head=self.heads,
                                            d_model=self.xlnet_metadata[self.language]['d_model'],
                                            d_inner=self.xlnet_metadata[self.language]['d_inner'])
            self.xlnet = pytrf.XLNetModel(self.config)
            self.xlnet.apply(self._init_weights)

        # classifier layers
        if classifier_layers == 1:
            classifier = [nn.Linear(self.config.hidden_size, self.out_dim)]
        else:
            classifier = []
            for i in range(classifier_layers - 1):
                if i == 0:
                    fc = nn.Linear(self.config.hidden_size, classifier_units)
                else:
                    fc = nn.Linear(classifier_units, classifier_units)
                classifier.append(fc)
                classifier.append(nn.ReLU())

            classifier.append(nn.Linear(classifier_units, self.out_dim))

        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(*classifier)
        self.classifier.apply(self._init_weights)

    def forward(self, x, mask=None):
        """
        input dim = (batch_size, sequence_length)
        output dim = (batch_size, num_classes)
        """
        x, _ = self.xlnet(x, attention_mask=mask)
        # average over sequence length
        x = torch.mean(x, dim=1)
        x = self.relu(x)
        x = self.classifier(x)
        return x

    def _load_pretrained(self, pretrained_path='./'):
        """
        Loads from the pretrained model stored in given folder
        path: folder path where model is stored
        """

        load_time = time.time()

        self.config = pytrf.XLNetConfig(n_layer=self.layers, n_head=self.heads,
                                        d_model=self.xlnet_metadata[self.language]['d_model'],
                                        d_inner=self.xlnet_metadata[self.language]['d_inner'])
        self.xlnet = pytrf.XLNetModel(self.config)

        # load pretrained model
        self.xlnet.half()
        model_name = "{}_{}.{}".format(self.xlnet_metadata[self.language]['file'].split('.')[0],
                                       self.xlnet_metadata[self.language]['layers'],
                                       self.xlnet_metadata[self.language]['file'].split('.')[1])
        self.xlnet.load_state_dict(torch.load(os.path.join(pretrained_path, model_name)))
        self.xlnet.float()

        # disable gradient for bert embedding layer
        for name, param in self.xlnet.named_parameters():
            if self.finetuning or 'embedding' in name:
                param.requires_grad = False

        print('Pretrained model load time:', time.time() - load_time)

    def disable_finetuning(self):
        self.finetuning = False
        for name, param in self.xlnet.named_parameters():
            param.requires_grad = False

    def enable_finetuning(self):
        self.finetuning = True
        for name, param in self.xlnet.named_parameters():
            if not 'embedding' in name:
                param.requires_grad = True

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.parameters())
        return tot_sum

    def trainable_parameters(self):
        tot_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tot_sum
