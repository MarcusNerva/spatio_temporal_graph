#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.layers import EncoderLayer, DecoderLayer
from transformer.positional_embedding import PositionalEmbedding

class Encoder(nn.Module):
    """
    Encoder part of transformer.
    As paper request, encoder is composed of 2 encoder layer.
    d_model == 512.
    d_hidden == 1024.
    n_heads == 8.
    dropout ratio == 0.3.
    """
    def __init__(self, n_layers=2, n_heads=8, d_model=512, d_hidden=1024, dropout=0.3):
        super(Encoder, self).__init__()
        self.position_emb = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads, dropout=dropout)
                                         for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, features, mask=None, return_attns=False):
        enc_slf_attn_list = []

        enc_output = self.dropout(self.position_emb(features))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    """
    Decoder part of transformer.
    As paper request, decoder is composed of 2 DecoderLayer, namely, n_layers==2.
    d_model == 512
    d_hidden == 1024
    n_heads == 8
    dropout ratio == 0.3
    """
    def __init__(self, n_vocab, pad_idx, n_layers=2, n_heads=8, d_model=512, d_hidden=1024, dropout=0.3):
        super(Decoder, self).__init__()
        
        self.word_emb = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)
        self.position_emb = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, word_seq, encoder_output, word_mask=None, encoder_mask=None, return_attns=False):
        slf_attn_list, enc_attn_list = [], []

        decoder_output = self.dropout(self.position_emb(self.word_emb(word_seq)))
        decoder_output = self.layer_norm(decoder_output)


        for decoder_layer in self.layer_stack:
            decoder_output, slf_attn, enc_attn = decoder_layer(encoder_output=encoder_output, decoder_input=decoder_output, self_attn_mask=word_mask, enc_attn_mask=encoder_mask)
            slf_attn_list += [slf_attn] if return_attns else []
            enc_attn_list += [enc_attn] if return_attns else []

        if return_attns:
            return decoder_output, slf_attn_list, enc_attn_list
        return decoder_output


class Transformer(nn.Module):
    """
    Transformer = Encoder + Decoder
    As paper request, encoder and decoder are composed of 2 layers, bothly.
    d_model == 512
    d_hidden == 1024
    n_heads == 8
    dropout ratio == 0.3
    """
    def __init__(self, n_vocab, pad_idx, n_layers=2, n_heads=8, d_model=512, 
                 d_hidden=1024, dropout=0.3, emb_prj_weight_sharing=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        self.decoder = Decoder(n_vocab=n_vocab, pad_idx=pad_idx, n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        self.emb2word = nn.Linear(d_model, n_vocab, bias=False)
        self.x_logit_scale = 1.
        self.n_vocab = n_vocab
        self.pad_idx = pad_idx

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if emb_prj_weight_sharing:
            self.emb2word.weight = self.decoder.word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

    def get_pad_mask(self, seq):
        return (seq != self.pad_idx).unsqueeze(-2)

    def get_subsequent_mask(self, seq):
        batch, length = seq.shape
        subsequent_mask = (1 - torch.triu(torch.ones((1, length, length), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def forward(self, features, gt_seq):
        encoder_output  = self.encoder(features=features, mask=None)

        encoder_mask = None
        word_mask = self.get_pad_mask(gt_seq) & self.get_subsequent_mask(gt_seq)
        decoder_output = self.decoder(word_seq=gt_seq, encoder_output=encoder_output, 
                                          word_mask=word_mask, encoder_mask=encoder_mask)
        seq_logit = self.emb2word(decoder_output) * self.x_logit_scale
        return seq_logit.view(-1, seq_logit.size(2))

    def _model_decode(self, generated_seq, generated_mask, encoder_output, encoder_mask):
        decoder_output = self.decoder(word_seq=generated_seq, encoder_output=encoder_output, 
                                          word_mask=generated_mask, encoder_mask=encoder_mask)
        return F.softmax(self.emb2word(decoder_output), dim=-1)

    def _get_init_state(self, features):
        """
        prepare initial state for sentence generating.
        
        Args:
            features: visual features.
        
        Shapes:
            features: (batch, T, d_model) because of beam_search, batch is set to 1 here.
            --------------------------------
            encoder_output: (beam_size, T, d_model)
            generated_seq: (beam_size, max_seq_len)
            scores: (beam_size)

        Returns:
            encoder_output: the processed visual features.
                            for the sake of beam searching, the batch_size is set to beam_size here.
            generated_seq: the initial sentences which prepared for beam searching. 
                            Although it's seq_len == max_seq_len, the first 2 position is meanful, 
                            which are [bos, first_word]
            scores: the scores of initial sentences.
        """
        beam_size = self.beam_size
        
        encoder_output = self.encoder(features)
        generated_mask = self.get_subsequent_mask(self.init_seq)
        decoder_output = self._model_decode(generated_seq=self.init_seq, generated_mask=generated_mask,
                                           encoder_output=encoder_output, encoder_mask=None)

        best_k_probs, best_k_idxs = decoder_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idxs[0]
        encoder_output = encoder_output.repeat(beam_size, 1, 1)
        return encoder_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, generated_seq, decoder_output, scores, step):
        """
        beam search function.

        Args:
            generated_seq: current generated_seq. Only meaningful in first step position.
            decoder_output: output from decoder. We utilized this features to provide information of next word. 
            scores: recorded scores of current generated_seq. 
            step: time step of generate sequence.

        Shapes:
            generated_seq: (beam_size, max_seq_len)
            decoder_output: (beam_size, step, d_model)
            scores: (beam_size)
            step: int
            -----------------------------------------
            generated_seq:(beam_size, max_seq_len)
            scores: (beam_size)
        
        Return:
            generated_seq: new generated_seq. Meaningful in first step + 1 postion.
            scores: recorded scores of new generated_seq.
            
        """
        beam_size = self.beam_size
        
        # Shape are both (beam_size, beam_size). 
        # Get beam_size candidates for each beam, beam_size^2 candidates in total.
        best_beam2_probs, best_beam2_idxs = decoder_output[:, -1, :].topk(beam_size)
        
        # previous scores + new generated scores
        scores = torch.log(best_beam2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # get best beam_size candidates from beam_size^2 candidates.
        scores, best_beam_idxs_linear = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best beam_size candidiates.
        best_beam_r_idxs, best_beam_c_idxs = best_beam_idxs_linear // beam_size, best_beam_idxs_linear % beam_size
        best_beam_idxs = best_beam2_idxs[best_beam_r_idxs, best_beam_c_idxs]

        # Copy the corresponding previous tokens.
        generated_seq[:, :step] = generated_seq[best_beam_r_idxs, :step]
        # Set the best tokens in this beam search step
        generated_seq[:, step] = best_beam_idxs

        return generated_seq, scores


    def generate_sentence(self, features, beam_size, max_seq_len, bos_idx, eos_idx):
        """
        generate sentence one by one. 
        So here batch == 1.

        Args:
            features: visual features
            beam_size: the size of beam search
            max_seq_len: the max length of generated sequence.
            bos_idx: the idx of begin of the sequence.
            eos_idx: the idx of end of the sequence.
        
        Shapes:
            features: (batch, T, d_model)
        
        Returns:
            gen_seq[ans_idx][:seq_lens[ans_idx]]: a list of generated word expressed by idx. 
        """
        
        batch = features.shape[0]
        assert batch == 1, 'the batch size is larger than 1!'

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.register_buffer('init_seq', torch.LongTensor([[bos_idx]]))
        self.register_buffer('blank_seqs', torch.full((beam_size, max_seq_len), self.pad_idx, dtype=torch.long))
        self.register_buffer('len_map', torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))
        self.blank_seqs[:, 0] = self.bos_idx

        with torch.no_grad():
            encoder_output, generated_seq, scores = self._get_init_state(features=features)
            ans_idx = 0

            for step in range(2, self.max_seq_len):
                generated_mask = self.get_subsequent_mask(seq=generated_seq[:step])
                decoder_output = self._model_decode(generated_seq=generated_seq, generated_mask=generated_mask,
                                                   encoder_output=encoder_output, encoder_mask=None)
                generated_seq, scores = self._get_the_best_score_and_idx(generated_seq=generated_seq,
                                                                        decoder_output=decoder_output,
                                                                        scores=scores,
                                                                        step=step)
                eos_locs = generated_seq == self.eos_idx
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break

        return generated_seq[ans_idx][:seq_lens[ans_idx]]


if __name__ == '__main__':
    n_vocab = 10000
    pad_idx = 0
    batch_size = 16
    beam_size = 5
    max_seq_len = 30
    bos_idx = 0
    eos_idx = 500
    features = torch.ones((batch_size, 10, 512))
    gt_seq = torch.ones((batch_size, max_seq_len)).long()

    temp_trans = Transformer(n_vocab=n_vocab, pad_idx=pad_idx)
    
    out = temp_trans(features=features, gt_seq=gt_seq)
    print(out.shape)

    seq = temp_trans.generate_sentence(features=features[0].unsqueeze(0), beam_size=beam_size, max_seq_len=max_seq_len,
                                      bos_idx=9998, eos_idx=9999)
    print(seq.shape)









