import torch
import torch.nn as nn
import torch.nn.functional as F
from independent_job.model.matrix_net.sub_model import *

class CloudMatrixModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.nT = self.model_params['nT']
        self.nM = self.model_params['nM']

        embedding_dim = self.model_params['embedding_dim']

        # self.position_embedding
        self.T_embedding = nn.Linear(self.nT, embedding_dim)
        self.M_embedding = nn.Linear(self.nM, embedding_dim)
        self.encoder = Matrix_Encoder(**model_params)
        self.decoder = MMatrix_Decoder(**model_params)

    def forward(self, machine_state, task_state, D_TM, ninf_mask):
        # machine_state : [B, M, Feature]
        # task_state : [B, T, Feature]
        # D_TM : [B, T, M]
        # ninf_mask : [B, M, T]

        batch_size = machine_state.size(0)
        # pomo_size = state.BATCH_IDX.size(1)
        
        # position embedding -> 일단은 linear로
        row_emb = self.T_embedding(task_state)
        col_emb = self.M_embedding(machine_state)

        encoded_task, encoded_machine = self.encoder(row_emb, col_emb, D_TM)
        # (B, T, embedding), (B, M, embedding)

        probs = self.decoder(encoded_machine, encoded_task, ninf_mask)
        # shape: (B, M*T)

        if self.training or self.model_params['eval_type'] == 'softmax':
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample().reshape(batch_size, 1)
            # [B, 1]
            logpa = dist.log_prob(task_selected)
            # [B, 1]
        else:  
            task_selected = probs.argmax(dim=1)
            logpa = None

        return task_selected, logpa
        

class OneStageModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.nT = self.model_params['nT']
        self.nM = self.model_params['nM']

        embedding_dim = self.model_params['embedding_dim']

        # self.position_embedding
        self.T_embedding = nn.Linear(self.nT, embedding_dim)
        self.M_embedding = nn.Linear(self.nM, embedding_dim)
        self.encoder = Matrix_Encoder(**model_params)
        self.decoder = Matrix_Decoder(**model_params)

    def forward(self, machine_state, task_state, D_TM, ninf_mask, machine_idx):
        # machine_state : [B, M, Feature]
        # task_state : [B, T, Feature]
        # D_TM : [B, T, M]
        # ninf_mask : [B, M, T]
        # machine_idx : int
        # machine_ninf_mask = ninf_mask[:, [machine_idx], :]
        # machine_ninf_mask_plus_1 = torch.cat((torch.tensor([[[0]]]), machine_ninf_mask), dim=-1)

        batch_size = machine_state.size(0)
        # pomo_size = state.BATCH_IDX.size(1)
        
        # position embedding -> 일단은 linear로
        row_emb = self.T_embedding(task_state)
        col_emb = self.M_embedding(machine_state)

        encoded_task, encoded_machine = self.encoder(row_emb, col_emb, D_TM)
        # (B, T, embedding), (B, M, embedding)

        all_job_probs = self.decoder(encoded_machine[:, [machine_idx], :], encoded_task, ninf_mask)
        # shape: (B, 1, T+1)

        probs = all_job_probs.reshape(batch_size * 1, -1)

        if self.training or self.model_params['eval_type'] == 'softmax':
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample().reshape(batch_size, 1)
            # [B, 1]
            logpa = dist.log_prob(task_selected)
            # [B, 1]
        else:
            task_selected = all_job_probs.argmax(dim=2)
            logpa = None
            # shape: (B, 1)
            # job_prob = torch.zeros(size=(batch_size, 1))  #

        return task_selected, logpa

class Matrix_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, cost_mat):
        # col_emb.shape: (B, T, embedding)
        # row_emb.shape: (B, M, embedding)
        # cost_mat.shape: (B, T, M)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        if model_params['MHA'] == 'depth':
            self.row_encoding_block = EncodingBlock2(**model_params)
        else:
            self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (B, T, embedding)
        # col_emb.shape: (B, M, embedding)
        # cost_mat.shape: (B, T, M)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out

class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # input1.shape: (batch, row_cnt:TorM, embedding)
        # input2.shape: (batch, col_cnt:MorT, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (B, T, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (B, T, embedding)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (B, T, embedding)

class EncodingBlock2(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = Depth_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # input1.shape: (batch, row_cnt:TorM, embedding)
        # input2.shape: (batch, col_cnt:MorT, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (B, T, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (B, T, embedding)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (B, T, embedding)

class Matrix_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))
        # no job action shape : (1, 1, embedding_dim)
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (B, T, embedding)
        batch_size = encoded_jobs.size(0)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        encoded_no_job = self.encoded_NO_JOB.expand(size=(batch_size, 1, embedding_dim))
        encoded_jobs_plus_1 = torch.cat((encoded_no_job, encoded_jobs), dim=1)
        # shape: (B, T+1, embedding)

        self.k = reshape_by_heads(self.Wk(encoded_jobs_plus_1), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs_plus_1), head_num=head_num)
        # shape: (B,H, T+1, qkv_dim)
        self.single_head_key = encoded_jobs_plus_1.transpose(1, 2)
        # shape: (B, embedding, T+1)

    def forward(self, encoded_machine, encoded_jobs, ninf_mask):
        # encoded_machine.shape: (B, 1, embedding)
        # encoded_jobs.shape: (B, T, embedding)
        # ninf_mask.shape: (B, 1, T+1)
        self.set_kv(encoded_jobs)
        task_num = encoded_jobs.size(1)
        machine_num = encoded_machine.size(1)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.Wq_3(encoded_machine), head_num=head_num)
        # shape: (B, H, M, qkv_dim)

        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v,
                                                            rank3_ninf_mask=ninf_mask)
        # shape: (B, 1, H*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (B, 1, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (B, 1, T+1)


        score_scaled = score / sqrt_embedding_dim
        # shape: (B, 1, T+1)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (B, 1, T+1)

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        # q shape: (B, H, 1, qkv_dim)   :
        # k,v shape: (B, H, T+1, qkv_dim)
        # rank2_ninf_mask.shape: (B, T+1)
        # rank3_ninf_mask.shape: (B, 1, T+1)

        batch_size = q.size(0)
        n = q.size(2)
        T_cnt_plus_1 = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (B, H, n:1, T+1)

        score_scaled = score / sqrt_qkv_dim
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, n, T_cnt_plus_1)
        
        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (B, H, n, T+1)

        out = torch.matmul(weights, v)
        # shape: (B, H, n, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (B, n, H, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)
        # shape: (B, n, H*qkv_dim)

        return out_concat

class MMatrix_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # no job action shape : (1, 1, embedding_dim)
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (B, T, embedding)
        batch_size = encoded_jobs.size(0)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_jobs), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), head_num=head_num)
        # shape: (B,H, T, qkv_dim)
        self.single_head_key = encoded_jobs.transpose(1, 2)
        # shape: (B, embedding, T)

    def forward(self, encoded_machine, encoded_jobs, ninf_mask):
        # encoded_machine.shape: (B, 1, embedding)
        # encoded_jobs.shape: (B, T, embedding)
        # ninf_mask.shape: (B, 1, T)
        self.set_kv(encoded_jobs)
        task_num = encoded_jobs.size(1)
        machine_num = encoded_machine.size(1)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.Wq_3(encoded_machine), head_num=head_num)
        # shape: (B, H, M, qkv_dim)

        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v)
        # shape: (B, M, H*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (B, M, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (B, 1, T+1)


        score_scaled = score / sqrt_embedding_dim
        # shape: (B, 1, T+1)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask
        score_masked = score_masked.reshape(-1, machine_num * task_num)
        probs = F.softmax(score_masked, dim=1)
        # shape: (B, M*T)

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v):
        # q shape: (B, H, M, qkv_dim)   :
        # k,v shape: (B, H, T, qkv_dim)
        # rank2_ninf_mask.shape: (B, T)
        # rank3_ninf_mask.shape: (B, M, T)

        batch_size = q.size(0)
        n = q.size(2)
        T_cnt_plus_1 = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (B, H, M, T)

        score_scaled = score / sqrt_qkv_dim
        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (B, H, M, T)

        out = torch.matmul(weights, v)
        # shape: (B, H, M, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (B, M, H, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)
        # shape: (B, M, H*qkv_dim)

        return out_concat


def reshape_by_heads(qkv, head_num):
    # q.shape: (B, T or M, head_num*key_dim)
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (B, T or M, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (B, head_num, T or M, key_dim)

    return q_transposed