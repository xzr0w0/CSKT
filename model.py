import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys, math
from copy import deepcopy as cp
import time
device = torch.device('cuda')
torch.cuda.set_device(1)

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nvlc,vw->nwlc',(x,A))
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,supports_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*supports_len+1)*c_in
        self.mlp = nn.Linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=-1) #[320, 112, 19, 160] [bz,n,t,hidden]
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h



class QKVAttention(nn.Module):
    def __init__(self, in_dim,hidden_size, dropout, num_heads = 1):
        super(QKVAttention, self).__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** 0.5

        # Linear layers to project input to query, key, value
        self.query = nn.Linear(in_dim, hidden_size)
        self.key = nn.Linear(in_dim, hidden_size)
        self.value = nn.Linear(in_dim, hidden_size)

        # Final projection layer after attention heads are concatenated
        self.proj = nn.Linear(in_dim, hidden_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv=None, mask=None):
        B, N, T, C = x.shape  # Batch size (B), Spatial/Temporal length (N, T), Embedding size (C=in_dim)

        if kv is None:
            kv = x
        # Linear projections
        query = self.query(x)
        key = self.key(kv)
        value = self.value(kv)

        # Reshape for multi-head attention
        query = query.view(B, N, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, N, T, head_dim)
        key = key.view(B, N, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)      # (B, num_heads, N, T, head_dim)
        value = value.view(B, N, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, N, T, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # (B, num_heads, N, T, T)
        # Apply mask (optional)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
           
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # print('2',attn_weights.shape,value.shape) 
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)  # (B, num_heads, N, T, head_dim)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.permute(0, 2, 3, 1, 4).contiguous().view(B, N, T, C)
        out = self.proj(attn_output)
        return out

    def create_anti_diagonal_triangular_matrix(n, upper=True):
        eye = torch.eye(n)

        if upper:
            tri_matrix = torch.triu(eye.flip(1), diagonal=0).flip(1)
        else:
            tri_matrix = torch.tril(eye.flip(1), diagonal=0).flip(1)
        
        return tri_matrix


class LayerNorm(nn.Module):
    #Assume input has shape B, N, T, C
    def __init__(self, normalized_shape, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(*normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        #mean --> shape :(B, C, H, W) --> (B)
        #mean with keepdims --> shape: (B, C, H, W) --> (B, 1, 1, 1)
        mean = x.mean(dim = dims, keepdims = True)
        std = x.std(dim = dims, keepdims = True, unbiased = False)
        #x_norm = (B, C, H, W)
        x_norm = (x - mean) / (std + self.eps)
        # print(x)
        out = x_norm * self.gamma + self.beta
        return out


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum = 0.1, eps = 1e-5, track_running_stats = True):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        dims = [i for i in range(x.dim() - 1)]
        mean = x.mean(dim = dims)
        var = x.var(dim = dims, correction = 0)
        if (self.training) and (self.running_mean is not None):
            avg_factor = self.momentum
            moving_avg = lambda prev, cur: (1 - avg_factor) * prev + avg_factor * cur.detach()
            dims = [i for i in range(x.dim() - 1)]
            self.running_mean = moving_avg(self.running_mean, mean)
            self.running_var = moving_avg(self.running_var, var)
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = x_norm * self.gamma + self.beta
        return out


class SkipConnection(nn.Module):
    def __init__(self, module, norm):
        super(SkipConnection, self).__init__()
        self.module = module
        self.norm = norm

    def forward(self, x, aux = None):
        return self.norm(x + self.module(x, aux))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout, activation = nn.GELU()):
        super(PositionwiseFeedForward, self).__init__()
        self.act = activation
        self.l1 = nn.Linear(in_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, in_dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, kv = None):
        return self.dropout(self.l2(self.act(self.l1(x))))



class STModel(nn.Module):
    def __init__(self, hidden_size, supports_len, c_num, dropout, layers, out_dim = 1, in_dim = 32, spatial = False, activation = nn.ReLU()):
        super(STModel, self).__init__()
        self.spatial = spatial
        self.act = activation

        s_gcn = gcn(c_in = hidden_size, c_out = hidden_size, dropout = dropout, supports_len = supports_len, order = 2)
        t_attn = QKVAttention(in_dim = hidden_size, hidden_size = hidden_size, dropout = dropout)
        ff = PositionwiseFeedForward(in_dim = hidden_size, hidden_size = 4 * hidden_size, dropout = dropout)
        norm = LayerNorm(normalized_shape = (hidden_size, ))
        self.start_linear = nn.Linear(in_dim, hidden_size)

        if out_dim == 1:
            self.proj = nn.Linear(hidden_size, hidden_size + out_dim)
        else:
            self.proj = nn.Linear(hidden_size, out_dim)
        self.out_dim = out_dim

        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()

        for _ in range(layers):
            self.temporal_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.spatial_layers.append(SkipConnection(cp(s_gcn), cp(norm)))
            self.ed_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.ff.append(SkipConnection(cp(ff), cp(norm)))

    def forward(self, x, prev_hidden, supports):
        x = self.start_linear(x)
        x_start = x
        hiddens = []
        for i, (temporal_layer, spatial_layer, ed_layer, ff) in enumerate(zip(self.temporal_layers, self.spatial_layers, self.ed_layers, self.ff)):
            if not self.spatial:
                x1 = temporal_layer(x) # B, N, T, C
                x_attn = spatial_layer(x1, supports) # B, N, T, C
            else:
                x1 = spatial_layer(x, supports)
                x_attn = temporal_layer(x1)
            if prev_hidden is not None:
                x_attn = ed_layer(x_attn, prev_hidden[-1])
            x = ff(x_attn)
            hiddens.append(x)

        out = self.act(x)

        return x_start - out[...,], out, hiddens


class AttentionModel(nn.Module):
    """
    Input shape B, N, T, in_dim
    Output shape B, N, T, out_dim

    """
    def __init__(self, hidden_size, layers, dropout, edproj = False, in_dim = 2, out_dim = 1, spatial = False, activation = nn.ReLU()):
        super(AttentionModel, self).__init__()
        self.spatial = spatial
        self.act = activation

        base_model = SkipConnection(QKVAttention(hidden_size, hidden_size, dropout = dropout), LayerNorm(normalized_shape = (hidden_size, )))
        ff = SkipConnection(PositionwiseFeedForward(hidden_size, 4 * hidden_size, dropout = dropout), LayerNorm(normalized_shape = (hidden_size, )))

        self.start_linear = nn.Linear(in_dim, hidden_size)

        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()

        for i in range(layers):
            self.spatial_layers.append(cp(base_model))
            self.temporal_layers.append(cp(base_model))
            self.ed_layers.append(cp(base_model))
            self.ff.append(cp(ff))

        self.proj = nn.Linear(hidden_size, out_dim)


    def forward(self, x, prev_hidden = None):
        x = self.start_linear(x)
        
        for i, (s_layer, t_layer, ff) in enumerate(zip(self.spatial_layers, self.temporal_layers, self.ff)):
            if not self.spatial:
                x1 = t_layer(x)
                x_attn = s_layer(x1.transpose(1,2))
            else:
                x1 = s_layer(x.transpose(1,2))
                x_attn = t_layer(x1.transpose(1,2)).transpose(1,2)

            if prev_hidden is not None:
                x_attn = self.ed_layers[i](x_attn.transpose(1,2), prev_hidden[-1])
                x_attn = x_attn.transpose(1,2)
            x = ff(x_attn.transpose(1,2))

        return self.act(x), x


class TemporalModel(nn.Module):
    def __init__(self, n_q, n_c, d_a, d_e, d_k, q_matrix, dropout=0.2):
        super(TemporalModel, self).__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e 
        self.q_matrix = q_matrix.float().to(device)
        self.n_c = n_c

        self.q_embed = nn.Embedding(n_q + 10, d_k)
        torch.nn.init.xavier_uniform_(self.q_embed.weight)
        self.linear_1 = nn.Linear(d_a + d_e, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)


        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_data, a_data): 
        batch_size, seq_len = q_data.size(0), q_data.size(1)
        q_embed_data = self.q_embed(q_data)
        a_data = a_data.reshape(-1, 1).repeat(1, self.d_a).reshape(batch_size, -1, self.d_a) # (bs, seq_len, d_a)
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_c, self.d_k)).repeat(batch_size, 1, 1).to(device)  # (bs, n_skill, d_k)
        h_tilde_pre = None
        qa = self.linear_1(torch.cat((q_embed_data,  a_data), 2)) # (bs, seq_len, d_k)
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)

        H = []
        pred = torch.zeros(batch_size, seq_len).to(device)
        
        for t in range(0, seq_len - 1):
            # Knowledge perception module.
            e = q_data[:, t].to(device)
            # q_e: (bs, 1, n_skill)
            q_e = self.q_matrix[e].view(batch_size, 1, -1).float().to(device)
            # it = it_embed_data[:, t]
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre.float()).view(batch_size, self.d_k)
            learning = qa[:, t]
            KP_t = self.linear_2(torch.cat((learning_pre, learning, h_tilde_pre), 1))
            # print(KP_t.shape) # bz,dim
            KP_t = self.tanh(KP_t)
            #Knowledge comprehension module.
            R_t =self.sig(self.linear_3(torch.cat((learning_pre,  learning, h_tilde_pre), 1)))
            KC_t = R_t * ((KP_t + 1) / 2)
            KC_t_tilde = self.dropout(q_e.transpose(1, 2).bmm(KC_t.view(batch_size, 1, -1)))
            
            H.append(h_pre)
            n_skill = KC_t_tilde.size(1)
            KR_t = self.sig(self.linear_4(torch.cat((
                h_pre,
                KC_t.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                # it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))* h_pre
            h = KC_t_tilde + KR_t   # (bs, n_skill, d_k)
            h_tilde = self.q_matrix[q_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k) # (bs, d_k)
           # prepare for next prediction
            y = self.sig(self.linear_5(torch.cat((q_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        H = self.sig(torch.tensor([item.cpu().detach().numpy() for item in H])).cuda()  #[length_input, bz, concept, hid]
        return H, pred, q_embed_data, qa


class Gate(nn.Module):
    """
    Input
     - input: B, T, in_dim, original input
     - hidden: hidden states from each expert, shape: E-length list of (B, T, C) tensors, where E is the number of experts
    Output
     - similarity score (i.e., routing probability before softmax function)
    Arguments
     - mem_hid, memory_size: hidden size and total number of memory units
     - sim: similarity function to evaluate routing probability
     - nodewise: flag to determine routing level
    """
    def __init__(self, hidden_size, c_num, memory_size, mem_hid=32, input_dim=32, output_dim=1,  sim=nn.CosineSimilarity(dim=-1), ind_proj=True, attention_type='attention'):
        super(Gate, self).__init__()
        self.attention_type = attention_type
        self.sim = sim
        
        # Memory-related parameters
        self.memory = nn.Parameter(torch.empty(memory_size, mem_hid))
        self.hid_query = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(3)])
        self.key = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(3)])
        self.value = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(3)])
        self.input_query = nn.Parameter(torch.empty(input_dim, mem_hid))

        # Routing weights
        self.We1 = nn.Parameter(torch.empty(c_num, memory_size))
        self.We2 = nn.Parameter(torch.empty(c_num, memory_size))

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
    
    def forward(self, input, hidden):
        if self.attention_type == 'attention':
            attention = self.attention
        else:
            attention = self.topk_attention
            
        B, T, _ = input.size()  # Adjusted for B, T, in_dim input
        # Query memory based on input
        memories = self.query_mem(input)  #B,T,D_mem

        scores = []
        for i, h in enumerate(hidden):
            hidden_att = attention(h, i)
            scores.append(self.sim(memories, hidden_att))

        scores = torch.stack(scores, dim=-1)
        return scores

    def attention(self, x, i):
        B, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i])
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])

        energy = torch.matmul(query, key.transpose(-1, -2))
        score = torch.softmax(energy, dim=-1)
        out = torch.matmul(score, value)
        return out

    def topk_attention(self, x, i, k=3):
        B, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i])
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])

        energy = torch.matmul(query, key.transpose(-1, -2))
        values, indices = torch.topk(energy, k=k, dim=-1)
        score = energy.zero_().scatter_(-1, indices, torch.relu(values))
        out = torch.matmul(score, value)
        return out

    def query_mem(self, input):
        B, T, _ = input.size()
        mem = self.memory  # memory_size, D_mem
        query = torch.matmul(input, self.input_query) # B,T,D_in * D_in, D_men
        energy = torch.matmul(query, mem.T)  # B,T,memory_size 
        score = torch.softmax(energy, dim=-1) 
        out = torch.matmul(score, mem)  #B,T,D_mem
        return out

    def reset_queries(self):
        with torch.no_grad():
            for p in self.hid_query:
                nn.init.xavier_uniform_(p)
            nn.init.xavier_uniform_(self.input_query)
    
    def reset_params(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in "We1 We2 memory".split():
                    continue
                else:
                    nn.init.xavier_uniform_(p)

          

class TESTAM(nn.Module):
    """
    TESTAM model
    """
    def __init__(self, device,num_timesteps_input,num_timesteps_output, q_num, c_num,q_matrix, dropout=0.3, in_dim=2, out_dim=12,hidden_size=32, layers=5, prob_mul = False, **args):
        super(TESTAM, self).__init__()
        self.dropout = dropout
        self.prob_mul = prob_mul
        self.supports_len = 2
        self.c_num = c_num
        self.q_matrix = q_matrix
        self.hidden_size = hidden_size
        self.memory_size = 20
        self.temporalmodel = TemporalModel(n_q = q_num, n_c = c_num, d_a = 2, d_e = 32, d_k = 32, q_matrix = q_matrix)
        self.adaptive_expert = STModel(hidden_size, self.supports_len, c_num, in_dim = in_dim, layers = 1, dropout = dropout, spatial = True,)
        self.attention_expert = AttentionModel(hidden_size, in_dim = in_dim, layers = layers, dropout = dropout)
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.num_experts = 3

        self.gate_network = Gate(hidden_size, c_num, self.memory_size)
        for model in [self.temporalmodel, self.adaptive_expert, self.attention_expert]:
            for n, p in model.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        self.linear_out = nn.Linear(2*hidden_size,hidden_size)
        self.sig = nn.Sigmoid()
    
    def forward(self, q_data, a_data, gate_out = False):
        """
        input: B, in_dim, N, T
        o_identity shape B, N, T, 1
        """
        t0 = time.time()
        input,lpkt_out,qy_embed, qa = self.temporalmodel(q_data, a_data)
        t1 = time.time()
        # print('lpkt',t1-t0)
        input = input.permute(1,2,0,3) #[length_input, bz, concept, hid]
        # print(input.shape,lpkt_out.shape)  #[bz, n, t, hid]
        print(self.gate_network.We1.shape, self.gate_network.We1)
        n1 = torch.matmul(self.gate_network.We1, self.gate_network.memory) 
        n2 = torch.matmul(self.gate_network.We2, self.gate_network.memory)
        g1 = torch.softmax(torch.relu(torch.mm(n1, n2.T)), dim = -1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        g2 = torch.softmax(torch.relu(torch.mm(n2, n1.T)), dim = -1)
        # print(g1.shape) #[c_num,c_num]
        new_supports = [g1, g2] #[1, (c_num,c_num), (c_num,c_num)]
        
        _, o_adaptive, h_adaptive = self.adaptive_expert(input, None, new_supports)
        t2 = time.time()
        # print('expert2',t2-t1)
        # print(o_adaptive.shape)
        o_attention, h_attention = self.attention_expert(input)
        t3 = time.time()
        # print('expert3',t3-t2)
        # print(torch.sigmoid(o_adaptive))
        # print(torch.sigmoid(o_attention))

        q_y = q_data[:,-self.num_timesteps_output:]
        bz,seq_len = q_y.shape[0], q_y.shape[1]
        q_skill = torch.zeros((bz, seq_len, self.c_num),device = 'cuda') #[bz,len,skill]
        for i in range(bz):
            for j in range(seq_len):
                skill_index = q_y[i, j]  
                q_skill[i, j] = self.q_matrix[skill_index].cuda() ## torch.Size([bz,seq_len,node])
        # print(o_adaptive.shape,q_skill.shape)
        # print(o_adaptive.squeeze(-1).permute(0,2,1) * q_skill)
        expert2_out = torch.sum(o_adaptive.permute(3,0,2,1) * q_skill, dim = -1).permute(1,2,0)\
         / torch.sum(q_skill,dim=-1).unsqueeze(-1).expand(-1, -1, self.hidden_size)
        expert3_out = torch.sum(o_attention.permute(3,0,2,1) * q_skill, dim = -1).permute(1,2,0)\
         / torch.sum(q_skill,dim=-1).unsqueeze(-1).expand(-1, -1, self.hidden_size) #[B,T,dim]
        # print(expert2_out,qy_embed.shape)
        res2 = self.sig(self.linear_out(torch.cat((qy_embed[:,-1:,], expert2_out[:,-self.num_timesteps_output:,]), -1))).sum(2)/self.hidden_size
        res3 = self.sig(self.linear_out(torch.cat((qy_embed[:,-1:,], expert3_out[:,-self.num_timesteps_output:,]), -1))).sum(2)/self.hidden_size
        # print('res1',lpkt_out[:,-1:])
        # print('res2',res2.shape) 
        # print('res3',res3)
        ind_out = torch.cat([lpkt_out[:,-self.num_timesteps_output:], res2, res3], dim = -1) #[160, 1, 3]
        #[bz,out_len,3]
        ind_out = ind_out.view(bz,seq_len,self.num_experts)

        B, N, T, D = input.size()
        q_e = self.q_matrix[q_data[:,:-self.num_timesteps_output]].view(B, T, N)
        gate1 = torch.einsum('btn, bntd->btd', q_e.float(), input.float())
        gate2 = torch.einsum('btn, bntd->btd', q_e.float(), h_adaptive[-1].float())
        gate3 = torch.einsum('btn, bntd->btd', q_e.float(), h_attention.float())
        gate_in = [gate1, gate2, gate3]
        e_a = qa[:,:19,:]
        gate = torch.softmax(self.gate_network(e_a, gate_in), dim = -1)
        # print('gate',gate.shape) #([160, 19, 3])
        gate = gate[:,-self.num_timesteps_output:,:]

        out = torch.zeros_like(res2).view(-1, 1)


        outs = [lpkt_out[:, -self.num_timesteps_output:], res2, res3]  

        gate = gate.view(-1, self.num_experts)  


        for i in range(len(outs)):
            cur_out = outs[i].view(-1, 1)
            out += cur_out * gate[:, i].unsqueeze(dim=-1)
        if self.prob_mul:
            route_prob_max, _ = torch.max(gate, dim=-1)
            out = out * route_prob_max.unsqueeze(dim=-1)
        # print('output_t',time.time()-t3)

        # print(out.shape,out)
        if self.training or gate_out:
            return out, gate, ind_out,lpkt_out
        else:
            return out,lpkt_out
