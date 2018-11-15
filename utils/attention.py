import torch
import torch.nn as nn
import torch.nn.functional as F

# rewriten to our purpose ~ also i abandon performance, and favor my logic flow with transposing ( self-educational purpose, not production one .. )

class AttentionNN(nn.Module):
    """
        paper : https://arxiv.org/abs/1706.03762
        source : https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_subtasks = cfg['n_simulations']
        self.output = nn.Linear(2 * self.n_subtasks, self.n_subtasks)

# lots of transposing, need to refactor lol .. but just for poc it is OK
    def forward(self, data, attention):
        inputs = data.view(data.size(0) // self.n_subtasks, self.n_subtasks, -1)
        attention = attention.view(attention.size(0) // self.n_subtasks, self.n_subtasks, -1)

        attn = torch.bmm(inputs.transpose(1, 2), attention)
        attn = F.log_softmax(attn, dim=1)
        attn = torch.bmm(attn, attention.transpose(1, 2))

        concat = torch.cat([attn.transpose(1, 2), inputs], dim=1)
        concat = concat.transpose(1, 2).reshape(-1, self.n_subtasks * 2)
        output = torch.sigmoid(self.output(concat))
        return torch.mul(output.t().contiguous().view(data.shape), data) * self.cfg['attention_amplifier']

class SimulationAttention(AttentionNN):
    def __init__(self, state_size, action_size, cfg):
        super().__init__(cfg)
        self.encoding = nn.Linear(
                action_size + cfg['her_state_size'] + state_size * cfg['history_count'],
                cfg['attention_hidden'])

    def forward(self, gradients, state, action):
        attention = F.relu(self.encoding(torch.cat([state, action], dim=1)))
        return super().forward(gradients, attention)
