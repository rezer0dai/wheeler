import torch
import torch.nn as nn
import torch.nn.functional as F

# reference : https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
# rewriten to our purpose ~ also i abandon performance, and favor my logic flow with transposing ( self-educational purpose, not production one .. )

class AttentionNN(nn.Module):
    def __init__(self, task, cfg):
        super(AttentionNN, self).__init__()
        self.cfg = cfg
        self.sims_count = task.subtasks_count()
        self.output = nn.Linear(2 * self.sims_count, self.sims_count)

# lots of transposing, need to refactor lol .. but just for poc it is OK
    def forward(self, data, attention):
        inputs = data.view(data.size(0) // self.sims_count, self.sims_count, -1)
        attention = attention.view(attention.size(0) // self.sims_count, self.sims_count, -1)

        attn = torch.bmm(inputs.transpose(1, 2), attention)
        attn = F.log_softmax(attn, dim=1)
        attn = torch.bmm(attn, attention.transpose(1, 2))

        concat = torch.cat([attn.transpose(1, 2), inputs], dim=1)
        concat = concat.transpose(1, 2).reshape(-1, self.sims_count * 2)
        output = F.sigmoid(self.output(concat))
        return torch.mul(output.t().contiguous().view(data.shape), data) * self.cfg['attention_amplifier']

class SimulationAttention(AttentionNN):
    def __init__(self, task, cfg):
        super(SimulationAttention, self).__init__(task, cfg)
        self.encoding = nn.Linear(
                task.action_size() + cfg['her_state_size'] + task.state_size() * cfg['history_count'],
                cfg['attention_hidden'])

    def forward(self, gradients, state, action):
        attention = F.relu(self.encoding(torch.cat([state, action], dim=1)))
        return super(SimulationAttention, self).forward(gradients, attention)
