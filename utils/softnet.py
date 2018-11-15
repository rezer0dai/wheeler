import os
import torch
import numpy as np

def filter_dict(full_dict, blacklist):
    sheeps = list(filter(
        lambda k: any(b in k for b in blacklist),
        full_dict.keys()))
    for sheep in sheeps:
        full_dict.pop(sheep)
    return full_dict

class SoftUpdateNetwork:
    def share_memory(self):
        self.ac_target.share_memory()
        self.ac_explorer.share_memory()

    def soft_mean_update(self, tau):
        if not tau:
            return

        params = np.mean([list(explorer.parameters()) for explorer in self.ac_explorer.actor], 0)

        for target_w, explorer_w in zip(self.ac_target.actor[0].parameters(), params):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)

        self.ac_target.actor[0].remove_noise()
        for explorer in self.ac_explorer.actor:
            explorer.sample_noise()

    def soft_update(self, ind, tau):
        if not tau:
            return

        for target_w, explorer_w in zip(self.ac_target.critic[ind].parameters(), self.ac_explorer.critic[ind].parameters()):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)# / len(self.explorer))

    def sync_explorers(self, update_critics = False):
        self._sync_explorers(self.ac_target.actor, self.ac_explorer.actor, 1)
        if not update_critics:
            return
        count = len(self.ac_explorer.critic) // len(self.ac_target.critic)
        self._sync_explorers(self.ac_target.critic, self.ac_explorer.critic, count)

    def _sync_explorers(self, targets, explorers, count):
        for i, target in enumerate(targets):
            for explorer in explorers[i*count:(i+1)*count]:
                for target_w, explorer_w in zip(target.parameters(), explorer.parameters()):
                    explorer_w.data.copy_(target_w.data)

    def save_models(self, cfg, model_id, prefix):
        if not cfg['save']:
            return

        target = os.path.join(cfg['model_path'], '%s_target_%s'%(prefix, model_id))
        torch.save(self.ac_target.state_dict(), target)

        explorer = os.path.join(cfg['model_path'], '%s_explorer_%s'%(prefix, model_id))
        torch.save(self.ac_explorer.state_dict(), explorer)

    def load_models(self, cfg, model_id, prefix, blacklist = []):
        if not cfg['load']:
            return

        target = os.path.join(cfg['model_path'], '%s_target_%s'%(prefix, model_id))
        if not os.path.exists(target):
            return
        model = filter_dict(torch.load(target), blacklist)
        self.ac_target.load_state_dict(model, strict=False)

        path = os.path.join(cfg['model_path'], '%s_explorer_%s'%(prefix, model_id))
        if not os.path.exists(path):
            return
        model = filter_dict(torch.load(path), blacklist)
        self.ac_explorer.load_state_dict(model, strict=False)
