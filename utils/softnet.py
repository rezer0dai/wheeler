import os
import torch

def filter_dict(full_dict, blacklist):
    sheeps = list(filter(
        lambda k: any(b in k for b in blacklist),
        full_dict.keys()))
    for sheep in sheeps:
        full_dict.pop(sheep)
    return full_dict

class SoftUpdateNetwork:
    def soft_update(self, ind, tau):
        if not tau:
            return

        for target_w, explorer_w in zip(self.target.parameters(), self.explorer[ind].parameters()):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau / len(self.explorer))

        self.target.remove_noise()
        self.explorer[ind].sample_noise()

    def sync_explorers(self):
        for explorer in self.explorer:
            for target_w, explorer_w in zip(self.target.parameters(), explorer.parameters()):
                explorer_w.data.copy_(target_w.data)
            explorer.sample_noise()

    def share_memory(self):
        self.target.share_memory()
        for explorer in self.explorer:
            explorer.share_memory()

    def _load_models(self, cfg, model_id, prefix, blacklist = []):
        if not cfg['load']:
            return
        
        target = os.path.join(cfg['model_path'], '%s_target_%s'%(prefix, model_id))
        if not os.path.exists(target):
            return

        model = filter_dict(torch.load(target), blacklist)
        self.target.load_state_dict(model, strict=False)

        for i, explorer in enumerate(self.explorer):
            path = os.path.join(cfg['model_path'], '%s_explorer_%s_%i'%(prefix, model_id, i))
            if not os.path.exists(path):
                continue
            model = filter_dict(torch.load(path), blacklist)
            explorer.load_state_dict(model, strict=False)

    def _save_models(self, cfg, model_id, prefix, xid):
        if not cfg['save']:
            return
        target = os.path.join(cfg['model_path'], '%s_target_%s'%(prefix, model_id))
        explorer = os.path.join(cfg['model_path'], '%s_explorer_%s_%i'%(prefix, model_id, xid))

        torch.save(self.target.state_dict(), target)
        torch.save(self.explorer[xid].state_dict(), explorer)
