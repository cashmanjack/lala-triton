"""GPU profile = JSON. Keys: name, n_sm, l2_eff, hbm_bw, l2_bw, peak_tflops{dtype}, lat{dtype} (Little's-law latency, s),
eff{"dtype|layout": {tile_key: eff}}, meta{"dtype|layout": {tile_key: {occ,n_regs,shared}}}, notes."""
import json, os
from .config import PROFILES

class Profile(dict):
    @property
    def n_sm(self): return self['n_sm']
    def eff_for(self, dtype, layout, tile, default=None):
        t = self['eff'].get(f'{dtype}|{layout}', {})
        if tile in t: return t[tile]
        if default is not None: return default
        v = sorted(t.values()); return v[len(v)//2] if v else 0.3
    def occ_for(self, dtype, layout, tile):
        return self['meta'].get(f'{dtype}|{layout}', {}).get(tile, {}).get('occ')

def load(path_or_name):
    p = path_or_name if os.path.exists(path_or_name) else os.path.join(PROFILES, path_or_name + ('' if path_or_name.endswith('.json') else '.json'))
    with open(p) as f: return Profile(json.load(f))

def save(prof, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f: json.dump(prof, f, indent=1)
