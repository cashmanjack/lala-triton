"""Single place for filesystem roots. Override with LALA_TRITON_ROOT."""
import os
LEGACY_ROOT = '/home/min/a/cashman3/lala-triton-killtest'      # the old hardcoded path from rounds 1-3, kept here as the one documented constant
# default: the repo root that contains this package (== LEGACY_ROOT on the original machine); override with LALA_TRITON_ROOT
ROOT = os.environ.get('LALA_TRITON_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILES = os.path.join(ROOT, 'profiles')
RUNS = os.path.join(ROOT, 'runs')
TRITON_CACHE = os.environ.get('TRITON_CACHE_DIR', os.path.join(ROOT, '.triton_cache'))
