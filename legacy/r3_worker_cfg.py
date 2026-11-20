def cfg_from_key(k):
    t, g, w, s = k.split('_'); bm, bn, bk = map(int, t.split('x'))
    return dict(BM=bm, BN=bn, BK=bk, G=int(g[1:]), num_warps=int(w[1:]), num_stages=int(s[1:]))
