import csv, json, glob, os, pandas as pd
UNIT = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9, 'Tbyte': 1e12, '%': 1, 'sector': 1, 'Ksector': 1e3, 'Msector': 1e6, 'Gsector': 1e9}

def parse_dir(d, pred_csv):
    pred = pd.read_csv(pred_csv); recs = []
    for man in glob.glob(f'{d}/*.manifest.json'):
        base = man[:-len('.manifest.json')]
        idx = json.load(open(man))
        rows = [r for r in csv.reader(open(base + '.csv')) if len(r) > 12 and r[0].isdigit()]
        ids = sorted(set(int(r[0]) for r in rows))
        assert len(ids) == len(idx), (base, len(ids), len(idx))
        for kid, pi in zip(ids, idx):
            m = {r[12]: float(r[14].replace(',', '')) * UNIT.get(r[13], 1) for r in rows if int(r[0]) == kid}
            recs.append(dict(pred_idx=pi, meas_read=m['dram__bytes_read.sum'], meas_write=m['dram__bytes_write.sum'],
                             l2_read_sectors=m['lts__t_sectors_op_read.sum'], l2_read_hit=m['lts__t_sector_op_read_hit_rate.pct'],
                             l2_hit=m['lts__t_sector_hit_rate.pct']))
    meas = pd.DataFrame(recs).set_index('pred_idx')
    return pred.join(meas, how='inner')
