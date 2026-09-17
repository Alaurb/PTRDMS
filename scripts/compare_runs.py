"""Compare saved pipeline observations without treating either run as truth."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripeness_demo.detectors import _box_iou


def compare(old, new):
    def box(row):
        return tuple(float(row[k]) for k in ('bbox_x1','bbox_y1','bbox_x2','bbox_y2'))
    pairs = sorted((-_box_iou(box(a),box(b)),i,j) for i,a in enumerate(old) for j,b in enumerate(new)
                   if (a['frame'],a['side']) == (b['frame'],b['side']) and _box_iou(box(a),box(b)) >= .5)
    used_old, used_new, changes = set(), set(), []
    for _,i,j in pairs:
        if i in used_old or j in used_new:
            continue
        used_old.add(i); used_new.add(j)
        if old[i]['class_name'] != new[j]['class_name']:
            changes.append(dict(frame=new[j]['frame'],side=new[j]['side'],old=old[i]['class_name'],new=new[j]['class_name']))
    return dict(old_count=len(old), new_count=len(new), old_classes=dict(Counter(r['class_name'] for r in old)),
                new_classes=dict(Counter(r['class_name'] for r in new)),matched=len(used_old),class_changes=changes,
                added=[r for j,r in enumerate(new) if j not in used_new],removed=[r for i,r in enumerate(old) if i not in used_old],
                interpretation='Run differences, not accuracy gains. Matched greedily by IoU >= 0.5 within frame/side.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',required=True); parser.add_argument('--new',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    def load(path):
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    result=compare(load(args.old),load(args.new))
    result['input_sha256']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('old',args.old),('new',args.new)]}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('added','removed')},indent=2))
    print(f"Added: {len(result['added'])}; removed: {len(result['removed'])}")
