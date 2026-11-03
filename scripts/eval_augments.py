"""已选强化符文：整列表（含顺序）与名字多重集合精确率/召回率。"""
from collections import Counter


def evaluate(predictions, truth):
    exact = frames = tp = fp = fn = 0
    diffs = []
    for file, row in truth.items():
        if 'augments_taken' not in row:
            continue
        want = row['augments_taken']
        got = predictions.get(file, {}).get('augments_taken', [])
        frames += 1
        exact += want == got
        a, b = Counter(want), Counter(got)
        tp += sum((a & b).values())
        fp += sum((b-a).values())
        fn += sum((a-b).values())
        if want != got:
            diffs.append(dict(file=file, expected=want, predicted=got))
    return dict(frames=frames, exact=exact, accuracy=exact/frames if frames else None,
                tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None, diffs=diffs)
