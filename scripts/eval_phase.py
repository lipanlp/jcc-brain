"""独立的phase/next_round/round_type分类型统计；未知类和未评分帧显式保留。"""
from collections import Counter


def frame_key(file, samples_root):
    try:
        return file.resolve().relative_to(samples_root.resolve()).as_posix()
    except ValueError:
        return f'{file.parent.name}/{file.name}'


def evaluate(predictions: dict[str, dict], truth: dict[str, dict]) -> dict:
    report = {}
    for key in ('phase', 'next_round', 'round_type'):
        counts, correct, confusion = Counter(), Counter(), Counter()
        for file, pred in predictions.items():
            if file not in truth or key not in truth[file]:
                continue
            want, got = truth[file][key], pred.get(key, 'unknown')
            counts[want] += 1
            correct[want] += want == got
            confusion[(want, got)] += 1
        report[key] = dict(classes={name: dict(correct=correct[name], total=n,
                                               accuracy=correct[name] / n) for name, n in sorted(counts.items())},
                           correct=sum(correct.values()), total=sum(counts.values()),
                           confusion=[dict(expected=a, predicted=b, count=n) for (a, b), n in sorted(confusion.items())])
    return report
