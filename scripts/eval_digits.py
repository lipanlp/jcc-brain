"""只评人工核对过的数字字段；缺失和本应不可见的误报同样计错。"""
from collections import defaultdict


def get_field(state: dict, field: str):
    value = state
    for part in field.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def evaluate(predictions: dict[str, dict], truth: dict[str, dict]) -> dict:
    fields, splits = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    errors = []
    for file, row in truth.items():
        prediction = predictions.get(file, {})
        for name, expected in row['fields'].items():
            actual = get_field(prediction, name)
            ok = actual == expected
            fields[name][0] += ok
            fields[name][1] += 1
            splits[row['split']][0] += ok
            splits[row['split']][1] += 1
            if not ok:
                errors.append(dict(file=file, field=name, expected=expected, actual=actual))
    return dict(correct=sum(v[0] for v in fields.values()), total=sum(v[1] for v in fields.values()),
                fields=dict(fields), splits=dict(splits), errors=errors)
