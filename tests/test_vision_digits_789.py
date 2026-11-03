import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from jccbrain.vision.eye.digits import TEMPLATE_FILE, read_digits

FIXTURE = Path(__file__).parent / 'fixtures/digit_cells_789.npz'


def test_supplement_preserves_previous_templates_and_only_extends_789():
    with np.load(FIXTURE, allow_pickle=False) as fixture, np.load(TEMPLATE_FILE, allow_pickle=False) as data:
        old = str(fixture['base_chars'])
        chars = str(data['chars'])
        assert chars.startswith(old)
        assert np.array_equal(data['glyphs'][:len(old)], fixture['base_glyphs'])
        assert Counter(chars[len(old):]) == {'7': 1, '8': 3, '9': 1}


@pytest.mark.parametrize('file,field,expected', [
    ('0006', 'timer', '18'), ('0006', 'level', '7'),
    ('0008', 'gold', '39'), ('0008', 'level', '8'), ('0008', 'xp', '10/68'),
    ('0009', 'xp', '58/68'), ('0011', 'gold', '57'), ('0012', 'xp', '8/68'),
])
def test_real_new_digits_and_non_template_frame(file, field, expected):
    with np.load(FIXTURE, allow_pickle=False) as data:
        row = next(r for r in json.loads(str(data['metadata']))
                   if r['file'] == f'digits/{file}.png' and r['field'] == field)
        assert read_digits(data[row['key']], row['height']) == expected


def test_digit_metrics_count_missing_values_and_invisible_field_false_positives():
    from scripts.eval_digits import evaluate
    result = evaluate({'a': {'stage': '2-7', 'gold': 4, 'unscored': 99}}, {
        'a': {'split': 'template', 'fields': {'stage': '2-7', 'gold': None, 'level': 7}},
        'b': {'split': 'validation', 'fields': {'xp': [1, 8]}},
    })
    assert (result['correct'], result['total']) == (1, 4)
    assert result['splits'] == {'template': [1, 3], 'validation': [0, 1]}
    assert len(result['errors']) == 3 and 'unscored' not in result['fields']
