import logging
from types import SimpleNamespace

import pytest

from jccbrain.vision.eye import ocr, rapid
from jccbrain.vision.pyeye import ImageReplayProvider, PyEyeProvider


def reader_with(engine, directory):
    return SimpleNamespace(ocr=engine, units=SimpleNamespace(gallery=SimpleNamespace(dir=directory, names={})))


@pytest.mark.parametrize('kind', ['live', 'replay'])
def test_missing_rapid_warns_and_status_reports_actual_vision(monkeypatch, tmp_path, caplog, kind):
    provider = PyEyeProvider(gallery_dir=tmp_path) if kind == 'live' else ImageReplayProvider(tmp_path)
    before = provider.status()
    assert before['ocr'] is None and before['ocr_requested'] == 'rapid' and before['ocr_fallback'] is None

    def missing_dependency():
        raise ImportError('No module named rapidocr_onnxruntime')

    monkeypatch.setattr(rapid, 'RapidOCR', missing_dependency)
    with caplog.at_level(logging.WARNING):
        engine = ocr.create_engine('rapid')
    provider._reader = reader_with(engine, tmp_path)
    status = provider.status()
    assert status['ocr'] == 'vision' and status['ocr_requested'] == 'rapid' and status['ocr_fallback'] is True
    assert any(r.levelno == logging.WARNING and '[OCR FALLBACK]' in r.message and 'actual=vision' in r.message
               for r in caplog.records)


@pytest.mark.parametrize('requested', ['rapid', 'vision'])
def test_explicit_primary_is_reported_without_degradation(monkeypatch, tmp_path, caplog, requested):
    monkeypatch.setattr(rapid, 'RapidOCR', lambda: SimpleNamespace(name='rapid'))
    engine = ocr.create_engine(requested)
    provider = PyEyeProvider(gallery_dir=tmp_path, ocr=requested)
    provider._reader = reader_with(engine, tmp_path)
    assert provider.status()['ocr'] == requested and provider.status()['ocr_fallback'] is False
    assert not caplog.records
