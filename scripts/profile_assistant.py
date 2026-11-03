"""Offline-only assistant diagnostics. Requires explicit inputs, safe JCC_HOME and new output.

Never starts a provider, mode loop, learning transaction, native window or executor.
Measurements are tagged by clock/provenance; nested spans are not additive.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time


def summary(values):
    """Nearest-rank p95; no interpolation, no adding unrelated percentiles."""
    values = sorted(float(v) for v in values)
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    n = len(values)
    return {'n': n, 'min': values[0], 'median': (values[(n-1)//2]+values[n//2])/2,
            'p95': values[math.ceil(.95*n)-1], 'max': values[-1]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def validate(home, output, inputs):
    """Reject unsafe settings before importing AppContext or any native backend."""
    home, output, inputs = [Path(p).expanduser().resolve() for p in (home, output, inputs)]
    if not inputs.is_file() or output.exists():
        raise ValueError('显式输入必须存在，输出必须是新目录')
    if home == inputs.parent or home.is_relative_to(inputs.parent) or inputs.is_relative_to(home):
        raise ValueError('输入与独立JCC_HOME不能重叠')
    if output == home or output.is_relative_to(home) or home.is_relative_to(output):
        raise ValueError('输出与独立JCC_HOME不能重叠')
    marker = json.loads((home/'.assistant-profile-offline.json').read_text())
    if marker.get('purpose') != 'offline-assistant-profile' or marker.get('home') != str(home):
        raise ValueError('必须使用显式准备的独立离线home标记')
    cfg = json.loads((home/'settings.json').read_text())
    if (cfg.get('llm', {}).get('enabled', True) or cfg.get('scout', {}).get('enabled', True)
            or cfg.get('data', {}).get('auto_sync', True)
            or not cfg.get('actuator', {}).get('dry_run', False)
            or cfg.get('vision', {}).get('unit_learning_mode', 'off') != 'off'):
        raise ValueError('诊断要求LLM/侦察/同步/学习关闭，dry_run开启')
    if not (home/'jcc.sqlite3').is_file():
        raise ValueError('独立数据库副本缺失')
    os.environ['JCC_HOME'] = str(home)
    output.mkdir(parents=True)
    return home, output, inputs


@contextmanager
def no_network():
    """In-process diagnostic guard; not a security sandbox."""
    original = socket.socket.connect
    def blocked(*args, **kw):
        raise RuntimeError('offline diagnostic: network disabled')
    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.socket.connect = original


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns()-start)/1e6


def images(home, out, manifest, repeats):
    import cv2
    from jccbrain.context import AppContext
    from jccbrain.core.models import GameState
    from jccbrain.core import paths
    from jccbrain.vision.pyeye import build_reader
    rows = json.loads(manifest.read_text())
    # Never let gallery seeding read a main checkout/legacy runtime.
    paths.legacy_gallery_dir = lambda: home/'gallery'
    records = []
    with no_network():
        ctx = AppContext()
        try:
            for entry in rows:
                path = Path(entry['path']).resolve()
                if path.suffix.lower() not in ('.jpg', '.png', '.jpeg') or sha(path) != entry['sha256']:
                    raise ValueError('固定图片路径/hash不匹配')
                image, decode_ms = timed(lambda: cv2.imread(str(path)))
                if image is None:
                    raise ValueError('图片解码失败')
                reader = None
                try:
                    reader, init_ms = timed(lambda: build_reader(ctx.db, home/'gallery', 'rapid'))
                    for index in range(repeats+1):
                        started = time.perf_counter_ns()
                        raw, read_ms = timed(lambda: reader.read(image))
                        state, mapping_ms = timed(lambda: GameState.from_dict(raw))
                        record = {'path': str(path), 'sha256': entry['sha256'], 'facts': entry['facts'],
                                  'iteration': index, 'temperature': 'cold_object_first_read' if index == 0 else 'warm_same_image',
                                  'clock': 'python_perf_counter_ns', 'decode_ms': decode_ms,
                                  'model_reader_init_ms': init_ms if index == 0 else None,
                                  'read_call_ms': read_ms, 'mapping_ms': mapping_ms,
                                  'read_plus_mapping_ms': (time.perf_counter_ns()-started)/1e6,
                                  'reader_overlapping_spans_ms': dict(reader.last_ms),
                                  'ocr_internal_accumulated_work_ms': dict(getattr(reader.ocr, 'last_ms', {})),
                                  'ocr_requested': 'rapid', 'ocr_actual': getattr(reader.ocr, 'name', None),
                                  'ocr_cells': reader.last_ocr_cells, 'cell_cache_hits': reader.last_cell_cache_hits,
                                  'raw': raw, 'mapped': {'stage': state.stage, 'phase': str(state.phase), 'gold': state.gold, 'level': state.level}}
                        records.append(record)
                except Exception as exc:
                    records.append({'path': str(path), 'sha256': entry['sha256'], 'error': repr(exc), 'facts': entry['facts']})
                finally:
                    if reader is not None and hasattr(reader.ocr, 'pool'):
                        reader.ocr.pool.shutdown(wait=True)
                with (out/'images.jsonl').open('w') as f:
                    for row in records:
                        f.write(json.dumps(row, ensure_ascii=False)+'\n')
        finally:
            ctx.close()
    result = {'provenance': 'fixed saved real JPEG offline re-read; NOT live capture, NOT accuracy truth',
              'cold_note': 'new reader/model object per image; OS/disk/runtime caches not cleared; no provider.start/download/warmup/learning',
              'nested_spans_overlap': True, 'records': len(records), 'errors': [r for r in records if 'error' in r], 'stats_ms': {}}
    for temperature in ['cold_object_first_read', 'warm_same_image']:
        valid = [r for r in records if r.get('temperature') == temperature]
        for field in ['model_reader_init_ms', 'decode_ms', 'read_call_ms', 'mapping_ms', 'read_plus_mapping_ms']:
            result['stats_ms'][temperature+'.'+field] = summary(r[field] for r in valid if r.get(field) is not None)
        for field in ['ocr', 'units', 'items', 'total']:
            result['stats_ms'][temperature+'.reader_'+field] = summary(r['reader_overlapping_spans_ms'][field] for r in valid if field in r['reader_overlapping_spans_ms'])
    write(out/'summary.json', result)


def observations(home, out, source, repeats):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    valid = [r for r in rows if r.get('file') and (source.parent/r['file']).is_file()]
    stats = {}
    for game in [True, False]:
        sample = [r for r in valid if r['game'] is game]
        key = 'game' if game else 'hud'
        stats[key] = {'n': len(sample), 'historical_sampler_capture_ms': summary((r['mono_capture_end']-r['mono_start'])*1000 for r in sample),
                      'historical_sampler_total_ms': summary(r['elapsed_ms'] for r in sample),
                      'historical_start_interval_ms': summary((b['mono_start']-a['mono_start'])*1000 for a,b in zip(sample,sample[1:]))}
    write(out/'summary.json', {'provenance': 'historical separate sampler CLI captures; not user GUI backend timing', 'total': len(rows), 'successful': len(valid), 'failed': len(rows)-len(valid), 'stats': stats})
    write(out/'source-files.json', [{'path': str(source.parent/r['file']), 'sha256': sha(source.parent/r['file']), 'metadata': r} for r in valid])


def sequence(home, out, manifest, repeats):
    from jccbrain.context import AppContext
    from jccbrain.core.models import GameState
    from jccbrain.modes.assistant import AssistantMode
    from jccbrain.webui.api import Api
    from jccbrain.webui.events import EventBus
    from jccbrain.webui import serialize
    from jccbrain.engine import items
    steps = json.loads(manifest.read_text())
    traces, spans, delivered = [], [], []
    current = {'frame': None, 'active': False}
    class Bus(EventBus):
        def emit(self, event_type, payload):
            if event_type == 'advice':
                current['published_frame'] = current['frame']
            if current.get('record') is not None:
                current['record']['emitted'].append({'kind': event_type, 't_ns': time.perf_counter_ns(), 'advice_frame': current.get('published_frame'), 'payload': deepcopy(payload)})
            tagged = {**payload, '_profile_frame': current['frame']} if event_type in ('state','advice','hud','buy_projection') else payload
            super().emit(event_type, tagged)
    class Hud:
        status = {'overlay': True, 'panel': True}
        def set_refresh(self, *args): pass
        def set_scout_notice(self, *args): pass
        def close(self): pass
    def wrap(obj, key, label):
        original = getattr(obj, key)
        def measured(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                if current['active']:
                    spans.append({'frame': current['frame'], 'span': label, 'start_ns': start, 'end_ns': time.perf_counter_ns(), 'ms': (time.perf_counter_ns()-start)/1e6})
        setattr(obj, key, measured)
        return lambda: setattr(obj, key, original)
    with no_network():
        for repeat in range(repeats):
            ctx = AppContext(); ctx.game_lock_id = '104'
            class OfflineOnly:
                def __getattr__(self, name):
                    raise RuntimeError('offline diagnostic: native vision/executor forbidden: '+name)
            ctx.vision = OfflineOnly(); ctx.executor = OfflineOnly()
            bus = Bus(lambda js: delivered.append({'t_ns': time.perf_counter_ns(), 'js': js}))
            api = Api(ctx, bus, hud=Hud()); api._icons.close(); api._icons.prefetch = lambda *a: None
            mode = AssistantMode(ctx, api._sink); api._runner.mode = mode
            restores = [wrap(mode.lifecycle, 'observe', 'lifecycle'), wrap(mode.stabilizer, 'update', 'stabilizer'),
                        wrap(mode, 'compute_advice', 'compute_advice_including_items'), wrap(items, 'plan', 'items_plan_nested'),
                        *[wrap(serialize, key, 'serialize_'+key) for key in ['state', 'advice', 'hud']]]
            published = None
            current['published_frame'] = None
            try:
                for i, step in enumerate(steps):
                    fid = f'{repeat}:{i+1}'
                    record = {'frame': fid, 'label': step['label'], 'provenance': 'explicit controlled RAW input; production _process_frame, NOT captured OCR',
                              'clock': 'python_perf_counter_ns', 'logical_poll400_ms': i*400, 'raw': deepcopy(step['raw']), 'emitted': []}
                    current.update(frame=fid, record=record, active=True)
                    record['start_ns'] = time.perf_counter_ns()
                    if step.get('config'):
                        config = step['config']
                        ctx.settings.assistant.comp_presets = {ctx.db.season_id: {'selected': config['id'], 'configs': [config]}}
                        mode.refresh_selection()
                    accepted = mode._process_frame(GameState.from_dict(deepcopy(step['raw'])))
                    record['end_ns'] = time.perf_counter_ns(); current['active'] = False
                    adv_events = [e for e in record['emitted'] if e['kind'] == 'advice']
                    if adv_events: published = fid
                    record.update(accepted=accepted is not None, lifecycle=mode.lifecycle.status,
                                  stable_raw=deepcopy(mode.stabilizer.acc),
                                  stable_state=serialize.state(accepted[0], ctx.db) if accepted else None,
                                  computed_advice=serialize.advice(accepted[1]) if accepted else None,
                                  last_published_advice_frame=published,
                                  latest_published_advice=serialize.advice(api._last_advice) if api._last_advice else None,
                                  latest_buy_projection=serialize.buy_projection(api._buy_projection) if api._buy_projection else None,
                                  latest_hud=serialize.hud(api._last_state, api._last_advice, ctx.db, ctx.layout, ctx.settings.hud, projection=api._buy_projection),
                                  process_ms=(record['end_ns']-record['start_ns'])/1e6)
                    record['hud_pairs'] = [{'state_frame': fid, 'advice_frame': e.get('advice_frame'), 'payload': e['payload']} for e in record['emitted'] if e['kind']=='hud']
                    traces.append(record)
                current['record'] = None
                time.sleep(.25)  # drain this isolated EventBus; not part of process spans
            finally:
                current.update(record=None, active=False)
                for restore in reversed(restores): restore()
                bus.close(); api._runner.mode = None; api._shutdown(); ctx.__dict__.pop('vision',None); ctx.__dict__.pop('executor',None); ctx.close()
    for name, rows in [('frames',traces),('spans',spans),('deliveries',delivered)]:
        with (out/(name+'.jsonl')).open('w') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(out/'summary.json', {'provenance':'controlled raw input, real stabilizer/AssistantMode/WebSink/Api/EventBus; callbacks instrumented, icon prefetch excluded; not live input-to-pixel',
        'clock':'Python perf_counter_ns only; logical_poll400_ms is a MODEL, not wall measurement', 'frames':len(traces),
        'stats_ms':{'process_frame_instrumented':summary(r['process_ms']for r in traces), **{label: summary(r['ms']for r in spans if r['span']==label)for label in sorted({r['span']for r in spans})}},
        'no_additive_percentiles':True})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['images','sequence','observations'])
    for option in ['home','out','inputs']: parser.add_argument('--'+option, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 20: parser.error('repeats must be 1..20')
    home, out, inputs = validate(args.home,args.out,args.inputs)
    write(out/'run.json',{'kind':args.kind,'home':str(home),'inputs':str(inputs),'sha256':sha(inputs),'clock':'perf_counter_ns','script_sha256':sha(Path(__file__)), 'network_guard':'socket connect disabled in process; not a sandbox','no_native_actions':True})
    globals()[args.kind](home,out,inputs,args.repeats)

if __name__ == '__main__':
    main()
