"""Explicit provenance for isolated formatting/late-reply tests (no IO)."""
from jccbrain.core.buy_source import BuySource
from jccbrain.modes.ai_display import AiDisplay


def authorize_ai(mode, advice, state=None):
    if not hasattr(mode, '_ai'):
        mode._ai = AiDisplay(1, lambda p: None)
        mode._ai_frames = {}
        mode._input_epoch = 0
    epoch = mode._input_epoch
    source = BuySource(mode._ai.projection.source.session, 1, 0, advice.selection.get('revision', ''),
                       '', advice.stage, advice.phase.value, 'formatting-fixture', False, input_epoch=epoch)
    mode._ai.observe_raw('formatting-fixture', usable=True, input_epoch=epoch)
    binding = mode._ai.confirm('formatting-fixture', {}, source, usable=True)
    mode._llm_source = binding
    advice.buy_source = source
    if state is not None:
        state.buy_source = source
    mode._ai_frames[(epoch, 1)] = binding
    return binding
