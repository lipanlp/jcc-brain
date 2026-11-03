// Run the actual TypeScript consumers with Vue refs and their production bus.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'web/node_modules/typescript'));
const vue = require(path.join(root, 'web/node_modules/vue'));
const sourceRoot = process.env.FRONTEND_AI_SOURCE_ROOT || path.join(root, 'web/src');
const cache = new Map();
function load(file) {
    if (cache.has(file)) return cache.get(file).exports;
    const module = { exports: {} };
    cache.set(file, module);
    const js = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
    }).outputText;
    const requireSource = name => name === 'vue' ? vue : load(path.resolve(path.dirname(file), name + '.ts'));
    vm.runInThisContext('(function(require,module,exports){' + js + '\n})', { filename: file })(requireSource, module, module.exports);
    return module.exports;
}
const { bus } = load(path.join(sourceRoot, 'bridge/events.ts'));
const { visibleAiText, visibleAiAugment, purchaseHeadline, visibleCompReasons } = load(path.join(sourceRoot, 'composables/purchasePresentation.ts'));
let frame = 10, revision = 1, sequence = 0;
const source = (overrides = {}) => ({ session: 1000, frame: ++frame, generation: 1, selection: 'selection-A', season: 'season-A', stage: '3-2', phase: 'augment', fingerprint: 'options-A', eligible: false, decision: 'decision-A', input_epoch: 1, ...overrides });
const state = s => ({ buy_source: s, stage: s.stage, phase: s.phase });
const advice = (s, text = '任意说明') => ({ buy_source: s, ai_source:{...s, revision, sequence, ready:true, fingerprint:'semantic-context'}, stage: s.stage, phase: s.phase, llm_text: text, buys: [], augments: [{ name: '甲' }, { name: '乙' }], headline: '存钱；买 旧英雄；走 阵容甲' });
function snapshot(s, valid = false) { bus.emit('state', state(s)); bus.emit('buy_projection', { source: s, valid, buys: [] }); }
function publish(s, text, valid = true) {
 const source = { ...s, revision, sequence: ++sequence, fingerprint: 'semantic-context', ready: valid };
 const p = { source, valid, text, augment: valid ? '乙' : null, options: ['甲','乙'], buy_frame: s.frame };
 bus.emit('ai_projection',p);return p;
}
let checks=0;function check(name,a,b){assert.equal(a,b,name);checks++;}
let s=source();snapshot(s);let a=advice(s);const p=publish(s,'当前有效说明');
check('new nonpurchase explanation',visibleAiText(a,state(s)),p.text);check('AI tag shares valid source',visibleAiAugment(a,state(s)),'乙');
for(let i=0;i<20;i++) {s={...s,frame:++frame};snapshot(s);check('unchanged sample '+i+' preserves semantic reply',visibleAiText(a,state(s)),p.text);}
check('snapshot envelope stage mismatch',visibleAiText({...a,stage:'3-3'},{...state(s),stage:'3-3'}),'');
for(const field of ['session','generation','selection','season','stage','phase','fingerprint','input_epoch']) {
 const value=s[field];const wrong={...a,buy_source:{...a.buy_source,[field]:typeof value==='number'?value-1:value+'-old'}};
 check('wrong full source '+field,visibleAiText(wrong,state(s)),'');
}
const comp={why:['本局锁定','商店有 旧英雄']};check('no purchase header in augment',purchaseHeadline(a),'');check('nonshop reasons retained',visibleCompReasons(a,comp).join('；'),'本局锁定');
revision++;publish(s,'',false);check('semantic revocation clears body',visibleAiText(a,state(s)),'');check('semantic revocation clears AI label',visibleAiAugment(a,state(s)),undefined);
bus.emit('ai_projection',p);check('late old projection cannot revive',visibleAiText(a,state(s)),'');
revision++;const recovered=publish(s,'新语义回复');check('new semantic reply recovers',visibleAiText(a,state(s)),recovered.text);
const oldStatus={ai_projection:p};bus.emit('status',oldStatus);check('old reconnect status cannot downgrade semantic source',visibleAiText(a,state(s)),recovered.text);
revision++;s=source({input_epoch:2});snapshot(s);a=advice(s);const epoch=publish(s,'新epoch回复');check('new epoch reply visible',visibleAiText(a,state(s)),epoch.text);
for(const phase of ['planning','combat']) {
 revision++;s=source({input_epoch:2,phase,eligible:true});snapshot(s,true);a=advice(s);const current=publish(s,'当前经济解释');check(phase+' valid empty current AI',visibleAiText(a,state(s)),current.text);
 const empty={...s,frame:++frame};snapshot(empty,true);check(phase+' newer buyer projection cannot use old purchase AI',visibleAiText(a,state(empty)),'');check(phase+' empty no purchase headline',purchaseHeadline(a),'');
 revision++;const renewed=publish(empty,'对应空购买新解释');a=advice(empty);check(phase+' matching new response can recover',visibleAiText(a,state(empty)),renewed.text);
 const invalid={...empty,frame:++frame,eligible:false};snapshot(invalid);check(phase+' invalid buy authority remains protected',visibleAiText(a,state(invalid)),'');
}
revision++;s=source({input_epoch:2,phase:'carousel'});snapshot(s);a=advice(s);const carousel=publish(s,'选秀说明');check('carousel has independent semantic authority',visibleAiText(a,state(s)),carousel.text);
revision++;publish(s,'',false);check('stop clears carousel text',visibleAiText(a,state(s)),'');bus.emit('status',{ai_projection:carousel});check('old reconnect after stop stays clear',visibleAiText(a,state(s)),'');
// Full strategy data can precede unchanged sample frames without pairing a new state.
const { currentFullAdvice }=load(path.join(sourceRoot,'composables/buyProjection.ts'));
revision++;s=source({input_epoch:3,phase:'augment'});snapshot(s);a=advice(s);publish(s,'当前语义');
const newer={...s,frame:++frame};snapshot(newer);
check('identical semantic full snapshot survives newer nonbuy sample',currentFullAdvice(a,null),true);
check('semantic metadata cannot excuse wrong full phase',currentFullAdvice({...a,buy_source:{...s,phase:'combat'}},null),false);
check('semantic metadata cannot excuse wrong full stage',currentFullAdvice({...a,buy_source:{...s,stage:'3-3'}},null),false);
check('missing semantic sidecar cannot bypass nonbuy frame fence',currentFullAdvice({...a,ai_source:undefined},null),false);
check('wrong semantic fingerprint cannot bypass full fence',currentFullAdvice({...a,ai_source:{...a.ai_source,fingerprint:'other'}},null),false);
revision++;publish(newer,'',false);check('revoked semantic full snapshot cannot reenter',currentFullAdvice(a,null),false);
revision++;s=source({input_epoch:4});snapshot(s);a=advice(s);publish(s,'新输入回复');
check('new input full snapshot recovers',currentFullAdvice(a,null),true);

console.log(JSON.stringify({checks}));
