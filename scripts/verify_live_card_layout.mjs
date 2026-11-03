/** Isolated UI regression check. Run against this worktree's web server; no game backend is used.
 * LIVE_LAYOUT_URL=http://127.0.0.1:5185 node scripts/verify_live_card_layout.mjs
 * Optional PLAYWRIGHT_MODULE, CHROME_EXECUTABLE, LIVE_LAYOUT_EVIDENCE, LIVE_LAYOUT_BROWSER=webkit.
 */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const playwright = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.LIVE_LAYOUT_URL ?? 'http://127.0.0.1:5185';
const output = path.resolve(process.env.LIVE_LAYOUT_EVIDENCE ?? '.task/evidence/browser');
const browserName = process.env.LIVE_LAYOUT_BROWSER ?? 'chromium';
await mkdir(output, { recursive: true });
const browser = await playwright[browserName].launch({ headless: true, ...(browserName === 'chromium' ? { executablePath: process.env.CHROME_EXECUTABLE ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' } : {}) });
const report = { browser: browserName, version: browser.version(), url: base, browserPlugin: 'Browser plugin not available', backend: 'independent synthetic pywebview bridge; no real backend', checks: [], console: [], network: [] };
const defaults = ['targets', 'comps', 'opponents', 'suggestions', 'units'];
const storageKey = 'jcc-live-card-layout';
let currentPage;
async function check(label, fn) { await fn(); report.checks.push({ label, result: 'pass' }); console.log(`PASS ${label}`); }
async function contextWithStub(options = {}) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await context.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (url.origin !== new URL(base).origin) { report.network.push(url.href); await route.abort(); }
        else await route.continue();
    });
    await context.addInitScript(({ options, storageKey }) => {
        const unit = { name: '阿狸', star: 2, zone: 'board', slot: 0, x: 0, y: 0, known: true, cost: 2, traits: [], items: [] };
        const status = {
            season: { id: 'synthetic', label: 'UI 合成数据', rules_note: '' }, llm: { enabled: false, preset_title: '', model: '' },
            mode: { key: 'assistant', title: '辅助大脑' }, modes: [], hud: { overlay: false, panel: false }, dry_run: true,
            stopping: false, syncing: false, db_empty: false, comp_source_notice: '', vision_backend: 'synthetic',
            hero_costs: { 阿狸: 2 }, scout_unavailable_reason: '', scout: null, watch: [], lock: null, default_lock: null, augments_taken: ['合成强化'],
            sources: { official: true, ingame: false, dataj: false }
        };
        const state = { stage: '3-2', timer: 28, phase: 'planning', gold: 50, interest: 5, level: 6, xp: [12, 24], hp: 82,
            streak: { kind: 'win', n: 2 }, board: [unit], bench: [unit], shop: [], item_bench: [{ name: '反曲弓' }], traits: [], summary: '浏览器合成数据', augments_taken: [] };
        const advice = { headline: '合成建议：保持经济，观察下一轮来牌', stage: '3-2', phase: 'planning',
            traits_view: { final: [], current: [] }, board: { source: '合成站位', population: 6, next_hint: '升级后补充前排', cells: [
                { row: 4, col: 3, name: '阿狸', cost: 2, is_carry: true, role: 'carry', owned_star: 2, on_board: true, items: [], item_details: [], traits: [], star_target: 3 }
            ] }, targets: Array.from({ length: 12 }, (_, i) => ({ name: i ? `目标${i}` : '阿狸', have: 3, need: 6, target_star: 3, cost: 2, remaining: 20, per_roll: .15, gold_for_50: 10 })),
            comps: Array.from({ length: 8 }, (_, i) => ({ id: `comp-${i}`, name: `合成阵容${i}`, tier: 'S', avg_placement: 3.5, fit: .8, seen: 4, seen_carry: 2, owned_copies: 3, why: ['来牌匹配，便于测试缩小后的内容滚动'], missing: ['阿狸'] })),
            opponents: Array.from({ length: 7 }, (_, i) => ({ key: `op-${i}`, name: `对手${i+1}`, hp: 80-i*5, level: 6, round: '3-2', stale_rounds: 0, contested: i===0, carries: [{ name: '阿狸', star: 2 }], traits: [], comp_guess: { name: '合成阵容', by: 'units' } })),
            econ: { reason: '存到 50 金币，保持利息', action: 'save', keep_gold: 50 }, buys: [{ slot: 0, name: '阿狸', reason: '补充目标卡' }], sells: [], augments: [], items: [],
            held_items: { components: {}, completed: [], emblems: [] }, item_plan: [], notes: ['所有数据均为独立浏览器合成数据'], llm_text: '下一回合继续观察来牌。\n'.repeat(15), comp_status: '探索', augments_taken: [] };
        window.__fixture = { status, state, advice };
        window.__bridgeCalls = [];
        function emit(type, payload) { window.__jcc_emit({ type, payload }); }
        const api = {
            status: () => { setTimeout(() => { emit('state', state); emit('advice', advice); }, 100); return status; },
            comps: () => ({ rows: advice.comps.map(comp => ({ ...comp, source: 'official', channels: ['web'], core: [] })), info: {} }),
            icons: keys => Object.fromEntries(keys.map(key => [key, null])),
            set_watch_comps: ids => { status.watch = ids.map(id => ({ id, name: id })); emit('watch_changed', { watch: status.watch }); return null; },
            lock_comp_for_game: id => { status.lock = id ? { id, name: id, source: 'game' } : null; emit('lock_changed', { lock: status.lock, default_lock: null }); return null; },
            scout_now: () => null,
            window_drag_regions: () => null,
            seasons: () => [], settings_schema: () => []
        };
        window.pywebview = { api: Object.fromEntries(Object.entries(api).map(([name, fn]) => [name, async (...args) => {
            window.__bridgeCalls.push({ name, args }); return { ok: true, data: fn(...args) };
        }])) };
        if (options.rawStorage !== undefined) localStorage.setItem(storageKey, options.rawStorage);
        if (options.storageUnavailable) {
            Storage.prototype.getItem = function () { throw new DOMException('Storage disabled', 'SecurityError'); };
            Storage.prototype.setItem = function () { throw new DOMException('Storage disabled', 'QuotaExceededError'); };
        }
    }, { options, storageKey });
    const page = await context.newPage();
    currentPage = page;
    page.on('console', msg => { if (['error', 'warning'].includes(msg.type())) report.console.push(`${msg.type()}: ${msg.text()}`); });
    page.on('pageerror', error => report.console.push(`pageerror: ${error.stack}`));
    await page.goto(`${base}/#/live`);
    await page.locator('[data-card-id="units"]').waitFor();
    return { context, page };
}
const order = page => page.locator('[data-card-id]').evaluateAll(nodes => nodes.map(node => node.dataset.cardId));
const layout = page => page.locator('[data-card-id]').evaluateAll(nodes => nodes.map(node => ({ id: node.dataset.cardId, span: Number(node.dataset.span), height: node.offsetHeight })));
async function screenshot(page, filename) { await page.screenshot({ path: path.join(output, filename) }); }
async function scrollCard(page, id, align = 'start') { await page.locator(`[data-card-id="${id}"]`).evaluate((node, align) => node.scrollIntoView({ block: align }), align); await page.waitForTimeout(100); }
async function drag(page, from, to, after = false) {
    await scrollCard(page, from);
    const handle = page.locator(`[data-card-id="${from}"] .drag-handle`);
    const a = await handle.boundingBox();
    await page.mouse.move(a.x+a.width/2, a.y+a.height/2); await page.mouse.down();
    // Real mouse stays captured; scroll movement keeps the destination reachable.
    const destination = page.locator(`[data-card-id="${to}"]`);
    await destination.evaluate(node => node.scrollIntoView({ block: 'start' }));
    const b = await destination.boundingBox();
    const targetY = after ? Math.min(920, b.y+b.height-35) : b.y+28;
    await page.mouse.move(b.x+b.width/2, targetY, { steps: 20 });
    await page.waitForTimeout(80);
    assert.equal(await page.locator('.layout-ghost').count(), 1);
    await page.mouse.up();
}
try {
    const { context, page } = await contextWithStub();
    await check('page identity, meaningful five cards, no overlay', async () => {
        assert.match(page.url(), /#\/live$/); assert.equal(await page.title(), '金铲铲大脑');
        assert.deepEqual(await order(page), defaults);
        assert.match(await page.locator('.live-page').innerText(), /合成建议/);
        assert.equal(await page.locator('vite-error-overlay').count(), 0);
        await screenshot(page, '01-default-desktop.png');
    });
    await check('mouse suggestions before opponents and back', async () => {
        await drag(page, 'suggestions', 'opponents');
        assert.deepEqual(await order(page), ['targets','comps','suggestions','opponents','units']);
        await screenshot(page, '02-reordered.png');
        await drag(page, 'suggestions', 'opponents', true);
        assert.deepEqual(await order(page), defaults);
    });
    await check('edge auto-scroll and Escape cancel cleanly', async () => {
        await scrollCard(page, 'suggestions');
        const a = await page.locator('[data-card-id="suggestions"] .drag-handle').boundingBox();
        const before = await page.locator('.live-page').evaluate(node => node.scrollTop);
        await page.mouse.move(a.x+10,a.y+10); await page.mouse.down();
        await page.mouse.move(a.x+10,65,{steps:12}); await page.waitForTimeout(500);
        assert.ok(await page.locator('.live-page').evaluate(node => node.scrollTop) < before - 100);
        await page.keyboard.press('Escape'); await page.mouse.up();
        assert.deepEqual(await order(page), defaults); assert.equal(await page.locator('.layout-ghost').count(),0);
        assert.equal(await page.evaluate(() => document.body.style.cursor),'');
    });
    await check('mouse resize snaps half/full width and changes height', async () => {
        await scrollCard(page,'suggestions','center');
        const frame = page.locator('[data-card-id="suggestions"]');
        const a = await frame.locator('.resize-handle').boundingBox();
        const initial = await frame.boundingBox();
        await page.mouse.move(a.x+12,a.y+12); await page.mouse.down();
        await page.mouse.move(a.x+initial.width*.6,a.y+110,{steps:20}); await page.mouse.up();
        assert.equal(await frame.getAttribute('data-span'),'2');
        assert.ok((await frame.boundingBox()).height>initial.height+60);
        await screenshot(page,'03-resized.png');
        await scrollCard(page,'suggestions','center');
        const b = await frame.locator('.resize-handle').boundingBox();
        const wide = await frame.boundingBox();
        await page.mouse.move(b.x+12,b.y+12); await page.mouse.down();
        await page.mouse.move(b.x-wide.width*.55,b.y-80,{steps:20}); await page.mouse.up();
        assert.equal(await frame.getAttribute('data-span'),'1');
    });
    await check('keyboard ordering and resize alternatives', async () => {
        const handle = page.locator('[data-card-id="suggestions"] .drag-handle');
        await handle.focus(); await handle.press('ArrowUp');
        assert.deepEqual(await order(page),['targets','comps','suggestions','opponents','units']);
        const resize = page.locator('[data-card-id="suggestions"] .resize-handle');
        await resize.focus(); await resize.press('ArrowRight'); await resize.press('ArrowDown');
        assert.equal(await page.locator('[data-card-id="suggestions"]').getAttribute('data-span'),'2');
    });
    const saved = await layout(page);
    await check('state/advice updates preserve layout', async () => {
        await page.evaluate(() => {
            window.__fixture.advice.headline = '更新后的建议';
            window.__jcc_emit({type:'state',payload:{...window.__fixture.state,gold:61}});
            window.__jcc_emit({type:'advice',payload:{...window.__fixture.advice}});
        });
        assert.deepEqual(await layout(page),saved);
        assert.match(await page.locator('.headline-banner').innerText(),/更新后的建议/);
    });
    await check('route return and reload preserve order and sizes', async () => {
        await page.locator('[data-nav="/logs"]').click();
        await page.locator('[data-nav="/live"]').click();
        await page.locator('[data-card-id="units"]').waitFor();
        assert.deepEqual(await layout(page),saved);
        await page.reload(); await page.locator('[data-card-id="units"]').waitFor();
        assert.deepEqual(await layout(page),saved);
    });
    await check('original selection/watch/lock/scout buttons do not reorder', async () => {
        await scrollCard(page,'targets');
        await page.locator('.target-table tbody tr').first().click();
        assert.equal(await page.locator('.target-table tr[aria-selected="true"]').count(),1);
        await page.locator('.board-mini button[data-hero="阿狸"]').click();
        assert.equal(await page.locator('.board-mini button[data-hero="阿狸"]').getAttribute('aria-pressed'),'true');
        await scrollCard(page,'comps');
        await page.locator('[aria-label="关注合成阵容0"]').check();
        await page.locator('[aria-label="本局锁定：合成阵容0"]').click();
        await scrollCard(page,'opponents'); await page.getByRole('button',{name:'立即侦察',exact:true}).click();
        assert.deepEqual(await layout(page),saved);
        const calls = await page.evaluate(()=>window.__bridgeCalls.map(call=>call.name));
        for(const name of ['set_watch_comps','lock_comp_for_game','scout_now']) assert.ok(calls.includes(name));
    });
    await check('800px single column keeps desktop size intentions, no horizontal overflow', async () => {
        await page.setViewportSize({width:800,height:900});
        await scrollCard(page,'suggestions');
        assert.deepEqual(await layout(page),saved);
        const dimensions = await page.locator('.live-page').evaluate(node=>({w:node.clientWidth,sw:node.scrollWidth}));
        assert.ok(dimensions.sw<=dimensions.w+1,JSON.stringify(dimensions));
        const documentWidth = await page.evaluate(()=>document.documentElement.scrollWidth);
        assert.ok(documentWidth<=800,`document width ${documentWidth}`);
        const boxes = await page.locator('[data-card-id]').evaluateAll(nodes=>nodes.map(node=>node.getBoundingClientRect().width));
        assert.ok(Math.max(...boxes)-Math.min(...boxes)<2);
        await screenshot(page,'04-narrow-800.png');
        await page.setViewportSize({width:1440,height:1000});
        assert.deepEqual(await layout(page),saved);
    });
    await check('restore default resets sizes and order persistently', async () => {
        await page.getByRole('button',{name:'恢复默认布局'}).click();
        assert.deepEqual(await order(page),defaults);
        assert.deepEqual((await layout(page)).map(card=>[card.span,card.height]),[[2,460],[2,340],[2,420],[1,360],[1,360]]);
        await page.reload(); await page.locator('[data-card-id="units"]').waitFor();
        assert.deepEqual(await order(page),defaults);
    });
    await check('layout menu sizes every card and bounds height; half-width board adapts', async () => {
        for(const id of defaults) {
            await scrollCard(page,id);
            const frame=page.locator(`[data-card-id="${id}"]`);
            await frame.locator('.layout-settings').click();
            const panel=page.locator('.u-popover').last();
            await panel.locator('select').selectOption('1');
            await page.waitForFunction(id=>document.querySelector(`[data-card-id="${id}"]`).dataset.span==='1',id);
            await panel.locator('input[type="number"]').fill('220');
            await panel.locator('input[type="number"]').press('Tab');
            assert.equal(await frame.getAttribute('data-span'),'1');
            await page.waitForFunction(id=>document.querySelector(`[data-card-id="${id}"]`).offsetHeight===220,id);
            assert.equal((await frame.boundingBox()).height,220);
            await page.keyboard.press('Escape');
        }
        await scrollCard(page,'targets');
        const dimensions=await page.locator('.live-page').evaluate(node=>({w:node.clientWidth,sw:node.scrollWidth}));
        assert.ok(dimensions.sw<=dimensions.w+1);
        const positions=await page.locator('[data-card-id]').evaluateAll(nodes=>nodes.map(node=>{const b=node.getBoundingClientRect();return {x:b.x,y:b.y,right:b.right,bottom:b.bottom};}));
        for(let i=0;i<positions.length;i++) for(let j=i+1;j<positions.length;j++) {
            const a=positions[i], b=positions[j]; assert.ok(a.right<=b.x||b.right<=a.x||a.bottom<=b.y||b.bottom<=a.y,'cards overlap');
        }
        await screenshot(page,'05-half-cards.png');
        await page.getByRole('button',{name:'恢复默认布局'}).click();
    });
    await check('pointercancel rolls resize back; leaving route cleans active drag', async () => {
        const baseline=await layout(page);
        await scrollCard(page,'suggestions','center');
        const handle=page.locator('[data-card-id="suggestions"] .resize-handle');
        const a=await handle.boundingBox();
        await page.mouse.move(a.x+10,a.y+10);await page.mouse.down();
        await page.mouse.move(a.x+120,a.y-80,{steps:10});
        await handle.evaluate(node=>node.dispatchEvent(new PointerEvent('pointercancel',{bubbles:true,pointerId:1,isPrimary:true})));
        await page.mouse.up();
        assert.deepEqual(await layout(page),baseline);
        const dragHandle=page.locator('[data-card-id="suggestions"] .drag-handle');
        const b=await dragHandle.boundingBox();
        await page.mouse.move(b.x+10,b.y+10);await page.mouse.down();await page.mouse.move(b.x+100,b.y-50,{steps:5});
        await page.evaluate(()=>{location.hash='#/logs';});
        await page.locator('[data-card-id]').first().waitFor({state:'detached'});
        await page.mouse.up();
        assert.equal(await page.locator('.layout-ghost').count(),0);
        assert.equal(await page.evaluate(()=>document.body.style.userSelect),'');
        await page.locator('[data-nav="/live"]').click();await page.locator('[data-card-id="units"]').waitFor();
        assert.deepEqual(await layout(page),baseline);
    });
    await check('390px cards keep their own content internally scrollable', async () => {
        await page.setViewportSize({width:390,height:844});
        await scrollCard(page,'targets');
        const dimensions=await page.locator('.live-page').evaluate(node=>({w:node.clientWidth,sw:node.scrollWidth}));
        assert.ok(dimensions.sw<=dimensions.w+1,JSON.stringify(dimensions));
        await page.locator('[data-card-id="targets"] .layout-settings').click();
        const menu=page.getByRole('dialog',{name:'阵容与概率布局',exact:true});
        await menu.waitFor({state:'visible'});
        const menuBounds=await menu.boundingBox();
        assert.ok(menuBounds.x>=0 && menuBounds.x+menuBounds.width<=390);
        await page.keyboard.press('Escape');
        report.narrow390 = { ...dimensions, documentWidth: await page.evaluate(()=>document.documentElement.scrollWidth), note: 'Existing fixed sidebar and toolbar are outside this task; this check covers card content only.' };
        const table = page.locator('.target-table');
        assert.ok(await table.evaluate(node=>node.scrollWidth>node.clientWidth));
        await table.evaluate(node=>{node.scrollLeft=node.scrollWidth;});
        assert.ok(await table.evaluate(node=>node.scrollLeft>0));
        await screenshot(page,'06-narrow-390.png');
        await page.setViewportSize({width:1440,height:1000});
    });
    await context.close();
    for (const [label,options] of [
        ['malformed JSON',{rawStorage:'{broken'}],
        ['unknown schema version',{rawStorage:JSON.stringify({version:999,cards:[]})}],
        ['whitelist, duplicates and numeric validation',{rawStorage:JSON.stringify({version:1,cards:[{id:'unknown',span:2,height:500},{id:'units',span:8,height:1e9},{id:'units',span:1,height:220},{id:'targets',span:1,height:'bad'},{id:'comps',span:2,height:-999}]})}],
        ['storage unavailable',{storageUnavailable:true}]
    ]) {
        const { context, page } = await contextWithStub(options);
        await check(label,async()=>{
            assert.equal(await page.locator('[data-card-id]').count(),5);
            const values = await layout(page); assert.equal(new Set(values.map(card=>card.id)).size,5);
            assert.ok(values.every(card=>(card.span===1||card.span===2)&&card.height>=220&&card.height<=900));
            if(label==='whitelist, duplicates and numeric validation') { assert.equal(values[0].id,'units'); assert.equal(values[0].height,900); assert.equal(values.find(card=>card.id==='targets').height,460); }
            else assert.deepEqual(await order(page),defaults);
            const beforeIndex = (await order(page)).indexOf('targets');
            await page.locator('[data-card-id="targets"] .drag-handle').press('ArrowDown');
            assert.equal((await order(page)).indexOf('targets'),beforeIndex+1);
            if(options.storageUnavailable) assert.match(await page.locator('.storage-notice').innerText(),/存储不可用/);
        });
        await context.close();
    }
    assert.deepEqual(report.console,[]); assert.deepEqual(report.network,[]);
    report.result='pass';
}
catch(error) { report.result='fail'; report.failure=error.stack; console.error(error); process.exitCode=1; if(currentPage && !currentPage.isClosed()) await screenshot(currentPage,'failure.png').catch(()=>{}); }
finally { await writeFile(path.join(output,'results.json'),JSON.stringify(report,null,2)); await browser.close(); }
