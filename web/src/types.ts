export interface BuySource {
    session: number; input_epoch?: number; frame: number; generation: number; selection: string; season: string;
    stage: string | null; phase: string; fingerprint: string; eligible: boolean; decision: string;
}
export interface BuyProjection {
    source: BuySource; valid: boolean;
    buys: { slot: number; name: string; cost: number; reason: string; priority: number }[];
}
export type ModeKey = 'assistant' | 'autoroll' | 'autopilot';
export interface ScoutNotice { id: string; stage: string; seconds: number; }
export type CompSource = 'official' | 'ingame' | 'dataj';
export type CompChannel = 'web' | 'ingame';
export type SourceFlags = Record<CompSource, boolean>;
export interface ScoutProgress {
    notice: ScoutNotice | null;
    scanning: boolean; processing: boolean; pending: boolean; captured: number; total: number;
    processed: number; recognized: number; interruption_s: number; background_s: number; reason: string;
}
export type EconomyPreference = 'adaptive' | 'health' | 'economy';
export interface CompIdentity {
    id: string; name: string; canonical_id?: string | null; available?: boolean;
}
export interface AssistantSelection {
    revision: string; preset_id: string; name: string; season_id: string;
    catalog_count: number; effective_count: number; candidate_count: number;
    preset_ids: string[]; candidate_ids: string[]; watch_ids: string[];
    allowed_comps: { id: string; name: string }[]; lock_id: string | null;
    lock_outside: boolean; watch_outside: { id: string; name: string }[]; error: string; notice: string;
    selection_map?: Record<string, string | null>; lock_selection_id?: string | null;
}
export interface CompPreset {
    id: string; name: string; members: string[]; builtin: boolean;
    selected_count: number; effective_count: number; unavailable: { id: string; name: string }[];
}
export interface CompPresetsResult {
    season_id: string; active_id: string; configs: CompPreset[];
    scope: AssistantSelection; error: string; catalog: (CompRow & { heroes: string[] })[]; saved_id?: string;
}
export interface AutopilotStatus {
    augment_selection: "manual" | "auto"; paused: boolean; token: string | null; stage: string | null; can_resume: boolean; reason: string;
}
export interface Status {
    ai_projection?: AiProjection | null;
    buy_projection?: BuyProjection | null;
    assistant_selection?: AssistantSelection;
    feature_settings: FeatureSettings;
    stopping: boolean;
    autoroll_task: (AutoRollOptions & { rolls: number; bought: number }) | null;
    autopilot: AutopilotStatus | null;
    augment_selection: "manual" | "auto";
    economy_preference: EconomyPreference;
    scout_unavailable_reason: string;
    scout: ScoutProgress | null;
    hud: {
        overlay: boolean;
        panel: boolean;
    };
    hero_costs: Record<string, number>;
    mode: {
        key: ModeKey;
        title: string;
    } | null;
    modes: {
        key: ModeKey;
        title: string;
        acts: boolean;
    }[];
    season: {
        id: string;
        label: string;
        rules_note: string;
    };
    llm: {
        enabled: boolean;
        preset: string;
        preset_title: string;
        model: string;
    };
    dry_run: boolean;
    vision_backend: string;
    db_empty: boolean;
    comp_source_notice: string;
    sources: SourceFlags;
    game_version: string | null;
    syncing: boolean;
    lock: {
        id: string;
        name: string;
        canonical_id?: string | null;
        available?: boolean;
        source: 'game' | 'default';
    } | null;
    default_lock: {
        id: string;
        name: string;
        canonical_id?: string | null;
        available?: boolean;
    } | null;
    watch: CompIdentity[];
    augments_taken: string[];
}
export type Phase = 'planning' | 'combat' | 'augment' | 'carousel' | 'ended' | 'unknown';
export interface EquippedItem {
    name?: string | null;
    [key: string]: unknown;
}
export interface Hud {
    buy_source?: BuySource;
    buy_projection?: BuyProjection;
    scout_notice: ScoutNotice | null;
    mode: 'board' | 'augments' | 'idle';
    opacity: number;
    font_size: number;
    positions: {
        key: string;
        x: number;
        y: number;
        w: number;
        h: number;
        name: string;
        state: 'ok' | 'misplaced' | 'bench' | 'missing';
        role: 'carry' | 'tank' | null;
    }[];
    arrows: {
        key: string;
        x1: number;
        y1: number;
        x2: number;
        y2: number;
        kind: 'move' | 'place';
    }[];
    units: {
        key: string;
        x: number;
        y_bottom: number;
        title: string;
        traits: string;
        cost: number;
        target: boolean;
        unknown: boolean;
    }[];
    shop: {
        key: number;
        x: number;
        y: number;
        w: number;
        h: number;
        kind: 'buy' | 'target';
        text: string;
    }[];
    augments: {
        name: string;
        key: number;
        x: number;
        y: number;
        text: string;
        best: boolean;
        ai_best: boolean;
        rule_text?: string;
    }[];
}
export interface Unit {
    items: EquippedItem[];
    name: string;
    star: number;
    zone: string;
    slot: number | null;
    x: number | null;
    y: number | null;
    known: boolean;
    cost: number | null;
    traits: string[];
}
export interface GameState {
    buy_source?: BuySource;
    item_bench: EquippedItem[];
    stage: string | null;
    timer: number | null;
    phase: Phase;
    gold: number | null;
    interest: number | null;
    level: number | null;
    xp: [
        number,
        number
    ] | null;
    hp: number | null;
    streak: {
        kind: string;
        n: number;
    } | null;
    shop_open: boolean;
    shop: ({
        name: string;
        cost: number | null;
        traits: string[];
        unsure: boolean;
        item: boolean;
        upgrade: number | null;
    } | null)[];
    traits: {
        name: string;
        count: number | null;
        active: boolean;
        next: number | null;
    }[];
    board: Unit[];
    bench: Unit[];
    units_stale: boolean;
    summary: string;
    augments_taken: string[];
}
export interface BoardCell {
    row: number;
    col: number;
    name: string;
    cost: number;
    is_carry: boolean;
    item: string | null;
    star_target: number | null;
    owned_star: number | null;
    on_board: boolean;
    items: string[];
    role: 'carry' | 'tank' | null;
    traits: string[];
    item_details: { name: string; components: string[]; held: string[]; ready: boolean }[];
}
export interface BoardPlan {
    source: string;
    population: number;
    cells: BoardCell[];
    next_hint: string | null;
}
export interface TraitView {
    name: string;
    count: number | null;
    tier_index: number;
    next_breakpoint: number | null;
    picture_key: string;
    breakpoints: number[];
    description: string;
    level_desc: string[];
    active?: boolean;
    in_comp?: boolean;
    emblem: { name: string; components: string[]; held: string[]; owned: boolean } | null;
}
export interface ScoutOpponent {
    key: string; name: string | null; slot: number | null; hp: number | null; round: string | null;
    level: number | null; stale_rounds: number; contested: boolean;
    comp_guess: { id: string; name: string; fit: number; by: 'traits' | 'units' } | null;
    carries: { name: string; star: number }[];
    traits: { name: string; count: number | null; active: boolean }[];
}
export interface AiProjection {
    source: { session: number; revision: number; sequence: number; generation: number; selection: string;
        season: string; input_epoch: number; stage: string | null; phase: string; fingerprint: string; ready: boolean };
    valid: boolean; text: string; augment: string | null; options: string[]; buy_frame: number;
}
export interface Advice {
    ai_source?: AiProjection['source'];
    buy_source?: BuySource;
    selection?: AssistantSelection | Record<string, never>;
    opponents: ScoutOpponent[];
    traits_view: { final: TraitView[]; current: TraitView[] };
    augment_situation: Record<string, unknown>;
    ai_augment: string | null;
    held_items: { components: Record<string, number>; completed: string[]; emblems: [string, string][] };
    augments_taken: string[];
    item_plan: {
        key: string; item: string; components: string[]; ready: boolean; give_to: string; final_to: string;
        action: '合成' | '佩戴' | '备料' | '暂留'; reason: string; missing: string[];
        current_unit: { key: string; index: number; name: string; star: number; label: string } | null;
        assets: { source: 'item_bench'; index: number; name: string }[];
        existing_item: { key: string; zone: 'board' | 'bench'; index: number; item_index: number; name: string; holder: string } | null;
    }[];
    board: BoardPlan | null;
    locked: boolean;
    comp_status: '探索' | '过渡' | '定型' | '锁定' | '关注' | null;
    switch_reason: string | null;
    stage: string | null;
    phase: Phase;
    headline: string;
    headline_summary?: string;
    target: {
        id: string;
        name: string;
        tier: string | null;
    } | null;
    comps: {
        id: string;
        name: string;
        tier: string | null;
        avg_placement: number | null;
        fit: number;
        seen: number;
        seen_carry: number;
        owned_copies: number;
        why: string[];
        why_parts?: { kind: 'shop' | 'strategy'; text: string }[];
        missing: string[];
    }[];
    econ: {
        action: 'save' | 'level' | 'roll' | 'all_in' | 'none';
        reason: string;
        keep_gold: number;
        xp_buys: number;
        roll_budget: number;
    } | null;
    buys: {
        slot: number;
        name: string;
        cost: number;
        reason: string;
    }[];
    sells: {
        slot: number;
        name: string;
        reason: string;
    }[];
    augments: {
        index: number;
        name: string;
        score: number;
        reason: string;
        category: string[];
        desc: string;
        base_score: number;
        situation_score: number;
        stats_weight: number;
        best: boolean;
        ai_best: boolean;
        rule_text?: string;
    }[];
    targets: {
        name: string;
        have: number;
        need: number;
        target_star: number;
        cost: number | null;
        remaining: number | null;
        per_roll: number | null;
        gold_for_50: number | null;
    }[];
    items: {
        hero: string;
        items: string[];
    }[];
    notes: string[];
    llm_text: string | null;
}
export interface PopulationPreview {
    selection_revision?: string | null;
    population: number | null;
    target_id: string;
    season_id: string;
    board: BoardPlan | null;
    targets: Advice['targets'];
}
export interface LogEntry {
    time: string;
    text: string;
    level: 'info' | 'warning' | 'error';
    source: 'mode' | 'logger' | 'app';
}
export interface CompRow {
    id: string;
    name: string;
    tier: string | null;
    source: string;
    channels: CompChannel[];
    source_label: string;
    style: 'reroll' | 'fast8' | 'fast9';
    style_label: string;
    avg_placement: number | null;
    top4_rate: number | null;
    top_rate: number | null;
    pick_rate: number | null;
    sample_count: number | null;
    core: {
        name: string;
        carry: boolean;
    }[];
}
export interface CompDetail extends CompRow {
    game_version: string | null;
    visualization: CompVisualization;
    traits: string[];
    heroes: {
        name: string;
        cost: number;
        role: string;
        star_target: number | null;
        position: [
            number,
            number
        ] | null;
        items: string[];
    }[];
    augments: {
        name: string;
        rating: string | null;
        avg_placement: number | null;
        placement_delta: number | null;
        play_rate: number | null;
    }[];
    game_code: string | null;
    guide: {
        id: string;
        name: string;
        channels: CompChannel[];
        text: string;
        author: string | null;
        recommended_augments: string[];
        equipment_order: string[];
        use_count: number | null;
        week_use: number | null;
        collect_count: number | null;
        comment_count: number | null;
        stat_date: string | null;
    } | null;
}
export interface CompUnit {
    key: string; name: string; hero_id: string; unit_type: string; is_hero: boolean; cost: number;
    role: string; target_star: number | null; source_star: number | null; position: [number, number] | null;
    position_raw: string; position_note: string; items: string[];
}
export interface CompTrait {
    name: string; count: number; known: boolean; activation: string; tier: number | null;
    breakpoints: number[]; description: string; levels: { count: number | null; text: string }[];
    level_desc: string[]; picture_key: string;
}
export interface CompStage {
    id: string; label: string; origin: 'comp' | 'guide'; source_name: string; population: number | null;
    hero_count: number; other_count: number; units: CompUnit[]; traits: CompTrait[];
}
export interface CompItemInfo {
    name: string; known: boolean; id: string | null; kind: string | null; description: string;
    components: { id: string; name: string; known: boolean }[];
}
export interface CompVisualization {
    stages: CompStage[];
    heroes: Record<string, { name: string; known: boolean; id: string | null; cost: number | null;
        traits: string[]; role_tag: string | null; attack_range: number | null; skill_name: string | null; skill_desc: string | null }>;
    items: Record<string, CompItemInfo>;
    augments: Record<string, { name: string; known: boolean; tier: number | null; description: string }>;
    mechanics: Record<string, { name: string; known: boolean; description: string; price: number | null; upgrades: string[] }>;
    guide: { name: string; sections: { key: string; title: string; text: string }[];
        carry_items: string[]; carry_items_backup: string[]; alternative_augments: string[];
        adventures: { stage: string; names: string[] }[] } | null;
}
export interface HeroOption {
    name: string;
    cost: number;
}
export interface SelectOption {
    value: string;
    label: string;
    hint?: string;
    color?: string;
}
export interface TableColumn {
    key: string;
    title: string;
    width?: string | number;
    align?: 'left' | 'center' | 'right';
    sortable?: boolean;
    wrap?: boolean;
    format?: (value: unknown, row: TableRow) => string;
}
export type TableRow = Record<string, unknown>;
export interface StoredLog extends LogEntry {
    id: number;
}
export interface AutoRollDefaults {
    comp_name: string | null;
    keep: number;
    max_rolls: number;
    dry_run: boolean;
    heroes: {
        name: string;
        cost: number;
        star: number;
        checked: boolean;
    }[];
    all_heroes: HeroOption[];
}
export interface AutoRollOptions {
    targets: {
        name: string;
        star: number;
    }[];
    keep: number;
    max_rolls: number;
}
export interface CompsResult {
    info: {
        game_version: string | null;
        official_count: number;
        ingame_count: number;
        dual_channel_count: number;
        catalog_count: number;
        dataj_count: number;
        synced_at: string | null;
        source_notice: string;
    };
    rows: CompRow[];
}
export type DatabaseKind = 'heroes' | 'traits' | 'items' | 'augments' | 'mechanics' | 'encounters' | 'guides';
/** Fields returned by db_list; only fields belonging to the requested category are populated. */
export interface DatabaseEntry {
    name: string;
    cost?: number | null;
    traits?: string[];
    attack_range?: number | null;
    skill_name?: string | null;
    skill_desc?: string | null;
    kind?: string | null;
    breakpoints?: number[];
    levels?: {
        count: number;
        text: string;
        values: number[];
    }[];
    description?: string | null;
    level_desc?: string[];
    recipe?: string[];
    trait_name?: string | null;
    tier?: number | null;
    tier_label?: string;
    category?: string | null;
    price?: number | null;
    rounds?: [
        string,
        string | null
    ][];
    upgrades?: string[];
    quality?: string | null;
    heroes?: string[];
    three_star?: string[];
    text?: string;
}
export interface OddsInput {
    name: string;
    need: number;
    others: number;
}
export interface OddsRow extends OddsInput {
    cost: number;
    have: number;
    remaining: number;
    slot_prob: number;
    per_roll: number;
    rolls_50: number | null;
    gold_50: number | null;
    rolls_80: number | null;
    gold_80: number | null;
    done: boolean;
}
export interface OddsResult {
    owned: Record<string, number>;
    rows: OddsRow[];
    any_hit: number | null;
}
export interface SimulationResult {
    all_hit_rate: number;
    hit_rate: Record<string, number>;
    avg_rolls: number;
    avg_gold_left: number;
}
export type SettingValue = string | number | boolean | string[];
export type SettingsValues = Record<string, Record<string, SettingValue>>;
export interface SettingsField {
    key: string;
    label: string;
    type: 'rounds' | 'bool' | 'int' | 'float' | 'str' | 'secret' | 'file' | 'choice' | 'comp';
    choices: (string | SelectOption)[] | null;
    value: SettingValue;
}
export interface SettingsSection {
    key: string;
    title: string;
    note: string;
    fields: SettingsField[];
}
export interface SeasonOption {
    id: string;
    label: string;
    game_version: string | null;
    has_dataj: boolean;
    has_ingame: boolean;
    current: boolean;
}
export interface LLMPreset {
    key: string;
    title: string;
    model: string;
    effort: string;
    effort_augment: string;
    note: string;
    active: boolean;
}
export interface LabelUnitsInfo {
    units: {
        key: string;
        where: string;
        star: number;
        name: string;
        known: boolean;
    }[];
    heroes: string[];
}
export interface CalibrationPoint {
    key: string;
    label: string;
}
export interface DatabaseStats {
    stats: Record<string, string | number | null>;
    pool_count_by_cost: Record<string, number>;
}

export interface FeatureSettings {
    assistant: { default_lock_comp_id: string; economy_preference: 'adaptive' | 'health' | 'economy' };
    scout: { enabled: boolean; schedule: 'key_rounds' | 'every_round' };
    autoroll: { keep_gold: number; max_rolls: number };
    hud: { show_names: boolean; show_traits: boolean; show_odds_panel: boolean; font_size: number; opacity: number; show_positions: boolean };
    autopilot: { augment_selection: 'manual' | 'auto' };
}
export interface FeatureSettingsResult { values: FeatureSettings; comp_options: SelectOption[]; scout_rounds: string[]; }
