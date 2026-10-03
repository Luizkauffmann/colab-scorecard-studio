// Scorecard Studio - scorecard alignment app.
// Strategy (cutoff), Scorecard (points editor), Statistics, Export.
// All requests use relative URLs so they resolve through Colab's port proxy.
'use strict';

async function api(path, body) {
  const opts = { method: body ? 'POST' : 'GET', headers: { Accept: 'application/json' } };
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    const t = await res.text();
    throw new Error('HTTP ' + res.status + ': ' + t.replace(/<[^>]+>/g, '').trim().slice(0, 160));
  }
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

const S = { state: null, strategy: null, scorecard: null, stats: null, tab: 'strategy', charts: {}, code: '' };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const pct = (x, d = 1) => (x == null || isNaN(x) ? '—' : (x * 100).toFixed(d) + '%');
const num = (x, d = 0) => (x == null || isNaN(x) ? '—' : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const money = (x) => (x == null || isNaN(x) ? '—' : Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 }));

function busy(on, msg) { $('busy').style.display = on ? 'flex' : 'none'; if (msg) $('busyMsg').textContent = msg; }
function toast(msg, type) {
  const el = $('toast'); el.textContent = msg; el.className = type === 'error' ? 'error' : '';
  el.style.opacity = '1'; clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.opacity = '0'; }, type === 'error' ? 7000 : 3000);
}
async function run(msg, fn) { busy(true, msg); try { return await fn(); } catch (e) { toast(e.message, 'error'); } finally { busy(false); } }

// Vertical line at a cutoff on linear-x charts.
const cutLinePlugin = {
  id: 'cutLine',
  afterDatasetsDraw(chart, _, opts) {
    if (opts == null || opts.value == null) return;
    const x = chart.scales.x; const a = chart.chartArea;
    if (!x || !a) return;
    const px = x.getPixelForValue(opts.value);
    if (px < a.left || px > a.right) return;
    const c = chart.ctx; c.save(); c.strokeStyle = opts.color || '#c0392b'; c.lineWidth = 1.5; c.setLineDash(opts.dash || []);
    c.beginPath(); c.moveTo(px, a.top); c.lineTo(px, a.bottom); c.stroke();
    if (opts.label) { c.fillStyle = opts.color || '#c0392b'; c.font = '10px sans-serif'; c.fillText(opts.label, px + 4, a.top + 10); }
    c.restore();
  },
};

window.addEventListener('DOMContentLoaded', async () => {
  if (typeof Chart !== 'undefined') Chart.register(cutLinePlugin);
  document.querySelectorAll('.main-tab').forEach((b) => { b.onclick = () => switchTab(b.dataset.tab); });
  document.querySelectorAll('#modeSeg button').forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
  $('btnApply').onclick = applySettings;
  $('btnFinal').onclick = finalize; $('btnFinal2').onclick = finalize;
  $('btnResetAll').onclick = () => resetPoints(null, null);
  $('btnCode').onclick = previewCode; $('btnCodeSave').onclick = saveCode; $('btnCodeDl').onclick = downloadCode;
  $('codeLang').onchange = () => { $('codeTable').style.display = $('codeLang').value.startsWith('sql') ? '' : 'none'; previewCode(); };
  $('codeTable').style.display = 'none';
  $('gainsSample').onchange = () => loadStats($('gainsSample').value);
  try {
    const r = await api('api/state');
    apply(r);
    fillSettings();
  } catch (e) { toast('Could not reach the app backend: ' + e.message, 'error'); }
});

function apply(r) {
  if (r.state) S.state = r.state;
  if (r.strategy) S.strategy = r.strategy;
  if (r.scorecard) S.scorecard = r.scorecard;
  renderSidebar();
  renderStrategy();
  if (r.scorecard && S.tab === 'scorecard') renderScorecard();
}

function switchTab(t) {
  S.tab = t;
  document.querySelectorAll('.main-tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === t));
  ['strategy', 'scorecard', 'stats', 'export'].forEach((x) => { $('tab-' + x).style.display = x === t ? '' : 'none'; });
  if (t === 'scorecard') loadScorecard();
  if (t === 'stats') loadStats();
  if (t === 'strategy') renderStrategy();
}

// ── Sidebar ──────────────────────────────────────────────────────────
function fillSettings() {
  const s = S.state;
  $('goodClass').value = String(s.good_class);
  $('stratSample').innerHTML = s.samples.concat(s.samples.length > 1 ? ['all'] : [])
    .map((x) => '<option value="' + esc(x) + '">' + esc(x) + (x === 'train' ? ' (in-sample)' : '') + '</option>').join('');
  $('stratSample').value = s.strategy_sample;
  $('pdo').value = s.pdo; $('baseScore').value = s.base_score; $('baseOdds').value = s.base_odds;
  $('basePoints').value = s.base_points; $('costBad').value = s.cost_bad; $('benefitGood').value = s.benefit_good;
  $('gainsSample').innerHTML = $('stratSample').innerHTML;
}

function renderSidebar() {
  const s = S.state; if (!s) return;
  $('dsInfo').innerHTML = (s.dataset ? '<b>' + esc(s.dataset) + '</b><br>' : '') +
    'Target <b>' + esc(s.target) + '</b> · ' + num(s.rows) + ' rows<br>' + s.variables.length + ' variables · samples: ' + esc(s.samples.join(', '));
  $('scaleInfo').innerHTML = 'Factor ' + Number(s.factor).toFixed(4) + ' · Offset ' + Number(s.offset).toFixed(4) +
    '<br>Score = Offset + Factor · ln(odds good:bad)';
  if (s.last_saved) { $('saveStatus').textContent = 'Saved to Drive at ' + s.last_saved; $('saveStatus').className = 'save-status ok'; }
}

function applySettings() {
  const body = {
    good_class: parseInt($('goodClass').value, 10), strategy_sample: $('stratSample').value,
    pdo: parseFloat($('pdo').value), base_score: parseFloat($('baseScore').value), base_odds: parseFloat($('baseOdds').value),
    base_points: $('basePoints').value, cost_bad: parseFloat($('costBad').value), benefit_good: parseFloat($('benefitGood').value),
  };
  run('Applying…', async () => {
    const r = await api('api/settings', body);
    apply(r);
    if (S.tab === 'scorecard') loadScorecard();
    if (S.tab === 'stats') loadStats();
    toast('Settings applied.' + (S.scorecard && S.scorecard.n_stale ? ' Some overrides were set under another scale: check them.' : ''));
  });
}

// ── Strategy ─────────────────────────────────────────────────────────
function setMode(mode) {
  const st = S.strategy;
  let value = null;
  if (mode === 'approval') value = st ? Math.round(st.kpis.approval_rate * 100) / 100 : 0.7;
  if (mode === 'bad_rate') value = st ? Math.round(st.kpis.bad_rate_approved * 1000) / 1000 : 0.05;
  if (mode === 'manual') value = st ? st.cutoff : 0;
  postStrategy({ mode, value });
}
function postStrategy(body) {
  return run('Updating…', async () => { apply(await api('api/settings', body)); });
}

function renderStrategy() {
  const st = S.strategy; if (!st) return;
  document.querySelectorAll('#modeSeg button').forEach((b) => b.classList.toggle('active', b.dataset.mode === st.mode));
  const box = $('modeInput');
  if (st.mode === 'profit') {
    box.innerHTML = '<span class="unit">Cutoff that maximises profit on <b>' + esc(st.sample) + '</b>.</span>';
  } else if (st.mode === 'approval' || st.mode === 'bad_rate') {
    const v = st.value != null ? +(st.value * 100).toFixed(2) : '';
    box.innerHTML = '<span class="unit">' + (st.mode === 'approval' ? 'Approve at least' : 'Bad rate among approved at most') +
      '</span><input type="number" id="modeVal" min="0" max="100" step="0.5" value="' + v + '"><span class="unit">%</span>';
    $('modeVal').onchange = () => postStrategy({ value: parseFloat($('modeVal').value) / 100 });
  } else {
    const [lo, hi] = st.score_range;
    box.innerHTML = '<input type="range" id="modeSlider" min="' + Math.floor(lo) + '" max="' + Math.ceil(hi) + '" step="1" value="' + Math.round(st.cutoff) +
      '"><input type="number" id="modeVal" step="1" value="' + st.cutoff + '">';
    $('modeSlider').oninput = () => { $('modeVal').value = $('modeSlider').value; };
    $('modeSlider').onchange = () => postStrategy({ value: parseFloat($('modeSlider').value) });
    $('modeVal').onchange = () => postStrategy({ value: parseFloat($('modeVal').value) });
  }
  const k = st.kpis; const sug = st.suggested_kpis;
  const be = st.break_even_score; const pdo = S.state.pdo;
  const gap = be != null ? st.suggested - be : null;
  const note = $('calibNote');
  if (gap == null || isNaN(gap)) { note.className = 'note'; note.textContent = 'Set a cost of a bad and a benefit of a good to get the profit-optimal cutoff.'; }
  else {
    const g = Math.abs(gap);
    const level = g <= pdo / 2 ? 'agree' : g <= pdo ? 'close' : 'off';
    note.className = 'note ' + (level === 'off' ? 'warn' : 'ok');
    note.innerHTML = 'Suggested cutoff (max profit): <b>' + num(st.suggested) + '</b> · approval ' + pct(sug.approval_rate) +
      ', profit per applicant ' + money(sug.profit_per_applicant) + '. Break-even score from the scale (odds = cost ÷ benefit): <b>' + num(be, 1) + '</b>. ' +
      (level === 'agree' ? 'They agree: the scale is calibrated on this sample.'
        : level === 'close' ? 'They are ' + num(g, 1) + ' points apart, within one PDO: the profit curve is flat near its maximum, so a gap this size is mostly sampling noise.'
        : 'They differ by ' + num(g, 1) + ' points, more than one PDO: the scale\'s odds don\'t match this sample. Check the realized PDO and odds in Statistics.');
  }
  const cards = [
    ['Cutoff', num(st.cutoff), 'approve if score ≥ cutoff', true],
    ['Approval rate', pct(k.approval_rate), num(k.approved) + ' of ' + num(k.rows) + ' rows'],
    ['Bad rate approved', pct(k.bad_rate_approved), 'overall ' + pct(k.bad_rate_all)],
    ['Bads declined', pct(k.bad_capture), num(k.declined_bad) + ' bads'],
    ['Goods declined', pct(k.good_reject_rate), num(k.declined_good) + ' goods'],
    ['Profit / applicant', money(k.profit_per_applicant), 'rows with known outcome'],
    ['Total profit', money(k.profit), 'on ' + esc(st.sample)],
  ];
  $('kpis').innerHTML = cards.map(([l, v, s, acc]) => '<div class="kpi' + (acc ? ' accent' : '') + '"><div class="v">' + v + '</div><div class="l">' + l + '</div><div class="s">' + s + '</div></div>').join('');
  $('cmSample').textContent = st.sample;
  $('cm').innerHTML = '<tr><th></th><th>Approved</th><th>Declined</th></tr>' +
    '<tr><th>Good</th><td class="good">' + num(k.approved_good) + '</td><td>' + num(k.declined_good) + '</td></tr>' +
    '<tr><th>Bad</th><td class="bad">' + num(k.approved_bad) + '</td><td class="good">' + num(k.declined_bad) + '</td></tr>';
  renderStrategyCharts();
}

function chart(id, cfg) {
  if (typeof Chart === 'undefined') return;
  if (S.charts[id]) S.charts[id].destroy();
  S.charts[id] = new Chart($(id), cfg);
}
const baseOpts = (extra) => Object.assign({ responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } }, extra || {});

function renderStrategyCharts() {
  const st = S.strategy; const c = st.curve; const cut = { value: st.cutoff, label: 'cutoff ' + num(st.cutoff) };
  const xy = (ys) => c.cutoff.map((x, i) => ({ x, y: ys[i] }));
  chart('chProfit', { type: 'line', data: { datasets: [{ label: 'profit per applicant', data: xy(c.profit_per_applicant), borderColor: '#1e8449', pointRadius: 0, borderWidth: 1.6 }] },
    options: baseOpts({ plugins: { legend: { display: false }, cutLine: cut, tooltip: { mode: 'nearest', intersect: false } },
      scales: { x: { type: 'linear', title: { display: true, text: 'cutoff', font: { size: 10 } } }, y: { ticks: { font: { size: 10 } } } } }) });
  chart('chRates', { type: 'line', data: { datasets: [
      { label: 'approval rate', data: xy(c.approval_rate), borderColor: '#1a6fc4', pointRadius: 0, borderWidth: 1.6, yAxisID: 'y' },
      { label: 'bad rate among approved', data: xy(c.bad_rate_approved), borderColor: '#c0392b', pointRadius: 0, borderWidth: 1.6, yAxisID: 'y2' }] },
    options: baseOpts({ plugins: { cutLine: cut, legend: { labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: { x: { type: 'linear', title: { display: true, text: 'cutoff', font: { size: 10 } } },
        y: { min: 0, max: 1, ticks: { callback: (v) => pct(v, 0), font: { size: 10 } } },
        y2: { position: 'right', min: 0, grid: { drawOnChartArea: false }, ticks: { callback: (v) => pct(v, 0), font: { size: 10 } } } } }) });
  const h = st.histogram; const w = h.from.length > 1 ? h.from[1] - h.from[0] : 1;
  chart('chDist', { type: 'bar', data: { datasets: [
      { label: 'goods', data: h.from.map((x, i) => ({ x: x + w / 2, y: h.goods[i] })), backgroundColor: '#1e844999', stack: 's', barPercentage: 1, categoryPercentage: 1, grouped: false },
      { label: 'bads', data: h.from.map((x, i) => ({ x: x + w / 2, y: h.bads[i] })), backgroundColor: '#c0392bcc', stack: 's', barPercentage: 1, categoryPercentage: 1, grouped: false }] },
    options: baseOpts({ plugins: { cutLine: cut, legend: { labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: { x: { type: 'linear', stacked: true, title: { display: true, text: 'score', font: { size: 10 } } }, y: { stacked: true, ticks: { font: { size: 10 } } } } }) });
  const d = st.discrimination;
  $('rocLabel').textContent = 'ROC (bads declined vs goods declined) · AUC ' + (d.AUC != null ? d.AUC.toFixed(3) : '—') + ' · Gini ' + (d.Gini != null ? d.Gini.toFixed(3) : '—') + ' · KS ' + (d.KS != null ? d.KS.toFixed(3) : '—');
  chart('chRoc', { type: 'line', data: { datasets: [
      { label: 'scorecard', data: st.roc.fpr.map((x, i) => ({ x, y: st.roc.tpr[i] })), borderColor: '#1a6fc4', pointRadius: 0, borderWidth: 1.6 },
      { label: 'random', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: '#bbb', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 }] },
    options: baseOpts({ scales: { x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'share of goods declined', font: { size: 10 } } },
      y: { min: 0, max: 1, title: { display: true, text: 'share of bads declined', font: { size: 10 } } } } }) });
}

// ── Scorecard editor ─────────────────────────────────────────────────
async function loadScorecard() {
  try { S.scorecard = await api('api/scorecard'); renderScorecard(); } catch (e) { toast(e.message, 'error'); }
}

function renderScorecard() {
  const sc = S.scorecard; if (!sc) return;
  $('scSummary').innerHTML = '<b>' + sc.n_manual + '</b> override(s) · <b>' + sc.n_stale + '</b> set under another scale · <b>' +
    sc.n_order_flags + '</b> order flag(s)' + (sc.base_points != null ? ' · base points <b>' + num(sc.base_points) + '</b>' : '');
  const rows = sc.rows;
  const vars = Array.from(new Set(rows.map((r) => r.Variable)));
  const maxShare = Math.max(0.01, ...Object.values(sc.ranges).map((r) => r.share));
  let html = '<tr><th>Bin</th><th>Kind</th><th class="r">Share</th><th class="r">Event rate</th><th class="r">WOE</th><th class="r">Coef.</th>' +
    '<th class="r">Model pts</th><th class="r">Points</th><th>Reason</th><th>Status</th><th></th></tr>';
  vars.forEach((v) => {
    const rg = sc.ranges[v] || {};
    html += '<tr class="var-head"><td colspan="11">' + esc(v) + ' <span class="muted">(' + esc(sc.representations[v]) + ') · points ' +
      num(rg.min) + ' to ' + num(rg.max) + ' · ' + pct(rg.share, 0) + ' of the score range</span><span class="range-bar" style="width:' +
      Math.round(80 * (rg.share || 0) / maxShare) + 'px"></span></td></tr>';
    rows.filter((r) => r.Variable === v).forEach((r) => {
      if (r.Representation === 'raw') {
        html += '<tr><td colspan="6">' + esc(r.Label) + '</td><td colspan="5" class="muted">linear term: not editable here</td></tr>';
        return;
      }
      const badges = (r.Manual ? '<span class="badge b-manual">manual</span>' : '') + (r.Stale ? '<span class="badge b-stale">old scale</span>' : '') +
        (r['Order flag'] ? '<span class="badge b-order">order</span>' : '');
      html += '<tr class="' + (r.Manual ? 'manual' : '') + '" data-var="' + esc(v) + '" data-group="' + r.Group + '">' +
        '<td style="max-width:260px;white-space:normal">' + esc(r.Label) + '</td>' +
        '<td><span class="badge b-kind">' + esc(r.Kind) + '</span></td>' +
        '<td class="r">' + pct(r.Share) + '</td>' +
        '<td class="r">' + pct(r['Event rate']) + '</td>' +
        '<td class="r">' + (r.WOE == null ? '—' : Number(r.WOE).toFixed(3)) + '</td>' +
        '<td class="r">' + (r.Coefficient == null ? '—' : Number(r.Coefficient).toFixed(3)) + '</td>' +
        '<td class="r">' + num(r['Model points']) + '</td>' +
        '<td class="r"><input class="pts" type="number" step="1" value="' + r.Points + '" aria-label="points"></td>' +
        '<td><input class="reason" type="text" value="' + esc(r.Reason || '') + '" placeholder="reason for the override"></td>' +
        '<td>' + badges + '</td>' +
        '<td>' + (r.Manual ? '<button class="row-btn" data-reset="1" title="Back to the model points">↺</button>' : '') + '</td></tr>';
    });
  });
  $('scTable').innerHTML = html;
  $('scTable').querySelectorAll('tr[data-var]').forEach((tr) => {
    const v = tr.dataset.var; const g = parseInt(tr.dataset.group, 10);
    const pts = tr.querySelector('input.pts'); const reason = tr.querySelector('input.reason');
    const commit = () => {
      const val = parseFloat(pts.value);
      if (isNaN(val)) { toast('Points must be a number.', 'error'); return; }
      const row = sc.rows.find((r) => r.Variable === v && r.Group === g);
      if (row && val === row.Points && (reason.value || '') === (row.Reason || '')) return;
      setPoints(v, g, val, reason.value);
    };
    pts.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
    pts.addEventListener('change', commit);
    reason.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
    reason.addEventListener('change', () => { const row = sc.rows.find((r) => r.Variable === v && r.Group === g); if (row && row.Manual) commit(); });
    const rb = tr.querySelector('button[data-reset]');
    if (rb) rb.onclick = () => resetPoints(v, g);
  });
}

function setPoints(variable, group, points, reason) {
  run('Recomputing scores…', async () => {
    const r = await api('api/points', { variable, group, points, reason });
    apply(r); renderScorecard();
    toast(variable + ': bin set to ' + points + ' points. Cutoff and KPIs updated.');
  });
}
function resetPoints(variable, group) {
  run('Resetting…', async () => {
    const r = await api('api/reset', { variable, group });
    apply(r); renderScorecard();
    toast(variable ? 'Back to the model points.' : 'All overrides removed.');
  });
}

// ── Statistics ───────────────────────────────────────────────────────
async function loadStats(sample) {
  try {
    S.stats = await api('api/stats' + (sample ? '?sample=' + encodeURIComponent(sample) : ''));
    $('gainsSample').value = S.stats.sample;
    renderStats();
  } catch (e) { toast(e.message, 'error'); }
}

function renderStats() {
  const s = S.stats; const d = s.design;
  $('designNote').textContent = '· design: PDO ' + d.pdo + ', ' + d.base_odds + ':1 at ' + d.base_score;
  let h = '<tr><th>Sample</th><th class="r">Rows</th><th class="r">Bad rate</th><th class="r">Mean score</th><th class="r">AUC</th><th class="r">Gini</th><th class="r">KS</th><th class="r">Realized PDO</th><th class="r">Realized odds at ' + d.base_score + '</th></tr>';
  s.per_sample.forEach((r) => {
    h += '<tr><td>' + esc(r.sample) + '</td><td class="r">' + num(r.rows) + '</td><td class="r">' + pct(r.bad_rate) + '</td><td class="r">' + num(r.mean_score, 1) +
      '</td><td class="r">' + (r.AUC != null ? r.AUC.toFixed(3) : '—') + '</td><td class="r">' + (r.Gini != null ? r.Gini.toFixed(3) : '—') +
      '</td><td class="r">' + (r.KS != null ? r.KS.toFixed(3) : '—') + '</td><td class="r">' + num(r.realized_pdo, 1) + '</td><td class="r">' + num(r.realized_odds_at_base, 1) + ':1</td></tr>';
  });
  $('perfTable').innerHTML = h;
  const psi = Object.entries(s.psi);
  $('psiNote').innerHTML = psi.length ? 'Score PSI from train: ' + psi.map(([k, v]) => esc(k) + ' <b>' + v.toFixed(4) + '</b>' + (v < 0.1 ? ' (stable)' : v < 0.25 ? ' (watch)' : ' (shifted)')).join(' · ') +
    '. Realized PDO and odds come from a weighted fit of observed ln(odds) on band mean score; close to the design means the scale holds on that sample.' : '';
  const g = s.gains;
  let gh = '<tr><th>Band</th><th class="r">Rows</th><th class="r">Share</th><th class="r">Goods</th><th class="r">Bads</th><th class="r">Bad rate</th><th class="r">Expected</th><th class="r">Odds</th><th class="r">ln(odds)</th><th class="r">Cum. approval</th><th class="r">Cum. bad rate</th></tr>';
  g.forEach((r) => {
    gh += '<tr><td>' + esc(r.band) + '</td><td class="r">' + num(r.rows) + '</td><td class="r">' + pct(r.share) + '</td><td class="r">' + num(r.goods) + '</td><td class="r">' + num(r.bads) +
      '</td><td class="r">' + pct(r.bad_rate) + '</td><td class="r">' + pct(r.expected_bad_rate) + '</td><td class="r">' + (r.odds == null ? '∞' : num(r.odds, 1)) + '</td><td class="r">' + num(r.ln_odds, 2) +
      '</td><td class="r">' + pct(r.cum_approval) + '</td><td class="r">' + pct(r.cum_bad_rate) + '</td></tr>';
  });
  $('gainsTable').innerHTML = gh;
  const labels = g.map((r) => r.band).reverse(); const gr = g.slice().reverse();
  chart('chCalib', { data: { labels, datasets: [
      { type: 'bar', label: 'observed bad rate', data: gr.map((r) => r.bad_rate), backgroundColor: '#c0392b88' },
      { type: 'line', label: 'expected by the scale', data: gr.map((r) => r.expected_bad_rate), borderColor: '#1a1a2e', pointRadius: 2, borderWidth: 1.5 }] },
    options: baseOpts({ scales: { x: { ticks: { font: { size: 9 }, maxRotation: 45 } }, y: { min: 0, ticks: { callback: (v) => pct(v, 0), font: { size: 10 } } } } }) });
  const cb = Object.entries(s.contribution).sort((a, b) => b[1].share - a[1].share);
  chart('chContrib', { type: 'bar', data: { labels: cb.map(([v]) => v), datasets: [{ label: 'share of score range', data: cb.map(([, r]) => r.share), backgroundColor: '#1a6fc499' }] },
    options: baseOpts({ indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { min: 0, ticks: { callback: (v) => pct(v, 0), font: { size: 10 } } }, y: { ticks: { font: { size: 10 } } } } }) });
}

// ── Export ───────────────────────────────────────────────────────────
function finalize() {
  run('Creating the final table…', async () => {
    const r = await api('api/finalize', {});
    S.state = r.state; renderSidebar();
    const f = r.final;
    $('finalInfo').className = 'note ok';
    $('finalInfo').innerHTML = '<b>Final table written at ' + esc(f.at) + '</b>: ' + num(f.rows) + ' rows × ' + f.columns + ' columns, cutoff ' + num(f.cutoff) +
      ', approval ' + pct(f.approval_rate) + ' (all samples).<br>' + Object.values(f.paths || {}).map(esc).join('<br>');
    const p = r.preview;
    $('finalPreview').innerHTML = '<tr>' + p.columns.map((c) => '<th>' + esc(c) + '</th>').join('') + '</tr>' +
      p.rows.map((row) => '<tr>' + row.map((x) => '<td>' + esc(typeof x === 'number' ? +x.toFixed(4) : x) + '</td>').join('') + '</tr>').join('');
    switchTab('export');
    toast('Final table and final scorecard saved.');
  });
}
function codeParams() {
  const v = $('codeLang').value;
  const [lang, dialect] = v.includes(':') ? v.split(':') : [v, 'standard'];
  return { lang, dialect, table: $('codeTable').value || 'input_table' };
}
function previewCode() {
  const p = codeParams();
  run('Generating code…', async () => {
    const r = await api('api/code?lang=' + p.lang + '&dialect=' + p.dialect + '&table=' + encodeURIComponent(p.table));
    S.code = r.code; $('codeBox').textContent = r.code;
    $('codeNote').textContent = (p.lang === 'python' ? 'Run: python scorer.py input.csv output.csv  ·  or import score_record()' : 'Reads ' + p.table + '; returns its columns plus pts_<var>, score, pd and decision.') +
      '  ·  ' + r.code.split('\n').length + ' lines, cutoff ' + num(S.strategy.cutoff) + '.';
  });
}
function saveCode() {
  const p = codeParams();
  run('Saving…', async () => { const r = await api('api/code/save', p); toast('Saved ' + r.path); previewCode(); });
}
function downloadCode() {
  const p = codeParams();
  const go = (text) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.download = p.lang === 'python' ? 'scorer.py' : 'scorer_' + p.dialect + '.sql';
    a.click();
  };
  if (S.code) go(S.code); else api('api/code?lang=' + p.lang + '&dialect=' + p.dialect + '&table=' + encodeURIComponent(p.table)).then((r) => go(r.code));
}
