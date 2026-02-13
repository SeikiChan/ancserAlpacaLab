/* ==========================================
   Factor Optimizer — Client JS
   ========================================== */

let pollTimer = null;

// ==========================================
// Start / Stop
// ==========================================

async function startOptimizer() {
    const nIter = parseInt(document.getElementById('nIterations').value) || 50;
    const minF = parseInt(document.getElementById('minFactors').value) || 1;
    const maxF = parseInt(document.getElementById('maxFactors').value) || 5;
    const universe = document.getElementById('optUniverse').value;

    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnStop').disabled = false;
    setStatus('running', 'Starting...');

    try {
        const res = await fetch('/api/optimizer/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                n_iterations: nIter,
                min_factors: minF,
                max_factors: maxF,
                universe_mode: universe
            })
        });
        const data = await res.json();

        if (data.status === 'error') throw new Error(data.message);

        document.getElementById('progressSection').style.display = '';
        startPolling();

    } catch (e) {
        setStatus('error', 'Failed: ' + e.message);
        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnStop').disabled = true;
    }
}


async function stopOptimizer() {
    try {
        await fetch('/api/optimizer/stop', { method: 'POST' });
        setStatus('ready', 'Stopped');
    } catch (e) {
        console.error(e);
    }
    stopPolling();
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnStop').disabled = true;
}


// ==========================================
// Polling
// ==========================================

function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollStatus, 1500);
    pollStatus(); // immediate first call
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

async function pollStatus() {
    try {
        const res = await fetch('/api/optimizer/status');
        const data = await res.json();
        renderProgress(data);
        renderBest(data.best);
        renderLeaderboard(data.top_results);

        if (data.status === 'done' || data.status === 'stopped' || data.status === 'error') {
            stopPolling();
            document.getElementById('btnStart').disabled = false;
            document.getElementById('btnStop').disabled = true;
            setStatus('ready', data.status === 'done' ? 'Complete' : data.status);

            if (data.status === 'done') {
                setStatus('ready', `Done! ${data.results_count} iterations — Top 10 saved`);
            }
        } else {
            setStatus('running', `Running ${data.progress}/${data.total}...`);
        }
    } catch (e) {
        console.error('Poll error:', e);
    }
}


// ==========================================
// Rendering
// ==========================================

function renderProgress(data) {
    const section = document.getElementById('progressSection');
    section.style.display = '';

    const pct = data.total > 0 ? (data.progress / data.total * 100) : 0;
    document.getElementById('progressFill').style.width = pct.toFixed(1) + '%';
    document.getElementById('progressLabel').textContent = `${data.progress} / ${data.total}`;

    if (data.eta_sec > 0) {
        const mins = Math.floor(data.eta_sec / 60);
        const secs = Math.floor(data.eta_sec % 60);
        document.getElementById('progressEta').textContent = `ETA: ${mins}m ${secs}s`;
    } else {
        document.getElementById('progressEta').textContent = data.status === 'done' ? 'Complete' : 'ETA: —';
    }
}


function renderBest(best) {
    const section = document.getElementById('bestSection');
    const card = document.getElementById('bestCard');
    if (!best) {
        section.style.display = 'none';
        return;
    }
    section.style.display = '';

    const factorTags = Object.entries(best.factors)
        .map(([k, w]) => `<span class="factor-tag">${k} <b>${(w * 100).toFixed(0)}%</b></span>`)
        .join('');

    card.innerHTML = `
        <div class="best-stats">
            <div class="best-stat">
                <span class="label">Calmar</span>
                <span class="value highlight">${best.calmar.toFixed(2)}</span>
            </div>
            <div class="best-stat">
                <span class="label">CAGR</span>
                <span class="value ${best.cagr >= 0 ? 'positive' : 'negative'}">${(best.cagr * 100).toFixed(2)}%</span>
            </div>
            <div class="best-stat">
                <span class="label">Max DD</span>
                <span class="value negative">${(best.max_dd * 100).toFixed(2)}%</span>
            </div>
            <div class="best-stat">
                <span class="label">Sharpe</span>
                <span class="value">${best.sharpe.toFixed(2)}</span>
            </div>
            <div class="best-stat">
                <span class="label">Top N</span>
                <span class="value">${best.top_n}</span>
            </div>
            <div class="best-stat">
                <span class="label">Rebalance</span>
                <span class="value">${best.rebalance}</span>
            </div>
            <div class="best-stat">
                <span class="label">Years</span>
                <span class="value">${best.years}</span>
            </div>
        </div>
        <div class="best-factors">
            <span class="label">Factors:</span> ${factorTags}
        </div>
        <button class="btn btn-primary" onclick="applyToMain()" style="margin-top:12px;">
            Apply to Dashboard →
        </button>
    `;
}


function renderLeaderboard(results) {
    const section = document.getElementById('leaderboardSection');
    const tbody = document.getElementById('leaderboardBody');
    const countEl = document.getElementById('resultsCount');
    if (!results || results.length === 0) {
        section.style.display = 'none';
        return;
    }
    section.style.display = '';
    countEl.textContent = `(${results.length} shown)`;

    tbody.innerHTML = results.map((r, idx) => {
        const factors = Object.entries(r.factors)
            .map(([k, w]) => `${k}:${(w * 100).toFixed(0)}%`)
            .join(', ');

        return `<tr class="${idx === 0 ? 'top-row' : ''}">
            <td>${idx + 1}</td>
            <td class="mono bold">${r.calmar.toFixed(2)}</td>
            <td class="mono ${r.cagr >= 0 ? 'positive' : 'negative'}">${(r.cagr * 100).toFixed(1)}%</td>
            <td class="mono negative">${(r.max_dd * 100).toFixed(1)}%</td>
            <td class="mono">${r.sharpe.toFixed(2)}</td>
            <td>${r.top_n}</td>
            <td>${r.rebalance}</td>
            <td>${r.years}y</td>
            <td class="factors-cell">${factors}</td>
            <td><button class="btn-mini" onclick="applyConfig(${idx})">Apply</button></td>
        </tr>`;
    }).join('');
}


// ==========================================
// Apply to Dashboard
// ==========================================

let _cachedResults = [];

// Override renderLeaderboard to cache results
const _origRenderLeaderboard = renderLeaderboard;
renderLeaderboard = function (results) {
    if (results) _cachedResults = results;
    _origRenderLeaderboard(results);
};


function applyToMain() {
    if (_cachedResults.length > 0) {
        _applyAndRedirect(_cachedResults[0]);
    }
}

function applyConfig(idx) {
    if (_cachedResults[idx]) {
        _applyAndRedirect(_cachedResults[idx]);
    }
}

function _applyAndRedirect(entry) {
    // Store in sessionStorage, main page will pick it up
    const strategy = {
        factors: entry.full_config,
        rebalance: entry.rebalance,
        years: entry.years,
        top_n: entry.top_n,
    };
    sessionStorage.setItem('optimizer_apply', JSON.stringify(strategy));
    window.location.href = '/';
}


// ==========================================
// Utilities
// ==========================================

function setStatus(type, text) {
    const pill = document.getElementById('statusPill');
    const textEl = document.getElementById('statusText');
    pill.className = 'status-pill';
    if (type === 'running') pill.classList.add('running');
    if (type === 'error') pill.classList.add('error');
    textEl.textContent = text;
}
