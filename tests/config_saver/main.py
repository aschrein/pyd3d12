# MIT License
# Copyright (c) 2025 Anton Schreiner

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from difflib import unified_diff
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


TMP_FOLDER = Path(__file__).parent / ".tmp"
TMP_FOLDER.mkdir(exist_ok=True)

DATABASE = TMP_FOLDER / "configs.db"


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            config TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON configs(name)")
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ML Config Tracker", lifespan=lifespan)


# -------------------- Pydantic Models --------------------

class ConfigCreate(BaseModel):
    name: str
    config: dict[str, Any]


class ConfigResponse(BaseModel):
    id: int
    name: str
    config: dict[str, Any]
    created_at: str


class ConfigListItem(BaseModel):
    id: int
    name: str
    created_at: str


class DiffResponse(BaseModel):
    config1: dict[str, str]
    config2: dict[str, str]
    diff: str


# -------------------- API Routes --------------------

@app.post("/api/configs", status_code=201)
def save_config(data: ConfigCreate):
    """Save a new configuration."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO configs (name, config) VALUES (?, ?)",
            (data.name, json.dumps(data.config, indent=2, sort_keys=True)),
        )
        conn.commit()
        return {"message": f"Config '{data.name}' saved successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail=f"Config with name '{data.name}' already exists")
    finally:
        conn.close()


@app.get("/api/configs", response_model=list[ConfigListItem])
def list_configs(search: str = Query(default="")):
    """List all configs, optionally filtered by search query."""
    conn = get_db()
    try:
        if search:
            rows = conn.execute(
                "SELECT id, name, created_at FROM configs WHERE name LIKE ? ORDER BY created_at DESC",
                (f"%{search}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, created_at FROM configs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@app.get("/api/configs/{name}", response_model=ConfigResponse)
def get_config(name: str):
    """Get a specific config by name."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM configs WHERE name = ?", (name,)
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail=f"Config '{name}' not found")

        result = dict(row)
        result["config"] = json.loads(result["config"])
        return result
    finally:
        conn.close()


@app.delete("/api/configs/{name}")
def delete_config(name: str):
    """Delete a config by name."""
    conn = get_db()
    try:
        cursor = conn.execute("DELETE FROM configs WHERE name = ?", (name,))
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Config '{name}' not found")

        return {"message": f"Config '{name}' deleted successfully"}
    finally:
        conn.close()


@app.get("/api/diff", response_model=DiffResponse)
def diff_configs(config1: str = Query(...), config2: str = Query(...)):
    """Compare two configs and return a unified diff."""
    conn = get_db()
    try:
        row1 = conn.execute("SELECT config FROM configs WHERE name = ?", (config1,)).fetchone()
        row2 = conn.execute("SELECT config FROM configs WHERE name = ?", (config2,)).fetchone()

        if row1 is None:
            raise HTTPException(status_code=404, detail=f"Config '{config1}' not found")
        if row2 is None:
            raise HTTPException(status_code=404, detail=f"Config '{config2}' not found")

        config1_str = row1["config"]
        config2_str = row2["config"]

        diff = list(unified_diff(
            config1_str.splitlines(keepends=True),
            config2_str.splitlines(keepends=True),
            fromfile=config1,
            tofile=config2,
        ))

        return {
            "config1": {"name": config1, "content": config1_str},
            "config2": {"name": config2, "content": config2_str},
            "diff": "".join(diff),
        }
    finally:
        conn.close()


# -------------------- UI --------------------

HTML_TEMPLATE = """
//js
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Config Tracker</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        h2 { color: #8b949e; font-size: 1rem; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .grid { display: grid; grid-template-columns: 300px 1fr; gap: 20px; }
        
        .panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
        }
        
        input, textarea, button, select {
            font-family: inherit;
            font-size: 14px;
        }
        
        input[type="text"], textarea, select {
            width: 100%;
            padding: 8px 12px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            margin-bottom: 10px;
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #58a6ff;
        }
        
        button {
            padding: 8px 16px;
            background: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #2ea043; }
        button.secondary { background: #30363d; }
        button.secondary:hover { background: #484f58; }
        button.danger { background: #da3633; }
        button.danger:hover { background: #f85149; }
        
        .config-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .config-item {
            padding: 10px;
            border-bottom: 1px solid #30363d;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .config-item:hover { background: #1f2937; }
        .config-item.selected { background: #1f6feb33; border-left: 3px solid #58a6ff; }
        .config-item .name { font-weight: 500; }
        .config-item .date { font-size: 12px; color: #8b949e; }
        
        .diff-container { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .diff-panel { flex: 1; }
        
        .diff-view {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-all;
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
            max-height: 500px;
            overflow: auto;
            border: 1px solid #30363d;
        }
        
        .diff-line { padding: 1px 4px; }
        .diff-add { background: #2ea04333; color: #3fb950; }
        .diff-remove { background: #f8514933; color: #f85149; }
        .diff-header { color: #8b949e; }
        .diff-range { color: #a371f7; }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .tab {
            padding: 8px 16px;
            background: transparent;
            border: 1px solid #30363d;
            color: #8b949e;
        }
        .tab.active { background: #30363d; color: #c9d1d9; border-color: #58a6ff; }
        
        .compare-selection {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }
        .compare-selection select { margin-bottom: 0; flex: 1; }
        
        .message {
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 10px;
            margin-top: 10px;
        }
        .message.success { background: #2ea04333; color: #3fb950; }
        .message.error { background: #f8514933; color: #f85149; }
        
        .checkbox-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }
        .checkbox-row input { width: auto; margin: 0; }
        
        textarea { min-height: 150px; font-family: monospace; }
        
        .empty-state { color: #8b949e; text-align: center; padding: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 ML Config Tracker</h1>
        
        <div class="grid">
            <div class="sidebar">
                <div class="panel">
                    <h2>Search Configs</h2>
                    <input type="text" id="searchInput" placeholder="Search by name..." oninput="searchConfigs()">
                    
                    <div class="checkbox-row">
                        <input type="checkbox" id="selectMode" onchange="toggleSelectMode()">
                        <label for="selectMode">Select for comparison</label>
                    </div>
                    
                    <div class="config-list" id="configList">
                        <div class="empty-state">Loading...</div>
                    </div>
                </div>
                
                <div class="panel" style="margin-top: 20px;">
                    <h2>Save New Config</h2>
                    <input type="text" id="configName" placeholder="e.g., model-run-20-09-2022">
                    <textarea id="configJson" placeholder='{"learning_rate": 0.001, "batch_size": 32}'></textarea>
                    <button onclick="saveConfig()">Save Config</button>
                    <div id="saveMessage"></div>
                </div>
            </div>
            
            <div class="main">
                <div class="panel">
                    <div class="tabs">
                        <button class="tab active" onclick="showTab('view')">View Config</button>
                        <button class="tab" onclick="showTab('compare')">Compare Configs</button>
                    </div>
                    
                    <div id="viewTab">
                        <div id="configView" class="diff-view empty-state">
                            Select a config from the list to view
                        </div>
                        <div style="margin-top: 10px;">
                            <button class="danger" id="deleteBtn" onclick="deleteConfig()" style="display: none;">Delete Config</button>
                        </div>
                    </div>
                    
                    <div id="compareTab" style="display: none;">
                        <div class="compare-selection">
                            <select id="config1Select" onchange="updateDiff()">
                                <option value="">Select first config...</option>
                            </select>
                            <span>vs</span>
                            <select id="config2Select" onchange="updateDiff()">
                                <option value="">Select second config...</option>
                            </select>
                            <button class="secondary" onclick="swapConfigs()">⇄ Swap</button>
                        </div>
                        
                        <h2>Diff View</h2>
                        <div id="diffOutput" class="diff-view empty-state">
                            Select two configs to compare
                        </div>
                        
                        <div class="diff-container" style="margin-top: 15px;">
                            <div class="diff-panel">
                                <h2 id="config1Label">Config 1</h2>
                                <div id="config1Content" class="diff-view"></div>
                            </div>
                            <div class="diff-panel">
                                <h2 id="config2Label">Config 2</h2>
                                <div id="config2Content" class="diff-view"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let allConfigs = [];
        let selectedConfig = null;
        let selectMode = false;
        let selectedForCompare = [];
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function loadConfigs() {
            const response = await fetch('/api/configs');
            allConfigs = await response.json();
            renderConfigList(allConfigs);
            updateCompareSelects();
        }
        
        function renderConfigList(configs) {
            const list = document.getElementById('configList');
            if (configs.length === 0) {
                list.innerHTML = '<div class="empty-state">No configs found</div>';
                return;
            }
            
            list.innerHTML = configs.map(c => {
                const date = new Date(c.created_at).toLocaleDateString();
                const isSelected = selectedConfig === c.name;
                const isCompareSelected = selectedForCompare.includes(c.name);
                return `
                    <div class="config-item ${isSelected || isCompareSelected ? 'selected' : ''}" 
                         onclick="selectConfig('${escapeHtml(c.name)}')">
                        <div>
                            <div class="name">${escapeHtml(c.name)}</div>
                            <div class="date">${date}</div>
                        </div>
                        ${selectMode && isCompareSelected ? '<span>✓</span>' : ''}
                    </div>
                `;
            }).join('');
        }
        
        async function searchConfigs() {
            const query = document.getElementById('searchInput').value;
            const response = await fetch(`/api/configs?search=${encodeURIComponent(query)}`);
            const configs = await response.json();
            renderConfigList(configs);
        }
        
        function toggleSelectMode() {
            selectMode = document.getElementById('selectMode').checked;
            selectedForCompare = [];
            renderConfigList(allConfigs);
        }
        
        async function selectConfig(name) {
            if (selectMode) {
                const idx = selectedForCompare.indexOf(name);
                if (idx > -1) {
                    selectedForCompare.splice(idx, 1);
                } else if (selectedForCompare.length < 2) {
                    selectedForCompare.push(name);
                } else {
                    selectedForCompare = [selectedForCompare[1], name];
                }
                renderConfigList(allConfigs.filter(c => 
                    c.name.includes(document.getElementById('searchInput').value)
                ));
                
                if (selectedForCompare.length === 2) {
                    document.getElementById('config1Select').value = selectedForCompare[0];
                    document.getElementById('config2Select').value = selectedForCompare[1];
                    showTab('compare');
                    updateDiff();
                }
                return;
            }
            
            selectedConfig = name;
            const response = await fetch(`/api/configs/${encodeURIComponent(name)}`);
            const data = await response.json();
            
            document.getElementById('configView').textContent = 
                JSON.stringify(data.config, null, 2);
            document.getElementById('configView').classList.remove('empty-state');
            document.getElementById('deleteBtn').style.display = 'inline-block';
            
            renderConfigList(allConfigs.filter(c => 
                c.name.includes(document.getElementById('searchInput').value)
            ));
        }
        
        async function saveConfig() {
            const name = document.getElementById('configName').value.trim();
            const jsonStr = document.getElementById('configJson').value.trim();
            const msgEl = document.getElementById('saveMessage');
            
            if (!name || !jsonStr) {
                msgEl.innerHTML = '<div class="message error">Name and config are required</div>';
                return;
            }
            
            let config;
            try {
                config = JSON.parse(jsonStr);
            } catch (e) {
                msgEl.innerHTML = '<div class="message error">Invalid JSON: ' + escapeHtml(e.message) + '</div>';
                return;
            }
            
            const response = await fetch('/api/configs', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name, config})
            });
            
            const data = await response.json();
            
            if (response.ok) {
                msgEl.innerHTML = '<div class="message success">' + escapeHtml(data.message) + '</div>';
                document.getElementById('configName').value = '';
                document.getElementById('configJson').value = '';
                loadConfigs();
            } else {
                msgEl.innerHTML = '<div class="message error">' + escapeHtml(data.detail) + '</div>';
            }
            
            setTimeout(() => msgEl.innerHTML = '', 3000);
        }
        
        async function deleteConfig() {
            if (!selectedConfig) return;
            if (!confirm(`Delete config "${selectedConfig}"?`)) return;
            
            await fetch(`/api/configs/${encodeURIComponent(selectedConfig)}`, {method: 'DELETE'});
            selectedConfig = null;
            document.getElementById('configView').innerHTML = 'Select a config from the list to view';
            document.getElementById('configView').classList.add('empty-state');
            document.getElementById('deleteBtn').style.display = 'none';
            loadConfigs();
        }
        
        function updateCompareSelects() {
            const html = '<option value="">Select config...</option>' + 
                allConfigs.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('');
            document.getElementById('config1Select').innerHTML = html;
            document.getElementById('config2Select').innerHTML = html;
        }
        
        async function updateDiff() {
            const name1 = document.getElementById('config1Select').value;
            const name2 = document.getElementById('config2Select').value;
            
            if (!name1 || !name2) {
                document.getElementById('diffOutput').innerHTML = 'Select two configs to compare';
                document.getElementById('diffOutput').classList.add('empty-state');
                document.getElementById('config1Content').textContent = '';
                document.getElementById('config2Content').textContent = '';
                return;
            }
            
            const response = await fetch(`/api/diff?config1=${encodeURIComponent(name1)}&config2=${encodeURIComponent(name2)}`);
            const data = await response.json();
            
            document.getElementById('config1Label').textContent = name1;
            document.getElementById('config2Label').textContent = name2;
            document.getElementById('config1Content').textContent = data.config1.content;
            document.getElementById('config2Content').textContent = data.config2.content;
            
            if (data.diff) {
                const diffHtml = data.diff.split('\\n').map(line => {
                    let cls = '';
                    if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-add';
                    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-remove';
                    else if (line.startsWith('@@')) cls = 'diff-range';
                    else if (line.startsWith('---') || line.startsWith('+++')) cls = 'diff-header';
                    return `<div class="diff-line ${cls}">${escapeHtml(line)}</div>`;
                }).join('');
                document.getElementById('diffOutput').innerHTML = diffHtml || '<div class="empty-state">No differences found</div>';
                document.getElementById('diffOutput').classList.remove('empty-state');
            } else {
                document.getElementById('diffOutput').innerHTML = 'No differences found';
                document.getElementById('diffOutput').classList.add('empty-state');
            }
        }
        
        function swapConfigs() {
            const s1 = document.getElementById('config1Select');
            const s2 = document.getElementById('config2Select');
            [s1.value, s2.value] = [s2.value, s1.value];
            updateDiff();
        }
        
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab[onclick="showTab('${tab}')"]`).classList.add('active');
            document.getElementById('viewTab').style.display = tab === 'view' ? 'block' : 'none';
            document.getElementById('compareTab').style.display = tab === 'compare' ? 'block' : 'none';
        }
        
        // Initialize
        loadConfigs();
    </script>
</body>
</html>
;//
"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the main UI."""
    return HTML_TEMPLATE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)