"""
Web-based Real-time Pareto Frontier Viewer

Serves a web interface at localhost:3000 that displays real-time Pareto graphs
using Plotly.js for visualization.

Usage:
    python web_pareto_viewer.py --data-url http://localhost:8888/data
    python web_pareto_viewer.py --data-file test_pareto_data.jsonl
"""

import argparse
import http.server
import socketserver
import json
import os
import threading
import time
import subprocess
from urllib.request import urlopen
from urllib.error import URLError
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FASOP root directory (parent of demo/)
FASOP_DIR = os.path.dirname(SCRIPT_DIR)

# Default data file path (relative to FASOP_DIR)
DEFAULT_DATA_FILE = os.path.join(FASOP_DIR, "test.jsonl")

def get_default_fasop_command(data_file: str) -> list:
    """Get the default FASOP command with proper paths."""
    return [
        "python", os.path.join(FASOP_DIR, "FASOP.py"),
        "--gpus", "A40", "8",
        "--model-type", "llama70b",
        "--dataset", "squad",
        "--gpu-per-node", "8",
        "--pp-partition-method", "minmax",
        "--pareto",
        "--pareto-gbs-max", "64",
        "--parsing",
        "--parsing-file", data_file,
        "--no-save-csv"
    ]

# HTML template with Plotly.js
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Pareto Frontier</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }
        h1 {
            font-size: 2rem;
            font-weight: 600;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.live { background: #10b981; }
        .status-dot.error { background: #ef4444; animation: none; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .chart-container {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0,0,0,0.3);
        }
        #pareto-chart {
            width: 100%;
            height: 600px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00d4ff;
        }
        .stat-label {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.6);
            margin-top: 5px;
        }
        .best-config {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .best-config h3 {
            color: #10b981;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .config-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }
        .config-item {
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .config-item .label {
            font-size: 0.75rem;
            color: rgba(255,255,255,0.5);
        }
        .config-item .value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
        }
        .search-btn {
            background: linear-gradient(135deg, #3b82f6 0%, #7c3aed 100%);
            border: none;
            color: #fff;
            padding: 12px 28px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .search-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        }
        .search-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .search-btn.running {
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
        }
        .search-btn .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .search-section {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
            padding: 15px 0;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .progress-section {
            margin-top: 15px;
            padding: 15px 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        .progress-stats {
            display: flex;
            gap: 20px;
        }
        .progress-stat {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .progress-stat .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .progress-stat .dot.valid { background: #3b82f6; }
        .progress-stat .dot.pareto { background: #10b981; }
        .progress-stat .dot.oom { background: #ef4444; }
        .progress-bar-container {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #10b981);
            border-radius: 4px;
            transition: width 0.3s ease;
            width: 0%;
        }
        .progress-text {
            text-align: center;
            margin-top: 8px;
            font-size: 0.85rem;
            color: rgba(255,255,255,0.7);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Real-time Pareto Frontier Analysis</h1>
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-dot live" id="status-dot"></div>
                    <span id="status-text">Connecting...</span>
                </div>
                <div class="status-item">
                    <span>⏱</span>
                    <span id="update-time">Elapsed: 0s</span>
                </div>
                <div class="status-item">
                    <span id="data-source">DATA_SOURCE_PLACEHOLDER</span>
                </div>
            </div>
            <div class="search-section">
                <button class="search-btn" id="search-btn" onclick="startSearch()">
                    <span id="search-icon">🔍</span>
                    <span id="search-text">Search</span>
                </button>
            </div>
            <div class="progress-section">
                <div class="progress-header">
                    <span id="progress-label">Search Progress</span>
                    <div class="progress-stats">
                        <div class="progress-stat">
                            <span class="dot valid"></span>
                            <span id="valid-count">Valid: 0</span>
                        </div>
                        <div class="progress-stat">
                            <span class="dot pareto"></span>
                            <span id="pareto-count">Pareto: 0</span>
                        </div>
                        <div class="progress-stat">
                            <span class="dot oom"></span>
                            <span id="oom-count">OOM: 0</span>
                        </div>
                    </div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
                <div class="progress-text" id="progress-text">0 / ? combinations</div>
            </div>
        </header>

        <div class="chart-container">
            <div id="pareto-chart"></div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-points">0</div>
                <div class="stat-label">Total Configurations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="pareto-points">0</div>
                <div class="stat-label">Pareto Frontier Points</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="best-throughput">0</div>
                <div class="stat-label">Best Throughput (samples/s)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="min-cost">0</div>
                <div class="stat-label">Minimum Cost ($)</div>
            </div>
        </div>

        <div class="best-config" id="best-config-section" style="display: none;">
            <h3>⭐ Best Configuration (Highest Throughput)</h3>
            <div class="config-details" id="best-config-details"></div>
        </div>
    </div>

    <script>
        const DATA_URL = '/api/data';
        const REFRESH_INTERVAL = 1000; // 1 second

        let chart = null;
        let lastDataCount = 0;
        let searchStartTime = null;
        let searchIntervalId = null;
        let noChangeCount = 0;
        let totalSearchCount = null; // Total combinations from meta
        let totalDataCount = 0; // Current data count (including OOM)
        let oomCount = 0;
        const STOP_AFTER_NO_CHANGE = 30; // Stop after 30 consecutive no-change fetches (30 seconds)
        let isSearchRunning = false;

        // Start FASOP search
        async function startSearch() {
            const btn = document.getElementById('search-btn');
            const icon = document.getElementById('search-icon');
            const text = document.getElementById('search-text');

            if (isSearchRunning) {
                // Stop search
                try {
                    const response = await fetch('/api/search/stop', { method: 'POST' });
                    const result = await response.json();
                    if (result.status === 'stopped') {
                        isSearchRunning = false;
                        btn.classList.remove('running');
                        icon.innerHTML = '🔍';
                        text.textContent = 'Search';
                        btn.disabled = false;
                    }
                } catch (error) {
                    console.error('Stop error:', error);
                }
                return;
            }

            // Start search
            isSearchRunning = true;
            btn.classList.add('running');
            icon.innerHTML = '<div class="spinner"></div>';
            text.textContent = 'Running...';

            // Reset state for new search
            lastDataCount = 0;
            searchStartTime = null;
            noChangeCount = 0;
            totalSearchCount = null;
            totalDataCount = 0;
            oomCount = 0;

            // Restart interval if stopped
            if (searchIntervalId === null) {
                searchIntervalId = setInterval(fetchData, REFRESH_INTERVAL);
            }

            try {
                const response = await fetch('/api/search', { method: 'POST' });
                const result = await response.json();

                if (result.status === 'started') {
                    document.getElementById('status-text').textContent = 'Search started...';
                    document.getElementById('status-dot').className = 'status-dot live';
                    // Update button to show "Stop" option
                    text.textContent = 'Stop';
                    icon.innerHTML = '⏹';
                } else if (result.status === 'already_running') {
                    document.getElementById('status-text').textContent = 'Search already running';
                    text.textContent = 'Stop';
                    icon.innerHTML = '⏹';
                } else {
                    throw new Error(result.error || 'Unknown error');
                }
            } catch (error) {
                console.error('Search error:', error);
                isSearchRunning = false;
                btn.classList.remove('running');
                icon.innerHTML = '🔍';
                text.textContent = 'Search';
                document.getElementById('status-dot').className = 'status-dot error';
                document.getElementById('status-text').textContent = 'Search failed: ' + error.message;
            }
        }

        // Check search status periodically
        async function checkSearchStatus() {
            if (!isSearchRunning) return;

            try {
                const response = await fetch('/api/search/status');
                const result = await response.json();

                if (!result.running && result.status === 'completed') {
                    // Search completed
                    isSearchRunning = false;
                    const btn = document.getElementById('search-btn');
                    const icon = document.getElementById('search-icon');
                    const text = document.getElementById('search-text');
                    btn.classList.remove('running');
                    icon.innerHTML = '🔍';
                    text.textContent = 'Search';

                    if (result.exit_code === 0) {
                        document.getElementById('status-text').textContent = 'Search completed successfully';
                    } else {
                        document.getElementById('status-text').textContent = `Search finished (exit code: ${result.exit_code})`;
                    }
                }
            } catch (error) {
                console.error('Status check error:', error);
            }
        }

        // Compute Pareto frontier
        function computeParetoFrontier(costs, throughputs) {
            const n = costs.length;
            if (n === 0) return [];

            // Create array of indices sorted by cost, then by throughput (descending)
            const indices = Array.from({length: n}, (_, i) => i);
            indices.sort((a, b) => {
                if (costs[a] !== costs[b]) return costs[a] - costs[b];
                return throughputs[b] - throughputs[a];
            });

            const isPareto = new Array(n).fill(false);
            let maxThroughput = -Infinity;

            for (const idx of indices) {
                if (throughputs[idx] > maxThroughput) {
                    isPareto[idx] = true;
                    maxThroughput = throughputs[idx];
                }
            }

            return isPareto;
        }

        // Initialize chart
        function initChart() {
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.02)',
                font: { color: '#fff', family: '-apple-system, BlinkMacSystemFont, sans-serif' },
                xaxis: {
                    title: { text: 'Training Cost ($)', font: { size: 14 } },
                    gridcolor: 'rgba(255,255,255,0.1)',
                    zerolinecolor: 'rgba(255,255,255,0.2)',
                },
                yaxis: {
                    title: { text: 'Throughput (samples/s)', font: { size: 14 } },
                    gridcolor: 'rgba(255,255,255,0.1)',
                    zerolinecolor: 'rgba(255,255,255,0.2)',
                },
                showlegend: true,
                legend: {
                    x: 1,
                    xanchor: 'right',
                    y: 1,
                    bgcolor: 'rgba(0,0,0,0.5)',
                    bordercolor: 'rgba(255,255,255,0.2)',
                    borderwidth: 1
                },
                margin: { t: 40, r: 40, b: 60, l: 60 },
                hovermode: 'closest'
            };

            Plotly.newPlot('pareto-chart', [], layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            });
            chart = document.getElementById('pareto-chart');
        }

        // Normalize field names (support both FASOP JSONL and CSV formats)
        function normalizeData(data) {
            return data.map(d => ({
                ...d,
                'cost($)': d['cost($)'] ?? d.cost ?? 0,
                'throughput(samples/s)': d['throughput(samples/s)'] ?? d.throughput ?? 0,
                'total_time(s)': d['total_time(s)'] ?? d.total_time_seconds ?? 0,
                is_oom: d.is_oom ?? false
            }));
        }

        // Update chart with new data
        function updateChart(data) {
            if (!data || data.length === 0) return;

            // Normalize field names
            const normalizedData = normalizeData(data);

            // Filter out OOM
            const validData = normalizedData.filter(d => !d.is_oom && d['cost($)'] > 0 && d['throughput(samples/s)'] > 0);
            if (validData.length === 0) return;

            const costs = validData.map(d => d['cost($)']);
            const throughputs = validData.map(d => d['throughput(samples/s)']);

            // Compute Pareto frontier
            const isPareto = computeParetoFrontier(costs, throughputs);

            // Separate Pareto and non-Pareto points
            const paretoData = validData.filter((_, i) => isPareto[i]);
            const otherData = validData.filter((_, i) => !isPareto[i]);

            // Create hover text
            const createHoverText = (d) =>
                `tp=${d.tp} dp=${d.dp} pp=${d.pp}<br>` +
                `gbs=${d.gbs} mbs=${d.mbs}<br>` +
                `Cost: $${d['cost($)'].toFixed(2)}<br>` +
                `Throughput: ${d['throughput(samples/s)'].toFixed(2)}`;

            // Non-Pareto points (gray)
            const traceOther = {
                x: otherData.map(d => d['cost($)']),
                y: otherData.map(d => d['throughput(samples/s)']),
                mode: 'markers',
                type: 'scatter',
                name: 'Other Configurations',
                marker: {
                    color: 'rgba(150, 150, 150, 0.5)',
                    size: 8
                },
                text: otherData.map(createHoverText),
                hoverinfo: 'text'
            };

            // Pareto frontier points (blue)
            const paretoSorted = [...paretoData].sort((a, b) => a['cost($)'] - b['cost($)']);
            const tracePareto = {
                x: paretoSorted.map(d => d['cost($)']),
                y: paretoSorted.map(d => d['throughput(samples/s)']),
                mode: 'markers+lines',
                type: 'scatter',
                name: 'Pareto Frontier',
                marker: {
                    color: '#3b82f6',
                    size: 12,
                    line: { color: '#1d4ed8', width: 2 }
                },
                line: {
                    color: 'rgba(59, 130, 246, 0.5)',
                    width: 2,
                    dash: 'dot'
                },
                text: paretoSorted.map(createHoverText),
                hoverinfo: 'text'
            };

            // Best point (highest throughput)
            const bestIdx = throughputs.indexOf(Math.max(...throughputs));
            const bestPoint = validData[bestIdx];
            const traceBest = {
                x: [bestPoint['cost($)']],
                y: [bestPoint['throughput(samples/s)']],
                mode: 'markers',
                type: 'scatter',
                name: 'Best Throughput',
                marker: {
                    color: '#10b981',
                    size: 20,
                    symbol: 'star',
                    line: { color: '#fff', width: 2 }
                },
                text: [createHoverText(bestPoint)],
                hoverinfo: 'text'
            };

            // Latest point (most recent search result) - Orange
            const latestPoint = validData[validData.length - 1];
            const traceLatest = {
                x: [latestPoint['cost($)']],
                y: [latestPoint['throughput(samples/s)']],
                mode: 'markers',
                type: 'scatter',
                name: 'Latest Search',
                marker: {
                    color: '#f97316',
                    size: 16,
                    symbol: 'diamond',
                    line: { color: '#fff', width: 2 }
                },
                text: [createHoverText(latestPoint)],
                hoverinfo: 'text'
            };

            Plotly.react('pareto-chart', [traceOther, tracePareto, traceBest, traceLatest], chart.layout);

            // Update stats
            document.getElementById('total-points').textContent = validData.length;
            document.getElementById('pareto-points').textContent = paretoData.length;
            document.getElementById('best-throughput').textContent = Math.max(...throughputs).toFixed(2);
            document.getElementById('min-cost').textContent = '$' + Math.min(...costs).toFixed(2);

            // Update progress bar and counts
            totalDataCount = normalizedData.length; // All data including OOM (global var)
            oomCount = normalizedData.filter(d => d.is_oom).length;
            const validCount = validData.length;
            const paretoCount = paretoData.length;

            document.getElementById('valid-count').textContent = `Valid: ${validCount}`;
            document.getElementById('pareto-count').textContent = `Pareto: ${paretoCount}`;
            document.getElementById('oom-count').textContent = `OOM: ${oomCount}`;

            // Update progress bar
            if (totalSearchCount !== null && totalSearchCount > 0) {
                const progressPercent = Math.min(100, (totalDataCount / totalSearchCount) * 100);
                document.getElementById('progress-bar').style.width = `${progressPercent}%`;
                document.getElementById('progress-text').textContent =
                    `${totalDataCount} / ${totalSearchCount} combinations (${progressPercent.toFixed(1)}%)`;
            } else {
                document.getElementById('progress-text').textContent =
                    `${totalDataCount} combinations (total unknown)`;
            }

            // Update best config section
            const bestConfigSection = document.getElementById('best-config-section');
            const bestConfigDetails = document.getElementById('best-config-details');
            bestConfigSection.style.display = 'block';

            const configItems = [
                { label: 'TP', value: bestPoint.tp },
                { label: 'DP', value: bestPoint.dp },
                { label: 'PP', value: bestPoint.pp },
                { label: 'GBS', value: bestPoint.gbs },
                { label: 'MBS', value: bestPoint.mbs },
                { label: 'Cost', value: '$' + bestPoint['cost($)'].toFixed(2) },
                { label: 'Throughput', value: bestPoint['throughput(samples/s)'].toFixed(2) },
                { label: 'Time', value: (bestPoint['total_time(s)'] || 0).toFixed(1) + 's' }
            ];

            bestConfigDetails.innerHTML = configItems.map(item => `
                <div class="config-item">
                    <div class="label">${item.label}</div>
                    <div class="value">${item.value}</div>
                </div>
            `).join('');

            // Update status
            const newCount = validData.length - lastDataCount;

            // Track search elapsed time
            if (searchStartTime === null && validData.length > 0) {
                searchStartTime = Date.now();
            }

            // Update elapsed time display
            if (searchStartTime !== null) {
                const elapsedMs = Date.now() - searchStartTime;
                const elapsedSec = Math.floor(elapsedMs / 1000);
                const hours = Math.floor(elapsedSec / 3600);
                const minutes = Math.floor((elapsedSec % 3600) / 60);
                const seconds = elapsedSec % 60;
                const timeStr = hours > 0
                    ? `${hours}h ${minutes}m ${seconds}s`
                    : minutes > 0
                        ? `${minutes}m ${seconds}s`
                        : `${seconds}s`;
                document.getElementById('update-time').textContent = `Elapsed: ${timeStr}`;
            }

            // Check if search is complete
            const isComplete = totalSearchCount !== null && totalDataCount >= totalSearchCount;

            if (isComplete) {
                // Search reached total count - definitely complete
                if (searchIntervalId !== null) {
                    clearInterval(searchIntervalId);
                    searchIntervalId = null;
                }
                document.getElementById('status-dot').className = 'status-dot';
                document.getElementById('status-dot').style.background = '#10b981';
                document.getElementById('status-dot').style.animation = 'none';
                document.getElementById('status-text').textContent = `Complete (${validData.length} valid / ${totalDataCount} total)`;
            } else if (newCount === 0 && lastDataCount > 0) {
                // No new data - might be computing or complete
                noChangeCount++;
                if (noChangeCount >= STOP_AFTER_NO_CHANGE && searchIntervalId !== null) {
                    clearInterval(searchIntervalId);
                    searchIntervalId = null;
                    document.getElementById('status-dot').className = 'status-dot';
                    document.getElementById('status-dot').style.background = '#6b7280';
                    document.getElementById('status-dot').style.animation = 'none';
                    document.getElementById('status-text').textContent = `Stopped (${validData.length} points, no updates for 30s)`;
                }
            } else {
                noChangeCount = 0;
            }

            lastDataCount = validData.length;

            document.getElementById('status-text').textContent =
                `Live (${validData.length} points${newCount > 0 ? ', +' + newCount + ' new' : ''})`;
        }

        // Fetch data from server
        async function fetchData() {
            try {
                const response = await fetch(DATA_URL);
                if (!response.ok) throw new Error('Network response was not ok');

                const text = await response.text();
                const lines = text.trim().split('\\n').filter(line => line.trim());
                const allParsed = lines.map(line => {
                    try {
                        return JSON.parse(line);
                    } catch (e) {
                        return null;
                    }
                }).filter(d => d !== null);

                // Separate meta data and actual data
                const metaData = allParsed.filter(d => d.type === 'meta' || d.type === 'pareto_meta');
                const data = allParsed.filter(d => !d.type);

                // Extract total count and config info from meta
                for (const meta of metaData) {
                    if (meta.pareto_total_count && totalSearchCount === null) {
                        totalSearchCount = meta.pareto_total_count;
                    } else if (meta.total_count && totalSearchCount === null) {
                        totalSearchCount = meta.total_count;
                    }
                    // Update data source display with model_type and gpu_cluster
                    if (meta.model_type && meta.gpu_cluster) {
                        const gpuStr = Object.entries(meta.gpu_cluster)
                            .map(([k, v]) => `${k}:${v}`).join(', ');
                        document.getElementById('data-source').textContent =
                            `model: ${meta.model_type}, gpus: ${gpuStr}`;
                    }
                }

                updateChart(data);

                document.getElementById('status-dot').className = 'status-dot live';
            } catch (error) {
                console.error('Fetch error:', error);
                document.getElementById('status-dot').className = 'status-dot error';
                document.getElementById('status-text').textContent = 'Connection Error';
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initChart();
            fetchData();
            searchIntervalId = setInterval(fetchData, REFRESH_INTERVAL);
            // Check search status periodically
            setInterval(checkSearchStatus, 2000);
        });
    </script>
</body>
</html>
"""


class WebParetoHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for web-based Pareto viewer."""

    data_source = None  # Will be set by server
    data_url = None
    data_file = None
    fasop_process = None  # Track running FASOP process
    fasop_command = None  # FASOP command to run

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_html()
        elif self.path == '/api/data':
            self.serve_data()
        elif self.path == '/api/search/status':
            self.get_search_status()
        elif self.path == '/health':
            self.send_json({'status': 'ok'})
        else:
            self.send_error(404)

    def get_search_status(self):
        """Get status of FASOP search process."""
        if WebParetoHandler.fasop_process is None:
            self.send_json({'running': False, 'status': 'not_started'})
        elif WebParetoHandler.fasop_process.poll() is None:
            self.send_json({'running': True, 'status': 'running', 'pid': WebParetoHandler.fasop_process.pid})
        else:
            exit_code = WebParetoHandler.fasop_process.returncode
            WebParetoHandler.fasop_process = None
            self.send_json({'running': False, 'status': 'completed', 'exit_code': exit_code})

    def do_POST(self):
        if self.path == '/api/search':
            self.start_search()
        elif self.path == '/api/search/stop':
            self.stop_search()
        else:
            self.send_error(404)

    def start_search(self):
        """Start FASOP search process."""
        # Check if already running
        if WebParetoHandler.fasop_process is not None:
            if WebParetoHandler.fasop_process.poll() is None:
                self.send_json({'status': 'already_running'})
                return
            else:
                WebParetoHandler.fasop_process = None

        # Clear existing data file
        data_file = WebParetoHandler.data_file or DEFAULT_DATA_FILE
        try:
            if os.path.exists(data_file):
                os.remove(data_file)
        except Exception as e:
            print(f"Warning: Could not clear {data_file}: {e}")

        # Get the command to run (use absolute path for data file)
        command = WebParetoHandler.fasop_command or get_default_fasop_command(data_file)

        try:
            # Start FASOP process in background (run from FASOP_DIR)
            WebParetoHandler.fasop_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=FASOP_DIR  # Run FASOP from the correct directory
            )

            # Start a thread to monitor the process output
            def monitor_process():
                process = WebParetoHandler.fasop_process
                if process and process.stdout:
                    for line in process.stdout:
                        print(f"[FASOP] {line.rstrip()}")
                    process.stdout.close()

            monitor_thread = threading.Thread(target=monitor_process, daemon=True)
            monitor_thread.start()

            self.send_json({'status': 'started', 'pid': WebParetoHandler.fasop_process.pid})

        except Exception as e:
            self.send_json({'status': 'error', 'error': str(e)}, status=500)

    def stop_search(self):
        """Stop FASOP search process."""
        if WebParetoHandler.fasop_process is not None:
            if WebParetoHandler.fasop_process.poll() is None:
                WebParetoHandler.fasop_process.terminate()
                try:
                    WebParetoHandler.fasop_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    WebParetoHandler.fasop_process.kill()
            WebParetoHandler.fasop_process = None
            self.send_json({'status': 'stopped'})
        else:
            self.send_json({'status': 'not_running'})

    def serve_html(self):
        """Serve the main HTML page."""
        html = HTML_TEMPLATE.replace('DATA_SOURCE_PLACEHOLDER',
                                     WebParetoHandler.data_source or 'Unknown')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def serve_data(self):
        """Serve data from URL or file."""
        try:
            if WebParetoHandler.data_url:
                # Fetch from URL
                with urlopen(WebParetoHandler.data_url, timeout=5) as response:
                    content = response.read().decode('utf-8')
            elif WebParetoHandler.data_file:
                # Read from file (return empty if not exists yet)
                if os.path.exists(WebParetoHandler.data_file):
                    with open(WebParetoHandler.data_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    content = ''
            else:
                content = ''

            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        except Exception as e:
            self.send_json({'error': str(e)}, status=500)

    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def run_server(port: int, data_url: str = None, data_file: str = None):
    """Run the web server."""
    # Configure handler
    WebParetoHandler.data_url = data_url
    WebParetoHandler.data_file = data_file
    WebParetoHandler.data_source = data_url or data_file or 'None'

    with ThreadedTCPServer(("", port), WebParetoHandler) as server:
        print("=" * 60)
        print("  Real-time Pareto Frontier Web Viewer")
        print("=" * 60)
        print()
        print(f"  Data Source: {WebParetoHandler.data_source}")
        print()
        print(f"  Open in browser:")
        print(f"    http://localhost:{port}")
        print()
        print("  Press Ctrl+C to stop.")
        print("=" * 60)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Web-based Real-time Pareto Frontier Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python web_pareto_viewer.py
      Start viewer with default settings (data file: {DEFAULT_DATA_FILE})

  python web_pareto_viewer.py --data-file ../main_logs/results.jsonl
      Monitor a specific JSONL file

  python web_pareto_viewer.py --port 8080
      Use a different port

Note: FASOP will run from {FASOP_DIR} when using the Search button.
"""
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=3000,
        help='Server port (default: 3000)'
    )

    parser.add_argument(
        '--data-url', '-u',
        type=str,
        help='URL to fetch data from (e.g., http://localhost:8888/data)'
    )

    parser.add_argument(
        '--data-file', '-f',
        type=str,
        help=f'Local file to read data from (default: {DEFAULT_DATA_FILE})'
    )

    args = parser.parse_args()

    # Default to DEFAULT_DATA_FILE if no data source specified
    data_file = args.data_file
    if not args.data_url and not args.data_file:
        data_file = DEFAULT_DATA_FILE
        print(f"No data source specified, defaulting to: {data_file}")
    elif data_file and not os.path.isabs(data_file):
        # Convert relative paths to absolute (relative to current working directory)
        data_file = os.path.abspath(data_file)

    # File doesn't need to exist yet - it will be created by FASOP search
    if data_file and os.path.exists(data_file):
        print(f"Data file exists: {data_file}")
    elif data_file:
        print(f"Data file will be created when search starts: {data_file}")

    print(f"FASOP directory: {FASOP_DIR}")

    run_server(args.port, args.data_url, data_file)
    return 0


if __name__ == '__main__':
    exit(main())
