/* ================================================
   AI Heatwave & Anomaly Prediction — Frontend JS
   Improved with model metrics, confidence bands,
   and IMD severity classification
   ================================================ */

const PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(21,28,44,0.6)",
    font: { family: "Inter, sans-serif", color: "#94a3b8", size: 12 },
    margin: { t: 10, r: 24, b: 44, l: 52 },
    xaxis: {
        gridcolor: "rgba(255,255,255,0.04)",
        linecolor: "rgba(255,255,255,0.08)",
        tickfont: { size: 11 },
    },
    yaxis: {
        gridcolor: "rgba(255,255,255,0.04)",
        linecolor: "rgba(255,255,255,0.08)",
        tickfont: { size: 11 },
        title: { text: "Temperature (°C)", font: { size: 12 } },
    },
    legend: {
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "left",
        x: 0,
        font: { size: 11 },
        bgcolor: "rgba(0,0,0,0)",
    },
    hovermode: "x unified",
    hoverlabel: {
        bgcolor: "#1e293b",
        bordercolor: "rgba(255,255,255,0.1)",
        font: { family: "Inter", size: 12, color: "#f1f5f9" },
    },
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    displaylogo: false,
};

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchAnalyzeWithRetry(payload, maxRetries = 2) {
    let lastError = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const controller = new AbortController();
        const timeoutMs = 180000;
        const timer = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal: controller.signal,
            });
            clearTimeout(timer);
            return res;
        } catch (err) {
            clearTimeout(timer);
            lastError = err;
            if (attempt < maxRetries) {
                await sleep((attempt + 1) * 800);
            }
        }
    }

    throw lastError || new Error("Request failed");
}

// =========== Main Analysis ===========
async function runAnalysis() {
    const city = document.getElementById("citySelect").value;
    const startYear = document.getElementById("startYear").value;
    const endYear = document.getElementById("endYear").value;
    const btn = document.getElementById("analyzeBtn");
    const overlay = document.getElementById("loadingOverlay");

    btn.disabled = true;
    overlay.classList.remove("hidden");

    try {
        const res = await fetchAnalyzeWithRetry({ city, startYear, endYear }, 2);

        let data = {};
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            data = await res.json();
        }

        if (!res.ok) {
            const errMsg = data.error || `Server error (${res.status})`;
            alert("Error: " + errMsg);
            return;
        }

        if (!data.success) {
            alert("Error: " + (data.error || "Unknown error"));
            return;
        }

        updateStats(data.stats);
        updateModelMetrics(data.model_metrics, data.imd_thresholds);
        renderTempChart(data);
        renderAnomalyChart(data);
        renderHeatwaveEvents(data.heatwave_events);
    } catch (err) {
        const msg =
            err && err.name === "AbortError"
                ? "Analysis timed out. Please try a smaller date range."
                : "Could not reach local server. It may be starting or restarting. Please try again.";
        alert(msg);
    } finally {
        btn.disabled = false;
        overlay.classList.add("hidden");
    }
}

// =========== Stat Cards ===========
function updateStats(stats) {
    animateValue("valLatestTemp", stats.latest_temp, "°C");
    animateValue("valAnomalies", stats.anomaly_count, "");
    animateValue("valHeatwave", stats.heatwave_days, "");
    animateValue("valForecastMax", stats.forecast_max, "°C");
    animateValue("valRecordHigh", stats.record_high, "°C");
    animateValue("valThreshold", stats.anomaly_threshold, "°C");
}

function animateValue(elementId, targetValue, suffix) {
    const el = document.getElementById(elementId);
    const isFloat = String(targetValue).includes(".");
    const start = 0;
    const duration = 800;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        const current = start + (targetValue - start) * ease;

        if (isFloat) {
            el.textContent = current.toFixed(1) + suffix;
        } else {
            el.textContent = Math.round(current) + suffix;
        }

        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// =========== Model Metrics ===========
function updateModelMetrics(metrics, imdThresholds) {
    const section = document.getElementById("metricsSection");
    section.classList.remove("hidden");

    if (metrics && metrics.mae !== undefined) {
        document.getElementById("metricMAEBase").textContent =
            metrics.mae_baseline !== undefined ? metrics.mae_baseline.toFixed(2) : "—";
        document.getElementById("metricMAE").textContent = metrics.mae.toFixed(2);
        document.getElementById("metricRMSE").textContent = metrics.rmse.toFixed(2);
        document.getElementById("metricR2").textContent = `${(metrics.r2 * 100).toFixed(2)}%`;
        document.getElementById("metricF1").textContent =
            metrics.f1 !== undefined ? metrics.f1.toFixed(2) : "—";
        document.getElementById("metricTrain").textContent =
            metrics.train_size ? metrics.train_size.toLocaleString() : "—";
        document.getElementById("metricTest").textContent =
            metrics.test_size ? metrics.test_size.toLocaleString() : "—";
        document.getElementById("metricFeatures").textContent =
            metrics.n_features || "—";

        // Color-code R² value
        const r2El = document.getElementById("metricR2");
        if (metrics.r2 >= 0.9) r2El.classList.add("metric-excellent");
        else if (metrics.r2 >= 0.7) r2El.classList.add("metric-good");
        else r2El.classList.add("metric-fair");
    }

    // IMD thresholds info
    if (imdThresholds) {
        const imdEl = document.getElementById("imdInfo");
        const typeLabel =
            imdThresholds.station_type === "coastal" ? "🌊 Coastal" : "🏜️ Plains";
        imdEl.innerHTML = `
            <span class="imd-tag">${typeLabel} Station</span>
            <span class="imd-detail">Min: ${imdThresholds.min_actual}°C</span>
            <span class="imd-detail">HW: +${imdThresholds.heatwave_departure}°C</span>
            <span class="imd-detail severe-tag">Severe: +${imdThresholds.severe_departure}°C</span>
        `;
    }
}

// =========== Temperature Chart ===========
function renderTempChart(data) {
    const hist = data.historical;
    const fc = data.forecast;

    const traces = [];

    // IMD Absolute Minimum Threshold Line
    if (data.imd_thresholds && data.imd_thresholds.min_actual) {
        const threshold = data.imd_thresholds.min_actual;
        const allDates = hist.dates.concat(fc.dates);
        traces.push({
            x: [allDates[0], allDates[allDates.length - 1]],
            y: [threshold, threshold],
            type: "scatter",
            mode: "lines",
            name: "IMD Min Threshold",
            line: { color: "#ef4444", width: 1.5, dash: "dot" },
            hovertemplate: "%{y:.1f}°C<extra>IMD Threshold</extra>",
            showlegend: false,
        });
    }

    // Historical temperature
    traces.push({
        x: hist.dates,
        y: hist.temps,
        type: "scatter",
        mode: "lines",
        name: "Historical Temperature",
        line: { color: "#3b82f6", width: 1.5 },
        hovertemplate: "%{y:.1f}°C<extra>Historical</extra>",
    });

    // Forecast confidence interval (upper)
    traces.push({
        x: fc.dates,
        y: fc.upper,
        type: "scatter",
        mode: "lines",
        name: "95% CI Upper",
        line: { color: "rgba(239,68,68,0.0)", width: 0 },
        showlegend: false,
        hoverinfo: "skip",
    });

    // Forecast confidence interval (lower) — fill to upper
    traces.push({
        x: fc.dates,
        y: fc.lower,
        type: "scatter",
        mode: "lines",
        name: "95% Confidence Interval",
        line: { color: "rgba(239,68,68,0.0)", width: 0 },
        fill: "tonexty",
        fillcolor: "rgba(239,68,68,0.12)",
        hoverinfo: "skip",
    });

    // Forecast temperature
    traces.push({
        x: fc.dates,
        y: fc.temps,
        type: "scatter",
        mode: "lines",
        name: "Forecast Temperature",
        line: { color: "#ef4444", width: 2 },
        hovertemplate: "%{y:.1f}°C<extra>Forecast</extra>",
    });

    // Climatology (avg)
    traces.push({
        x: hist.dates,
        y: hist.climatology,
        type: "scatter",
        mode: "lines",
        name: "Climatology (Avg)",
        line: { color: "#64748b", width: 1, dash: "dash" },
        hovertemplate: "%{y:.1f}°C<extra>Climatology</extra>",
    });

    // Historical heatwave days as triangles
    const hwHistDates = [], hwHistTemps = [];
    for (let i = 0; i < hist.dates.length; i++) {
        if (hist.heatwave[i]) {
            hwHistDates.push(hist.dates[i]);
            hwHistTemps.push(hist.temps[i]);
        }
    }
    traces.push({
        x: hwHistDates,
        y: hwHistTemps,
        type: "scatter",
        mode: "markers",
        name: "Heatwave Days",
        marker: { color: "#f97316", size: 8, symbol: "triangle-up", line: { color: "#ea580c", width: 1 } },
        hovertemplate: "%{y:.1f}°C<extra>Heatwave</extra>",
    });

    // Forecast heatwave days
    const hwFcDates = [], hwFcTemps = [];
    for (let i = 0; i < fc.dates.length; i++) {
        if (fc.heatwave[i]) {
            hwFcDates.push(fc.dates[i]);
            hwFcTemps.push(fc.temps[i]);
        }
    }
    if (hwFcDates.length > 0) {
        traces.push({
            x: hwFcDates,
            y: hwFcTemps,
            type: "scatter",
            mode: "markers",
            name: "Heatwave (Forecast)",
            marker: { color: "#fb923c", size: 8, symbol: "triangle-up", line: { color: "#f97316", width: 1 } },
            hovertemplate: "%{y:.1f}°C<extra>Heatwave (FC)</extra>",
        });
    }

    // Anomaly markers (statistical — Z-score + Isolation Forest)
    const anomDates = [], anomTemps = [];
    for (let i = 0; i < hist.dates.length; i++) {
        if (hist.anomalies[i]) {
            anomDates.push(hist.dates[i]);
            anomTemps.push(hist.temps[i]);
        }
    }
    traces.push({
        x: anomDates,
        y: anomTemps,
        type: "scatter",
        mode: "markers",
        name: "Anomalies (Z+IF)",
        marker: { color: "#a855f7", size: 6, opacity: 0.7 },
        hovertemplate: "%{y:.1f}°C<extra>Anomaly</extra>",
    });

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        showlegend: false,
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: { text: "Temperature (°C)", font: { size: 12 } } },
    };

    Plotly.newPlot("tempChart", traces, layout, PLOTLY_CONFIG);
}

// =========== Anomaly Chart ===========
function renderAnomalyChart(data) {
    const hist = data.historical;
    const fc = data.forecast;
    const threshold = data.threshold;

    const allDates = hist.dates.concat(fc.dates);
    const allVals = hist.anomaly_vals.concat(fc.anomaly_vals);

    const colors = allVals.map((v) =>
        v > threshold ? "rgba(249,115,22,0.8)" : "rgba(59,130,246,0.5)"
    );

    const traces = [
        {
            x: allDates,
            y: allVals,
            type: "bar",
            name: "Anomaly",
            marker: { color: colors },
            hovertemplate: "%{y:.2f}°C<extra>Anomaly</extra>",
        },
        {
            x: [allDates[0], allDates[allDates.length - 1]],
            y: [threshold, threshold],
            type: "scatter",
            mode: "lines",
            name: "95th Percentile Threshold",
            line: { color: "#f97316", width: 1.5, dash: "dash" },
            hovertemplate: "%{y:.2f}°C<extra>Threshold</extra>",
        },
    ];

    // Add Z-score line if available
    if (hist.zscore) {
        traces.push({
            x: hist.dates,
            y: hist.zscore,
            type: "scatter",
            mode: "lines",
            name: "Z-score",
            yaxis: "y2",
            line: { color: "#a855f7", width: 1, dash: "dot" },
            opacity: 0.6,
            hovertemplate: "Z: %{y:.2f}<extra>Z-score</extra>",
        });
    }

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: { text: "Anomaly (°C)", font: { size: 12 } } },
        yaxis2: {
            title: { text: "Z-score", font: { size: 11, color: "#a855f7" } },
            overlaying: "y",
            side: "right",
            gridcolor: "rgba(0,0,0,0)",
            tickfont: { size: 10, color: "#a855f7" },
        },
        bargap: 0.05,
        showlegend: true,
    };

    Plotly.newPlot("anomalyChart", traces, layout, PLOTLY_CONFIG);
}

// =========== Heatwave Event Cards ===========
function renderHeatwaveEvents(events) {
    const grid = document.getElementById("eventsGrid");

    if (!events || events.length === 0) {
        grid.innerHTML = '<div class="events-placeholder">No heatwave events detected in the selected range.</div>';
        return;
    }

    let html = "";
    events.forEach((ev, i) => {
        const isHist = ev.label === "HISTORICAL";
        const badgeClass = isHist ? "badge-historical" : "badge-predicted";
        const accentClass = isHist ? "hist-accent" : "pred-accent";

        // Severity badge
        let severityHtml = "";
        if (ev.severity === "SEVERE") {
            severityHtml = '<span class="severity-badge severity-severe">SEVERE</span>';
        } else if (ev.severity === "HEATWAVE") {
            severityHtml = '<span class="severity-badge severity-hw">IMD HW</span>';
        }

        html += `
            <div class="event-card" style="animation-delay: ${i * 0.04}s">
                <div class="event-card-accent ${accentClass}"></div>
                <div class="event-card-header">
                    <div class="event-badges">
                        <span class="event-badge ${badgeClass}">${ev.label}</span>
                        ${severityHtml}
                    </div>
                    <span class="event-duration">${ev.duration} days</span>
                </div>
                <div class="event-dates">
                    <span class="date-icon">📅</span>
                    ${ev.start_display} → ${ev.end_display}
                </div>
                <div class="event-stats">
                    <span class="event-stat">Peak: <strong>${ev.peak}°C</strong></span>
                    <span class="event-stat">Avg: <strong>${ev.avg}°C</strong></span>
                </div>
            </div>
        `;
    });

    grid.innerHTML = html;
}
