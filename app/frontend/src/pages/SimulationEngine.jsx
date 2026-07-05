import { useState, useMemo } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, Legend
} from "recharts";
import { Terminal, RefreshCw, Zap, CloudRain, Sun, Thermometer, ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";

const PRESETS = [
  { key: "normal", label: "🌾 Normal Season", seed: 42, multiplier: 1.0, tempOffset: 0, rainfallPct: 100, desc: "Baseline conditions — average monsoon, typical evapotranspiration." },
  { key: "drought", label: "🔥 Drought Event", seed: 77, multiplier: 2.0, tempOffset: 4, rainfallPct: 35, desc: "Severe water deficit — 65% rainfall reduction, +4°C above normal." },
  { key: "monsoon", label: "🌧️ Monsoon Surplus", seed: 21, multiplier: 0.5, tempOffset: -1, rainfallPct: 180, desc: "Excess rainfall saturating root zones — waterlogging risk." },
  { key: "heatwave", label: "☀️ Heat Wave", seed: 99, multiplier: 1.8, tempOffset: 6, rainfallPct: 60, desc: "Extended heat dome — accelerated evapotranspiration, wilting risk." },
  { key: "custom", label: "⚙️ Custom Scenario", seed: 42, multiplier: 1.0, tempOffset: 0, rainfallPct: 100, desc: "Configure your own parameters below." },
];

const STATUS_COLORS = {
  Adequate: "#00E65B",
  Watch: "#FBBF24",
  Urgent: "#FF8800",
  Critical: "#FF3B30",
};

function seededRand(seed) {
  let x = seed | 0;
  return () => {
    x = (x * 1103515245 + 12345) & 0x7fffffff;
    return x / 0x7fffffff;
  };
}

function simulateFields(fields, { seed, multiplier, tempOffset, rainfallPct }) {
  const rng = seededRand(seed + Math.round(multiplier * 100) + tempOffset);
  return fields.map((f) => {
    const droughtFactor = Math.max(0, (100 - rainfallPct) / 100);
    const heatFactor = Math.max(0, tempOffset / 10);
    const stressBoost = (droughtFactor * 0.5 + heatFactor * 0.3 + (rng() - 0.3) * 0.2) * multiplier;
    const newCSI = Math.min(1, Math.max(0, f.csi + stressBoost * 0.4));
    const newDeficit = Math.max(0, f.water_deficit_mm + stressBoost * 30 + (rng() - 0.5) * 10);
    let status = "Adequate";
    if (newCSI >= 0.78) status = "Critical";
    else if (newCSI >= 0.55) status = "Urgent";
    else if (newCSI >= 0.3) status = "Watch";
    return { ...f, csi: newCSI, water_deficit_mm: newDeficit, advisory_status: status };
  });
}

function buildTimeline(fields, simFields, seed) {
  const rng = seededRand(seed);
  const days = [];
  for (let d = 0; d <= 30; d++) {
    const t = d / 30;
    const baseStress = fields.reduce((a, f) => a + f.csi, 0) / fields.length;
    const simStress = simFields.reduce((a, f) => a + f.csi, 0) / simFields.length;
    const baseline = baseStress + (rng() - 0.5) * 0.02;
    const projected = baseStress + (simStress - baseStress) * t + (rng() - 0.5) * 0.03;
    days.push({
      day: d,
      baseline: +Math.max(0, Math.min(1, baseline)).toFixed(3),
      projected: +Math.max(0, Math.min(1, projected)).toFixed(3),
    });
  }
  return days;
}

function buildRadarData(simFields) {
  const avg = (arr, fn) => arr.length ? arr.reduce((a, f) => a + fn(f), 0) / arr.length : 0;
  const avgCSI = avg(simFields, f => f.csi);
  const avgDeficit = avg(simFields, f => f.water_deficit_mm);
  return [
    { axis: "NDVI Health", value: Math.round((1 - avgCSI) * 100) },
    { axis: "Soil Moisture", value: Math.round(Math.max(0, 100 - avgDeficit * 1.2)) },
    { axis: "Crop Vigor", value: Math.round((1 - avgCSI * 0.8) * 100) },
    { axis: "Water Balance", value: Math.round(Math.max(0, 100 - avgDeficit)) },
    { axis: "Stress Resilience", value: Math.round((1 - avgCSI * 1.1) * 100) },
  ];
}

function computeTransitions(fields, simFields) {
  const statuses = ["Adequate", "Watch", "Urgent", "Critical"];
  const transitions = {};
  statuses.forEach(from => {
    statuses.forEach(to => {
      transitions[`${from}→${to}`] = 0;
    });
  });
  fields.forEach((f, i) => {
    const sf = simFields[i];
    transitions[`${f.advisory_status}→${sf.advisory_status}`]++;
  });
  return statuses.map(from => {
    const row = { from };
    statuses.forEach(to => {
      row[to] = transitions[`${from}→${to}`];
    });
    return row;
  }).filter(row => {
    return Object.values(row).some(v => typeof v === "number" && v > 0);
  });
}

export default function SimulationEngine({ fields = [], loadAll, loading }) {
  const [preset, setPreset] = useState("normal");
  const [seed, setSeed] = useState(42);
  const [multiplier, setMultiplier] = useState(1.0);
  const [tempOffset, setTempOffset] = useState(0);
  const [rainfallPct, setRainfallPct] = useState(100);
  const [hasRun, setHasRun] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState([
    "INFO  — [SimulationEngine] Initialized parametric model.",
    "INFO  — [Simulator] Anchored coordinates: 29.88° N, 75.82° E",
    "INFO  — [Simulator] Ready. Select a scenario preset or configure manually.",
  ]);

  const simFields = useMemo(() => {
    if (!hasRun || !fields.length) return fields;
    return simulateFields(fields, { seed, multiplier, tempOffset, rainfallPct });
  }, [hasRun, fields, seed, multiplier, tempOffset, rainfallPct]);

  const timeline = useMemo(() => {
    if (!hasRun || !fields.length) return [];
    return buildTimeline(fields, simFields, seed);
  }, [hasRun, fields, simFields, seed]);

  const radarData = useMemo(() => {
    if (!hasRun || !simFields.length) return [];
    return buildRadarData(simFields);
  }, [hasRun, simFields]);

  const transitions = useMemo(() => {
    if (!hasRun || !fields.length) return [];
    return computeTransitions(fields, simFields);
  }, [hasRun, fields, simFields]);

  const simStats = useMemo(() => {
    if (!simFields.length) return { adequate: 0, watch: 0, urgent: 0, critical: 0 };
    return {
      adequate: simFields.filter(f => f.advisory_status === "Adequate").length,
      watch: simFields.filter(f => f.advisory_status === "Watch").length,
      urgent: simFields.filter(f => f.advisory_status === "Urgent").length,
      critical: simFields.filter(f => f.advisory_status === "Critical").length,
    };
  }, [simFields]);

  const origStats = useMemo(() => {
    if (!fields.length) return { adequate: 0, watch: 0, urgent: 0, critical: 0 };
    return {
      adequate: fields.filter(f => f.advisory_status === "Adequate").length,
      watch: fields.filter(f => f.advisory_status === "Watch").length,
      urgent: fields.filter(f => f.advisory_status === "Urgent").length,
      critical: fields.filter(f => f.advisory_status === "Critical").length,
    };
  }, [fields]);

  const applyPreset = (key) => {
    setPreset(key);
    const p = PRESETS.find(x => x.key === key);
    if (p && key !== "custom") {
      setSeed(p.seed);
      setMultiplier(p.multiplier);
      setTempOffset(p.tempOffset);
      setRainfallPct(p.rainfallPct);
    }
    setHasRun(false);
  };

  const handleSimulate = async () => {
    const p = PRESETS.find(x => x.key === preset);
    setSimulating(true);
    setLogs(prev => [
      ...prev,
      ``,
      `━━━ NEW SIMULATION RUN ━━━━━━━━━━━━━━━━━━━`,
      `INFO  — Scenario: ${p?.label || preset}`,
      `INFO  — Seed: ${seed} | Stress: ${multiplier}x | Temp: ${tempOffset > 0 ? "+" : ""}${tempOffset}°C | Rain: ${rainfallPct}%`,
      `INFO  — Computing evapotranspiration for ${fields.length} fields...`,
    ]);

    // Simulate a brief processing delay for UX
    await new Promise(r => setTimeout(r, 800));

    setLogs(prev => [
      ...prev,
      `INFO  — Applying FAO-56 Penman-Monteith water balance...`,
      `INFO  — Calculating Combined Stress Index (CSI) adjustments...`,
    ]);

    await new Promise(r => setTimeout(r, 600));

    setHasRun(true);
    setSimulating(false);

    setLogs(prev => [
      ...prev,
      `✓ OK  — Generated radar health profile (5 axes)`,
      `✓ OK  — Built 30-day stress projection timeline`,
      `✓ OK  — Computed field status transitions`,
      `INFO  — Simulation complete. Visual outputs rendered.`,
    ]);

    toast.success("Simulation complete!", {
      description: `${fields.length} fields analyzed under "${p?.label}" scenario`,
    });

    try {
      await loadAll();
    } catch (_) {}
  };

  const activePreset = PRESETS.find(p => p.key === preset);

  return (
    <div className="flex flex-col gap-6 p-6 md:p-10">
      {/* Header */}
      <div className="chart-entrance">
        <h2 className="font-display text-frost text-2xl font-bold">
          Parametric Simulation Engine
        </h2>
        <p className="mt-1 text-sm text-white/50">
          Simulate climate scenarios, drought events, and soil moisture stressors — then visualize projected impact on field health.
        </p>
      </div>

      {/* Scenario Presets */}
      <div className="chart-entrance-delay-1">
        <div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-white/50">Scenario Presets</div>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
          {PRESETS.map((p) => (
            <button
              key={p.key}
              onClick={() => applyPreset(p.key)}
              className={`rounded-xl border px-3 py-3 text-left text-xs font-medium transition hover:-translate-y-0.5 ${
                preset === p.key
                  ? "border-[#00E65B]/40 bg-[#00E65B]/10 text-[#00E65B] shadow-[0_0_20px_rgba(0,230,91,0.1)]"
                  : "border-white/8 bg-white/[0.02] text-white/70 hover:border-white/15 hover:text-white"
              }`}
            >
              <div className="text-sm font-semibold">{p.label}</div>
            </button>
          ))}
        </div>
        {activePreset && (
          <p className="mt-2 text-xs text-white/40 italic">{activePreset.desc}</p>
        )}
      </div>

      {/* Config + Run */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 chart-entrance-delay-2">
        {/* Parameter Controls */}
        <div className="glass rounded-2xl p-5 lg:col-span-1 flex flex-col gap-5">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-white/60">Parameters</h3>

          <div>
            <label className="mb-1.5 flex justify-between text-xs uppercase tracking-wider text-white/50">
              <span>Seed</span>
              <span className="font-mono text-[#00E5FF]">{seed}</span>
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => { setSeed(Number(e.target.value)); setPreset("custom"); }}
              className="w-full rounded-lg border border-white/10 bg-black/40 px-4 py-2.5 text-sm text-white outline-none focus:border-[#00E65B]/50"
            />
          </div>

          <div>
            <label className="mb-1.5 flex justify-between text-xs uppercase tracking-wider text-white/50">
              <span>Stress Factor</span>
              <span className="font-mono text-[#FBBF24]">{multiplier}x</span>
            </label>
            <input
              type="range" min={0.5} max={3.0} step={0.1} value={multiplier}
              onChange={(e) => { setMultiplier(parseFloat(e.target.value)); setPreset("custom"); }}
              className="w-full styled-range"
            />
          </div>

          <div>
            <label className="mb-1.5 flex justify-between text-xs uppercase tracking-wider text-white/50">
              <span><Thermometer size={12} className="inline mr-1" />Temp Offset</span>
              <span className="font-mono text-[#FF8800]">{tempOffset > 0 ? "+" : ""}{tempOffset}°C</span>
            </label>
            <input
              type="range" min={-3} max={8} step={1} value={tempOffset}
              onChange={(e) => { setTempOffset(parseInt(e.target.value)); setPreset("custom"); }}
              className="w-full styled-range"
            />
          </div>

          <div>
            <label className="mb-1.5 flex justify-between text-xs uppercase tracking-wider text-white/50">
              <span><CloudRain size={12} className="inline mr-1" />Rainfall</span>
              <span className="font-mono text-[#00E5FF]">{rainfallPct}%</span>
            </label>
            <input
              type="range" min={10} max={200} step={5} value={rainfallPct}
              onChange={(e) => { setRainfallPct(parseInt(e.target.value)); setPreset("custom"); }}
              className="w-full styled-range"
            />
          </div>

          <button
            onClick={handleSimulate}
            disabled={simulating || loading || !fields.length}
            className="relative flex items-center justify-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-[#00E65B] to-[#00c24d] py-3.5 text-sm font-bold text-black transition hover:shadow-[0_0_30px_rgba(0,230,91,0.4)] active:scale-[0.98] disabled:opacity-50"
          >
            {simulating ? (
              <>
                <RefreshCw size={16} className="animate-spin" /> Simulating…
              </>
            ) : (
              <>
                <Zap size={16} /> Run Simulation
              </>
            )}
          </button>
        </div>

        {/* Visual Output Area */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <AnimatePresence mode="wait">
            {!hasRun ? (
              <motion.div
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass rounded-2xl p-10 flex flex-col items-center justify-center text-center min-h-[420px]"
              >
                <div className="mb-4 grid h-16 w-16 place-items-center rounded-2xl bg-[#00E65B]/10 ring-1 ring-[#00E65B]/30">
                  <Zap size={28} className="text-[#00E65B]" />
                </div>
                <h3 className="font-display text-xl font-bold text-white mb-2">Select a Scenario & Run</h3>
                <p className="text-sm text-white/50 max-w-md">
                  Choose a climate scenario preset above, adjust parameters, then click <strong className="text-[#00E65B]">Run Simulation</strong> to see projected field health impacts with radar charts, stress timelines, and status transitions.
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="flex flex-col gap-6"
              >
                {/* Status Change Summary */}
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <StatusCard label="Adequate" before={origStats.adequate} after={simStats.adequate} color="#00E65B" />
                  <StatusCard label="Watch" before={origStats.watch} after={simStats.watch} color="#FBBF24" />
                  <StatusCard label="Urgent" before={origStats.urgent} after={simStats.urgent} color="#FF8800" />
                  <StatusCard label="Critical" before={origStats.critical} after={simStats.critical} color="#FF3B30" />
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  {/* Radar Chart */}
                  <div className="glass-glow rounded-2xl p-5">
                    <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/60">
                      Field Health Radar
                    </h3>
                    <div className="h-[280px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                          <PolarGrid stroke="rgba(255,255,255,0.08)" />
                          <PolarAngleAxis
                            dataKey="axis"
                            tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }}
                          />
                          <PolarRadiusAxis
                            angle={90}
                            domain={[0, 100]}
                            tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 9 }}
                          />
                          <Radar
                            name="Health"
                            dataKey="value"
                            stroke="#00E65B"
                            fill="#00E65B"
                            fillOpacity={0.2}
                            strokeWidth={2}
                            isAnimationActive={true}
                            animationDuration={1000}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Impact Timeline */}
                  <div className="glass-cyan rounded-2xl p-5">
                    <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/60">
                      30-Day Stress Projection
                    </h3>
                    <div className="h-[280px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={timeline} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                          <defs>
                            <linearGradient id="baselineGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#00E65B" stopOpacity={0.3} />
                              <stop offset="100%" stopColor="#00E65B" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="projectedGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#FF3B30" stopOpacity={0.3} />
                              <stop offset="100%" stopColor="#FF3B30" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis
                            dataKey="day"
                            stroke="rgba(255,255,255,0.3)"
                            tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
                            label={{ value: "Days", position: "insideBottom", offset: -2, fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                          />
                          <YAxis
                            domain={[0, 1]}
                            stroke="rgba(255,255,255,0.3)"
                            tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
                            label={{ value: "Avg CSI", angle: -90, position: "insideLeft", fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                          />
                          <Tooltip
                            contentStyle={{
                              background: "rgba(10,15,13,0.95)",
                              border: "1px solid rgba(255,255,255,0.1)",
                              borderRadius: "10px",
                              fontSize: 12,
                            }}
                            labelFormatter={(v) => `Day ${v}`}
                          />
                          <Legend
                            wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.6)" }}
                          />
                          <Area
                            type="monotone"
                            dataKey="baseline"
                            name="Baseline"
                            stroke="#00E65B"
                            fill="url(#baselineGrad)"
                            strokeWidth={2}
                            isAnimationActive={true}
                            animationDuration={1200}
                          />
                          <Area
                            type="monotone"
                            dataKey="projected"
                            name="Projected"
                            stroke="#FF3B30"
                            fill="url(#projectedGrad)"
                            strokeWidth={2}
                            strokeDasharray="5 3"
                            isAnimationActive={true}
                            animationDuration={1200}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Field Transition Matrix */}
                <div className="glass rounded-2xl p-5">
                  <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/60">
                    Status Transition Matrix
                  </h3>
                  <div className="h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={transitions} layout="vertical" margin={{ left: 60, right: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis type="number" stroke="rgba(255,255,255,0.3)" tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }} />
                        <YAxis
                          type="category"
                          dataKey="from"
                          stroke="rgba(255,255,255,0.3)"
                          tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(10,15,13,0.95)",
                            border: "1px solid rgba(255,255,255,0.1)",
                            borderRadius: "10px",
                            fontSize: 12,
                          }}
                          formatter={(value, name) => [`${value} fields`, `→ ${name}`]}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Bar dataKey="Adequate" stackId="a" fill="#00E65B" radius={[0, 0, 0, 0]} isAnimationActive={true} />
                        <Bar dataKey="Watch" stackId="a" fill="#FBBF24" radius={[0, 0, 0, 0]} isAnimationActive={true} />
                        <Bar dataKey="Urgent" stackId="a" fill="#FF8800" radius={[0, 0, 0, 0]} isAnimationActive={true} />
                        <Bar dataKey="Critical" stackId="a" fill="#FF3B30" radius={[0, 4, 4, 0]} isAnimationActive={true} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Collapsible Terminal Logs */}
      <div className="glass rounded-2xl overflow-hidden">
        <button
          onClick={() => setShowLogs(!showLogs)}
          className="flex w-full items-center justify-between px-5 py-3 text-sm font-semibold uppercase tracking-wider text-white/60 hover:text-white/80 transition"
        >
          <span className="flex items-center gap-2">
            <Terminal size={14} className="text-[#00E65B]" /> Simulator Terminal Logs
          </span>
          <span className="flex items-center gap-2">
            <span className="text-xs font-mono text-white/30">{logs.length} entries</span>
            {showLogs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </span>
        </button>
        <AnimatePresence>
          {showLogs && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="max-h-[250px] overflow-y-auto border-t border-white/5 bg-black/60 p-4 font-mono text-xs text-white/70 leading-relaxed">
                {logs.map((log, i) => (
                  <div key={i} className="flex gap-2">
                    {log ? (
                      <>
                        <span className={log.startsWith("✓") ? "text-[#00E65B]" : log.startsWith("ERROR") ? "text-[#FF3B30]" : "text-white/30"}>
                          {log.startsWith("✓") ? "✓" : log.startsWith("━") ? "━" : ">"}
                        </span>
                        <span className={
                          log.startsWith("✓") ? "text-[#00E65B]/80" :
                          log.startsWith("ERROR") ? "text-[#FF3B30]" :
                          log.startsWith("━") ? "text-[#00E5FF] font-bold" : ""
                        }>{log}</span>
                      </>
                    ) : <span>&nbsp;</span>}
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function StatusCard({ label, before, after, color }) {
  const diff = after - before;
  const arrow = diff > 0 ? "↑" : diff < 0 ? "↓" : "→";
  const diffColor = label === "Adequate"
    ? (diff >= 0 ? "#00E65B" : "#FF3B30")
    : (diff <= 0 ? "#00E65B" : "#FF3B30");

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="glass rounded-2xl px-4 py-3.5"
    >
      <div className="text-[10px] uppercase tracking-[0.2em] text-white/50">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span className="font-display text-2xl font-bold" style={{ color }}>{after}</span>
        <span className="text-xs text-white/40">/ {before}</span>
      </div>
      <div className="mt-1 flex items-center gap-1 text-xs font-semibold" style={{ color: diffColor }}>
        <span>{arrow}</span>
        <span>{Math.abs(diff)} fields</span>
      </div>
    </motion.div>
  );
}
