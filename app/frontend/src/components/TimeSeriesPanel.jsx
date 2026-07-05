import { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

/* Multi-temporal 8-day composite time-series with a scrubbable time-step slider
   and phenological-stage badges. Uses deterministic pseudo-random series seeded
   by the selected field id so charts feel field-specific yet stable. */

const TS_LEN = 24; // 24 × 8-day = 192-day season
const SERIES = [
  { key: "ndvi", label: "NDVI", color: "#00E65B" },
  { key: "evi", label: "EVI", color: "#00E5FF" },
  { key: "ndwi", label: "NDWI", color: "#79C2FF" },
  { key: "sar_vh", label: "SAR VH", color: "#A78BFA" },
];

const PHENOLOGY = [
  { key: "germ", label: "Germination", start: 0,  end: 4,  months: "Jun–Jul", color: "#FBBF24" },
  { key: "veg",  label: "Vegetative",  start: 4,  end: 10, months: "Jul–Aug", color: "#00E65B" },
  { key: "repr", label: "Reproductive",start: 10, end: 18, months: "Aug–Oct", color: "#00E5FF" },
  { key: "mat",  label: "Maturity",    start: 18, end: 24, months: "Oct–Nov", color: "#FF8A00" },
];

// Deterministic sinusoidal-with-noise curves per series/field
function seededRand(seed) {
  let x = seed | 0;
  return () => {
    x = (x * 1103515245 + 12345) & 0x7fffffff;
    return x / 0x7fffffff;
  };
}

function hashString(s = "") {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return h;
}

function buildSeries(field) {
  const seed = hashString(field?.field_id || "SEED");
  const rng = seededRand(seed);
  const csi = field?.csi ?? 0.4;
  const rows = [];
  for (let t = 0; t < TS_LEN; t++) {
    // Double-logistic-ish curve peaking mid-season
    const growth = 0.9 / (1 + Math.exp(-0.6 * (t - 8)));
    const senesc = 1 - 1 / (1 + Math.exp(-0.6 * (t - 18)));
    const base = growth * senesc; // 0..~0.9
    const stress = 1 - Math.min(csi, 0.6); // higher CSI → dampens
    const ndvi = Math.max(0.05, base * 0.9 * stress + (rng() - 0.5) * 0.04);
    const evi = Math.max(0.05, base * 0.75 * stress + (rng() - 0.5) * 0.04);
    const ndwi = Math.max(-0.2, base * 0.5 * stress - 0.1 + (rng() - 0.5) * 0.05);
    // SAR VH backscatter in dB, negative — inverse relationship to soil moisture
    const sar_vh = -18 + base * 6 * (1 - stress * 0.7) + (rng() - 0.5) * 1.4;
    rows.push({
      step: t,
      day: t * 8,
      ndvi: +ndvi.toFixed(3),
      evi: +evi.toFixed(3),
      ndwi: +ndwi.toFixed(3),
      sar_vh: +sar_vh.toFixed(2),
    });
  }
  return rows;
}

function TooltipBox({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-black/85 px-3 py-2 text-xs backdrop-blur-md">
      <div className="mb-1 font-mono text-[10px] uppercase tracking-[0.18em] text-white/60">
        Step {label} · Day {label * 8}
      </div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
            {SERIES.find((s) => s.key === p.dataKey)?.label || p.dataKey}
          </span>
          <span className="font-mono">{p.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function TimeSeriesPanel({ field }) {
  const [step, setStep] = useState(12);
  const data = useMemo(() => buildSeries(field), [field]);
  const activeRow = data[step] || data[0];
  const stage = PHENOLOGY.find((p) => step >= p.start && step < p.end) || PHENOLOGY[0];

  return (
    <section
      data-testid="timeseries-panel"
      className="glass overflow-hidden rounded-2xl"
    >
      <div className="flex items-center justify-between gap-3 border-b border-white/5 px-5 py-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/50">
            Multi-temporal composites
          </div>
          <div className="font-display mt-0.5 text-lg font-semibold">
            {field?.field_id ? `${field.field_id} · ${field.crop_type}` : "Season timeline"}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {SERIES.map((s) => (
            <span
              key={s.key}
              className="flex items-center gap-1.5 rounded-full border border-white/10 bg-black/40 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-white/85"
              style={{ borderColor: `${s.color}55`, color: s.color }}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      </div>

      {/* Phenology badges */}
      <div className="flex gap-1.5 px-5 pt-4" data-testid="phenology-track">
        {PHENOLOGY.map((p) => {
          const isActive = p.key === stage.key;
          return (
            <div
              key={p.key}
              data-testid={`phenology-${p.key}`}
              className={`flex-1 rounded-lg border px-3 py-2 text-center transition ${
                isActive
                  ? "border-white/25 bg-white/[0.05]"
                  : "border-white/8 bg-black/30"
              }`}
              style={
                isActive
                  ? { boxShadow: `inset 0 0 0 1px ${p.color}aa, 0 0 18px ${p.color}22` }
                  : {}
              }
            >
              <div
                className="font-display text-[11px] font-semibold"
                style={{ color: isActive ? p.color : "rgba(255,255,255,0.75)" }}
              >
                {p.label}
              </div>
              <div className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.15em] text-white/45">
                {p.months} · steps {p.start}-{p.end - 1}
              </div>
            </div>
          );
        })}
      </div>

      {/* Chart */}
      <div className="px-3 pt-2" data-testid="timeseries-chart">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="3 4" />
            <XAxis
              dataKey="step"
              tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10 }}
              stroke="rgba(255,255,255,0.15)"
              tickFormatter={(v) => `${v}`}
              label={{ value: "8-day composite step", position: "insideBottom", offset: -4, fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
            />
            <YAxis
              yAxisId="left"
              tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10 }}
              stroke="rgba(255,255,255,0.15)"
              domain={[-0.2, 1]}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10 }}
              stroke="rgba(255,255,255,0.15)"
              domain={[-24, -8]}
            />
            <Tooltip content={<TooltipBox />} />
            <Legend wrapperStyle={{ display: "none" }} />
            {SERIES.filter((s) => s.key !== "sar_vh").map((s) => (
              <Line
                key={s.key}
                yAxisId="left"
                type="monotone"
                dataKey={s.key}
                stroke={s.color}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="sar_vh"
              stroke="#A78BFA"
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
            <ReferenceLine
              x={step}
              stroke="#ffffff"
              strokeOpacity={0.7}
              strokeWidth={1}
              yAxisId="left"
              label={{ value: `t=${step}`, position: "top", fill: "#fff", fontSize: 10 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Slider + current values */}
      <div className="flex flex-wrap items-center gap-4 border-t border-white/5 px-5 py-4">
        <div className="flex flex-1 min-w-[240px] items-center gap-3">
          <span className="text-[10px] uppercase tracking-[0.18em] text-white/50">
            Time step
          </span>
          <input
            type="range"
            min={0}
            max={TS_LEN - 1}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            data-testid="timeseries-slider"
            className="flex-1 accent-[#00E65B]"
          />
          <span className="font-mono text-xs text-white/80">
            {step}/{TS_LEN - 1}
          </span>
        </div>
        <div className="flex gap-3 font-mono text-[11px] text-white/80">
          <span data-testid="ts-val-ndvi">
            <span className="text-[#00E65B]">NDVI</span> {activeRow.ndvi.toFixed(2)}
          </span>
          <span data-testid="ts-val-evi">
            <span className="text-[#00E5FF]">EVI</span> {activeRow.evi.toFixed(2)}
          </span>
          <span data-testid="ts-val-ndwi">
            <span className="text-[#79C2FF]">NDWI</span> {activeRow.ndwi.toFixed(2)}
          </span>
          <span data-testid="ts-val-sar">
            <span className="text-[#A78BFA]">SAR VH</span> {activeRow.sar_vh.toFixed(1)} dB
          </span>
        </div>
      </div>
    </section>
  );
}
