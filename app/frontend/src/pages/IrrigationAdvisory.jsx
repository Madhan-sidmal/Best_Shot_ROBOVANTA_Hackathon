import { useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from "recharts";
import ActionPanel from "@/components/ActionPanel";
import AdvisoryTable from "@/components/AdvisoryTable";
import { motion } from "framer-motion";
import { Droplets, AlertCircle, CheckCircle2, TrendingUp } from "lucide-react";

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

function deficitColor(deficit) {
  if (deficit > 60) return "#FF3B30";
  if (deficit > 40) return "#FF8800";
  if (deficit > 20) return "#FBBF24";
  return "#00E65B";
}

function DeficitTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-xl border border-white/10 bg-black/90 px-4 py-3 text-xs backdrop-blur-md shadow-xl">
      <div className="font-semibold text-white mb-1">{d.field_id}</div>
      <div className="text-white/60">{d.crop_type} · {d.growth_stage}</div>
      <div className="mt-1 font-mono" style={{ color: deficitColor(d.deficit) }}>
        {d.deficit.toFixed(1)} mm deficit
      </div>
    </div>
  );
}

export default function IrrigationAdvisory({ fields = [], loading, selectedFieldId, setSelectedFieldId, setFilteredFieldIds }) {
  const totalDeficit = useMemo(() => fields.reduce((acc, f) => acc + f.water_deficit_mm, 0).toFixed(0), [fields]);
  const avgDeficit = useMemo(() => fields.length ? (fields.reduce((a, f) => a + f.water_deficit_mm, 0) / fields.length).toFixed(1) : 0, [fields]);
  const criticalDeficitCount = useMemo(() => fields.filter((f) => f.water_deficit_mm > 50).length, [fields]);
  const adequateCount = useMemo(() => fields.filter((f) => f.advisory_status === "Adequate").length, [fields]);

  // Sorted deficit bar data
  const deficitBars = useMemo(() =>
    [...fields]
      .sort((a, b) => b.water_deficit_mm - a.water_deficit_mm)
      .slice(0, 15)
      .map(f => ({
        field_id: f.field_id,
        deficit: f.water_deficit_mm,
        crop_type: f.crop_type,
        growth_stage: f.growth_stage,
      })),
    [fields]
  );

  // Deficit by crop type
  const deficitByCrop = useMemo(() => {
    const agg = {};
    fields.forEach(f => {
      if (!agg[f.crop_type]) agg[f.crop_type] = { total: 0, count: 0 };
      agg[f.crop_type].total += f.water_deficit_mm;
      agg[f.crop_type].count++;
    });
    return Object.entries(agg).map(([crop, { total, count }]) => ({
      crop,
      avg_deficit: +(total / count).toFixed(1),
      total_deficit: +total.toFixed(0),
    }));
  }, [fields]);

  if (loading) {
    return (
      <div className="p-10 flex flex-col items-center justify-center min-h-[400px]">
        <div className="h-8 w-8 rounded-full border-2 border-[#00E5FF] border-t-transparent animate-spin" />
        <div className="mt-4 text-sm text-white/50">Loading advisories…</div>
      </div>
    );
  }

  const CROP_COLORS = { Rice: "#00E65B", Wheat: "#FBBF24", Cotton: "#A78BFA", Sugarcane: "#00E5FF" };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-col gap-6 p-6 md:p-10"
    >
      {/* Header */}
      <motion.div variants={item}>
        <h2 className="font-display text-frost text-2xl font-bold">Irrigation Advisory (FAO-56)</h2>
        <p className="mt-1 text-sm text-white/50">
          Evapotranspiration calculations & irrigation schedules computed daily using satellite crop coefficients & local weather models.
        </p>
      </motion.div>

      {/* KPI Cards */}
      <motion.div variants={item} className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Cumulative Deficit" value={`${totalDeficit} mm`} icon={<Droplets size={16} />} accent="#00E5FF" glassClass="glass-cyan" />
        <StatCard label="Avg Deficit" value={`${avgDeficit} mm`} icon={<TrendingUp size={16} />} accent="#FBBF24" glassClass="glass" />
        <StatCard label="High Deficit (>50mm)" value={criticalDeficitCount} icon={<AlertCircle size={16} />} accent="#FF3B30" glassClass="glass-danger" />
        <StatCard label="Fields Adequate" value={adequateCount} icon={<CheckCircle2 size={16} />} accent="#00E65B" glassClass="glass-glow" />
      </motion.div>

      {/* Charts */}
      <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Top 15 Fields by Deficit */}
        <div className="glass rounded-2xl p-5">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/60">
            Top 15 Fields by Water Deficit
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={deficitBars} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="deficitGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00E5FF" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#00E5FF" stopOpacity={0.2} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="field_id"
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 9 }}
                  angle={-35}
                  textAnchor="end"
                  height={50}
                />
                <YAxis
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
                  label={{ value: "Deficit (mm)", angle: -90, position: "insideLeft", fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                />
                <Tooltip content={<DeficitTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                <Bar dataKey="deficit" radius={[6, 6, 0, 0]} isAnimationActive={true} animationDuration={1000}>
                  {deficitBars.map((d, i) => (
                    <Cell key={i} fill={deficitColor(d.deficit)} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Deficit by Crop Type */}
        <div className="glass rounded-2xl p-5">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/60">
            Average Deficit by Crop Type
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={deficitByCrop} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  {deficitByCrop.map(d => (
                    <linearGradient key={d.crop} id={`defCrop-${d.crop}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CROP_COLORS[d.crop] || "#888"} stopOpacity={0.9} />
                      <stop offset="100%" stopColor={CROP_COLORS[d.crop] || "#888"} stopOpacity={0.3} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="crop" stroke="rgba(255,255,255,0.3)" tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }} />
                <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: "rgba(10,15,13,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: 12 }}
                  formatter={(val) => [`${val} mm`, "Avg Deficit"]}
                />
                <Bar dataKey="avg_deficit" name="Avg Deficit (mm)" radius={[8, 8, 0, 0]} isAnimationActive={true} animationDuration={1000}>
                  {deficitByCrop.map(d => (
                    <Cell key={d.crop} fill={`url(#defCrop-${d.crop})`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* Advisory Table + Action Panel */}
      <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="glass rounded-2xl p-5">
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/60">
              Canal Command Field Advisory Map
            </h3>
            <AdvisoryTable
              fields={fields}
              loading={loading}
              selectedFieldId={selectedFieldId}
              onRowSelect={setSelectedFieldId}
              onFilteredIdsChange={setFilteredFieldIds}
            />
          </div>
        </div>

        <div className="lg:col-span-1">
          <ActionPanel
            fields={fields}
            selectedFieldId={selectedFieldId}
            onFieldChange={setSelectedFieldId}
          />
        </div>
      </motion.div>
    </motion.div>
  );
}

function StatCard({ label, value, icon, accent, glassClass }) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      className={`${glassClass} rounded-2xl px-5 py-4 transition`}
    >
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-white/50">
        <span style={{ color: accent }}>{icon}</span>
        {label}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="font-display text-3xl font-bold" style={{ color: accent }}>
          {value}
        </span>
      </div>
    </motion.div>
  );
}
