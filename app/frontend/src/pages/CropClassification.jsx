import { useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend
} from "recharts";
import { motion } from "framer-motion";
import { Layers, Sprout, TrendingUp } from "lucide-react";

const CROP_COLORS = {
  Rice: "#00E65B",
  Wheat: "#FBBF24",
  Cotton: "#A78BFA",
  Sugarcane: "#00E5FF",
};

const STAGE_COLORS = [
  "#00E65B", "#00E5FF", "#A78BFA", "#FBBF24", "#FF8800",
];

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-xl border border-white/10 bg-black/90 px-4 py-3 text-xs backdrop-blur-md shadow-xl">
      <div className="font-semibold text-white mb-1">{d.name}</div>
      <div className="text-white/60">{d.count} fields ({d.pct}%)</div>
    </div>
  );
}

function PieLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) {
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  if (percent < 0.08) return null;
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}

export default function CropClassification({ fields = [], loading }) {
  const cropCounts = useMemo(() => {
    return fields.reduce((acc, f) => {
      acc[f.crop_type] = (acc[f.crop_type] || 0) + 1;
      return acc;
    }, {});
  }, [fields]);

  const total = fields.length;

  const chartData = useMemo(() =>
    Object.entries(cropCounts).map(([name, count]) => ({
      name,
      count,
      pct: total ? ((count / total) * 100).toFixed(1) : "0",
      color: CROP_COLORS[name] || "#888",
    })),
    [cropCounts, total]
  );

  // Growth stage distribution per crop
  const stageData = useMemo(() => {
    const stages = {};
    fields.forEach((f) => {
      if (!stages[f.growth_stage]) stages[f.growth_stage] = {};
      stages[f.growth_stage][f.crop_type] = (stages[f.growth_stage][f.crop_type] || 0) + 1;
    });
    return Object.entries(stages).map(([stage, crops]) => ({
      stage,
      ...crops,
    }));
  }, [fields]);

  if (loading) {
    return (
      <div className="p-10 flex flex-col items-center justify-center min-h-[400px]">
        <div className="h-8 w-8 rounded-full border-2 border-[#00E65B] border-t-transparent animate-spin" />
        <div className="mt-4 text-sm text-white/50">Loading classification data…</div>
      </div>
    );
  }

  const cropNames = Object.keys(CROP_COLORS);

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-col gap-6 p-6 md:p-10"
    >
      {/* Header */}
      <motion.div variants={item}>
        <h2 className="font-display text-frost text-2xl font-bold">
          Crop Classification & Distribution
        </h2>
        <p className="mt-1 text-sm text-white/50">
          Satellite-derived classification based on multi-temporal optical imagery and spectral signatures.
        </p>
      </motion.div>

      {/* KPI Cards */}
      <motion.div variants={item} className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Total Fields" value={total} icon={<Layers size={16} />} glassClass="glass-glow" accent="#00E65B" />
        <StatCard label="Rice Fields" value={cropCounts["Rice"] || 0} icon={<Sprout size={16} />} glassClass="glass" accent="#00E65B" />
        <StatCard label="Wheat Fields" value={cropCounts["Wheat"] || 0} icon={<Sprout size={16} />} glassClass="glass" accent="#FBBF24" />
        <StatCard label="Cotton & Cane" value={(cropCounts["Cotton"] || 0) + (cropCounts["Sugarcane"] || 0)} icon={<TrendingUp size={16} />} glassClass="glass" accent="#A78BFA" />
      </motion.div>

      {/* Charts Row */}
      <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Gradient Bar Chart */}
        <div className="glass rounded-2xl p-5">
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/60">
            Crop Distribution
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  {chartData.map((d) => (
                    <linearGradient key={d.name} id={`grad-${d.name}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={d.color} stopOpacity={0.9} />
                      <stop offset="100%" stopColor={d.color} stopOpacity={0.3} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" stroke="rgba(255,255,255,0.3)" tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }} />
                <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                <Bar dataKey="count" radius={[8, 8, 0, 0]} isAnimationActive={true} animationDuration={1000}>
                  {chartData.map((d) => (
                    <Cell key={d.name} fill={`url(#grad-${d.name})`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Donut Pie Chart */}
        <div className="glass rounded-2xl p-5">
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/60">
            Crop Proportions
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%" cy="50%"
                  innerRadius={60}
                  outerRadius={110}
                  dataKey="count"
                  nameKey="name"
                  labelLine={false}
                  label={PieLabel}
                  strokeWidth={2}
                  stroke="rgba(10,15,13,0.8)"
                  isAnimationActive={true}
                  animationDuration={1000}
                >
                  {chartData.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Pie>
                <Legend
                  verticalAlign="bottom"
                  iconType="circle"
                  wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.6)" }}
                />
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* Growth Stage Distribution */}
      <motion.div variants={item} className="glass rounded-2xl p-5">
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/60">
          Growth Stage Distribution by Crop
        </h3>
        <div className="h-[260px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stageData} layout="vertical" margin={{ left: 100, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" stroke="rgba(255,255,255,0.3)" tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }} />
              <YAxis
                type="category"
                dataKey="stage"
                stroke="rgba(255,255,255,0.3)"
                tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }}
                width={95}
              />
              <Tooltip
                contentStyle={{ background: "rgba(10,15,13,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {cropNames.map((crop) => (
                <Bar key={crop} dataKey={crop} stackId="a" fill={CROP_COLORS[crop]} isAnimationActive={true} animationDuration={800} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Field Details Table */}
      <motion.div variants={item} className="glass overflow-hidden rounded-2xl">
        <div className="p-5 border-b border-white/5">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-white/60">
            Field Classification Registry
          </h3>
        </div>
        <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
          <table className="w-full text-left text-sm text-white/80">
            <thead className="sticky top-0 border-b border-white/5 bg-[#0a0f0d]/95 text-xs uppercase tracking-wider text-white/50 backdrop-blur-md">
              <tr>
                <th className="px-6 py-3">Field ID</th>
                <th className="px-6 py-3">Crop Type</th>
                <th className="px-6 py-3">Growth Stage</th>
                <th className="px-6 py-3">CSI</th>
                <th className="px-6 py-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {fields.map((f) => {
                const color = CROP_COLORS[f.crop_type] || "#888";
                const statusColor =
                  f.advisory_status === "Critical" ? "#FF3B30" :
                  f.advisory_status === "Urgent" ? "#FF8800" :
                  f.advisory_status === "Watch" ? "#FBBF24" : "#00E65B";
                return (
                  <tr key={f.field_id} className="border-b border-white/5 transition hover:bg-white/[0.02]">
                    <td className="px-6 py-3 font-mono font-bold">{f.field_id}</td>
                    <td className="px-6 py-3">
                      <span className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                        {f.crop_type}
                      </span>
                    </td>
                    <td className="px-6 py-3">
                      <span className="rounded-full bg-white/5 px-2.5 py-1 text-xs font-semibold text-[#00E5FF]">
                        {f.growth_stage}
                      </span>
                    </td>
                    <td className="px-6 py-3 font-mono">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-12 overflow-hidden rounded-full bg-white/10">
                          <div className="h-full rounded-full" style={{ width: `${f.csi * 100}%`, backgroundColor: statusColor }} />
                        </div>
                        {f.csi.toFixed(3)}
                      </div>
                    </td>
                    <td className="px-6 py-3">
                      <span
                        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${
                          f.advisory_status === "Critical" ? "pulse-critical" : ""
                        }`}
                        style={{ backgroundColor: `${statusColor}18`, color: statusColor }}
                      >
                        <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: statusColor }} />
                        {f.advisory_status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
}

function StatCard({ label, value, icon, glassClass, accent }) {
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
        <span className="text-xs text-white/40">fields</span>
      </div>
    </motion.div>
  );
}
