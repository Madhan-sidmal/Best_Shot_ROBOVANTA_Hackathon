import { useMemo } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";
import MapPanel from "@/components/MapPanel";
import { motion } from "framer-motion";
import { AlertTriangle, ShieldAlert, Activity, TrendingDown } from "lucide-react";

const STATUS_CONFIG = {
  Adequate: { color: "#00E65B", label: "Adequate" },
  Watch: { color: "#FBBF24", label: "Watch" },
  Urgent: { color: "#FF8800", label: "Urgent" },
  Critical: { color: "#FF3B30", label: "Critical" },
};

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

function DonutLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }) {
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  if (percent < 0.06) return null;
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}

export default function StressDetection({ fields = [], geojson, loading, selectedFieldId, setSelectedFieldId }) {
  const stressedFields = useMemo(() => fields.filter((f) => f.advisory_status !== "Adequate"), [fields]);
  const criticalFields = useMemo(() => fields.filter((f) => f.advisory_status === "Critical"), [fields]);
  const stressedRate = useMemo(() => fields.length ? ((stressedFields.length / fields.length) * 100).toFixed(1) : 0, [fields, stressedFields]);
  const avgCSI = useMemo(() => fields.length ? (fields.reduce((acc, f) => acc + f.csi, 0) / fields.length).toFixed(3) : 0, [fields]);

  // CSI Distribution Histogram
  const csiDistribution = useMemo(() => {
    const bins = Array.from({ length: 20 }, (_, i) => ({
      range: `${(i * 0.05).toFixed(2)}`,
      count: 0,
      label: `${(i * 0.05).toFixed(2)}-${((i + 1) * 0.05).toFixed(2)}`,
    }));
    fields.forEach((f) => {
      const idx = Math.min(19, Math.floor(f.csi * 20));
      bins[idx].count++;
    });
    return bins;
  }, [fields]);

  // Status breakdown for donut
  const statusData = useMemo(() => {
    const counts = {};
    fields.forEach((f) => {
      counts[f.advisory_status] = (counts[f.advisory_status] || 0) + 1;
    });
    return Object.entries(STATUS_CONFIG).map(([key, cfg]) => ({
      name: cfg.label,
      value: counts[key] || 0,
      color: cfg.color,
    })).filter(d => d.value > 0);
  }, [fields]);

  if (loading) {
    return (
      <div className="p-10 flex flex-col items-center justify-center min-h-[400px]">
        <div className="h-8 w-8 rounded-full border-2 border-[#FF8800] border-t-transparent animate-spin" />
        <div className="mt-4 text-sm text-white/50">Loading stress metrics…</div>
      </div>
    );
  }

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-col gap-6 p-6 md:p-10"
    >
      {/* Header */}
      <motion.div variants={item}>
        <h2 className="font-display text-frost text-2xl font-bold">Stress Detection Analysis</h2>
        <p className="mt-1 text-sm text-white/50">
          Soil moisture anomalies and crop stress indexes (CSI) computed using synthetic aperture radar (SAR) backscatter.
        </p>
      </motion.div>

      {/* KPI Cards */}
      <motion.div variants={item} className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Stressed Fields" value={stressedFields.length} icon={<AlertTriangle size={16} />} accent="#FF8800" glassClass="glass-warn" />
        <StatCard label="Critical Fields" value={criticalFields.length} icon={<ShieldAlert size={16} />} accent="#FF3B30" glassClass="glass-danger" />
        <StatCard label="Stressed Rate" value={`${stressedRate}%`} icon={<Activity size={16} />} accent="#00E5FF" glassClass="glass-cyan" />
        <StatCard label="Avg Stress Index" value={avgCSI} icon={<TrendingDown size={16} />} accent="#FBBF24" glassClass="glass" />
      </motion.div>

      {/* Charts Row */}
      <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* CSI Distribution Histogram */}
        <div className="glass rounded-2xl p-5 lg:col-span-2">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/60">
            CSI Distribution Across Fields
          </h3>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={csiDistribution} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="csiGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#00E65B" stopOpacity={0.6} />
                    <stop offset="50%" stopColor="#FBBF24" stopOpacity={0.6} />
                    <stop offset="100%" stopColor="#FF3B30" stopOpacity={0.6} />
                  </linearGradient>
                  <linearGradient id="csiFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#FF8800" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#FF8800" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="range"
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 9 }}
                  interval={3}
                  label={{ value: "CSI Range", position: "insideBottom", offset: -2, fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                />
                <YAxis
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
                  label={{ value: "Fields", angle: -90, position: "insideLeft", fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{ background: "rgba(10,15,13,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: 12 }}
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.label || ""}
                  formatter={(value) => [`${value} fields`, "Count"]}
                />
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="#FF8800"
                  fill="url(#csiFill)"
                  strokeWidth={2}
                  isAnimationActive={true}
                  animationDuration={1200}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Status Breakdown Donut */}
        <div className="glass rounded-2xl p-5 lg:col-span-1">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/60">
            Status Breakdown
          </h3>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%" cy="45%"
                  innerRadius={45}
                  outerRadius={85}
                  dataKey="value"
                  nameKey="name"
                  labelLine={false}
                  label={DonutLabel}
                  strokeWidth={2}
                  stroke="rgba(10,15,13,0.8)"
                  isAnimationActive={true}
                  animationDuration={1000}
                >
                  {statusData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Pie>
                <Legend
                  verticalAlign="bottom"
                  iconType="circle"
                  wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.6)" }}
                />
                <Tooltip
                  contentStyle={{ background: "rgba(10,15,13,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: 12 }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* Map */}
      <motion.div variants={item} className="glass rounded-2xl p-2">
        <MapPanel
          fields={fields}
          geojson={geojson}
          selectedFieldId={selectedFieldId}
          onSelect={setSelectedFieldId}
          loading={loading}
        />
      </motion.div>

      {/* Stressed Fields Table */}
      <motion.div variants={item} className="glass overflow-hidden rounded-2xl">
        <div className="p-5 border-b border-white/5">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-white/60">Stressed Fields Registry</h3>
          <p className="mt-1 text-xs text-white/30">{stressedFields.length} fields requiring attention</p>
        </div>
        <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
          <table className="w-full text-left text-sm text-white/80">
            <thead className="sticky top-0 border-b border-white/5 bg-[#0a0f0d]/95 text-xs uppercase tracking-wider text-white/50 backdrop-blur-md">
              <tr>
                <th className="px-6 py-3">Field ID</th>
                <th className="px-6 py-3">Crop</th>
                <th className="px-6 py-3">Combined Stress Index (CSI)</th>
                <th className="px-6 py-3">Advisory Status</th>
                <th className="px-6 py-3">Action Urgency</th>
              </tr>
            </thead>
            <tbody>
              {stressedFields.map((f) => {
                const statusColor = STATUS_CONFIG[f.advisory_status]?.color || "#888";
                return (
                  <tr key={f.field_id} className="border-b border-white/5 transition hover:bg-white/[0.02]">
                    <td className="px-6 py-3.5 font-mono font-bold">{f.field_id}</td>
                    <td className="px-6 py-3.5">{f.crop_type}</td>
                    <td className="px-6 py-3.5 font-mono">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-20 overflow-hidden rounded-full bg-white/10">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${f.csi * 100}%` }}
                            transition={{ duration: 0.8, delay: 0.2 }}
                            className="h-full rounded-full"
                            style={{ backgroundColor: statusColor }}
                          />
                        </div>
                        {f.csi.toFixed(3)}
                      </div>
                    </td>
                    <td className="px-6 py-3.5">
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
                    <td className="px-6 py-3.5 text-xs">
                      {f.advisory_status === "Critical" ? (
                        <span className="text-[#FF3B30] font-semibold">Deploy water within 12h</span>
                      ) : f.advisory_status === "Urgent" ? (
                        <span className="text-[#FF8800]">Deploy water within 24h</span>
                      ) : (
                        <span className="text-white/50">Monitor daily</span>
                      )}
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
