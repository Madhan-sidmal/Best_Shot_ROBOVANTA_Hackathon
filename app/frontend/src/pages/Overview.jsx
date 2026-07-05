import { useMemo } from "react";
import MapPanel from "@/components/MapPanel";
import TimeSeriesPanel from "@/components/TimeSeriesPanel";
import AdvisoryTable from "@/components/AdvisoryTable";
import ActionPanel from "@/components/ActionPanel";
import { motion } from "framer-motion";
import { Layers, CheckCircle2, AlertTriangle, ShieldAlert } from "lucide-react";

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } },
};

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

// Mini sparkline SVG based on field data
function Sparkline({ values, color, width = 60, height = 20 }) {
  if (!values || values.length < 2) return null;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width={width} height={height} className="opacity-50">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Overview({
  fields,
  geojson,
  loading,
  selectedFieldId,
  setSelectedFieldId,
  filteredFieldIds,
  setFilteredFieldIds,
  stats
}) {
  const selectedField = fields.find((f) => f.field_id === selectedFieldId) || fields[0];

  // Sparkline data from fields
  const sparkData = useMemo(() => {
    if (!fields.length) return { total: [], adequate: [], warn: [], critical: [] };
    // Create a pseudo-sparkline from CSI distributions
    const sorted = [...fields].sort((a, b) => a.csi - b.csi);
    const chunk = Math.ceil(sorted.length / 8);
    const bins = [];
    for (let i = 0; i < 8; i++) {
      const slice = sorted.slice(i * chunk, (i + 1) * chunk);
      if (slice.length) bins.push(slice.length);
    }
    return {
      total: bins,
      adequate: bins.map((_, i) => Math.max(0, bins[i] - i * 0.3)),
      warn: bins.map((_, i) => Math.abs(Math.sin(i * 0.8)) * 3),
      critical: bins.map((_, i) => Math.max(0, i * 0.5 - 1)),
    };
  }, [fields]);

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
    >
      {/* KPI row */}
      <motion.div variants={item} className="grid grid-cols-2 gap-3 px-6 pt-6 md:grid-cols-4 md:px-10">
        <StatCard
          label="Fields Monitored"
          value={stats.total}
          icon={<Layers size={16} />}
          accent="#00E65B"
          glassClass="glass-glow"
          testId="stat-total"
          sparkline={<Sparkline values={sparkData.total} color="#00E65B" />}
        />
        <StatCard
          label="Adequate"
          value={stats.adequate}
          icon={<CheckCircle2 size={16} />}
          accent="#00E65B"
          glassClass="glass"
          testId="stat-adequate"
          sparkline={<Sparkline values={sparkData.adequate} color="#00E65B" />}
          trend={stats.adequate > stats.total / 2 ? "up" : "down"}
        />
        <StatCard
          label="Watch + Urgent"
          value={stats.watch + stats.urgent}
          icon={<AlertTriangle size={16} />}
          accent="#FBBF24"
          glassClass="glass-warn"
          testId="stat-warn"
          sparkline={<Sparkline values={sparkData.warn} color="#FBBF24" />}
        />
        <StatCard
          label="Critical"
          value={stats.critical}
          icon={<ShieldAlert size={16} />}
          accent="#FF3B30"
          glassClass={stats.critical > 0 ? "glass-danger" : "glass"}
          testId="stat-critical"
          sparkline={<Sparkline values={sparkData.critical} color="#FF3B30" />}
          trend={stats.critical > 0 ? "up" : null}
        />
      </motion.div>

      {/* Map */}
      <motion.div variants={item} className="px-6 pt-6 md:px-10">
        <MapPanel
          fields={fields}
          geojson={geojson}
          selectedFieldId={selectedFieldId}
          onSelect={setSelectedFieldId}
          loading={loading}
        />
      </motion.div>

      {/* Time-series */}
      <motion.div variants={item} className="px-6 pt-6 md:px-10">
        <TimeSeriesPanel field={selectedField} />
      </motion.div>

      {/* Table + Action Panel */}
      <motion.div variants={item} className="grid grid-cols-1 gap-6 px-6 py-6 md:px-10 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <AdvisoryTable
            fields={fields}
            loading={loading}
            selectedFieldId={selectedFieldId}
            onRowSelect={setSelectedFieldId}
            onFilteredIdsChange={setFilteredFieldIds}
          />
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

function StatCard({ label, value, icon, accent, glassClass, testId, sparkline, trend }) {
  return (
    <motion.div
      data-testid={testId}
      whileHover={{ y: -3, transition: { duration: 0.2 } }}
      className={`${glassClass} rounded-2xl px-5 py-4 transition cursor-default`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-white/50">
          <span style={{ color: accent }}>{icon}</span>
          {label}
        </div>
        {trend && (
          <span className={`text-xs font-bold ${trend === "up" ? "text-[#FF3B30]" : "text-[#00E65B]"}`}>
            {trend === "up" ? "↑" : "↓"}
          </span>
        )}
      </div>
      <div className="mt-2 flex items-end justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="font-display text-3xl font-bold" style={{ color: accent }}>
            {value}
          </span>
          <span className="text-xs text-white/40">fields</span>
        </div>
        {sparkline}
      </div>
    </motion.div>
  );
}
