import { useEffect, useState } from "react";
import axios from "axios";
import Sidebar from "@/components/Sidebar";
import MapPanel from "@/components/MapPanel";
import TimeSeriesPanel from "@/components/TimeSeriesPanel";
import AdvisoryTable from "@/components/AdvisoryTable";
import ActionPanel from "@/components/ActionPanel";
import { Sprout, RefreshCw, Download, FileSpreadsheet, Printer } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function Dashboard() {
  const navigate = useNavigate();
  const [fields, setFields] = useState([]);
  const [geojson, setGeojson] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeNav, setActiveNav] = useState("Overview");
  const [selectedFieldId, setSelectedFieldId] = useState(null);
  const [filteredFieldIds, setFilteredFieldIds] = useState(null);

  const loadAll = async () => {
    setLoading(true);
    try {
      const [f, g] = await Promise.all([
        axios.get(`${API}/fields`),
        axios.get(`${API}/pipeline/geojson`),
      ]);
      setFields(f.data);
      setGeojson(g.data);
      if (f.data.length && !selectedFieldId) setSelectedFieldId(f.data[0].field_id);
    } catch (e) {
      console.error("load failed", e);
      toast.error("Failed to load pipeline data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const stats = computeStats(fields);
  const selectedField = fields.find((f) => f.field_id === selectedFieldId) || fields[0];

  // Export helpers
  const downloadGeoJSON = () => {
    if (!geojson) return;
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: "application/geo+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "krishidrishti_advisory.geojson";
    a.click();
    URL.revokeObjectURL(url);
    toast.success("GeoJSON exported", { description: `${geojson.features?.length || 0} features` });
  };

  const downloadCSV = () => {
    const rows = filteredFieldIds
      ? fields.filter((f) => filteredFieldIds.includes(f.field_id))
      : fields;
    if (!rows.length) {
      toast.error("Nothing to export");
      return;
    }
    const headers = ["field_id", "crop_type", "growth_stage", "csi", "water_deficit_mm", "advisory_status", "latitude", "longitude"];
    const csv = [
      headers.join(","),
      ...rows.map((r) => headers.map((h) => JSON.stringify(r[h] ?? "")).join(",")),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "krishidrishti_advisories.csv";
    a.click();
    URL.revokeObjectURL(url);
    toast.success("CSV exported", { description: `${rows.length} rows` });
  };

  const printPDF = () => {
    toast.info("Preparing PDF summary…", { description: "Use browser Save-as-PDF" });
    setTimeout(() => window.print(), 300);
  };

  return (
    <div data-testid="dashboard-page" className="flex min-h-screen w-full bg-[#0a0f0d] text-white">
      <Sidebar active={activeNav} onSelect={setActiveNav} onHome={() => navigate("/")} />

      <main className="flex-1 overflow-x-hidden">
        {/* Topbar */}
        <div className="flex items-center justify-between border-b border-white/5 px-6 py-4 md:px-10 print:hidden">
          <div>
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-white/50">
              <Sprout size={13} className="text-[#00E65B]" />
              KrishiDrishti · {activeNav}
            </div>
            <h1 className="font-display mt-1 text-2xl font-bold tracking-tight md:text-3xl">
              Field intelligence, live.
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <ExportBtn
              testId="export-geojson-button"
              onClick={downloadGeoJSON}
              icon={<Download size={13} />}
              label="GeoJSON"
              accent="#00E65B"
            />
            <ExportBtn
              testId="export-csv-button"
              onClick={downloadCSV}
              icon={<FileSpreadsheet size={13} />}
              label="CSV"
              accent="#00E5FF"
            />
            <ExportBtn
              testId="export-pdf-button"
              onClick={printPDF}
              icon={<Printer size={13} />}
              label="PDF"
              accent="#FBBF24"
            />
            <button
              onClick={loadAll}
              data-testid="refresh-fields-button"
              className="ml-2 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-xs font-medium text-white/80 transition hover:border-[#00E65B]/40 hover:text-[#00E65B]"
            >
              <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
              Re-simulate
            </button>
          </div>
        </div>

        {/* KPI row */}
        <div className="grid grid-cols-2 gap-3 px-6 pt-6 md:grid-cols-4 md:px-10">
          <StatCard label="Fields monitored" value={stats.total} accent="#00E65B" testId="stat-total" />
          <StatCard label="Adequate" value={stats.adequate} accent="#00E65B" testId="stat-adequate" />
          <StatCard label="Watch + Urgent" value={stats.watch + stats.urgent} accent="#FBBF24" testId="stat-warn" />
          <StatCard label="Critical" value={stats.critical} accent="#FF3B30" testId="stat-critical" />
        </div>

        {/* Map */}
        <div className="px-6 pt-6 md:px-10">
          <MapPanel
            fields={fields}
            geojson={geojson}
            selectedFieldId={selectedFieldId}
            onSelect={setSelectedFieldId}
            loading={loading}
          />
        </div>

        {/* Time-series */}
        <div className="px-6 pt-6 md:px-10">
          <TimeSeriesPanel field={selectedField} />
        </div>

        {/* Table + Action Panel */}
        <div className="grid grid-cols-1 gap-6 px-6 py-6 md:px-10 lg:grid-cols-3">
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
        </div>

        <footer className="border-t border-white/5 px-6 py-4 text-center text-[11px] uppercase tracking-[0.2em] text-white/50 md:px-10">
          KrishiDrishti — Empowering Canal Command Agriculture. (Simulated Data Preview)
        </footer>
      </main>
    </div>
  );
}

function computeStats(fields) {
  const s = { total: fields.length, adequate: 0, watch: 0, urgent: 0, critical: 0 };
  for (const f of fields) {
    const k = f.advisory_status?.toLowerCase();
    if (k in s) s[k] += 1;
  }
  return s;
}

function StatCard({ label, value, accent, testId }) {
  return (
    <div data-testid={testId} className="glass rounded-2xl px-5 py-4 transition hover:-translate-y-0.5">
      <div className="text-[10px] uppercase tracking-[0.2em] text-white/50">{label}</div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="font-display text-3xl font-bold" style={{ color: accent }}>
          {value}
        </span>
        <span className="text-xs text-white/40">fields</span>
      </div>
    </div>
  );
}

function ExportBtn({ testId, onClick, icon, label, accent }) {
  return (
    <button
      data-testid={testId}
      onClick={onClick}
      className="glass inline-flex items-center gap-2 rounded-full px-3.5 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-white/85 transition hover:-translate-y-0.5"
      style={{ borderColor: `${accent}44`, boxShadow: `inset 0 0 0 1px ${accent}33` }}
    >
      <span style={{ color: accent }}>{icon}</span> {label}
    </button>
  );
}
