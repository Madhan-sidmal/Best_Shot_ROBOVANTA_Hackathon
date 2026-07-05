import { useEffect, useState } from "react";
import axios from "axios";
import Sidebar from "@/components/Sidebar";
import Overview from "@/pages/Overview";
import CropClassification from "@/pages/CropClassification";
import StressDetection from "@/pages/StressDetection";
import IrrigationAdvisory from "@/pages/IrrigationAdvisory";
import SimulationEngine from "@/pages/SimulationEngine";
import { useAuth } from "@/contexts/AuthContext";
import { Sprout, RefreshCw, Download, FileSpreadsheet, Printer, LogOut } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function Dashboard() {
  const navigate = useNavigate();
  const { logout, user } = useAuth();
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

  const handleLogout = () => {
    logout();
    toast.success("Logged out successfully");
    navigate("/login");
  };

  return (
    <div data-testid="dashboard-page" className="relative flex min-h-screen w-full bg-[#0a0f0d] text-white">
      {/* Cinematic video backdrop matching the hero section */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-0 h-screen w-screen overflow-hidden"
      >
        <video
          className="h-full w-full object-cover opacity-20 filter blur-[1px]"
          src="https://customer-assets.emergentagent.com/job_366a7cf7-6ad0-4593-a0be-cf5f3d61ff7d/artifacts/5v2yklkl_The_Master_Continuous_Video_Pr.mp4"
          autoPlay
          loop
          muted
          playsInline
        />
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 50% 50%, rgba(10,15,13,0.3) 0%, rgba(10,15,13,0.7) 60%, rgba(10,15,13,0.95) 100%)",
          }}
        />
      </div>

      <Sidebar active={activeNav} onSelect={setActiveNav} onHome={() => navigate("/")} />

      <main className="relative z-10 flex-1 overflow-x-hidden">
        {/* Topbar */}
        <div className="flex flex-wrap items-center justify-between border-b border-white/5 px-6 py-4 md:px-10 print:hidden gap-4">
          <div>
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-white/50">
              <Sprout size={13} className="text-[#00E65B]" />
              KrishiDrishti · {activeNav}
            </div>
            <h1 className="font-display text-frost mt-1 text-2xl font-bold tracking-tight md:text-3xl">
              {user ? `Welcome, ${user.name}` : "Field intelligence, live."}
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
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
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-xs font-medium text-white/80 transition hover:border-[#00E65B]/40 hover:text-[#00E65B]"
            >
              <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
              Re-simulate
            </button>
            <button
              onClick={handleLogout}
              className="inline-flex items-center gap-2 rounded-full border border-red-500/20 bg-red-500/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-red-400 transition hover:bg-red-500/10 hover:text-red-300"
            >
              <LogOut size={13} />
              Logout
            </button>
          </div>
        </div>

        {/* Conditional rendering of subpages */}
        {activeNav === "Overview" && (
          <Overview
            fields={fields}
            geojson={geojson}
            loading={loading}
            selectedFieldId={selectedFieldId}
            setSelectedFieldId={setSelectedFieldId}
            filteredFieldIds={filteredFieldIds}
            setFilteredFieldIds={setFilteredFieldIds}
            stats={stats}
          />
        )}

        {activeNav === "Crop Classification" && (
          <CropClassification
            fields={fields}
            loading={loading}
          />
        )}

        {activeNav === "Stress Detection" && (
          <StressDetection
            fields={fields}
            geojson={geojson}
            loading={loading}
            selectedFieldId={selectedFieldId}
            setSelectedFieldId={setSelectedFieldId}
          />
        )}

        {activeNav === "Irrigation Advisory" && (
          <IrrigationAdvisory
            fields={fields}
            loading={loading}
            selectedFieldId={selectedFieldId}
            setSelectedFieldId={setSelectedFieldId}
            setFilteredFieldIds={setFilteredFieldIds}
          />
        )}

        {activeNav === "Simulation Engine" && (
          <SimulationEngine
            fields={fields}
            loadAll={loadAll}
            loading={loading}
          />
        )}

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
