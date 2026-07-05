import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";

const STATUS_STYLES = {
  Adequate: "bg-[#00E65B]/12 text-[#00E65B] border-[#00E65B]/30",
  Watch: "bg-[#FBBF24]/12 text-[#FBBF24] border-[#FBBF24]/30",
  Urgent: "bg-[#FF8A00]/12 text-[#FF8A00] border-[#FF8A00]/30",
  Critical: "bg-[#FF3B30]/12 text-[#FF3B30] border-[#FF3B30]/30",
};

const CROPS = ["All", "Rice", "Wheat", "Cotton", "Sugarcane"];
const STATUSES = ["All", "Adequate", "Watch", "Urgent", "Critical"];

export default function AdvisoryTable({ fields, loading, selectedFieldId, onRowSelect, onFilteredIdsChange }) {
  const [crop, setCrop] = useState("All");
  const [status, setStatus] = useState("All");
  const [stage, setStage] = useState("All");
  const [q, setQ] = useState("");

  const stages = useMemo(() => {
    const s = new Set(fields.map((f) => f.growth_stage));
    return ["All", ...Array.from(s).sort()];
  }, [fields]);

  const rows = useMemo(() => {
    return fields.filter((f) => {
      if (crop !== "All" && f.crop_type !== crop) return false;
      if (status !== "All" && f.advisory_status !== status) return false;
      if (stage !== "All" && f.growth_stage !== stage) return false;
      if (q && !f.field_id.toLowerCase().includes(q.toLowerCase())) return false;
      return true;
    });
  }, [fields, crop, status, stage, q]);

  useEffect(() => {
    if (onFilteredIdsChange) onFilteredIdsChange(rows.map((r) => r.field_id));
  }, [rows, onFilteredIdsChange]);

  return (
    <section
      data-testid="advisory-table"
      className="glass overflow-hidden rounded-2xl"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/5 px-5 py-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/50">
            Field Advisories
          </div>
          <div className="font-display mt-0.5 text-lg font-semibold">
            {rows.length} of {fields.length} fields
          </div>
        </div>
        <div className="relative">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40" />
          <input
            data-testid="filter-search-input"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search field ID"
            className="w-52 rounded-lg border border-white/10 bg-black/30 py-2 pl-8 pr-3 text-xs text-white placeholder:text-white/40 focus:border-[#00E65B]/60 focus:outline-none"
          />
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2 px-5 py-3">
        <FilterPills label="Crop" options={CROPS} value={crop} onChange={setCrop} testIdPrefix="filter-crop" />
        <FilterPills label="Stage" options={stages} value={stage} onChange={setStage} testIdPrefix="filter-stage" />
        <FilterPills label="Status" options={STATUSES} value={status} onChange={setStatus} testIdPrefix="filter-status" />
      </div>

      <div className="max-h-[440px] overflow-auto">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-[#0d1712]/95 backdrop-blur">
            <tr className="text-[10px] uppercase tracking-[0.16em] text-white/50">
              <th className="px-5 py-3 font-semibold">Field ID</th>
              <th className="px-5 py-3 font-semibold">Crop</th>
              <th className="px-5 py-3 font-semibold">Growth Stage</th>
              <th className="px-5 py-3 font-semibold">CSI</th>
              <th className="px-5 py-3 font-semibold">Water Deficit</th>
              <th className="px-5 py-3 font-semibold">Advisory</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={6} className="px-5 py-10 text-center text-white/40">
                  Loading synthetic field data…
                </td>
              </tr>
            )}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-5 py-10 text-center text-white/40">
                  No fields match your filters.
                </td>
              </tr>
            )}
            {!loading &&
              rows.map((f) => {
                const isSel = f.field_id === selectedFieldId;
                return (
                  <tr
                    key={f.field_id}
                    data-testid={`row-${f.field_id}`}
                    onClick={() => onRowSelect(f.field_id)}
                    className={`cursor-pointer border-t border-white/5 transition ${
                      isSel ? "bg-[#00E65B]/[0.06]" : "hover:bg-white/[0.02]"
                    }`}
                  >
                    <td className="px-5 py-3 font-mono text-xs text-white/90">{f.field_id}</td>
                    <td className="px-5 py-3 text-white/85">{f.crop_type}</td>
                    <td className="px-5 py-3 text-white/70">{f.growth_stage}</td>
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${f.csi * 100}%`,
                              background:
                                f.csi > 0.78
                                  ? "#FF3B30"
                                  : f.csi > 0.55
                                    ? "#FF8A00"
                                    : f.csi > 0.3
                                      ? "#FBBF24"
                                      : "#00E65B",
                            }}
                          />
                        </div>
                        <span className="font-mono text-xs text-white/70">
                          {f.csi.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="px-5 py-3 font-mono text-xs text-[#00E5FF]">
                      {f.water_deficit_mm.toFixed(1)} mm
                    </td>
                    <td className="px-5 py-3">
                      <span
                        className={`inline-flex rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] ${STATUS_STYLES[f.advisory_status]}`}
                      >
                        {f.advisory_status}
                      </span>
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function FilterPills({ label, options, value, onChange, testIdPrefix }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-[0.2em] text-white/40">{label}</span>
      <div className="flex flex-wrap gap-1">
        {options.map((opt) => {
          const active = value === opt;
          return (
            <button
              key={opt}
              data-testid={`${testIdPrefix}-${opt.toLowerCase().replace(/\s+/g, "-")}`}
              onClick={() => onChange(opt)}
              className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${
                active
                  ? "border-[#00E65B]/50 bg-[#00E65B]/10 text-[#00E65B]"
                  : "border-white/10 bg-white/[0.02] text-white/60 hover:text-white"
              }`}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}
