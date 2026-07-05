import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

/* FarmScene — cinematic aerial farm view.
   Real photo of green fields as backdrop + SVG grid + colored field polygons
   representing the four target crops + animated scan sweep. */

const FARM_AERIAL =
  "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=2400&q=90";

// Approximate field polygons — laid out as a patchwork over the aerial photo.
// Each field has: crop type, coordinates (percent of container), CSI (stress index).
const FIELDS = [
  { id: "FLD-1000", crop: "Rice",       color: "#00E65B", poly: "8,22 32,18 34,42 10,46",   x: 20, y: 32, csi: 0.22 },
  { id: "FLD-1004", crop: "Wheat",      color: "#FBBF24", poly: "36,16 62,14 65,38 38,42",  x: 50, y: 28, csi: 0.55 },
  { id: "FLD-1009", crop: "Cotton",     color: "#FF8A00", poly: "68,12 92,10 94,36 70,40",  x: 80, y: 24, csi: 0.68 },
  { id: "FLD-1012", crop: "Sugarcane",  color: "#00E5FF", poly: "6,50 34,48 32,74 8,76",    x: 20, y: 62, csi: 0.34 },
  { id: "FLD-1015", crop: "Rice",       color: "#00E65B", poly: "36,46 64,44 66,72 38,74",  x: 50, y: 60, csi: 0.12 },
  { id: "FLD-1019", crop: "Wheat",      color: "#FBBF24", poly: "68,44 94,42 96,72 70,72",  x: 82, y: 58, csi: 0.78 },
  { id: "FLD-1023", crop: "Cotton",     color: "#FF8A00", poly: "10,80 40,78 38,94 12,96",  x: 24, y: 88, csi: 0.44 },
  { id: "FLD-1026", crop: "Sugarcane",  color: "#00E5FF", poly: "44,78 78,76 80,94 46,96",  x: 62, y: 87, csi: 0.28 },
];

const LEGEND = [
  { crop: "Rice",      color: "#00E65B" },
  { crop: "Wheat",     color: "#FBBF24" },
  { crop: "Cotton",    color: "#FF8A00" },
  { crop: "Sugarcane", color: "#00E5FF" },
];

export default function FarmScene() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [hovered, setHovered] = useState(FIELDS[4].id);

  const active = FIELDS.find((f) => f.id === hovered) || FIELDS[0];
  const status =
    active.csi < 0.3 ? "Adequate" : active.csi < 0.55 ? "Watch" : active.csi < 0.78 ? "Urgent" : "Critical";
  const statusColor =
    status === "Adequate" ? "#00E65B" : status === "Watch" ? "#FBBF24" : status === "Urgent" ? "#FF8A00" : "#FF3B30";

  return (
    <section
      ref={ref}
      data-testid="scene-farm"
      className="relative min-h-screen w-full overflow-hidden"
    >
      <img
        src={FARM_AERIAL}
        alt="aerial farm view"
        className="absolute inset-0 h-full w-full object-cover"
        style={{ filter: "brightness(0.72) contrast(1.05) saturate(0.9)" }}
      />
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.65) 0%, rgba(0,0,0,0.25) 40%, rgba(0,0,0,0.75) 100%)",
        }}
      />

      {/* Header */}
      <div className="relative z-10 mx-auto max-w-7xl px-6 pt-24 md:px-12">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9 }}
        >
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-white/70">
            Chapter Three · Zooming in
          </div>
          <h3 className="font-display text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
            From orbit to <span className="text-frost">every field.</span>
          </h3>
          <p className="mt-4 max-w-2xl text-white/75">
            Sentinel-2 10 m resolution + Sentinel-1 SAR fuse into a per-pixel crop map. Four
            crops classified with a Random-Forest + XGBoost ensemble at{" "}
            <span className="text-[#00E65B] font-semibold">≥85% overall accuracy</span>.
          </p>
        </motion.div>
      </div>

      {/* Field polygon overlay */}
      <div
        data-testid="farm-map"
        className="pointer-events-auto absolute left-1/2 top-1/2 aspect-[16/8] w-[min(1200px,92vw)] -translate-x-1/2 -translate-y-1/2"
      >
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
          <defs>
            <pattern id="crop-grid" width="4" height="4" patternUnits="userSpaceOnUse">
              <path d="M4 0H0V4" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="0.15" />
            </pattern>
            <linearGradient id="scan-sweep" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#00E5FF" stopOpacity="0" />
              <stop offset="50%" stopColor="#00E5FF" stopOpacity="0.5" />
              <stop offset="100%" stopColor="#00E5FF" stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect width="100" height="100" fill="url(#crop-grid)" />

          {/* Field polygons */}
          {FIELDS.map((f) => {
            const isActive = f.id === hovered;
            return (
              <g key={f.id}>
                <polygon
                  data-testid={`farm-field-${f.id}`}
                  points={f.poly}
                  fill={f.color}
                  fillOpacity={isActive ? 0.55 : 0.28}
                  stroke={f.color}
                  strokeOpacity={isActive ? 1 : 0.6}
                  strokeWidth={isActive ? 0.55 : 0.3}
                  onMouseEnter={() => setHovered(f.id)}
                  style={{ cursor: "pointer", transition: "all 0.25s" }}
                />
              </g>
            );
          })}

          {/* Sweep animation */}
          <motion.rect
            initial={{ x: -20 }}
            animate={{ x: 120 }}
            transition={{ duration: 6, ease: "linear", repeat: Infinity }}
            y="0"
            width="18"
            height="100"
            fill="url(#scan-sweep)"
          />
        </svg>

        {/* Field pins for each polygon (positioned in %) */}
        {FIELDS.map((f) => {
          const isActive = f.id === hovered;
          return (
            <button
              key={f.id}
              onMouseEnter={() => setHovered(f.id)}
              onClick={() => setHovered(f.id)}
              className="absolute -translate-x-1/2 -translate-y-1/2 focus:outline-none"
              style={{ left: `${f.x}%`, top: `${f.y}%` }}
              data-testid={`farm-pin-${f.id}`}
            >
              <span
                className={`absolute inset-0 -m-3 rounded-full transition ${
                  isActive ? "opacity-70" : "opacity-30"
                }`}
                style={{ background: f.color, filter: "blur(6px)" }}
              />
              <span
                className="relative block h-3 w-3 rounded-full ring-2"
                style={{ background: f.color, borderColor: "#fff" }}
              />
            </button>
          );
        })}

        {/* Selected field tooltip */}
        <motion.div
          key={active.id}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute -translate-x-1/2 -translate-y-full rounded-xl border border-white/15 bg-black/70 px-4 py-2 text-left backdrop-blur"
          style={{ left: `${active.x}%`, top: `${active.y - 3}%` }}
        >
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/50">
            {active.id}
          </div>
          <div className="font-display text-sm font-semibold text-white">{active.crop}</div>
          <div className="mt-1 flex items-center gap-2 text-[10px] uppercase tracking-[0.16em]">
            <span className="text-white/50">Status</span>
            <span
              className="rounded-full px-1.5 py-0.5"
              style={{ background: `${statusColor}22`, color: statusColor, border: `1px solid ${statusColor}55` }}
            >
              {status}
            </span>
            <span className="font-mono text-white/50 normal-case tracking-normal">
              CSI {active.csi.toFixed(2)}
            </span>
          </div>
        </motion.div>
      </div>

      {/* Legend + metrics bottom-left */}
      <div className="pointer-events-none absolute bottom-6 left-6 z-10 flex flex-wrap items-center gap-3 rounded-2xl border border-white/10 bg-black/55 px-4 py-3 text-[10px] uppercase tracking-[0.18em] text-white/85 backdrop-blur md:left-12">
        {LEGEND.map((l) => (
          <span key={l.crop} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: l.color }} />
            {l.crop}
          </span>
        ))}
        <span className="opacity-40">|</span>
        <span className="text-[#00E65B]">≥85% OA</span>
        <span className="opacity-40">|</span>
        <span className="font-mono normal-case tracking-normal text-white/60">
          {FIELDS.length} fields · Cohen&rsquo;s κ &gt; 0.84
        </span>
      </div>

      {/* Bottom-right stat pills */}
      <div className="pointer-events-none absolute bottom-6 right-6 z-10 flex flex-col gap-2 md:right-12">
        <StatPill label="Sentinel-2 · 10 m" tone="#00E65B" />
        <StatPill label="Sentinel-1 · C-band" tone="#00E5FF" />
        <StatPill label="MODIS · 8-day ET" tone="#FBBF24" />
      </div>
    </section>
  );
}

function StatPill({ label, tone }) {
  return (
    <div
      className="rounded-full border bg-black/55 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] backdrop-blur"
      style={{ borderColor: `${tone}55`, color: tone }}
    >
      {label}
    </div>
  );
}
