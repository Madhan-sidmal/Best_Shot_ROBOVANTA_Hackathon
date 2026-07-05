/* SatelliteScene v2 — full-width cinematic Earth + realistic satellite,
   with a scan sweep and a rich right-hand telemetry stack. */
import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

const EARTH_IMG =
  "https://images.unsplash.com/photo-1446776653964-20c1d3a81b06?crop=entropy&cs=srgb&fm=jpg&w=2600&q=90";

const LAYERS = [
  {
    key: "optical",
    label: "Sentinel-2 Optical",
    tone: "#00E65B",
    detail: "10 m · 5-day revisit · NDVI · EVI · NDWI",
    band: "B2/B3/B4/B8/B11/B12 · SCL cloud mask",
  },
  {
    key: "sar",
    label: "Sentinel-1 SAR",
    tone: "#00E5FF",
    detail: "C-band · all-weather · Refined-Lee speckle filter",
    band: "VV · VH · VH/VV · RVI",
  },
  {
    key: "weather",
    label: "MODIS MOD16A2 ET₀",
    tone: "#FBBF24",
    detail: "8-day reference evapotranspiration",
    band: "ETo · ETc = Kc × ETo",
  },
  {
    key: "rainfall",
    label: "IMD Rainfall Grid",
    tone: "#79C2FF",
    detail: "0.25° gridded · effective rainfall",
    band: "Pe = 0.8 × P",
  },
];

function RealisticSatellite({ tone }) {
  return (
    <svg
      viewBox="-160 -80 320 160"
      className="pointer-events-none absolute left-1/2 top-[38%] h-[300px] w-[600px] -translate-x-1/2 -translate-y-1/2 drop-shadow-[0_20px_60px_rgba(0,0,0,0.75)]"
    >
      <defs>
        <linearGradient id="body-grad2" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f5f6f8" />
          <stop offset="45%" stopColor="#c2c6cc" />
          <stop offset="100%" stopColor="#5a5f66" />
        </linearGradient>
        <linearGradient id="foil-grad2" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#f8c15b" />
          <stop offset="45%" stopColor="#c88a2a" />
          <stop offset="100%" stopColor="#8a5411" />
        </linearGradient>
        <linearGradient id="panel-grad2" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2a4b78" />
          <stop offset="55%" stopColor="#132a4a" />
          <stop offset="100%" stopColor="#0a1a30" />
        </linearGradient>
        <radialGradient id="lens-grad2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.95" />
          <stop offset="60%" stopColor={tone} stopOpacity="0.6" />
          <stop offset="100%" stopColor={tone} stopOpacity="0" />
        </radialGradient>
        <pattern id="panel-cells2" width="7" height="7" patternUnits="userSpaceOnUse">
          <rect width="7" height="7" fill="url(#panel-grad2)" />
          <rect x="0" y="0" width="6.6" height="6.6" fill="none" stroke="rgba(140,190,240,0.32)" strokeWidth="0.4" />
        </pattern>
      </defs>

      <g transform="translate(-105,0) rotate(-3)">
        <rect x="-52" y="-18" width="52" height="36" fill="url(#panel-cells2)" stroke="#7ea6c9" strokeOpacity="0.4" strokeWidth="0.7" />
        <rect x="-52" y="-18" width="52" height="36" fill="rgba(255,255,255,0.05)" />
      </g>
      <rect x="-53" y="-1.3" width="8" height="2.6" fill="#8a8f96" />
      <g transform="translate(105,0) rotate(3)">
        <rect x="0" y="-18" width="52" height="36" fill="url(#panel-cells2)" stroke="#7ea6c9" strokeOpacity="0.4" strokeWidth="0.7" />
        <rect x="0" y="-18" width="52" height="36" fill="rgba(255,255,255,0.05)" />
      </g>
      <rect x="45" y="-1.3" width="8" height="2.6" fill="#8a8f96" />
      <rect x="-30" y="-22" width="60" height="44" rx="4" fill="url(#body-grad2)" stroke="#40454c" strokeWidth="0.6" />
      <rect x="-27" y="-19" width="54" height="16" fill="url(#foil-grad2)" opacity="0.95" />
      <rect x="-27" y="-19" width="54" height="16" fill="url(#foil-grad2)" opacity="0.6" transform="translate(0,3) skewX(-6)" />
      {[-14, -10, -6, -2, 2, 6, 10, 14].map((x) => (
        <line key={x} x1={x} y1="-19" x2={x + 2} y2="-3" stroke="rgba(255,235,180,0.4)" strokeWidth="0.4" />
      ))}
      <rect x="-27" y="4" width="54" height="4" fill="rgba(255,255,255,0.18)" />
      <rect x="-22" y="10" width="10" height="9" fill="#22252a" stroke="#40454c" strokeWidth="0.4" />
      <circle cx="-13" cy="14.5" r="1.6" fill="#00E65B" />
      <rect x="0" y="10" width="10" height="9" fill="#22252a" stroke="#40454c" strokeWidth="0.4" />
      <circle cx="9" cy="14.5" r="1.6" fill="#FBBF24" />
      <path d="M-4,22 L4,22 L2,32 L-2,32 Z" fill="#2b2f36" stroke="#5a5f66" strokeWidth="0.5" />
      <circle cx="0" cy="32" r="4" fill="#0a0f12" stroke="#3a3f46" strokeWidth="0.6" />
      <circle cx="0" cy="32" r="3" fill="url(#lens-grad2)" />
      <g transform="translate(20,-28)">
        <ellipse cx="0" cy="0" rx="10" ry="4.5" fill="#dadde1" />
        <ellipse cx="0" cy="-0.4" rx="9" ry="3.6" fill="#8a8f96" />
        <line x1="0" y1="0" x2="0" y2="-8" stroke="#8a8f96" strokeWidth="0.6" />
        <circle cx="0" cy="-8" r="1.2" fill="#00E65B" />
      </g>
      <line x1="-20" y1="-22" x2="-20" y2="-32" stroke="#c2c6cc" strokeWidth="0.8" />
      <circle cx="-20" cy="-33" r="1.5" fill={tone} opacity="0.9" />
    </svg>
  );
}

export default function SatelliteScene() {
  const sectionRef = useRef(null);
  const inView = useInView(sectionRef, { once: true, margin: "-10% 0px" });
  const [activeLayer, setActiveLayer] = useState("optical");
  const layerObj = LAYERS.find((l) => l.key === activeLayer) || LAYERS[0];

  return (
    <section
      ref={sectionRef}
      data-testid="scene-satellite"
      className="relative min-h-screen w-full overflow-hidden"
    >
      {/* Full-width Earth backdrop */}
      <img
        src={EARTH_IMG}
        alt="Earth from orbit"
        className="absolute inset-0 h-full w-full object-cover"
        style={{ filter: "brightness(0.85) contrast(1.08) saturate(0.95)" }}
      />
      {/* Deep-space tint */}
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(circle at 50% 100%, rgba(0,60,120,0.35) 0%, rgba(0,0,0,0.85) 60%), linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.55) 50%, rgba(0,0,0,0.85) 100%)",
        }}
      />
      {/* Star sparks */}
      <svg className="absolute inset-0 h-full w-full opacity-60" xmlns="http://www.w3.org/2000/svg">
        {Array.from({ length: 85 }).map((_, i) => (
          <circle
            key={i}
            cx={`${(i * 71) % 100}%`}
            cy={`${(i * 37) % 65}%`}
            r={i % 8 === 0 ? 1.5 : 0.55}
            fill="rgba(255,255,255,0.85)"
          />
        ))}
      </svg>

      {/* Orbit path — big and cinematic */}
      <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none" viewBox="0 0 100 100">
        <defs>
          <linearGradient id="orbit-line2" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={layerObj.tone} stopOpacity="0" />
            <stop offset="50%" stopColor={layerObj.tone} stopOpacity="0.85" />
            <stop offset="100%" stopColor={layerObj.tone} stopOpacity="0" />
          </linearGradient>
        </defs>
        <ellipse cx="50" cy="42" rx="46" ry="7.5" fill="none" stroke="url(#orbit-line2)" strokeWidth="0.28" strokeDasharray="0.5 0.5" />
        <ellipse cx="50" cy="42" rx="46" ry="7.5" fill="none" stroke={layerObj.tone} strokeOpacity="0.12" strokeWidth="0.55" />
      </svg>

      {/* Animated satellite drifting along orbit */}
      <motion.div
        className="absolute inset-x-0 top-0 h-full"
        animate={{ x: [-80, 0, 80, 0, -80], y: [-8, 0, 8, 0, -8] }}
        transition={{ duration: 28, ease: "linear", repeat: Infinity }}
      >
        <RealisticSatellite tone={layerObj.tone} />
      </motion.div>

      {/* Scanning cone — wider, more cinematic */}
      <motion.svg
        key={activeLayer}
        className="pointer-events-none absolute inset-0 h-full w-full"
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <defs>
          <linearGradient id="scan-cone2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={layerObj.tone} stopOpacity="0.55" />
            <stop offset="100%" stopColor={layerObj.tone} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d="M 46 40 L 54 40 L 68 100 L 32 100 Z" fill="url(#scan-cone2)" />
      </motion.svg>

      {/* Ground target ping — bottom center */}
      <div
        className="absolute left-1/2 top-[86%] -translate-x-1/2"
        style={{
          width: "34px",
          height: "34px",
          borderRadius: "999px",
          border: `1.5px solid ${layerObj.tone}`,
          boxShadow: `0 0 32px ${layerObj.tone}aa`,
        }}
      >
        <div
          className="pulse-dot absolute inset-2 rounded-full"
          style={{ background: layerObj.tone }}
        />
      </div>

      {/* Header text */}
      <div className="relative z-10 mx-auto max-w-7xl px-6 pt-24 md:px-12">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9 }}
        >
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-white/70">
            Chapter Three · Eyes in orbit
          </div>
          <h3 className="font-display text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
            One farm. <span className="text-frost">Four data streams.</span>
          </h3>
          <p className="mt-4 max-w-2xl text-white/75">
            A live-imaging constellation of Sentinel-1, Sentinel-2 and MODIS fuses every
            8&#8209;day composite — optical vegetation, radar backscatter, evapotranspiration
            and rainfall — scanning through every cloud.
          </p>
        </motion.div>
      </div>

      {/* Telemetry stack — bottom left */}
      <div className="pointer-events-none absolute bottom-24 left-6 z-10 flex flex-wrap items-center gap-3 rounded-2xl border border-white/10 bg-black/50 px-4 py-3 text-[10px] uppercase tracking-[0.2em] text-white/80 backdrop-blur md:left-12">
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-[#00E65B] pulse-dot" />
          Live orbit · LEO 786 km
        </span>
        <span className="opacity-40">|</span>
        <span style={{ color: layerObj.tone }}>{layerObj.label} streaming</span>
        <span className="opacity-40">|</span>
        <span className="font-mono normal-case tracking-normal text-white/60">{layerObj.band}</span>
      </div>

      {/* Layer buttons — bottom right, glass */}
      <div
        data-testid="satellite-canvas"
        className="pointer-events-auto absolute bottom-6 right-6 z-10 flex flex-col gap-2 md:right-12"
      >
        {LAYERS.map((l) => {
          const active = l.key === activeLayer;
          return (
            <button
              key={l.key}
              data-testid={`sat-layer-${l.key}`}
              onClick={() => setActiveLayer(l.key)}
              className={`group flex w-[280px] items-start gap-3 rounded-2xl border p-3 text-left transition ${
                active
                  ? "border-white/30 bg-black/60"
                  : "border-white/10 bg-black/40 hover:border-white/20 hover:bg-black/55"
              }`}
              style={active ? { boxShadow: `inset 0 0 0 1px ${l.tone}aa, 0 0 22px ${l.tone}26` } : {}}
            >
              <span
                className="mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: l.tone, boxShadow: active ? `0 0 12px ${l.tone}` : "none" }}
              />
              <div className="min-w-0">
                <div className="font-display text-sm font-semibold text-white">{l.label}</div>
                <div className="mt-0.5 font-mono text-[9.5px] uppercase tracking-[0.14em] text-white/60">
                  {l.detail}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
