import { useRef } from "react";
import {
  motion,
  useScroll,
  useTransform,
  useInView,
  AnimatePresence,
} from "framer-motion";
import { Loader2, AlertCircle } from "lucide-react";
import useInsights from "@/hooks/useInsights";

/* Realistic plant scene.
   - Photo-realistic plant SVG (multi-stop gradients, veins, curl, ambient shadows)
   - Photo backdrop of live foliage for depth (subtle, behind pot)
   - Scroll animation is *sticky-aware*: uses ["start start","end end"] offset so
     leaves only detach while the sticky stage is actually on-screen.
   - Cards fade in staged; leaves fade fully to 0 by end. Values pulled from
     /api/pipeline/insights, gracefully labeled Demo if pipeline files missing. */

const FOLIAGE_BG =
  "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?auto=format&fit=crop&w=1800&q=85";

const LEAVES = [
  {
    key: "crop",
    label: "Crop Detected",
    accent: "#00E65B",
    anchor: { x: -32, y: -110, rot: -42 },
    fall: { x: -50, y: -40, rot: -70 },
    card: { top: "6%", left: "3%" },
  },
  {
    key: "growth_stage",
    label: "Growth Stage",
    accent: "#00E65B",
    anchor: { x: 32, y: -110, rot: 42 },
    fall: { x: 50, y: -40, rot: 70 },
    card: { top: "6%", right: "3%" },
  },
  {
    key: "moisture_stress",
    label: "Moisture Stress",
    accent: "#00E5FF",
    anchor: { x: -48, y: -40, rot: -72 },
    fall: { x: -80, y: 30, rot: -110 },
    card: { top: "42%", left: "3%" },
  },
  {
    key: "water_deficit_mm",
    label: "Water Deficit",
    accent: "#00E5FF",
    anchor: { x: 48, y: -40, rot: 72 },
    fall: { x: 80, y: 30, rot: 110 },
    card: { top: "42%", right: "3%" },
  },
  {
    key: "irrigation_advisory",
    label: "Irrigation Advisory",
    accent: "#FBBF24",
    anchor: { x: 0, y: 30, rot: 0 },
    fall: { x: 0, y: 80, rot: 25 },
    card: { bottom: "4%", left: "50%", transform: "translateX(-50%)" },
  },
];

function formatValue(key, entry) {
  if (!entry) return "—";
  const v = entry.value;
  if (v === null || v === undefined) return "—";
  if (key === "moisture_stress") {
    const label = entry.label || "";
    return `${label} · CSI ${Number(v).toFixed(2)}`.trim();
  }
  if (key === "water_deficit_mm") return `${Number(v).toFixed(1)} mm`;
  return String(v);
}

function InsightCard({ label, value, source, accent, testId }) {
  const isDemo = source === "demo";
  return (
    <div
      data-testid={testId}
      className="w-[220px] rounded-2xl bg-black/55 p-4 shadow-[0_16px_48px_rgba(0,0,0,0.6)] backdrop-blur-xl"
      style={{
        border: `1px solid ${accent}55`,
        boxShadow: `0 16px 48px rgba(0,0,0,0.65), inset 0 0 0 1px ${accent}22, 0 0 24px ${accent}18`,
      }}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[9px] font-semibold uppercase tracking-[0.22em]" style={{ color: accent }}>
          {label}
        </span>
        <span
          className="rounded-full border px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-[0.14em]"
          style={{
            borderColor: isDemo ? "rgba(251,191,36,0.4)" : "rgba(0,230,91,0.4)",
            color: isDemo ? "#FBBF24" : "#00E65B",
            background: isDemo ? "rgba(251,191,36,0.06)" : "rgba(0,230,91,0.06)",
          }}
        >
          {isDemo ? "Demo" : "Live"}
        </span>
      </div>
      <div className="mt-2 font-display text-base font-semibold leading-snug text-white">{value}</div>
    </div>
  );
}

/* Photorealistic leaf shape via multi-stop radial + linear gradients + veins */
function LeafShape({ index, accent }) {
  const gradId = `leaf-grad-${index}`;
  const highlightId = `leaf-hi-${index}`;
  const dropId = `leaf-drop-${index}`;
  return (
    <g>
      <defs>
        <radialGradient id={gradId} cx="30%" cy="35%" r="80%">
          <stop offset="0%" stopColor="#a6f0b6" />
          <stop offset="35%" stopColor="#4fd06f" />
          <stop offset="72%" stopColor="#188a3a" />
          <stop offset="100%" stopColor="#053017" />
        </radialGradient>
        <linearGradient id={highlightId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(255,255,255,0.55)" />
          <stop offset="60%" stopColor="rgba(255,255,255,0.05)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0)" />
        </linearGradient>
        <filter id={dropId} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feOffset dx="0" dy="4" result="offsetBlur" />
          <feMerge>
            <feMergeNode in="offsetBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <g filter={`url(#${dropId})`}>
        {/* leaf body */}
        <path
          d="M0,-42
             C 26,-38 42,-22 40,4
             C 36,26 18,42 0,42
             C -18,42 -36,26 -40,4
             C -42,-22 -26,-38 0,-42 Z"
          fill={`url(#${gradId})`}
        />
        {/* subtle top highlight */}
        <path
          d="M-14,-30 C -6,-38 8,-38 16,-32 C 12,-22 4,-18 -4,-20 C -12,-22 -16,-26 -14,-30 Z"
          fill={`url(#${highlightId})`}
          opacity="0.6"
        />
        {/* midrib */}
        <path
          d="M0,-40 C 0.5,-20 0.5,20 0,40"
          stroke="rgba(10,40,20,0.85)"
          strokeWidth="1.2"
          fill="none"
        />
        {/* secondary veins */}
        {[-28, -18, -8, 6, 18, 28].map((y, i) => (
          <g key={i} stroke="rgba(6,50,25,0.55)" strokeWidth="0.7" fill="none">
            <path d={`M0,${y} Q ${16 * (y > 0 ? 1 : 1)},${y + 4} ${28},${y + 8}`} />
            <path d={`M0,${y} Q ${-16},${y + 4} ${-28},${y + 8}`} />
          </g>
        ))}
        {/* accent glow tint (subtle) */}
        <path
          d="M0,-42 C 26,-38 42,-22 40,4 C 36,26 18,42 0,42 C -18,42 -36,26 -40,4 C -42,-22 -26,-38 0,-42 Z"
          fill={accent}
          fillOpacity="0.08"
        />
      </g>
    </g>
  );
}

function Leaf({ leaf, index, progress }) {
  // Compressed sticky-aware windows: all leaves detach WHILE the plant is
  // stuck in the viewport centre (progress 0.10-0.55 of the section scroll).
  const start = 0.10 + index * 0.075;
  const end = start + 0.14;
  const p = useTransform(progress, [start, end], [0, 1], { clamp: true });
  const tx = useTransform(p, [0, 1], [leaf.anchor.x, leaf.anchor.x + leaf.fall.x]);
  const ty = useTransform(p, [0, 1], [leaf.anchor.y, leaf.anchor.y + leaf.fall.y]);
  const rot = useTransform(p, [0, 1], [leaf.anchor.rot, leaf.anchor.rot + leaf.fall.rot]);
  const opacity = useTransform(p, [0, 0.7, 1], [1, 0.5, 0]);
  const scale = useTransform(p, [0, 1], [1, 0.55]);
  return (
    <motion.g
      style={{
        x: tx,
        y: ty,
        rotate: rot,
        opacity,
        scale,
        transformOrigin: "0 0",
      }}
    >
      <LeafShape index={index} accent={leaf.accent} />
    </motion.g>
  );
}

function InsightSlot({ leaf, index, progress, insight }) {
  // Card appears just as its leaf finishes detaching – all within sticky range.
  const start = 0.16 + index * 0.075;
  const end = start + 0.11;
  const p = useTransform(progress, [start, end], [0, 1], { clamp: true });
  const opacity = useTransform(p, [0, 1], [0, 1]);
  const scale = useTransform(p, [0, 1], [0.85, 1]);
  const y = useTransform(p, [0, 1], [12, 0]);
  return (
    <motion.div
      data-testid={`insight-card-${leaf.key}`}
      className="absolute"
      style={{ ...leaf.card, opacity, scale, y }}
    >
      <InsightCard
        label={leaf.label}
        value={insight ? insight.value : "—"}
        source={insight ? insight.source : "demo"}
        accent={leaf.accent}
        testId={`insight-card-inner-${leaf.key}`}
      />
    </motion.div>
  );
}

export default function PlantScene() {
  const containerRef = useRef(null);
  const inView = useInView(containerRef, { once: true, margin: "-10% 0px" });
  const { data, loading, error } = useInsights();

  // STICKY-AWARE scroll offset: progress = 0 when section top hits viewport top,
  // progress = 1 when section bottom hits viewport bottom. This means the leaves
  // only animate while the sticky stage is actually in view — freezing (at 0 or 1)
  // when the user is above or below the section.
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  const insightsMap = data?.insights || {};

  return (
    <section
      ref={containerRef}
      data-testid="scene-plant"
      className="relative bg-gradient-to-b from-black/70 via-black/60 to-black/70 px-6 pt-24 pb-24 md:px-12"
      style={{ minHeight: "160vh" }}
    >
      <div className="relative mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9 }}
          className="mb-8 text-center"
        >
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-white/60">
            Chapter Five · Living intelligence
          </div>
          <h3 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl">
            Every leaf, an insight.
          </h3>
          <p className="mx-auto mt-4 max-w-xl text-white/70">
            Scroll — and watch the plant reveal itself, one insight-leaf at a time.
          </p>
        </motion.div>

        <AnimatePresence>
          {loading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mb-4 flex items-center justify-center gap-2 text-sm text-white/70"
            >
              <Loader2 size={14} className="animate-spin text-[#00E65B]" /> Loading insights…
            </motion.div>
          )}
          {error && !loading && (
            <motion.div
              key="err"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mb-4 flex items-center justify-center gap-2 text-sm text-[#FBBF24]"
            >
              <AlertCircle size={14} /> Insight service unavailable — showing demo values.
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Sticky stage — sibling of the heading so containing block = section
          (min-height: 160vh), which gives 60vh+ of stuck-scroll room. */}
      <div className="sticky top-24 flex h-[82vh] items-center justify-center">
        <div
          data-testid="plant-stage"
          className="relative mx-auto h-[600px] w-full max-w-[900px] overflow-hidden rounded-3xl"
        >
            {/* Real foliage photo backdrop */}
            <img
              src={FOLIAGE_BG}
              alt="live foliage"
              className="absolute inset-0 h-full w-full object-cover"
              style={{ filter: "brightness(0.35) contrast(1.05) blur(2px)" }}
            />
            {/* Dark vignette for readability */}
            <div
              aria-hidden
              className="absolute inset-0"
              style={{
                background:
                  "radial-gradient(circle at 50% 45%, rgba(0,0,0,0.15) 0%, rgba(0,0,0,0.75) 75%)",
              }}
            />
            {/* Ambient green glow behind plant */}
            <div
              aria-hidden
              className="pointer-events-none absolute left-1/2 top-1/2 h-[420px] w-[420px] -translate-x-1/2 -translate-y-1/2 rounded-full"
              style={{
                background:
                  "radial-gradient(circle at center, rgba(0,230,91,0.16), transparent 70%)",
              }}
            />

            {/* Plant SVG in center */}
            <svg
              data-testid="plant-svg"
              viewBox="-150 -230 300 470"
              className="absolute left-1/2 top-1/2 h-[520px] w-[340px] -translate-x-1/2 -translate-y-1/2"
            >
              <defs>
                <linearGradient id="pot-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3a2418" />
                  <stop offset="45%" stopColor="#2a180f" />
                  <stop offset="100%" stopColor="#160b06" />
                </linearGradient>
                <linearGradient id="pot-rim" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4d321f" />
                  <stop offset="100%" stopColor="#2c1a10" />
                </linearGradient>
                <radialGradient id="soil-grad" cx="50%" cy="35%" r="65%">
                  <stop offset="0%" stopColor="#5a4028" />
                  <stop offset="55%" stopColor="#2b1c11" />
                  <stop offset="100%" stopColor="#160b06" />
                </radialGradient>
                <linearGradient id="stem-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#5fd177" />
                  <stop offset="40%" stopColor="#2a9a45" />
                  <stop offset="100%" stopColor="#0a3a1d" />
                </linearGradient>
                <filter id="stem-shadow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="1.4" />
                </filter>
              </defs>

              {/* Ground shadow under pot */}
              <ellipse cx="0" cy="220" rx="90" ry="10" fill="rgba(0,0,0,0.45)" />

              {/* Pot base + rim */}
              <path
                d="M-58,150 L58,150 L48,215 L-48,215 Z"
                fill="url(#pot-grad)"
                stroke="rgba(255,255,255,0.06)"
              />
              <rect x="-60" y="146" width="120" height="10" rx="2" fill="url(#pot-rim)" />
              {/* Soil */}
              <ellipse cx="0" cy="150" rx="55" ry="7" fill="url(#soil-grad)" />
              <ellipse cx="0" cy="150" rx="55" ry="7" fill="none" stroke="rgba(255,255,255,0.06)" />

              {/* Stem shadow */}
              <path
                d="M0,150 C -10,80 8,20 -4,-40 C 6,-90 -4,-160 0,-200"
                stroke="rgba(0,0,0,0.6)"
                strokeWidth="9"
                fill="none"
                strokeLinecap="round"
                filter="url(#stem-shadow)"
              />
              {/* Stem */}
              <path
                d="M0,150 C -10,80 8,20 -4,-40 C 6,-90 -4,-160 0,-200"
                stroke="url(#stem-grad)"
                strokeWidth="6"
                fill="none"
                strokeLinecap="round"
              />
              {/* Top sprout */}
              <circle cx="0" cy="-200" r="9" fill="#4fd06f" />
              <circle cx="0" cy="-200" r="9" fill="url(#stem-grad)" />
              <circle cx="0" cy="-200" r="16" fill="none" stroke="#00E65B" strokeOpacity="0.3" />

              {/* Detachable realistic leaves */}
              {LEAVES.map((leaf, i) => (
                <Leaf key={leaf.key} leaf={leaf} index={i} progress={scrollYProgress} />
              ))}
            </svg>

            {/* Insight cards */}
            {LEAVES.map((leaf, i) => (
              <InsightSlot
                key={leaf.key}
                leaf={leaf}
                index={i}
                progress={scrollYProgress}
                insight={
                  insightsMap[leaf.key]
                    ? {
                        ...insightsMap[leaf.key],
                        value: formatValue(leaf.key, insightsMap[leaf.key]),
                      }
                    : null
                }
              />
            ))}

            {/* Source badge (inside stage, so it hides with the plant) */}
            {data && (
              <div
                data-testid="insights-source-badge"
                className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-black/55 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-white/70 backdrop-blur"
              >
                {data.meta.all_sources_demo
                  ? "All values · Demo simulator"
                  : "Live pipeline attached"}
              </div>
            )}
          </div>
        </div>
    </section>
  );
}
