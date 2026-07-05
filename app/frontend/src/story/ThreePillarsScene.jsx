import { useRef, useState } from "react";
import { motion, useInView, AnimatePresence } from "framer-motion";
import { Cpu, Activity, Droplets, CheckCircle2 } from "lucide-react";

/* ThreePillarsScene — the actual solution.
   Interactive tabbed view of the three pillars from the architecture doc:
   1) Crop Classification (RF + XGBoost + SHAP)
   2) Phenology-Aware Stress Detection (VCI + NDWI + SAR → CSI)
   3) FAO-56 Irrigation Advisory. */

const PILLARS = [
  {
    key: "classification",
    number: "01",
    icon: Cpu,
    tone: "#00E65B",
    title: "Crop Classification",
    subtitle: "Random Forest + XGBoost · SHAP explainability",
    tagline: "The right crop, in the right pixel.",
    body:
      "40+ multi-temporal features per pixel — NDVI, EVI, NDWI, NDMI, SAVI, LSWI, VV, VH, VH/VV, RVI, GLCM textures — are fed to a soft-voting ensemble of Random Forest (500 trees, balanced) and XGBoost (300 estimators, depth 8). SHAP TreeExplainer opens the black box.",
    metrics: [
      { k: "≥85%", v: "Overall accuracy" },
      { k: ">0.84", v: "Cohen's κ" },
      { k: "5-fold", v: "Stratified CV" },
    ],
    steps: [
      "40+ features per pixel",
      "RF (500 trees) · XGB (300 est.)",
      "Soft-voting ensemble",
      "SHAP-explained predictions",
    ],
  },
  {
    key: "stress",
    number: "02",
    icon: Activity,
    tone: "#00E5FF",
    title: "Phenology-Aware Stress Detection",
    subtitle: "VCI · NDWI · SAR → Combined Stress Index",
    tagline: "Same drop, different consequence.",
    body:
      "A Savitzky-Golay filter smooths the NDVI curve; SOS/Peak/EOS mark the phenology. Stress is then computed against the crop's current growth stage using three signals — VCI (40%), NDWI anomaly (30%) and SAR VH z-score (30%) — normalised into a sigmoidal Combined Stress Index.",
    metrics: [
      { k: "4", v: "Growth stages" },
      { k: "3", v: "Stress signals" },
      { k: "CSI", v: "Sigmoid-normalised" },
    ],
    steps: [
      "Savitzky-Golay smoothing",
      "SOS · Peak · EOS extraction",
      "VCI + NDWI + SAR fusion",
      "Stage-wise severity",
    ],
  },
  {
    key: "advisory",
    number: "03",
    icon: Droplets,
    tone: "#FBBF24",
    title: "FAO-56 Irrigation Advisory",
    subtitle: "ETc = Kc × ETo · Deficit = ETc − Pe",
    tagline: "Millimetres, not maybe.",
    body:
      "For each crop × growth-stage pair, we lookup Kc from FAO Irrigation Paper 56. Multiply by MODIS 8-day ETo, subtract effective rainfall (0.8 × P), and route through 4-level thresholds to produce a farmer-ready advisory.",
    metrics: [
      { k: "FAO-56", v: "Kc lookup" },
      { k: "8-day", v: "MODIS ETo" },
      { k: "4-level", v: "Advisory" },
    ],
    steps: [
      "Kc(crop, stage)",
      "ETc = Kc × ETo",
      "Water deficit = ETc − Pe",
      "Adequate → Critical",
    ],
  },
];

export default function ThreePillarsScene() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [active, setActive] = useState(PILLARS[0].key);
  const p = PILLARS.find((x) => x.key === active) || PILLARS[0];
  const Icon = p.icon;

  return (
    <section
      ref={ref}
      data-testid="scene-pillars"
      className="relative min-h-screen bg-gradient-to-b from-black/70 via-black/55 to-black/70 px-6 py-24 md:px-12"
    >
      <div className="relative mx-auto max-w-7xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9 }}
          className="mb-14 max-w-3xl"
        >
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-white/60">
            Chapter Five · The solution
          </div>
          <h3 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl">
            Three pillars. <span className="text-frost">One pipeline.</span>
          </h3>
          <p className="mt-4 text-white/70">
            Classification feeds phenology-aware stress feeds FAO-56 water balance. Each pillar
            hands off richer context to the next — the reason KrishiDrishti works where
            single-model tools stall.
          </p>
        </motion.div>

        {/* Pillar tabs */}
        <div className="mb-8 flex flex-wrap gap-3">
          {PILLARS.map((pillar) => {
            const isActive = pillar.key === active;
            const PIcon = pillar.icon;
            return (
              <button
                key={pillar.key}
                data-testid={`pillar-tab-${pillar.key}`}
                onClick={() => setActive(pillar.key)}
                className={`flex flex-1 min-w-[240px] items-center gap-3 rounded-2xl border p-4 text-left transition ${
                  isActive
                    ? "border-white/25 bg-white/[0.05]"
                    : "border-white/10 bg-black/40 hover:border-white/20 hover:bg-black/50"
                }`}
                style={isActive ? { boxShadow: `inset 0 0 0 1px ${pillar.tone}aa, 0 0 26px ${pillar.tone}22` } : {}}
              >
                <span
                  className="grid h-10 w-10 shrink-0 place-items-center rounded-xl"
                  style={{ background: `${pillar.tone}22`, color: pillar.tone }}
                >
                  <PIcon size={18} strokeWidth={2.2} />
                </span>
                <div className="min-w-0">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/50">
                    Pillar {pillar.number}
                  </div>
                  <div className="font-display text-sm font-semibold text-white">{pillar.title}</div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Active pillar detail */}
        <AnimatePresence mode="wait">
          <motion.div
            key={active}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.45 }}
            className="grid grid-cols-1 gap-6 lg:grid-cols-3"
          >
            {/* Left detail card */}
            <div
              data-testid={`pillar-detail-${active}`}
              className="glass relative overflow-hidden rounded-3xl bg-black/50 p-6 backdrop-blur-xl lg:col-span-2"
              style={{ boxShadow: `inset 0 0 0 1px ${p.tone}22` }}
            >
              <div
                className="absolute -right-16 -top-16 h-64 w-64 rounded-full opacity-25 blur-3xl"
                style={{ background: p.tone }}
              />
              <div className="relative">
                <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-[0.22em]" style={{ color: p.tone }}>
                  <Icon size={13} strokeWidth={2.4} /> Pillar {p.number}
                </div>
                <h4 className="font-display text-2xl font-bold text-white sm:text-3xl">{p.title}</h4>
                <div className="mt-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-white/60">
                  {p.subtitle}
                </div>
                <p className="mt-6 text-lg italic text-white/90">&ldquo;{p.tagline}&rdquo;</p>
                <p className="mt-4 max-w-2xl text-white/70">{p.body}</p>

                <ul className="mt-6 grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {p.steps.map((s, i) => (
                    <li
                      key={s}
                      data-testid={`pillar-step-${active}-${i}`}
                      className="flex items-start gap-2 rounded-xl border border-white/8 bg-white/[0.02] px-3 py-2 text-sm text-white/80"
                    >
                      <CheckCircle2 size={14} className="mt-0.5 shrink-0" style={{ color: p.tone }} />
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Right metric column */}
            <div className="flex flex-col gap-4">
              {p.metrics.map((m, i) => (
                <div
                  key={m.k}
                  data-testid={`pillar-metric-${active}-${i}`}
                  className="rounded-2xl border border-white/10 bg-black/45 p-5 backdrop-blur-xl"
                  style={{ boxShadow: `inset 0 0 0 1px ${p.tone}18` }}
                >
                  <div className="font-display text-4xl font-bold" style={{ color: p.tone }}>
                    {m.k}
                  </div>
                  <div className="mt-1 text-[11px] uppercase tracking-[0.2em] text-white/60">{m.v}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Footer differentiators strip */}
        <div className="mt-12 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { k: "Optical + SAR", v: "Cloud-safe fusion" },
            { k: "Phenology-Aware", v: "Stage-sensitive" },
            { k: "FAO-56", v: "Peer-reviewed" },
            { k: "NISAR-Ready", v: "L-band forward-compat" },
          ].map((d) => (
            <div
              key={d.k}
              data-testid={`differentiator-${d.k.replace(/\s+/g, "-").toLowerCase()}`}
              className="rounded-xl border border-white/10 bg-black/40 px-4 py-3 backdrop-blur"
            >
              <div className="font-display text-sm font-semibold text-white">{d.k}</div>
              <div className="mt-0.5 text-[10px] uppercase tracking-[0.18em] text-white/55">{d.v}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
