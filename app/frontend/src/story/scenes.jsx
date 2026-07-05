import { useRef } from "react";
import { motion, useInView } from "framer-motion";

/* All non-3D story scenes. Each scene sits on top of the fixed hero video and
   uses a semi-transparent dark backdrop so the video subtly bleeds through. */

const SCENE_BG =
  "bg-gradient-to-b from-black/65 via-black/55 to-black/70 backdrop-blur-md";

/* ============================================================ TaglineScene */
export function TaglineScene() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-25% 0px -25% 0px" });
  return (
    <section
      ref={ref}
      data-testid="scene-tagline"
      className={`relative flex min-h-[95vh] w-full items-center justify-center overflow-hidden ${SCENE_BG}`}
    >
      {/* Golden sunrise horizon line */}
      <div
        aria-hidden
        className="absolute bottom-[30%] left-0 right-0 h-px opacity-70"
        style={{
          background:
            "linear-gradient(90deg, transparent, rgba(255,190,90,0.55), transparent)",
        }}
      />
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 1.2, ease: "easeOut" }}
        className="relative z-10 mx-auto max-w-4xl px-6 text-center"
      >
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/50 px-4 py-1.5 text-[10px] uppercase tracking-[0.28em] text-white/70 backdrop-blur-xl">
          <span className="h-1.5 w-1.5 rounded-full bg-[#00E65B] pulse-dot" />
          Chapter One
        </div>
        <h2
          data-testid="tagline-heading"
          className="font-display text-4xl font-bold leading-[1.05] tracking-tight text-white sm:text-5xl lg:text-6xl"
        >
          A healthy field doesn&rsquo;t always mean{" "}
          <span className="text-frost italic">a healthy crop.</span>
        </h2>
        <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-white/70 sm:text-lg">
          Behind every emerald canopy lies data invisible to the eye. Scroll deeper — the story
          begins where satellites see what farmers can&rsquo;t.
        </p>
      </motion.div>
    </section>
  );
}

/* ============================================================ ProblemsScene */
const PROBLEMS = [
  { title: "Moisture Stress", hint: "Invisible root-zone drought before leaves wilt.", color: "#00E5FF" },
  { title: "Wrong Crop ID", hint: "Neighbouring fields mistaken for wheat when they hold cotton.", color: "#00E65B" },
  { title: "Water Deficit", hint: "Under-irrigated stress silently caps yield by 20–40%.", color: "#FBBF24" },
  { title: "Cloud Cover Gaps", hint: "Weeks without optical data. SAR fills the blind spots.", color: "#FF8A00" },
];

export function ProblemsScene() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-20% 0px" });
  return (
    <section
      ref={ref}
      data-testid="scene-problems"
      className={`relative px-6 py-32 md:px-12 ${SCENE_BG}`}
    >
      <div className="relative mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9 }}
          className="mb-14 max-w-2xl"
        >
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-white/60">
            Chapter Two · Hidden failures
          </div>
          <h3 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl">
            Four blind spots the eye can&rsquo;t catch.
          </h3>
        </motion.div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PROBLEMS.map((p, i) => (
            <motion.div
              key={p.title}
              initial={{ opacity: 0, y: 40 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.15 + i * 0.12, ease: "easeOut" }}
              data-testid={`problem-card-${i}`}
              className="glass group relative overflow-hidden rounded-2xl bg-black/40 p-6 backdrop-blur-xl"
            >
              <div
                className="absolute -right-8 -top-8 h-32 w-32 rounded-full opacity-30 blur-3xl transition group-hover:opacity-60"
                style={{ background: p.color }}
              />
              <div
                className="mb-4 inline-flex h-8 w-8 items-center justify-center rounded-lg text-[13px] font-bold"
                style={{ background: `${p.color}22`, color: p.color }}
              >
                0{i + 1}
              </div>
              <div className="font-display text-lg font-semibold text-white">{p.title}</div>
              <div className="mt-2 text-sm leading-relaxed text-white/65">{p.hint}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ============================================================ PipelineScene */
const PIPELINE = [
  { key: "extract", label: "Feature Extraction", detail: "NDVI · EVI · Backscatter" },
  { key: "fuse", label: "Data Fusion", detail: "Optical + SAR + Weather" },
  { key: "predict", label: "Crop Prediction", detail: "Random Forest + XGBoost" },
  { key: "stress", label: "Stress Detection", detail: "Phenology-aware CSI" },
  { key: "advisory", label: "Irrigation Advisory", detail: "FAO-56 · Last-mile SMS" },
];

export function PipelineScene() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  return (
    <section
      ref={ref}
      data-testid="scene-pipeline"
      className={`relative px-6 py-32 md:px-12 ${SCENE_BG}`}
    >
      <div className="relative mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9 }}
          className="mb-16 text-center"
        >
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-white/60">
            Chapter Four · From orbit to advisory
          </div>
          <h3 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl">
            Five stages. One growth-stage-aware pipeline.
          </h3>
        </motion.div>
        <div className="relative grid grid-cols-1 gap-5 md:grid-cols-5 md:gap-3">
          <div
            aria-hidden
            className="pointer-events-none absolute top-9 hidden h-px w-full md:block"
            style={{
              background:
                "linear-gradient(90deg, transparent, rgba(0,230,91,0.4) 20%, rgba(0,229,255,0.4) 80%, transparent)",
            }}
          />
          {PIPELINE.map((step, i) => (
            <motion.div
              key={step.key}
              initial={{ opacity: 0, y: 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.7, delay: 0.2 + i * 0.15 }}
              data-testid={`pipeline-step-${step.key}`}
              className="glass relative rounded-2xl bg-black/40 p-5 text-center backdrop-blur-xl"
            >
              <div className="mx-auto mb-3 grid h-10 w-10 place-items-center rounded-full border border-[#00E65B]/30 bg-[#00E65B]/10 font-mono text-xs font-bold text-[#00E65B]">
                0{i + 1}
              </div>
              <div className="font-display text-sm font-semibold text-white">{step.label}</div>
              <div className="mt-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-white/60">
                {step.detail}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ============================================================ TransitionScene */
export function TransitionScene({ onEnter }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-20% 0px" });
  return (
    <section
      ref={ref}
      data-testid="scene-transition"
      className="relative flex min-h-[80vh] items-center justify-center overflow-hidden bg-gradient-to-b from-black/55 via-black/40 to-black/70 px-6 py-24 backdrop-blur-md"
    >
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(circle at 50% 50%, rgba(0,230,91,0.18), transparent 55%), radial-gradient(circle at 30% 70%, rgba(0,229,255,0.1), transparent 60%)",
        }}
      />
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={inView ? { opacity: 1, scale: 1 } : {}}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="relative z-10 text-center"
      >
        <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-white/60">
          The Living Dashboard
        </div>
        <h3 className="font-display mx-auto max-w-3xl text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
          The leaves settle. <span className="text-frost">Your dashboard begins.</span>
        </h3>
        <p className="mx-auto mt-6 max-w-xl text-base text-white/70 sm:text-lg">
          Every insight you just watched grow now sits, live, at your fingertips.
        </p>
        <button
          data-testid="enter-dashboard-button"
          onClick={onEnter}
          className="group mt-10 inline-flex items-center gap-3 rounded-full bg-[#00E65B] px-10 py-4 text-sm font-semibold uppercase tracking-[0.16em] text-[#062611] transition-all hover:bg-[#00FF66] hover:shadow-[0_0_36px_rgba(0,230,91,0.5)] active:scale-95"
        >
          Enter the Dashboard
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="transition-transform group-hover:translate-x-1">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </button>
      </motion.div>
    </section>
  );
}
