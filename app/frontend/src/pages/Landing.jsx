import { useNavigate } from "react-router-dom";
import { ArrowRight, Satellite, Droplets, Sprout, ChevronDown } from "lucide-react";
import { TaglineScene, ProblemsScene, PipelineScene, TransitionScene } from "@/story/scenes";
import SatelliteScene from "@/story/SatelliteScene";
import FarmScene from "@/story/FarmScene";
import ThreePillarsScene from "@/story/ThreePillarsScene";
import PlantScene from "@/story/PlantScene";

/* TODO: For production, download and host at /hero.mp4 */
const HERO_VIDEO =
  "https://customer-assets.emergentagent.com/job_366a7cf7-6ad0-4593-a0be-cf5f3d61ff7d/artifacts/5v2yklkl_The_Master_Continuous_Video_Pr.mp4";

export default function Landing() {
  const navigate = useNavigate();
  const goDashboard = () => navigate("/dashboard");

  return (
    <main data-testid="landing-page" className="relative w-full">
      {/* -------- FIXED cinematic backdrop (persists across all scenes) -------- */}
      <div
        aria-hidden
        data-testid="fixed-video-backdrop"
        className="pointer-events-none fixed inset-0 z-0 h-screen w-screen overflow-hidden bg-[#0a0f0d]"
      >
        <video
          data-testid="hero-video"
          className="h-full w-full object-cover opacity-90"
          src={HERO_VIDEO}
          autoPlay
          loop
          muted
          playsInline
        />
        {/* Subtle vignette so the video stays visible but never fights text */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 50% 50%, rgba(0,0,0,0.15) 0%, rgba(0,0,0,0.35) 60%, rgba(0,0,0,0.55) 100%)",
          }}
        />
      </div>

      {/* -------- Scene 1: Hero (video stays visible; text overlaid) -------- */}
      <section
        data-testid="scene-hero"
        className="relative z-10 min-h-screen w-full overflow-hidden"
      >
        {/* Extra dark bottom-fade only for hero readability of headline */}
        <div
          aria-hidden
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(180deg, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0.15) 40%, rgba(0,0,0,0.35) 100%)",
          }}
        />
        <header className="relative z-10 flex items-center justify-between px-6 py-5 md:px-12">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-[#00E65B]/15 ring-1 ring-[#00E65B]/40 backdrop-blur">
              <Sprout size={18} className="text-[#00E65B]" strokeWidth={2.4} />
            </div>
            <span className="font-display text-lg font-bold tracking-tight">KrishiDrishti</span>
          </div>
          <div className="hidden items-center gap-3 md:flex">
            <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs uppercase tracking-[0.18em] text-white/60 backdrop-blur-md">
              Canal Command Intelligence
            </span>
          </div>
        </header>

        <section className="relative z-10 mx-auto flex min-h-[calc(100vh-72px)] max-w-6xl flex-col items-start justify-center px-6 md:px-12">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-xs uppercase tracking-[0.22em] text-white/70 backdrop-blur-xl">
            <span className="h-1.5 w-1.5 rounded-full bg-[#00E65B] pulse-dot" />
            Growth-Stage Aware · Optical + SAR Fusion
          </div>
          <h1
            data-testid="hero-title"
            className="font-display text-frost text-6xl font-extrabold leading-[0.95] tracking-tight sm:text-7xl lg:text-[8.5rem]"
          >
            KrishiDrishti
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-white/70 sm:text-xl">
            Growth-Stage-Aware Water Deficit Advisory & Crop Classification — engineered
            from satellite intelligence to the last-mile farmer.
          </p>
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <button
              data-testid="launch-dashboard-button"
              onClick={goDashboard}
              className="group inline-flex items-center gap-3 rounded-full bg-[#00E65B] px-8 py-4 text-sm font-semibold uppercase tracking-[0.14em] text-[#062611] transition-all duration-300 hover:bg-[#00FF66] hover:shadow-[0_0_36px_rgba(0,230,91,0.5)] active:scale-95"
            >
              Launch Dashboard
              <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" strokeWidth={2.5} />
            </button>
            <a
              href="#story"
              data-testid="learn-more-link"
              className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/[0.03] px-6 py-4 text-sm font-medium text-white/80 backdrop-blur-md transition hover:border-white/25 hover:bg-white/[0.06]"
            >
              See the story
            </a>
          </div>

          <div id="pillars" className="mt-16 grid w-full max-w-4xl grid-cols-1 gap-4 sm:grid-cols-3">
            <PillarChip icon={<Satellite size={16} strokeWidth={2.2} />} label="Optical + SAR Fusion" accent="#00E65B" />
            <PillarChip icon={<Sprout size={16} strokeWidth={2.2} />} label="Phenology-Aware Detection" accent="#00E65B" />
            <PillarChip icon={<Droplets size={16} strokeWidth={2.2} />} label="FAO-56 Irrigation Model" accent="#00E5FF" />
          </div>
        </section>

        <a
          href="#story"
          className="absolute inset-x-0 bottom-8 z-10 mx-auto flex w-fit items-center gap-2 text-[10px] uppercase tracking-[0.28em] text-white/60 transition hover:text-[#00E65B]"
        >
          Scroll to begin
          <ChevronDown size={13} className="animate-bounce" />
        </a>
      </section>

      {/* -------- Story scenes (all sit on top of fixed video with per-scene tint) -------- */}
      <div id="story" className="relative z-10">
        <TaglineScene />
        <ProblemsScene />
        <SatelliteScene />
        <FarmScene />
        <PipelineScene />
        <ThreePillarsScene />
        <PlantScene />
        <TransitionScene onEnter={goDashboard} />
      </div>

      {/* Footer disclaimer */}
      <div className="relative z-10 border-t border-white/5 bg-black/70 px-6 py-3 text-center text-[11px] uppercase tracking-[0.2em] text-white/60 md:px-12">
        KrishiDrishti — Empowering Canal Command Agriculture. (Simulated Data Preview)
      </div>
    </main>
  );
}

function PillarChip({ icon, label, accent }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-black/40 px-4 py-3 backdrop-blur-xl transition hover:border-white/20 hover:bg-black/50">
      <span
        className="grid h-8 w-8 place-items-center rounded-lg"
        style={{ background: `${accent}22`, color: accent }}
      >
        {icon}
      </span>
      <span className="text-sm font-medium text-white/90">{label}</span>
    </div>
  );
}
