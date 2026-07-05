import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { toast } from "sonner";
import {
  Send, MessageSquare, Smartphone, Loader2, Sparkles, X, Bell,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function ActionPanel({ fields, selectedFieldId, onFieldChange }) {
  const [channel, setChannel] = useState("whatsapp");
  const [sending, setSending] = useState(false);
  const [ntfyTopic, setNtfyTopic] = useState("krishidrishti_demo");
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotLoading, setCopilotLoading] = useState(false);
  const [copilotData, setCopilotData] = useState(null);
  const [copilotTab, setCopilotTab] = useState("en");

  const selected = useMemo(
    () => fields.find((f) => f.field_id === selectedFieldId) || fields[0],
    [fields, selectedFieldId],
  );

  const defaultMessage = selected
    ? `KrishiDrishti Advisory: Field ${selected.field_id} (${selected.crop_type}, ${selected.growth_stage}) shows ${selected.advisory_status.toLowerCase()} water stress. Apply ${Math.max(
        20,
        Math.round(selected.water_deficit_mm),
      )}mm irrigation within 24h.`
    : "";

  const [message, setMessage] = useState(defaultMessage);
  useEffect(() => setMessage(defaultMessage), [defaultMessage]);

  const dispatch = async () => {
    if (!selected) {
      toast.error("Select a field first.");
      return;
    }
    setSending(true);
    try {
      const res = await axios.post(`${API}/alerts/dispatch`, {
        field_id: selected.field_id,
        channel,
        message,
        ntfy_topic: ntfyTopic || undefined,
        crop: selected.crop_type,
      });
      const ntfy = res.data.ntfy_status;
      if (ntfyTopic && ntfy === "sent") {
        toast.success("✅ Live Push Sent to Ntfy.sh & Mock SMS Logged to MongoDB!", {
          description: `${selected.field_id} · topic: ${ntfyTopic}`,
        });
      } else {
        toast.success(
          `${channel === "sms" ? "SMS" : "WhatsApp"} dispatched (MOCKED)`,
          {
            description: `${selected.field_id} · ${new Date(res.data.dispatched_at).toLocaleTimeString()}`,
          },
        );
      }
    } catch (e) {
      toast.error("Dispatch failed", { description: String(e?.message || e) });
    } finally {
      setSending(false);
    }
  };

  const askCopilot = async () => {
    if (!selected) {
      toast.error("Select a field first.");
      return;
    }
    setCopilotOpen(true);
    setCopilotLoading(true);
    setCopilotData(null);
    setCopilotTab("en");
    try {
      const res = await axios.post(`${API}/copilot/advisory`, {
        plot_id: selected.field_id,
        crop: selected.crop_type,
        stage: selected.growth_stage,
        deficit_mm: selected.water_deficit_mm,
        status: selected.advisory_status,
        etc_mm: Math.max(25, selected.water_deficit_mm + 8),
      });
      setCopilotData(res.data);
    } catch (e) {
      toast.error("Copilot failed", { description: String(e?.message || e) });
      setCopilotOpen(false);
    } finally {
      setCopilotLoading(false);
    }
  };

  return (
    <>
      <section
        data-testid="action-panel"
        className="glass flex flex-col gap-4 rounded-2xl p-5"
      >
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/50">
            Last-Mile Dispatch
          </div>
          <div className="font-display mt-0.5 text-lg font-semibold">Action Panel</div>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="text-[10px] uppercase tracking-[0.2em] text-white/45">Field</span>
          <select
            data-testid="action-field-select"
            value={selected?.field_id || ""}
            onChange={(e) => onFieldChange(e.target.value)}
            className="rounded-lg border border-white/10 bg-black/40 px-3 py-2.5 text-sm text-white focus:border-[#00E65B]/60 focus:outline-none"
          >
            {fields.map((f) => (
              <option key={f.field_id} value={f.field_id}>
                {f.field_id} · {f.crop_type} · {f.advisory_status}
              </option>
            ))}
          </select>
        </label>

        {selected && (
          <div className="grid grid-cols-2 gap-2 rounded-xl border border-white/8 bg-black/20 p-3 text-xs">
            <Meta label="Crop" value={selected.crop_type} />
            <Meta label="Stage" value={selected.growth_stage} />
            <Meta label="CSI" value={selected.csi.toFixed(2)} accent="#00E65B" />
            <Meta
              label="Deficit"
              value={`${selected.water_deficit_mm.toFixed(1)} mm`}
              accent="#00E5FF"
            />
          </div>
        )}

        {/* ✨ AI Kisan Copilot */}
        <motion.button
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.98 }}
          data-testid="ask-copilot-button"
          onClick={askCopilot}
          disabled={!selected}
          className="relative flex items-center justify-center gap-2 overflow-hidden rounded-xl border border-[#00E65B]/40 bg-gradient-to-r from-[#00E65B]/10 via-[#00E5FF]/10 to-[#00E65B]/10 px-4 py-3 text-sm font-semibold uppercase tracking-[0.14em] text-white transition disabled:opacity-50"
          style={{ boxShadow: "inset 0 0 0 1px rgba(0,230,91,0.25), 0 0 22px rgba(0,230,91,0.18)" }}
        >
          <Sparkles size={15} strokeWidth={2.4} className="text-[#00E65B]" />
          Ask AI Kisan Copilot
          <span className="ml-1 rounded-full border border-[#00E5FF]/40 bg-[#00E5FF]/10 px-1.5 py-0.5 text-[8px] font-bold tracking-[0.12em] text-[#00E5FF]">
            GEMINI
          </span>
        </motion.button>

        <div>
          <div className="mb-1.5 text-[10px] uppercase tracking-[0.2em] text-white/45">Channel</div>
          <div className="flex gap-2">
            <ChannelBtn active={channel === "whatsapp"} onClick={() => setChannel("whatsapp")} icon={<MessageSquare size={14} />} label="WhatsApp" testId="channel-whatsapp" />
            <ChannelBtn active={channel === "sms"} onClick={() => setChannel("sms")} icon={<Smartphone size={14} />} label="SMS" testId="channel-sms" />
          </div>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="text-[10px] uppercase tracking-[0.2em] text-white/45">Message</span>
          <textarea
            data-testid="action-message-input"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={3}
            className="resize-none rounded-lg border border-white/10 bg-black/40 px-3 py-2.5 text-sm leading-relaxed text-white/90 focus:border-[#00E65B]/60 focus:outline-none"
          />
        </label>

        {/* Ntfy topic */}
        <label className="flex flex-col gap-1.5">
          <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.2em] text-white/45">
            <Bell size={11} /> Ntfy Topic
          </span>
          <input
            data-testid="ntfy-topic-input"
            value={ntfyTopic}
            onChange={(e) => setNtfyTopic(e.target.value)}
            placeholder="krishidrishti_demo"
            className="rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-[#00E5FF]/60 focus:outline-none"
          />
          <span className="text-[10px] leading-relaxed text-[#00E5FF]/80">
            👉 Open{" "}
            <a
              href={`https://ntfy.sh/${ntfyTopic || "krishidrishti_demo"}`}
              target="_blank"
              rel="noreferrer"
              className="underline hover:text-[#00E5FF]"
            >
              ntfy.sh/{ntfyTopic || "krishidrishti_demo"}
            </a>{" "}
            on your mobile browser now to receive live push alerts.
          </span>
        </label>

        <button
          data-testid="dispatch-alert-button"
          onClick={dispatch}
          disabled={sending || !selected}
          className="mt-1 inline-flex items-center justify-center gap-2 rounded-lg bg-[#00E65B] px-4 py-3 text-sm font-semibold uppercase tracking-[0.14em] text-[#062611] transition hover:bg-[#00FF66] hover:shadow-[0_0_28px_rgba(0,230,91,0.45)] active:scale-[0.98] disabled:opacity-60"
        >
          {sending ? (
            <>
              <Loader2 size={15} className="animate-spin" /> Dispatching…
            </>
          ) : (
            <>
              <Send size={15} strokeWidth={2.4} /> Send SMS / WhatsApp Alert
            </>
          )}
        </button>

        <p className="text-[10px] leading-relaxed text-white/40">
          SMS/WhatsApp is <span className="text-[#00E5FF]">MOCKED</span> for the hackathon.
          Ntfy push is <span className="text-[#00E65B]">LIVE</span> via ntfy.sh.
        </p>
      </section>

      <CopilotDialog
        open={copilotOpen}
        onClose={() => setCopilotOpen(false)}
        loading={copilotLoading}
        data={copilotData}
        tab={copilotTab}
        setTab={setCopilotTab}
        field={selected}
      />
    </>
  );
}

function Meta({ label, value, accent }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] uppercase tracking-[0.18em] text-white/40">{label}</span>
      <span className="font-mono text-xs font-medium" style={{ color: accent || "#f3f4f6" }}>
        {value}
      </span>
    </div>
  );
}

function ChannelBtn({ active, onClick, icon, label, testId }) {
  return (
    <button
      data-testid={testId}
      onClick={onClick}
      className={`inline-flex flex-1 items-center justify-center gap-2 rounded-lg border px-3 py-2 text-xs font-semibold uppercase tracking-[0.14em] transition ${
        active
          ? "border-[#00E65B]/50 bg-[#00E65B]/10 text-[#00E65B]"
          : "border-white/10 bg-white/[0.02] text-white/60 hover:text-white"
      }`}
    >
      {icon} {label}
    </button>
  );
}

function CopilotDialog({ open, onClose, loading, data, tab, setTab, field }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          data-testid="copilot-dialog"
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-md p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.96 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-2xl rounded-2xl border border-white/10 bg-[#0d1712]/95 p-6 shadow-[0_30px_80px_rgba(0,0,0,0.7)] backdrop-blur-2xl"
            style={{ boxShadow: "0 30px 80px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(0,230,91,0.15)" }}
          >
            <button
              onClick={onClose}
              className="absolute right-4 top-4 rounded-full border border-white/10 bg-black/40 p-1.5 text-white/60 transition hover:bg-black/60 hover:text-white"
              aria-label="close"
              data-testid="copilot-close"
            >
              <X size={14} />
            </button>

            <div className="mb-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em] text-[#00E65B]">
                <Sparkles size={12} /> AI Kisan Copilot
              </div>
              <h3 className="font-display mt-1 text-2xl font-bold text-white">
                {field ? `${field.field_id} · ${field.crop_type}` : "Advisory"}
              </h3>
              {data?.source && (
                <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.16em] text-white/50">
                  Source: {data.source}
                </div>
              )}
            </div>

            {/* Tabs — now 4-language enabled */}
            <div className="mb-4 flex gap-1 rounded-full border border-white/10 bg-black/40 p-1">
              <TabBtn active={tab === "en"} onClick={() => setTab("en")} label="🇬🇧 EN" testId="copilot-tab-en" />
              <TabBtn active={tab === "hi"} onClick={() => setTab("hi")} label="🇮🇳 हिंदी" testId="copilot-tab-hi" />
              <TabBtn active={tab === "ta"} onClick={() => setTab("ta")} label="🇮🇳 தமிழ்" testId="copilot-tab-ta" />
              <TabBtn active={tab === "plan"} onClick={() => setTab("plan")} label="📋 Plan" testId="copilot-tab-plan" />
            </div>

            <div className="min-h-[240px]" data-testid="copilot-body">
              {loading && (
                <div className="flex h-40 flex-col items-center justify-center gap-2 text-white/70">
                  <Loader2 size={20} className="animate-spin text-[#00E65B]" />
                  <span className="text-xs uppercase tracking-[0.2em]">Consulting the Kisan Copilot…</span>
                </div>
              )}
              {!loading && data && tab === "en" && (
                <div data-testid="copilot-content-en">
                  <p className="text-sm leading-relaxed text-white/90">{data.advisory_en}</p>
                  <ul className="mt-4 space-y-2">
                    {(data.bullet_points_en || []).map((b, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-white/80">
                        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#00E65B]" />
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {!loading && data && tab === "hi" && (
                <div data-testid="copilot-content-hi" lang="hi">
                  <p className="text-sm leading-relaxed text-white/90">{data.advisory_hi}</p>
                  <ul className="mt-4 space-y-2">
                    {(data.bullet_points_hi || []).map((b, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-white/80">
                        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#00E5FF]" />
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {!loading && data && tab === "ta" && (
                <div data-testid="copilot-content-ta" lang="ta">
                  <p className="text-sm leading-relaxed text-white/90">{data.advisory_ta}</p>
                  <ul className="mt-4 space-y-2">
                    {(data.bullet_points_ta || []).map((b, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-white/80">
                        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#FBBF24]" />
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {!loading && data && tab === "plan" && (
                <ol data-testid="copilot-content-plan" className="space-y-2">
                  {(data.action_plan || []).map((step, i) => (
                    <li key={i} className="flex items-start gap-3 rounded-lg border border-white/8 bg-black/30 p-3 text-sm text-white/85">
                      <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-[#00E65B]/15 text-[10px] font-bold text-[#00E65B]">
                        {i + 1}
                      </span>
                      {step}
                    </li>
                  ))}
                </ol>
              )}
            </div>

            {data?.broadcast?.queued && (
              <div
                data-testid="copilot-broadcast-note"
                className="mt-4 flex items-center gap-2 rounded-lg border border-[#00E5FF]/25 bg-[#00E5FF]/8 px-3 py-2 text-[11px] text-[#00E5FF]"
              >
                <Bell size={12} />
                Live push broadcast to <span className="font-mono">ntfy.sh/krishidrishti_demo</span> (EN + हिंदी + தமிழ்).
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function TabBtn({ active, onClick, label, testId }) {
  return (
    <button
      data-testid={testId}
      onClick={onClick}
      className={`flex-1 rounded-full px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] transition ${
        active ? "bg-[#00E65B] text-black shadow-[0_0_18px_rgba(0,230,91,0.4)]" : "text-white/70 hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}
