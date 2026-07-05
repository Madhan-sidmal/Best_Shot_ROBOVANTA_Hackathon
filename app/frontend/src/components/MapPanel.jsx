import { useMemo, useState } from "react";
import { Satellite, Radar, MapPin, Layers } from "lucide-react";
import { MapContainer, TileLayer, GeoJSON, ZoomControl } from "react-leaflet";

/* Leaflet + OSM basemap with GeoJSON polygons color-coded by advisory status.
   Falls back to synthetic square polygons served by /api/pipeline/geojson when
   advisory.geojson is absent from backend/data/. */

const STATUS_FILL = {
  Adequate: "#00AA00",
  Watch: "#FFDD00",
  Urgent: "#FF8800",
  Critical: "#FF0000",
};

function polygonStyle(feature, selectedId) {
  const status = feature?.properties?.advisory_status || "Adequate";
  const isSel = feature?.properties?.field_id === selectedId;
  return {
    color: isSel ? "#ffffff" : STATUS_FILL[status] || "#00E65B",
    weight: isSel ? 3 : 1.2,
    fillColor: STATUS_FILL[status] || "#00E65B",
    fillOpacity: isSel ? 0.7 : 0.45,
  };
}

export default function MapPanel({
  fields,
  selectedFieldId,
  onSelect,
  geojson,
  loading,
}) {
  const [layer, setLayer] = useState("optical"); // optical | sar | leaflet

  const center = useMemo(() => {
    if (fields?.length) {
      const lats = fields.map((f) => f.latitude);
      const lngs = fields.map((f) => f.longitude);
      return [
        lats.reduce((a, b) => a + b, 0) / lats.length,
        lngs.reduce((a, b) => a + b, 0) / lngs.length,
      ];
    }
    return [30.7, 76.7];
  }, [fields]);

  // Normalized pin positions for optical/SAR overlays
  const points = useMemo(() => {
    if (!fields?.length) return [];
    const lats = fields.map((f) => f.latitude);
    const lngs = fields.map((f) => f.longitude);
    const minLat = Math.min(...lats),
      maxLat = Math.max(...lats);
    const minLng = Math.min(...lngs),
      maxLng = Math.max(...lngs);
    return fields.map((f) => ({
      ...f,
      x: ((f.longitude - minLng) / (maxLng - minLng || 1)) * 100,
      y: 100 - ((f.latitude - minLat) / (maxLat - minLat || 1)) * 100,
    }));
  }, [fields]);

  const isSAR = layer === "sar";
  const isLeaflet = layer === "leaflet";

  return (
    <section data-testid="map-panel" className="glass relative overflow-hidden rounded-2xl">
      <div className="flex items-center justify-between border-b border-white/5 px-5 py-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/50">
            Field Intelligence Map
          </div>
          <div className="font-display mt-0.5 text-lg font-semibold">
            {isLeaflet
              ? "Real-world (OSM) view"
              : isSAR
                ? "SAR moisture radar"
                : "Optical vegetation view"}
          </div>
        </div>
        <div
          data-testid="map-layer-toggle"
          className="flex items-center gap-1 rounded-full border border-white/10 bg-black/40 p-1 backdrop-blur"
        >
          <ToggleBtn active={layer === "optical"} onClick={() => setLayer("optical")} icon={<Satellite size={13} />} label="Optical" testId="layer-optical" accent="#00E65B" />
          <ToggleBtn active={isSAR} onClick={() => setLayer("sar")} icon={<Radar size={13} />} label="SAR" testId="layer-sar" accent="#00E5FF" />
          <ToggleBtn active={isLeaflet} onClick={() => setLayer("leaflet")} icon={<Layers size={13} />} label="Real" testId="layer-leaflet" accent="#FBBF24" />
        </div>
      </div>

      <div
        className="relative h-[420px] w-full overflow-hidden"
        style={
          isLeaflet
            ? undefined
            : {
                background: isSAR
                  ? "radial-gradient(circle at 30% 40%, rgba(0,229,255,0.18), transparent 55%), radial-gradient(circle at 75% 70%, rgba(0,229,255,0.14), transparent 60%), #05121a"
                  : "radial-gradient(circle at 30% 40%, rgba(0,230,91,0.15), transparent 55%), radial-gradient(circle at 75% 70%, rgba(0,230,91,0.12), transparent 60%), #0a1710",
              }
        }
      >
        {isLeaflet ? (
          <div className="h-full w-full">
            <MapContainer
              center={center}
              zoom={11}
              scrollWheelZoom
              zoomControl={false}
              className="h-full w-full"
              style={{ background: "#0a1710" }}
              data-testid="leaflet-map"
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <ZoomControl position="bottomright" />
              {geojson && (
                <GeoJSON
                  key={selectedFieldId + "-" + (geojson?.features?.length || 0)}
                  data={geojson}
                  style={(feat) => polygonStyle(feat, selectedFieldId)}
                  onEachFeature={(feat, mapLayer) => {
                    const p = feat.properties || {};
                    mapLayer.bindTooltip(
                      `<div style="font-family:'IBM Plex Sans';">
                         <b>${p.field_id}</b><br/>
                         ${p.crop_type} · ${p.growth_stage}<br/>
                         <span style="color:${STATUS_FILL[p.advisory_status] || "#00E65B"}">${p.advisory_status}</span>
                         · CSI ${Number(p.csi || 0).toFixed(2)} · ${Number(p.water_deficit_mm || 0).toFixed(1)} mm
                       </div>`,
                      { direction: "top", offset: [0, -6] },
                    );
                    mapLayer.on({
                      click: () => onSelect && onSelect(p.field_id),
                    });
                  }}
                />
              )}
            </MapContainer>
          </div>
        ) : (
          <>
            {/* Grid */}
            <svg className="absolute inset-0 h-full w-full opacity-40" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none"
                    stroke={isSAR ? "rgba(0,229,255,0.12)" : "rgba(0,230,91,0.1)"}
                    strokeWidth="1" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
            {isSAR && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden">
                <div className="sar-scan-line h-24 w-full"
                  style={{ background: "linear-gradient(180deg, transparent 0%, rgba(0,229,255,0.18) 50%, transparent 100%)" }} />
              </div>
            )}
            <div className="absolute inset-0 p-6">
              <div className="relative h-full w-full">
                {points.map((p) => {
                  const color = STATUS_FILL[p.advisory_status] || "#00E65B";
                  const isSel = p.field_id === selectedFieldId;
                  return (
                    <button
                      key={p.field_id}
                      data-testid={`map-pin-${p.field_id}`}
                      onClick={() => onSelect && onSelect(p.field_id)}
                      className="absolute -translate-x-1/2 -translate-y-1/2 transition hover:scale-110"
                      style={{ left: `${p.x}%`, top: `${p.y}%` }}
                      title={`${p.field_id} · ${p.crop_type} · ${p.advisory_status}`}
                    >
                      <span className="absolute inset-0 -m-2 rounded-full pulse-dot" style={{ background: `${color}33` }} />
                      <span className={`relative block h-3 w-3 rounded-full ring-2 ${isSel ? "ring-white" : "ring-black/40"}`}
                        style={{ background: color }} />
                    </button>
                  );
                })}
              </div>
            </div>
          </>
        )}

        <div className="pointer-events-none absolute bottom-4 left-4 flex flex-wrap items-center gap-3 rounded-xl border border-white/10 bg-black/60 px-3 py-2 text-[10px] uppercase tracking-[0.16em] text-white/80 backdrop-blur">
          {Object.entries(STATUS_FILL).map(([k, c]) => (
            <span key={k} className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ background: c }} />
              {k}
            </span>
          ))}
        </div>
        <div className="pointer-events-none absolute right-4 top-4 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-white/70 backdrop-blur">
          <MapPin size={11} />
          {isLeaflet
            ? geojson?.meta?.source === "demo"
              ? "Sim-Region · Demo GeoJSON"
              : "Live pipeline · advisory.geojson"
            : "Sim-Region · Punjab"}
        </div>
        {loading && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center text-xs text-white/60">
            Loading map…
          </div>
        )}
      </div>
    </section>
  );
}

function ToggleBtn({ active, onClick, icon, label, testId, accent }) {
  return (
    <button
      data-testid={testId}
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] transition ${
        active ? "text-black" : "text-white/70 hover:text-white"
      }`}
      style={{ background: active ? accent : "transparent", boxShadow: active ? `0 0 18px ${accent}55` : "none" }}
    >
      {icon} {label}
    </button>
  );
}
