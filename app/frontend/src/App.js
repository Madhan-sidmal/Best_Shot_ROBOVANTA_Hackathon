import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "sonner";
import Landing from "@/pages/Landing";
import Dashboard from "@/pages/Dashboard";

function App() {
  return (
    <div className="App dark">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </BrowserRouter>
      <Toaster
        theme="dark"
        position="top-right"
        toastOptions={{
          style: {
            background: "rgba(12,20,16,0.9)",
            border: "1px solid rgba(0,230,91,0.25)",
            color: "#f3f4f6",
            backdropFilter: "blur(20px)",
            zIndex: 10000,
          },
        }}
        style={{ zIndex: 10000 }}
      />
    </div>
  );
}

export default App;
