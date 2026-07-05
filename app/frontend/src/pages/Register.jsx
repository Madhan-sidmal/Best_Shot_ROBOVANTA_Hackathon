import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import axios from "axios";
import { toast } from "sonner";
import { Sprout } from "lucide-react";

export default function Register() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/api/auth/register`, {
        name,
        email,
        password,
      });
      login(res.data.token, res.data.user);
      toast.success("Account created successfully");
      navigate("/dashboard");
    } catch (err) {
      toast.error(err.response?.data?.detail || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen w-full flex-col items-center justify-center bg-[#0a0f0d] text-white">
      <div className="glass w-full max-w-md rounded-2xl p-8">
        <div className="mb-8 flex flex-col items-center text-center">
          <div className="mb-4 grid h-12 w-12 place-items-center rounded-xl bg-[#00E65B]/15 ring-1 ring-[#00E65B]/40">
            <Sprout size={24} className="text-[#00E65B]" />
          </div>
          <h2 className="font-display text-2xl font-bold">Create Account</h2>
          <p className="mt-2 text-sm text-white/50">Join KrishiDrishti platform</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label className="mb-1.5 block text-xs uppercase tracking-wider text-white/60">Full Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-lg border border-white/10 bg-black/40 px-4 py-2.5 text-sm outline-none transition focus:border-[#00E65B]/50 focus:ring-1 focus:ring-[#00E65B]/50"
              required
            />
          </div>
          <div>
            <label className="mb-1.5 block text-xs uppercase tracking-wider text-white/60">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-white/10 bg-black/40 px-4 py-2.5 text-sm outline-none transition focus:border-[#00E65B]/50 focus:ring-1 focus:ring-[#00E65B]/50"
              required
            />
          </div>
          <div>
            <label className="mb-1.5 block text-xs uppercase tracking-wider text-white/60">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-white/10 bg-black/40 px-4 py-2.5 text-sm outline-none transition focus:border-[#00E65B]/50 focus:ring-1 focus:ring-[#00E65B]/50"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="mt-2 rounded-lg bg-[#00E65B] px-4 py-2.5 text-sm font-bold text-black transition hover:bg-[#00c24d] disabled:opacity-50"
          >
            {loading ? "Registering..." : "Register"}
          </button>
        </form>

        <div className="mt-6 text-center text-sm text-white/50">
          Already have an account?{" "}
          <Link to="/login" className="text-[#00E65B] hover:underline">
            Sign in
          </Link>
        </div>
      </div>
    </div>
  );
}
