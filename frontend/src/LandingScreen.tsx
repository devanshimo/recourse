// src/LandingScreen.tsx

interface LandingScreenProps {
  onEnter: () => void;
}

export default function LandingScreen({ onEnter }: LandingScreenProps) {
  return (
    <div className="min-h-screen bg-[#FDFDFD] text-[#111111] font-sans flex items-center justify-center px-6">
      <div className="max-w-2xl w-full text-center space-y-8">
        <div>
          <div className="font-mono font-bold tracking-tight text-2xl text-black">
            RECOURSE
          </div>
          <div className="mt-1 font-mono text-[11px] uppercase tracking-wider text-neutral-400">
            AI-assisted chargeback defense
          </div>
        </div>

        <h1 className="text-2xl md:text-3xl font-semibold leading-snug text-neutral-900">
          Should a merchant defend this chargeback?
        </h1>

        <p className="text-sm text-neutral-600 leading-relaxed max-w-xl mx-auto">
          Recourse combines structured risk, economic value, and evidence from the
          customer&apos;s dispute narrative to decide what deserves attention.
        </p>

        <div className="flex items-center justify-center gap-6 font-mono text-xs uppercase tracking-wider pt-2">
          <span className="text-emerald-700 font-semibold">DEFEND</span>
          <span className="text-neutral-300">&middot;</span>
          <span className="text-rose-700 font-semibold">ACCEPT</span>
          <span className="text-neutral-300">&middot;</span>
          <span className="text-amber-700 font-semibold">REVIEW</span>
        </div>

        <div className="pt-4">
          <button
            type="button"
            onClick={onEnter}
            className="bg-black text-white font-mono text-xs uppercase tracking-wider px-6 py-2.5 hover:bg-neutral-800 transition-colors"
          >
            ENTER DECISION ENGINE
          </button>
        </div>
      </div>
    </div>
  );
}