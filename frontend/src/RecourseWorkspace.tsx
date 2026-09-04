import React, { useState } from 'react';

// ============================================================
// ENVIRONMENT-BASED API ROUTING
// ============================================================

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ============================================================
// CURRENCY FORMATTER
// ============================================================

const formatINR = (value: number): string =>
  new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);

// ============================================================
// EXACT BACKEND SCHEMA DEFINITIONS (POST /decide)
// ============================================================

export type DisputeType =
  | 'ITEM_NOT_RECEIVED'
  | 'UNAUTHORIZED_TRANSACTION'
  | 'PRODUCT_NOT_AS_DESCRIBED';

export interface ChargebackPayload {
  amount: number;
  transaction_age_days: number;
  payment_method: string;
  account_age_days: number;
  previous_orders: number;
  previous_chargebacks: number;
  previous_refunds: number;
  device_seen_before: boolean;
  location_consistent: boolean;
  velocity_24h: number;
  delivered: boolean;
  delivery_confirmed: boolean;
  delivery_age_days: number;
  customer_contacted: boolean;
  merchant_response_time_hours: number;
  refund_requested: boolean;
  dispute_type: DisputeType;
  dispute_text: string;
}

export interface LLMEvidence {
  customer_claim: string;
  claim_confidence: number;
  contradicts_merchant_evidence: boolean;
  contradiction_confidence: number;
  contradiction_detail: string;
  new_signal_present: boolean;
}

export interface DecisionResponse {
  decision: 'DEFEND' | 'ACCEPT' | 'REVIEW';
  p_win: number;
  amount: number;
  expected_recovery: number;
  estimated_defense_cost: number;
  expected_net_value: number;
  llm_evidence: LLMEvidence;
  review_reason: string | null;
  reasoning: string[];
}

// ============================================================
// PRESET CASES CONFORMING EXACTLY TO BACKEND SCHEMA
// ============================================================

const PRESETS: Record<string, ChargebackPayload> = {
  strong_defense: {
    amount: 14500,
    transaction_age_days: 14,
    payment_method: 'credit_card',
    account_age_days: 730,
    previous_orders: 18,
    previous_chargebacks: 0,
    previous_refunds: 1,
    device_seen_before: true,
    location_consistent: true,
    velocity_24h: 1,
    delivered: true,
    delivery_confirmed: true,
    delivery_age_days: 10,
    customer_contacted: true,
    merchant_response_time_hours: 2,
    refund_requested: false,
    dispute_type: 'UNAUTHORIZED_TRANSACTION',
    dispute_text: 'Customer claims they never authorized this transaction, but the order originated from a known device with a consistent location and completed confirmed delivery.'
  },
  weak_case: {
    amount: 850,
    transaction_age_days: 45,
    payment_method: 'debit_card',
    account_age_days: 12,
    previous_orders: 1,
    previous_chargebacks: 1,
    previous_refunds: 0,
    device_seen_before: false,
    location_consistent: false,
    velocity_24h: 4,
    delivered: false,
    delivery_confirmed: false,
    delivery_age_days: 0,
    customer_contacted: false,
    merchant_response_time_hours: 72,
    refund_requested: true,
    dispute_type: 'PRODUCT_NOT_AS_DESCRIBED',
    dispute_text: 'Item arrived with clear defects, completely misaligned with catalog specs. Merchant stopped replying to support messages after 3 days.'
  },
  delivery_contradiction: {
    amount: 3200,
    transaction_age_days: 20,
    payment_method: 'credit_card',
    account_age_days: 340,
    previous_orders: 6,
    previous_chargebacks: 0,
    previous_refunds: 0,
    device_seen_before: true,
    location_consistent: true,
    velocity_24h: 1,
    delivered: true,
    delivery_confirmed: true,
    delivery_age_days: 15,
    customer_contacted: false,
    merchant_response_time_hours: 4,
    refund_requested: false,
    dispute_type: 'ITEM_NOT_RECEIVED',
    dispute_text: 'Tracking says delivered, but I never received the package.'
  }
};

export default function RecourseWorkspace() {
  const [formData, setFormData] = useState<ChargebackPayload>(PRESETS.delivery_contradiction);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<DecisionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAudit, setShowAudit] = useState(false);

  const handleInputChange = (field: keyof ChargebackPayload, value: unknown) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const loadPreset = (key: string) => {
    setFormData(PRESETS[key]);
    setResponse(null);
    setError(null);
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data: DecisionResponse = await res.json();
      setResponse(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const getDecisionTheme = (decision: DecisionResponse['decision']) => {
    switch (decision) {
      case 'DEFEND':
        return {
          text: 'text-emerald-800',
          borderAccent: 'border-l-[3px] border-l-emerald-600',
          headerBg: 'bg-emerald-50/40 border-b border-emerald-100',
          bottomBorder: 'border-b-emerald-800',
          netHighlight: 'text-emerald-900',
        };
      case 'ACCEPT':
        return {
          text: 'text-rose-800',
          borderAccent: 'border-l-[3px] border-l-rose-600',
          headerBg: 'bg-rose-50/40 border-b border-rose-100',
          bottomBorder: 'border-b-rose-800',
          netHighlight: 'text-rose-900',
        };
      case 'REVIEW':
      default:
        return {
          text: 'text-amber-800',
          borderAccent: 'border-l-[3px] border-l-amber-600',
          headerBg: 'bg-amber-50/40 border-b border-amber-100',
          bottomBorder: 'border-b-amber-800',
          netHighlight: 'text-amber-900',
        };
    }
  };

  const decisionTheme = response ? getDecisionTheme(response.decision) : null;

  return (
    <div className="min-h-screen bg-[#FDFDFD] text-[#111111] font-sans text-xs selection:bg-neutral-200">
      {/* Precision Operational Header */}
      <div className="border-b border-neutral-200 bg-white px-6 py-2.5 flex items-center justify-between text-[11px]">
        <div className="flex items-center gap-3">
          <span className="font-mono font-bold tracking-tight text-black">REC0URSE</span>
          <span className="text-neutral-300">/</span>
          <span className="text-neutral-500 font-mono">DISPUTE-DECISION-ENGINE</span>
        </div>
        <div className="flex items-center gap-2 font-mono text-neutral-500">
          <span className="text-neutral-400">PRESETS:</span>
          <button
            type="button"
            onClick={() => loadPreset('strong_defense')}
            className="text-emerald-700 hover:text-emerald-900 underline underline-offset-2 decoration-emerald-200 hover:decoration-emerald-400 transition-colors"
          >
            Strong Defend
          </button>
          <span>·</span>
          <button
            type="button"
            onClick={() => loadPreset('delivery_contradiction')}
            className="text-amber-700 hover:text-amber-900 underline underline-offset-2 decoration-amber-200 hover:decoration-amber-400 transition-colors"
          >
            Contradiction
          </button>
          <span>·</span>
          <button
            type="button"
            onClick={() => loadPreset('weak_case')}
            className="text-rose-700 hover:text-rose-900 underline underline-offset-2 decoration-rose-200 hover:decoration-rose-400 transition-colors"
          >
            Accept
          </button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Input Deck */}
        <form onSubmit={handleAnalyze} className="lg:col-span-5 space-y-5">
          {/* Section 1: Transaction & Context */}
          <div className="space-y-3">
            <div className="pb-1 border-b border-neutral-200 flex justify-between items-baseline">
              <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">Transaction Profile</span>
              <span className="font-mono text-neutral-400 text-[10px]">P_METHOD: {formData.payment_method}</span>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Amount (₹)</label>
                <input
                  type="number"
                  step="any"
                  required
                  value={formData.amount}
                  onChange={(e) => handleInputChange('amount', parseFloat(e.target.value) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Tx Age (d)</label>
                <input
                  type="number"
                  required
                  value={formData.transaction_age_days}
                  onChange={(e) => handleInputChange('transaction_age_days', parseInt(e.target.value, 10) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Method</label>
                <select
                  value={formData.payment_method}
                  onChange={(e) => handleInputChange('payment_method', e.target.value)}
                  className="w-full bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                >
                  <option value="credit_card">credit_card</option>
                  <option value="debit_card">debit_card</option>
                  <option value="upi">upi</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2 pt-1">
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Account Age (d)</label>
                <input
                  type="number"
                  value={formData.account_age_days}
                  onChange={(e) => handleInputChange('account_age_days', parseInt(e.target.value, 10) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Orders</label>
                <input
                  type="number"
                  value={formData.previous_orders}
                  onChange={(e) => handleInputChange('previous_orders', parseInt(e.target.value, 10) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Disputes</label>
                <input
                  type="number"
                  value={formData.previous_chargebacks}
                  onChange={(e) => handleInputChange('previous_chargebacks', parseInt(e.target.value, 10) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Velocity (24h)</label>
                <input
                  type="number"
                  value={formData.velocity_24h}
                  onChange={(e) => handleInputChange('velocity_24h', parseInt(e.target.value, 10) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
            </div>

            <div className="flex gap-4 pt-1 text-neutral-600 font-mono">
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.device_seen_before}
                  onChange={(e) => handleInputChange('device_seen_before', e.target.checked)}
                  className="accent-black"
                />
                Device Known
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.location_consistent}
                  onChange={(e) => handleInputChange('location_consistent', e.target.checked)}
                  className="accent-black"
                />
                Location Match
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.refund_requested}
                  onChange={(e) => handleInputChange('refund_requested', e.target.checked)}
                  className="accent-black"
                />
                Refund Prior
              </label>
            </div>
          </div>

          {/* Section 2: Fulfillment & Support Telemetry */}
          <div className="space-y-3">
            <div className="pb-1 border-b border-neutral-200">
              <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">Fulfillment & Operations</span>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <label className="flex items-center gap-1.5 font-mono cursor-pointer pt-3">
                <input
                  type="checkbox"
                  checked={formData.delivered}
                  onChange={(e) => handleInputChange('delivered', e.target.checked)}
                  className="accent-black"
                />
                Delivered
              </label>
              <label className="flex items-center gap-1.5 font-mono cursor-pointer pt-3">
                <input
                  type="checkbox"
                  checked={formData.delivery_confirmed}
                  onChange={(e) => handleInputChange('delivery_confirmed', e.target.checked)}
                  className="accent-black"
                />
                POD Signature
              </label>
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Delivery Age (d)</label>
                <input
                  type="number"
                  value={formData.delivery_age_days}
                  onChange={(e) => handleInputChange('delivery_age_days', parseInt(e.target.value, 10) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 pt-1">
              <div>
                <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Response Latency (hrs)</label>
                <input
                  type="number"
                  step="any"
                  value={formData.merchant_response_time_hours}
                  onChange={(e) => handleInputChange('merchant_response_time_hours', parseFloat(e.target.value) || 0)}
                  className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
                />
              </div>
              <div className="flex items-center pt-3">
                <label className="flex items-center gap-1.5 font-mono cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.customer_contacted}
                    onChange={(e) => handleInputChange('customer_contacted', e.target.checked)}
                    className="accent-black"
                  />
                  Customer Contacted
                </label>
              </div>
            </div>
          </div>

          {/* Section 3: Dispute Claim */}
          <div className="space-y-2">
            <div className="pb-1 border-b border-neutral-200 flex justify-between items-baseline">
              <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">Dispute Narrative</span>
              <span className="font-mono text-neutral-400 text-[10px]">SCHEMA CLASSIFICATION</span>
            </div>

            <div>
              <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Dispute Type</label>
              <select
                value={formData.dispute_type}
                onChange={(e) => handleInputChange('dispute_type', e.target.value as DisputeType)}
                className="w-full font-mono bg-white border border-neutral-300 px-2 py-1 focus:border-black focus:outline-none"
              >
                <option value="ITEM_NOT_RECEIVED">ITEM_NOT_RECEIVED</option>
                <option value="UNAUTHORIZED_TRANSACTION">UNAUTHORIZED_TRANSACTION</option>
                <option value="PRODUCT_NOT_AS_DESCRIBED">PRODUCT_NOT_AS_DESCRIBED</option>
              </select>
            </div>

            <div>
              <label className="text-[10px] uppercase text-neutral-400 block mb-0.5">Narrative Payload</label>
              <textarea
                rows={3}
                required
                placeholder="Dispute statement..."
                value={formData.dispute_text}
                onChange={(e) => handleInputChange('dispute_text', e.target.value)}
                className="w-full bg-white border border-neutral-300 p-2 focus:border-black focus:outline-none leading-relaxed"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-black text-white font-mono text-xs uppercase tracking-wider py-2.5 hover:bg-neutral-800 disabled:bg-neutral-300 transition-colors"
          >
            {loading ? 'ANALYZING CASE...' : 'RUN DECISION ENGINE'}
          </button>
        </form>

        {/* Right: Decision Output & Forensic Audit Canvas */}
        <div className="lg:col-span-7">
          {error && (
            <div className="border border-neutral-300 p-3 bg-neutral-100 font-mono text-neutral-800 mb-6">
              SYSTEM_FAULT: {error}
            </div>
          )}

          {!response && !loading && (
            <div className="h-full min-h-[420px] flex items-center justify-center border border-neutral-200 border-dashed p-12 text-neutral-400 font-mono text-center">
              Load an operational preset or submit transaction data to compute defense posture.
            </div>
          )}

          {response && (
            <div className={`space-y-6 bg-white border border-neutral-200 ${decisionTheme?.borderAccent}`}>
              {/* Massive Unambiguous Decision Header */}
              <div className={`p-4 ${decisionTheme?.headerBg}`}>
                <div className="flex items-baseline justify-between">
                  <div className="flex items-baseline gap-4">
                    <span className={`text-3xl font-black font-mono tracking-tight ${decisionTheme?.text}`}>
                      {response.decision}
                    </span>
                    <span className="text-neutral-500 font-mono">
                      NET {response.expected_net_value >= 0 ? '+' : ''}{formatINR(response.expected_net_value)}
                    </span>
                  </div>
                  <span className="font-mono text-neutral-400 text-[11px]">
                    P(WIN): {(response.p_win * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Review Reason Banner (if escalated) */}
                {response.decision === 'REVIEW' && response.review_reason && (
                  <div className="mt-3 p-2.5 bg-amber-50/80 border border-amber-200 text-amber-950 font-mono text-[11px] leading-relaxed">
                    <span className="font-bold">ESCALATION NOTICE: </span>
                    {response.review_reason}
                  </div>
                )}

                {/* System Reasoning Stack */}
                <div className="mt-3 space-y-1">
                  {response.reasoning.map((item, idx) => (
                    <p key={idx} className="text-xs text-neutral-700 leading-relaxed">
                      • {item}
                    </p>
                  ))}
                </div>
              </div>

              <div className="p-4 pt-0 space-y-6">
                {/* Economic Ledger */}
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-400 mb-2">
                    Economic Reconciliation (Net Value Layer)
                  </div>
                  <table className="w-full text-left font-mono border-t border-b border-neutral-200">
                    <thead>
                      <tr className="text-neutral-400 text-[10px] border-b border-neutral-100">
                        <th className="py-1.5 font-normal">Dispute Total</th>
                        <th className="py-1.5 font-normal">P(Win)</th>
                        <th className="py-1.5 font-normal">Expected Recovery</th>
                        <th className="py-1.5 font-normal">Defense Cost</th>
                        <th className="py-1.5 font-normal text-right">Expected Net</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-100">
                      <tr>
                        <td className="py-2">{formatINR(response.amount)}</td>
                        <td className="py-2">{(response.p_win * 100).toFixed(1)}%</td>
                        <td className="py-2">{formatINR(response.expected_recovery)}</td>
                        <td className="py-2 text-neutral-500">-{formatINR(response.estimated_defense_cost)}</td>
                        <td className={`py-2 text-right font-bold ${decisionTheme?.netHighlight}`}>
                          {formatINR(response.expected_net_value)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Forensic Discrepancy Matrix */}
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-400 mb-2">
                    Forensic Cross-Check: Narrative vs Merchant Records
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 border border-neutral-200 divide-y md:divide-y-0 md:divide-x divide-neutral-200 bg-white">
                    <div className="p-3">
                      <span className="text-[10px] uppercase font-mono text-neutral-400 block mb-1">
                        Customer Narrative Assertion
                      </span>
                      <p className="text-neutral-700 italic leading-normal">
                        &quot;{formData.dispute_text}&quot;
                      </p>
                      <div className="mt-3 pt-2 border-t border-neutral-100 text-[10px] font-mono text-neutral-400">
                        TYPE: {formData.dispute_type}
                      </div>
                    </div>
                    <div className="p-3 font-mono">
                      <span className="text-[10px] uppercase text-neutral-400 block mb-1">
                        Merchant Evidence
                      </span>
                      <div className="space-y-1 text-[11px]">
                        <div className="flex justify-between">
                          <span className="text-neutral-500">Delivered</span>
                          <span className={formData.delivered ? 'text-black font-bold' : 'text-neutral-400'}>
                            {formData.delivered ? 'TRUE' : 'FALSE'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-500">POD Signature Confirmed</span>
                          <span className={formData.delivery_confirmed ? 'text-black font-bold' : 'text-neutral-400'}>
                            {formData.delivery_confirmed ? 'TRUE' : 'FALSE'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-500">Device Recognized</span>
                          <span className={formData.device_seen_before ? 'text-black font-bold' : 'text-neutral-400'}>
                            {formData.device_seen_before ? 'TRUE' : 'FALSE'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-500">Location Match</span>
                          <span className={formData.location_consistent ? 'text-black font-bold' : 'text-neutral-400'}>
                            {formData.location_consistent ? 'TRUE' : 'FALSE'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Gemini Evidence Layer (Restrained Slate-Blue Info Style) */}
                <div>
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-slate-500 mb-2">
                    <span className="text-slate-700 font-semibold">Gemini Evidence Extraction (Layer 3)</span>
                    <span className="text-slate-500">Extracted Signal Interpretation</span>
                  </div>
                  <div className="border border-slate-200 border-l-[3px] border-l-slate-400 p-3 bg-slate-50/70 space-y-2">
                    <div className="flex items-baseline justify-between font-mono text-[11px]">
                      <div>
                        <span className="text-slate-500">CLAIM: </span>
                        <span className="font-semibold text-slate-900">{response.llm_evidence.customer_claim}</span>
                        <span className="text-slate-500 ml-1">
                          ({(response.llm_evidence.claim_confidence * 100).toFixed(0)}%)
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500">CONTRADICTION: </span>
                        <span
                          className={
                            response.llm_evidence.contradicts_merchant_evidence
                              ? 'font-bold text-amber-800'
                              : 'text-slate-700'
                          }
                        >
                          {response.llm_evidence.contradicts_merchant_evidence ? 'DETECTED' : 'NONE'}
                        </span>
                      </div>
                    </div>
                    <p className="text-slate-800 leading-normal border-t border-slate-200 pt-2">
                      {response.llm_evidence.contradiction_detail}
                    </p>
                    <div className="flex items-center justify-between pt-1 border-t border-slate-200/60 font-mono text-[10px] text-slate-500">
                      <span>Contradiction Confidence: {(response.llm_evidence.contradiction_confidence * 100).toFixed(0)}%</span>
                      <span>New Signal Present: {response.llm_evidence.new_signal_present ? 'YES' : 'NO'}</span>
                    </div>
                  </div>
                </div>

                {/* Raw JSON Audit Toggle */}
                <div className="pt-1">
                  <button
                    type="button"
                    onClick={() => setShowAudit(!showAudit)}
                    className="font-mono text-[11px] text-neutral-400 hover:text-black transition-colors"
                  >
                    [{showAudit ? '−' : '+'}] View Raw API Response Schema
                  </button>
                  {showAudit && (
                    <pre className="mt-2 p-3 bg-neutral-100 text-neutral-800 font-mono text-[10px] overflow-x-auto border border-neutral-200">
                      {JSON.stringify(response, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}