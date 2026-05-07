"use client";

import { useEffect, useState } from "react";

import { getHealth, getModelInfo, predictOne } from "@/lib/api";
import { SAMPLES, type TransactionSample } from "@/lib/samples";
import type {
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
} from "@/lib/types";

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [activeSample, setActiveSample] = useState<TransactionSample | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showJson, setShowJson] = useState(false);

  // Fetch /health and /model-info on mount.
  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(String(e)));
    getModelInfo().then(setModelInfo).catch(() => {
      // model-info failing is non-fatal — the demo still works
    });
  }, []);

  async function onSample(sample: TransactionSample) {
    setLoading(true);
    setError(null);
    setActiveSample(sample);
    setResult(null);
    try {
      const r = await predictOne(sample.transaction);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="mx-auto max-w-3xl px-6 py-12">
        {/* Header */}
        <header className="mb-10 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">
              Credit Card Fraud Detection
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              XGBoost classifier &middot; test PR-AUC{" "}
              <span className="font-mono text-slate-200">
                {modelInfo
                  ? Number(modelInfo.test_metrics.pr_auc ?? 0).toFixed(3)
                  : "..."}
              </span>{" "}
              &middot; cost-tuned threshold
            </p>
          </div>
          <ApiBadge health={health} error={error} />
        </header>

        {/* Demo card */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-xl">
          <h2 className="text-lg font-medium">Try a transaction</h2>
          <p className="mt-1 text-sm text-slate-400">
            Click any sample to send its 30 features to{" "}
            <code className="font-mono text-slate-300">POST /predict</code>.
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            {SAMPLES.map((s) => (
              <button
                key={s.id}
                onClick={() => onSample(s)}
                disabled={loading}
                className={`rounded-lg border px-4 py-2 text-sm font-medium transition disabled:opacity-50 ${
                  activeSample?.id === s.id
                    ? "border-indigo-500 bg-indigo-950/40 text-indigo-100"
                    : "border-slate-700 bg-slate-800 text-slate-200 hover:bg-slate-700"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>

          {loading && (
            <div className="mt-6 rounded-lg bg-slate-800/50 p-4 text-sm text-slate-400">
              Calling /predict&hellip;
            </div>
          )}

          {error && (
            <div className="mt-6 rounded-lg border border-rose-900 bg-rose-950/30 p-4 text-sm text-rose-200">
              <p className="font-medium">Request failed</p>
              <p className="mt-1 break-all font-mono text-xs">{error}</p>
              <p className="mt-2 text-xs text-rose-300/70">
                Make sure the FastAPI server is running:{" "}
                <code className="rounded bg-rose-950/50 px-1.5 py-0.5">
                  python -m uvicorn app.main:app
                </code>
              </p>
            </div>
          )}

          {result && !loading && (
            <ResultCard result={result} sample={activeSample} />
          )}

          {result && (
            <button
              onClick={() => setShowJson((v) => !v)}
              className="mt-4 text-xs text-slate-400 underline-offset-4 hover:text-slate-200 hover:underline"
            >
              {showJson ? "Hide" : "Show"} raw JSON
            </button>
          )}
          {result && showJson && (
            <pre className="mt-2 overflow-x-auto rounded-lg bg-slate-950 p-4 text-xs text-slate-300">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}

          {activeSample && (
            <p className="mt-4 text-xs text-slate-500">
              {activeSample.description}
            </p>
          )}
        </section>

        {/* Model info */}
        {modelInfo && (
          <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
            <h2 className="text-lg font-medium">Model</h2>
            <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
              <dt className="text-slate-400">Name</dt>
              <dd className="font-mono">{modelInfo.model_name}</dd>
              <dt className="text-slate-400">Trained at (UTC)</dt>
              <dd className="font-mono text-xs">{modelInfo.trained_at_utc}</dd>
              <dt className="text-slate-400">Threshold</dt>
              <dd className="font-mono">{modelInfo.threshold}</dd>
              <dt className="text-slate-400">Features</dt>
              <dd className="font-mono">{modelInfo.feature_columns.length}</dd>
              <dt className="text-slate-400">Test PR-AUC</dt>
              <dd className="font-mono">
                {Number(modelInfo.test_metrics.pr_auc ?? 0).toFixed(4)}
              </dd>
              <dt className="text-slate-400">Test recall</dt>
              <dd className="font-mono">
                {Number(modelInfo.test_metrics.recall ?? 0).toFixed(4)}
              </dd>
            </dl>
          </section>
        )}

        <footer className="mt-12 text-center text-xs text-slate-500">
          FastAPI + Next.js &middot; learning project
        </footer>
      </div>
    </main>
  );
}

function ApiBadge({
  health,
  error,
}: {
  health: HealthResponse | null;
  error: string | null;
}) {
  if (error) {
    return (
      <span className="whitespace-nowrap rounded-full bg-rose-950 px-3 py-1 text-xs text-rose-300 ring-1 ring-rose-900">
        ● API down
      </span>
    );
  }
  if (!health) {
    return (
      <span className="whitespace-nowrap rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-400 ring-1 ring-slate-700">
        ● connecting&hellip;
      </span>
    );
  }
  return (
    <span className="whitespace-nowrap rounded-full bg-emerald-950 px-3 py-1 text-xs text-emerald-300 ring-1 ring-emerald-900">
      ● API ready
    </span>
  );
}

function ResultCard({
  result,
  sample,
}: {
  result: PredictionResponse;
  sample: TransactionSample | null;
}) {
  const isFraud = result.is_fraud;
  const proba = result.fraud_probability;
  const pct = (proba * 100).toFixed(2);
  const wasMistake =
    sample !== null && (sample.actualClass === 1) !== isFraud;
  const mistakeLabel =
    sample?.actualClass === 0 ? "false positive" : "false negative";

  return (
    <div
      className={`mt-6 rounded-xl border p-5 ${
        isFraud
          ? "border-rose-900 bg-rose-950/30"
          : "border-emerald-900 bg-emerald-950/30"
      }`}
    >
      <div className="flex items-baseline justify-between gap-4">
        <span
          className={`text-xl font-semibold ${
            isFraud ? "text-rose-200" : "text-emerald-200"
          }`}
        >
          {isFraud ? "🚨 FRAUD" : "✅ GENUINE"}
        </span>
        <span className="font-mono text-sm text-slate-300">
          P(fraud) = <strong className="text-slate-100">{pct}%</strong>
        </span>
      </div>

      {/* Probability bar with threshold marker */}
      <div className="relative mt-4 h-2 overflow-visible rounded-full bg-slate-800">
        <div
          className={`absolute inset-y-0 left-0 rounded-full transition-[width] duration-500 ${
            isFraud ? "bg-rose-500" : "bg-emerald-500"
          }`}
          style={{ width: `${Math.min(100, proba * 100)}%` }}
        />
        <div
          className="absolute -top-1 -bottom-1 w-px bg-slate-300"
          style={{ left: `${result.threshold * 100}%` }}
          title={`threshold = ${result.threshold}`}
        />
      </div>
      <p className="mt-2 text-xs text-slate-400">
        threshold (vertical line) ={" "}
        <span className="font-mono text-slate-300">{result.threshold}</span>{" "}
        &middot; model:{" "}
        <span className="font-mono text-slate-300">{result.model_name}</span>
      </p>

      {sample && (
        <p className="mt-4 text-xs text-slate-400">
          Ground truth:{" "}
          <span className="font-mono text-slate-300">
            {sample.actualClass === 1 ? "fraud" : "genuine"}
          </span>
          {wasMistake && (
            <span className="ml-2 inline-flex items-center rounded bg-amber-950/50 px-2 py-0.5 font-medium text-amber-300 ring-1 ring-amber-900">
              model error &middot; {mistakeLabel}
            </span>
          )}
        </p>
      )}
    </div>
  );
}
