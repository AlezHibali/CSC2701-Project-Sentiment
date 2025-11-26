'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';

interface DataItem {
  date: string; // YYYY-MM-DD
  content: string;
}

interface ClassificationResult {
  label: 'positive' | 'negative' | 'neutral' | string;
  score: number; // 0..1
}

interface MatchedWithScore {
  date: string;
  content: string;
  label: string;
  score: number;
  sentimentValue: number; // score * (+1|-1) or 0 for neutral
}

export default function Page() {
  const [dataset, setDataset] = useState<DataItem[]>([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<DataItem[]>([]);
  const [matchedWithScores, setMatchedWithScores] = useState<MatchedWithScore[]>([]);
  const [chartData, setChartData] = useState<{ date: string; sentiment: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --------- CSV Parsing (handles quoted fields with commas) ----------
  const splitCsvLine = (line: string) => {
    // split on commas not inside quotes
    return line.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/).map((s) => {
      let t = s.trim();
      // remove surrounding quotes if present and unescape double quotes
      if (t.startsWith('"') && t.endsWith('"')) {
        t = t.slice(1, -1).replace(/""/g, '"');
      }
      return t;
    });
  };

  const parseCSV = (text: string): DataItem[] => {
    const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return [];

    // Expect header including 'date' and 'content' in some order
    const headerParts = splitCsvLine(lines[0]).map((h) => h.toLowerCase());
    const dateIdx = headerParts.indexOf('date');
    const contentIdx = headerParts.indexOf('content');

    // fallback: assume first two columns are date, content
    const useDateIdx = dateIdx >= 0 ? dateIdx : 0;
    const useContentIdx = contentIdx >= 0 ? contentIdx : 1;

    const out: DataItem[] = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = splitCsvLine(lines[i]);
      if (parts.length <= Math.max(useDateIdx, useContentIdx)) continue;
      const date = parts[useDateIdx];
      const content = parts[useContentIdx];
      if (!date || !content) continue;
      out.push({ date, content });
    }
    return out;
  };

  // --------- load dataset once (on mount) ----------
  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const res = await fetch('/dataset.csv');
        if (!res.ok) {
          throw new Error(`Failed to load dataset.csv (${res.status})`);
        }
        const text = await res.text();
        const parsed = parseCSV(text);
        if (mounted) setDataset(parsed);
      } catch (err: any) {
        console.error('Failed to load dataset:', err);
        if (mounted) setError(String(err?.message ?? err));
      }
    };
    load();
    return () => { mounted = false; };
  }, []);

  // --------- utility: call classify endpoint for one text ----------
  const classifyOne = async (text: string): Promise<ClassificationResult | null> => {
    try {
      const resp = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tweet: text }),
      });
      if (!resp.ok) {
        console.warn('classify failed for text:', text, resp.status);
        return null;
      }
      const data = (await resp.json()) as ClassificationResult;
      return data;
    } catch (e) {
      console.error('classifyOne error', e);
      return null;
    }
  };

  // --------- handle search: find items, classify them, build chart ----------
  const handleSearch = async () => {
    setError(null);
    setMatchedWithScores([]);
    setChartData([]);
    setResults([]);
    if (!query.trim()) {
      setError('Please enter a search query.');
      return;
    }

    if (dataset.length === 0) {
      setError('Dataset not loaded yet.');
      return;
    }

    setLoading(true);

    try {
      const lower = query.toLowerCase();
      const matched = dataset.filter((d) => d.content.toLowerCase().includes(lower));
      setResults(matched);

      if (matched.length === 0) {
        setLoading(false);
        return;
      }

      // classify all matched items in parallel
      const promises = matched.map((m) => classifyOne(m.content));
      const settled = await Promise.allSettled(promises);

      const scored: MatchedWithScore[] = [];
      for (let i = 0; i < matched.length; i++) {
        const s = settled[i];
        if (s.status === 'fulfilled' && s.value) {
          const lab = s.value.label;
          const sc = typeof s.value.score === 'number' ? s.value.score : 0;
          let sentimentValue = 0;
          if (lab === 'positive') sentimentValue = sc * 1;
          else if (lab === 'negative') sentimentValue = sc * -1;
          else sentimentValue = 0;
          scored.push({
            date: matched[i].date,
            content: matched[i].content,
            label: lab,
            score: sc,
            sentimentValue,
          });
        } else {
          // classifier failed; push a placeholder with neutral=0
          scored.push({
            date: matched[i].date,
            content: matched[i].content,
            label: 'error',
            score: 0,
            sentimentValue: 0,
          });
        }
      }

      setMatchedWithScores(scored);

      // Aggregate by date -> average sentiment per date
      const byDate = new Map<string, { sum: number; count: number }>();
      for (const row of scored) {
        const d = row.date;
        const agg = byDate.get(d) ?? { sum: 0, count: 0 };
        agg.sum += row.sentimentValue;
        agg.count += 1;
        byDate.set(d, agg);
      }

      // Build chart array sorted by date
      const chartArr = Array.from(byDate.entries())
        .map(([date, agg]) => ({ date, sentiment: agg.count ? agg.sum / agg.count : 0 }))
        .sort((a, b) => {
          // try ISO date sort
          if (a.date < b.date) return -1;
          if (a.date > b.date) return 1;
          return 0;
        });

      setChartData(chartArr);
    } catch (err: any) {
      console.error(err);
      setError(String(err?.message ?? err));
    } finally {
      setLoading(false);
    }
  };

  const clearSearch = () => {
    setQuery('');
    setResults([]);
    setMatchedWithScores([]);
    setChartData([]);
    setError(null);
  };

  // nice memoized tooltip formatter
  const tooltipFormatter = (value: any) => {
    if (value == null) return '';
    return typeof value === 'number' ? value.toFixed(4) : String(value);
  };

  // ---------------- UI ----------------
  return (
    <main className="flex flex-col items-center p-8 min-h-screen bg-slate-50">
      <div className="w-full max-w-4xl space-y-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold">Dataset Sentiment Analyzer</h1>
          <p className="text-sm text-gray-600 mt-1">
            Search your dataset and visualize sentiment over time.
          </p>
        </header>

        <section className="bg-white p-4 rounded-lg shadow">
          <div className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search text (case-insensitive)..."
              className="flex-1 p-2 border rounded-md"
            />
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="px-4 py-2 bg-indigo-600 text-white rounded-md disabled:opacity-50"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
            <button
              onClick={clearSearch}
              className="px-3 py-2 border rounded-md"
            >
              Clear
            </button>
          </div>

          {error && (
            <div className="mt-3 text-sm text-red-700 bg-red-50 border border-red-200 p-2 rounded">
              {error}
            </div>
          )}
        </section>

        <section>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-3">Matched Items ({results.length})</h2>

              {results.length === 0 ? (
                <p className="text-sm text-gray-500">No matches yet.</p>
              ) : (
                <ul className="space-y-3 max-h-[420px] overflow-auto">
                  {matchedWithScores.length > 0 ? (
                    matchedWithScores.map((m, idx) => (
                      <li key={idx} className="p-3 border rounded-md bg-gray-50">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-sm text-gray-600">
                              <strong>Date:</strong> {m.date}
                            </div>
                            <div className="text-sm text-gray-800 mt-1">{m.content}</div>
                          </div>
                          <div className="text-right">
                            <div className={`px-2 py-1 rounded text-sm font-medium ${m.label === 'positive' ? 'bg-green-100 text-green-800' : m.label === 'negative' ? 'bg-red-100 text-red-800' : m.label === 'neutral' ? 'bg-amber-100 text-amber-800' : 'bg-gray-100 text-gray-800'}`}>
                              {m.label}
                            </div>
                            <div className="text-xs text-gray-600 mt-1">confidence score: {m.score.toFixed(4)}</div>
                          </div>
                        </div>
                      </li>
                    ))
                  ) : (
                    // results present but scores not yet available (classification in progress)
                    results.map((r, i) => (
                      <li key={i} className="p-3 border rounded-md bg-gray-50">
                        <div className="text-sm text-gray-600"><strong>Date:</strong> {r.date}</div>
                        <div className="text-sm text-gray-800 mt-1">{r.content}</div>
                      </li>
                    ))
                  )}
                </ul>
              )}
            </div>

            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-3">Sentiment Over Time</h2>

              {chartData.length === 0 ? (
                <div className="text-sm text-gray-500">No sentiment data to display.</div>
              ) : (
                <div style={{ width: '100%', height: 320 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                      <YAxis domain={[-1, 1]} />
                      <Tooltip formatter={tooltipFormatter} labelFormatter={(lab) => `Date: ${lab}`} />
                      <Line type="monotone" dataKey="sentiment" stroke="#6366f1" strokeWidth={3} dot />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              <div className="mt-3 text-xs text-gray-500">
                Note: sentiment value = confidence × (+1 for positive, −1 for negative). Neutral = 0. If multiple items share a date, chart uses the average sentiment for that date.
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
