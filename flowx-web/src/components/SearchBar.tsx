"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

interface SearchResult {
  ticker: string;
  name: string;
  grade: string;
  score: number;
  source: string;
  reason?: string;
}

const GRADE_COLOR: Record<string, string> = {
  "\uc801\uadf9\ub9e4\uc218": "text-red-400",
  "\ub9e4\uc218": "text-orange-400",
  "\uad00\uc2ec\ub9e4\uc218": "text-yellow-400",
  "\uad00\ucc30": "text-blue-400",
  AA: "text-red-400",
  A: "text-orange-400",
  B: "text-yellow-400",
  C: "text-gray-400",
};

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  // 외부 클릭 시 닫기
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const search = useCallback(async (q: string) => {
    if (!q.trim() || q.trim().length < 2) {
      setResults([]);
      setOpen(false);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q.trim())}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setResults(data);
        setOpen(data.length > 0);
      }
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    setSelectedIdx(-1);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => search(val), 300);
  };

  const navigate = (ticker: string) => {
    if (!ticker) return;
    setOpen(false);
    setQuery("");
    router.push(`/stock/${ticker}`);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIdx((prev) => Math.min(prev + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIdx((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter" && selectedIdx >= 0) {
      e.preventDefault();
      navigate(results[selectedIdx].ticker);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div ref={wrapperRef} className="relative">
      <div className="flex items-center gap-2 bg-gray-900 rounded-xl px-3 py-1.5">
        <svg
          className="w-4 h-4 text-gray-500 shrink-0"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
        <input
          type="text"
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder={"종목 검색"}
          className="bg-transparent text-sm text-gray-200 placeholder-gray-600 outline-none w-32 sm:w-48"
        />
        {loading && (
          <div className="w-3 h-3 border border-gray-500 border-t-transparent rounded-full animate-spin" />
        )}
      </div>

      {open && results.length > 0 && (
        <div className="absolute top-full mt-2 right-0 w-80 bg-gray-900 rounded-xl shadow-2xl overflow-hidden z-50 max-h-96 overflow-y-auto">
          {results.map((r, i) => (
            <button
              key={`${r.ticker}-${i}`}
              onClick={() => navigate(r.ticker)}
              className={`w-full text-left px-4 py-3 transition-colors ${
                i === selectedIdx ? "bg-gray-800" : "hover:bg-gray-800/50"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {r.grade && (
                    <span
                      className={`text-xs font-bold ${GRADE_COLOR[r.grade] || "text-gray-400"}`}
                    >
                      {r.grade}
                    </span>
                  )}
                  <span className="text-gray-200 text-sm font-medium">
                    {r.name}
                  </span>
                  <span className="text-gray-600 text-xs">{r.ticker}</span>
                </div>
                {r.score > 0 && (
                  <span className="text-blue-400 text-xs font-mono">
                    {r.score}점
                  </span>
                )}
              </div>
              {r.reason && (
                <p className="text-gray-500 text-xs mt-1 truncate">{r.reason}</p>
              )}
              <p className="text-gray-600 text-xs mt-0.5">{r.source}</p>
            </button>
          ))}
        </div>
      )}

      {open && results.length === 0 && !loading && query.trim().length >= 2 && (
        <div className="absolute top-full mt-2 right-0 w-80 bg-gray-900 rounded-xl shadow-2xl z-50 p-4">
          <p className="text-gray-400 text-sm text-center mb-1">
            {"\uAC80\uC0C9"} {"\uACB0\uACFC\uAC00"} {"\uC5C6\uC2B5\uB2C8\uB2E4"}
          </p>
          <p className="text-gray-600 text-xs text-center">
            FLOWX{"\uAC00"} {"\uBD84\uC11D\uD55C"} {"\uC885\uBAA9\uB9CC"} {"\uAC80\uC0C9"} {"\uAC00\uB2A5\uD569\uB2C8\uB2E4"}
          </p>
        </div>
      )}
    </div>
  );
}
