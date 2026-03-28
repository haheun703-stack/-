import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

interface StockResult {
  ticker: string;
  name: string;
  grade: string;
  score: number;
  source: string;
  reason?: string;
  signals?: string;
  entry_price?: number;
  stop_loss?: number;
  target_price?: number;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const q = (searchParams.get("q") || "").trim();

    if (!q || q.length < 1) {
      return NextResponse.json([]);
    }

    const results: StockResult[] = [];
    const seen = new Set<string>();

    // 1) quant_jarvis 최신 데이터에서 picks 검색
    const { data: jarvisRows } = await supabase
      .from("quant_jarvis")
      .select("data")
      .order("date", { ascending: false })
      .limit(1);

    const jarvis = jarvisRows?.[0]?.data;
    if (jarvis && typeof jarvis === "object") {
      const picks = (jarvis as Record<string, unknown>).picks as Record<string, unknown> | undefined;
      const pickList = (picks?.picks ?? []) as Record<string, unknown>[];

      for (const p of pickList) {
        const ticker = String(p.ticker || "");
        const name = String(p.name || "");
        if (!ticker) continue;

        const match =
          name.includes(q) ||
          ticker.includes(q) ||
          q.includes(name) ||
          q.includes(ticker);

        if (match && !seen.has(ticker)) {
          seen.add(ticker);
          const signals = Array.isArray(p.signals)
            ? (p.signals as string[]).join(", ")
            : typeof p.signals === "string"
              ? p.signals as string
              : "";
          const reasons = Array.isArray(p.reasons)
            ? (p.reasons as string[]).join(" | ")
            : typeof p.reason === "string"
              ? p.reason as string
              : "";

          results.push({
            ticker,
            name,
            grade: String(p.grade || ""),
            score: typeof p.total_score === "number" ? p.total_score : 0,
            source: "quant",
            reason: reasons || undefined,
            signals: signals || undefined,
            entry_price: typeof p.entry_price === "number" ? p.entry_price : undefined,
            stop_loss: typeof p.stop_loss === "number" ? p.stop_loss : undefined,
            target_price: typeof p.target_price === "number" ? p.target_price : undefined,
          });
        }
      }
    }

    // 2) morning_briefings 최근 7일 news_picks에서 검색
    const { data: briefings } = await supabase
      .from("morning_briefings")
      .select("date, news_picks")
      .order("date", { ascending: false })
      .limit(7);

    if (briefings) {
      for (const row of briefings) {
        const picks = row.news_picks;
        if (!Array.isArray(picks)) continue;
        for (const p of picks) {
          if (!p || typeof p !== "object") continue;
          const name = String(p.name || p.title || "");
          const ticker = String(p.ticker || "");
          if (!name && !ticker) continue;

          const match =
            name.includes(q) ||
            ticker.includes(q) ||
            q.includes(name) ||
            q.includes(ticker);

          if (match && !seen.has(ticker || name)) {
            seen.add(ticker || name);
            results.push({
              ticker: ticker || "",
              name: name || "N/A",
              grade: String(p.grade || ""),
              score: typeof p.score === "number" ? p.score : 0,
              source: `briefing (${row.date})`,
              reason: typeof p.reason === "string" ? p.reason : undefined,
            });
          }
        }
      }
    }

    // 3) signals 테이블에서 검색
    const { data: signals } = await supabase
      .from("signals")
      .select("ticker, ticker_name, signal_type, grade, score, entry_price, target_price, stop_price, status, signal_date")
      .or(`ticker_name.ilike.%${q}%,ticker.ilike.%${q}%`)
      .order("signal_date", { ascending: false })
      .limit(10);

    if (signals) {
      for (const s of signals) {
        if (!seen.has(s.ticker)) {
          seen.add(s.ticker);
          results.push({
            ticker: s.ticker,
            name: s.ticker_name || s.ticker,
            grade: s.grade || "",
            score: s.score || 0,
            source: `signal (${s.signal_type} ${s.signal_date})`,
            entry_price: s.entry_price ?? undefined,
            stop_loss: s.stop_price ?? undefined,
            target_price: s.target_price ?? undefined,
          });
        }
      }
    }

    // 점수 내림차순 정렬
    results.sort((a, b) => b.score - a.score);

    return NextResponse.json(results.slice(0, 20));
  } catch (e) {
    console.error("[API /search] Unexpected error:", e);
    return NextResponse.json(
      { error: "검색 중 오류가 발생했습니다" },
      { status: 500 },
    );
  }
}
