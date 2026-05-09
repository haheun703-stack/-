import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // 1. 요약 데이터 (JSONB — 최신 1건)
    const { data: summaryRows, error: sumErr } = await supabase
      .from("quant_surge_pullback_summary")
      .select("*")
      .order("date", { ascending: false })
      .limit(1);

    if (sumErr) {
      console.error("[API /surge-pullback] summary error:", sumErr.message);
    }

    // 2. Row 데이터 (최근 21일 — 3주)
    const { data: rows, error: rowErr } = await supabase
      .from("quant_surge_pullback")
      .select("*")
      .order("date", { ascending: false })
      .limit(500);

    if (rowErr) {
      console.error("[API /surge-pullback] rows error:", rowErr.message);
    }

    const summary = summaryRows?.[0];
    const summaryData =
      summary && typeof summary.data === "object" && summary.data !== null
        ? { ...summary.data, date: summary.date }
        : summary
          ? { date: summary.date }
          : null;

    // 날짜별 그룹핑 (히스토리용)
    const byDate: Record<string, typeof rows> = {};
    for (const r of rows ?? []) {
      const d = r.date;
      if (!byDate[d]) byDate[d] = [];
      byDate[d].push(r);
    }

    const dates = Object.keys(byDate).sort().reverse();

    return NextResponse.json({
      summary: summaryData,
      rows: rows ?? [],
      dates,
      byDate,
    });
  } catch (err) {
    console.error("[API /surge-pullback] error:", err);
    return NextResponse.json(
      { error: "데이터 조회 중 오류가 발생했습니다" },
      { status: 500 },
    );
  }
}
