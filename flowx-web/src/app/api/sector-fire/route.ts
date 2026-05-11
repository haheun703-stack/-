import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // 최신 날짜의 섹터 FIRE + Structure + Composite 데이터
    const { data: rows, error } = await supabase
      .from("quant_sector_fire")
      .select("*")
      .order("date", { ascending: false })
      .limit(30);

    if (error) {
      console.error("[API /sector-fire] error:", error.message);
      return NextResponse.json(
        { error: "데이터 조회 중 오류가 발생했습니다" },
        { status: 500 },
      );
    }

    if (!rows || rows.length === 0) {
      return NextResponse.json({ sectors: [], date: null });
    }

    // 최신 날짜만 필터
    const latestDate = rows[0].date;
    const latest = rows.filter((r) => r.date === latestDate);

    // composite_score 내림차순 정렬
    latest.sort(
      (a, b) => (b.composite_score ?? b.fire_score ?? 0) - (a.composite_score ?? a.fire_score ?? 0),
    );

    return NextResponse.json({
      sectors: latest,
      date: latestDate,
    });
  } catch (err) {
    console.error("[API /sector-fire] error:", err);
    return NextResponse.json(
      { error: "서버 오류가 발생했습니다" },
      { status: 500 },
    );
  }
}
