import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const date = searchParams.get("date");

  if (date) {
    if (!DATE_RE.test(date)) {
      return NextResponse.json({ error: "잘못된 date 형식 (YYYY-MM-DD)" }, { status: 400 });
    }

    const { data, error } = await supabase
      .from("morning_briefings")
      .select("*")
      .eq("date", date)
      .maybeSingle();

    if (error) {
      console.error("[API /briefing] Supabase error:", error.message);
      return NextResponse.json({ error: "브리핑 조회 중 오류가 발생했습니다" }, { status: 500 });
    }
    return NextResponse.json(data);
  }

  // 최신 브리핑 조회
  const { data, error } = await supabase
    .from("morning_briefings")
    .select("*")
    .order("date", { ascending: false })
    .limit(1);

  if (error) {
    console.error("[API /briefing] Supabase error:", error.message);
    return NextResponse.json({ error: "브리핑 조회 중 오류가 발생했습니다" }, { status: 500 });
  }

  return NextResponse.json(data?.[0] || null);
}
