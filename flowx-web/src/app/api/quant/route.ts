import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  // 최신 시나리오 대시보드 데이터 조회
  const { data, error } = await supabase
    .from("quant_scenario_dashboard")
    .select("*")
    .order("date", { ascending: false })
    .limit(1);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const row = data?.[0];
  if (!row) {
    return NextResponse.json(null);
  }

  // data JSONB 칼럼에 date 추가
  return NextResponse.json({ ...row.data, date: row.date });
}
