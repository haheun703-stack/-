import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const date = searchParams.get("date");

  if (date) {
    // 특정 날짜 조회
    const { data, error } = await supabase
      .from("morning_briefings")
      .select("*")
      .eq("date", date)
      .maybeSingle();

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
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
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data?.[0] || null);
}
