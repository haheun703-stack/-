"use client";

import { useEffect, useState } from "react";
import type { ScenarioDashboard } from "@/lib/types";
import MarketStatusBar from "./MarketStatusBar";
import ScenarioCard from "./ScenarioCard";
import CommodityTable from "./CommodityTable";
import ScenarioStockCard from "./ScenarioStockCard";
import ScenarioETFMap from "./ScenarioETFMap";
import ScenarioGuide from "./ScenarioGuide";

export default function QuantDashboard() {
  const [data, setData] = useState<ScenarioDashboard | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const res = await fetch("/api/quant", { signal: controller.signal });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const json = await res.json();
        if (!json || !json.market_status) {
          setData(null);
        } else {
          setData(json);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setData(null);
      }
      setLoading(false);
    }
    load();
    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-20 bg-gray-800 rounded-lg" />
          ))}
        </div>
        <div className="h-48 bg-gray-800 rounded-lg" />
        <div className="h-32 bg-gray-800 rounded-lg" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">시나리오 데이터가 아직 없습니다.</p>
        <p className="text-gray-600 text-sm mt-1">매일 장마감 후 업데이트됩니다.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* 1. 시장 상황판 */}
      <section>
        <MarketStatusBar status={data.market_status} />
      </section>

      {/* 2. 활성 시나리오 */}
      <section>
        <h2 className="text-white text-xl font-bold mb-4">활성 시나리오</h2>
        <ScenarioCard scenarios={data.active_scenarios} conflicts={data.conflicts} />
      </section>

      {/* 3. 원자재 원가갭 분석 */}
      <section>
        <h2 className="text-white text-xl font-bold mb-4">원자재 원가갭 분석</h2>
        <CommodityTable commodities={data.commodities} />
      </section>

      {/* 4. 시나리오 연동 종목 */}
      {(data.scenario_stocks?.length ?? 0) > 0 && (
        <section>
          <h2 className="text-white text-xl font-bold mb-4">시나리오 연동 종목</h2>
          <ScenarioStockCard stocks={data.scenario_stocks} />
        </section>
      )}

      {/* 5. ETF 매핑 */}
      {(data.etf_map?.length ?? 0) > 0 && (
        <section>
          <h2 className="text-white text-xl font-bold mb-4">시나리오별 ETF 매핑</h2>
          <ScenarioETFMap etfMap={data.etf_map} />
        </section>
      )}

      {/* 6. 초보자 가이드 */}
      <section>
        <h2 className="text-white text-xl font-bold mb-4">초보자 가이드</h2>
        <ScenarioGuide />
      </section>
    </div>
  );
}
