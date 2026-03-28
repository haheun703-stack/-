"use client";

import Link from "next/link";

const FEATURES = [
  {
    icon: "\u{1F916}",
    title: "AI \uBCF5\uD569 \uBD84\uC11D",
    desc: "\uD140\uD2B8+\uB2E8\uD0C0+\uC2DC\uB098\uB9AC\uC624 3\uAC1C \uBD07 \uAD50\uCC28\uAC80\uC99D",
  },
  {
    icon: "\u{1F4CA}",
    title: "\uC2E4\uC2DC\uAC04 \uC131\uC801\uD45C",
    desc: "30/60/90\uC77C \uC801\uC911\uB960 \uD22C\uBA85\uD558\uAC8C \uACF5\uAC1C",
  },
  {
    icon: "\u{1F50D}",
    title: "\uC885\uBAA9\uBCC4 \uADFC\uAC70",
    desc: "\u201C\uC65C \uC9C0\uAE08 \uC774 \uC885\uBAA9\uC778\uAC00\u201D AI \uBD84\uC11D",
  },
  {
    icon: "\u{1F6E1}",
    title: "\uC704\uD5D8 \uAD00\uB9AC",
    desc: "VIX+\uB808\uC9D0+SHIELD \uBCF5\uD569 \uC704\uD5D8 \uBAA8\uB4DC",
  },
];

export default function HeroSection() {
  return (
    <section className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-gray-900 via-gray-900 to-blue-950/30 p-8 md:p-12 mb-10">
      {/* 배경 장식 */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 left-0 w-48 h-48 bg-green-500/5 rounded-full blur-3xl" />

      <div className="relative z-10">
        {/* 타이틀 */}
        <div className="mb-8">
          <p className="text-blue-400 text-sm font-medium mb-2">
            AI {"\uAE30\uBC18"} {"\uC8FC\uC2DD"} {"\uC2DC\uADF8\uB110"} {"\uD50C\uB7AB\uD3FC"}
          </p>
          <h1 className="text-3xl md:text-4xl font-bold text-gray-100 leading-tight mb-3">
            {"\uD140\uD2B8"} + {"\uB2E8\uD0C0"} + AI{"\uAC00"}
            <br />
            <span className="text-blue-400">{"\uAD50\uCC28\uAC80\uC99D\uD55C"} {"\uC885\uBAA9\uB9CC"}</span>
          </h1>
          <p className="text-gray-400 text-sm md:text-base max-w-lg">
            {"\uB9E4\uC77C"} {"\uC544\uCE68"} 1,000+{"\uC885\uBAA9\uC744"} AI{"\uAC00"} {"\uBD84\uC11D\uD558\uACE0"},{" "}
            {"\uAD50\uCC28\uAC80\uC99D\uC744"} {"\uD1B5\uACFC\uD55C"} {"\uC2DC\uADF8\uB110\uB9CC"} {"\uC5C4\uC120\uD558\uC5EC"} {"\uC804\uB2EC\uD569\uB2C8\uB2E4"}.
            {"\uBAA8\uB4E0"} {"\uC131\uACFC\uB294"} {"\uC801\uC911\uB960\uB85C"} {"\uACF5\uAC1C\uB429\uB2C8\uB2E4"}.
          </p>
        </div>

        {/* 피처 그리드 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          {FEATURES.map((f) => (
            <div key={f.title} className="bg-gray-800/50 rounded-xl p-4">
              <p className="text-2xl mb-2">{f.icon}</p>
              <p className="text-gray-200 text-sm font-medium">{f.title}</p>
              <p className="text-gray-500 text-xs mt-1">{f.desc}</p>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className="flex items-center gap-4">
          <Link
            href="/quant"
            className="bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-6 py-2.5 rounded-xl transition-colors"
          >
            {"\uD140\uD2B8"} {"\uBD84\uC11D"} {"\uBCF4\uAE30"}
          </Link>
          <Link
            href="/signals"
            className="bg-gray-800 hover:bg-gray-700 text-gray-200 text-sm font-medium px-6 py-2.5 rounded-xl transition-colors"
          >
            {"\uC2DC\uADF8\uB110"} {"\uC131\uC801\uD45C"}
          </Link>
        </div>
      </div>
    </section>
  );
}
