"use client";

interface PaywallBlurProps {
  locked: boolean;
  children: React.ReactNode;
}

export default function PaywallBlur({ locked, children }: PaywallBlurProps) {
  if (!locked) return <>{children}</>;

  return (
    <div className="relative">
      <div className="blur-md select-none pointer-events-none">{children}</div>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="bg-gray-900/90 rounded-xl px-8 py-6 text-center shadow-2xl border border-gray-700">
          <div className="text-2xl mb-2">🔒</div>
          <p className="text-white font-bold text-lg">SIGNAL 구독 전용</p>
          <p className="text-gray-400 text-sm mt-1">
            전체 시그널과 성적표를 확인하세요
          </p>
          <button className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors">
            구독하기
          </button>
        </div>
      </div>
    </div>
  );
}
