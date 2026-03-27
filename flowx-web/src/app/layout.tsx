import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FLOWX | AI 주식 시그널",
  description: "퀀트 + 단타 AI 시그널 성적표",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-black min-h-screen`}
      >
        {/* 네비게이션 */}
        <nav className="border-b border-gray-800 bg-black/95 backdrop-blur sticky top-0 z-50">
          <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold text-white">FLOWX</span>
              <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                BETA
              </span>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="/"
                className="text-gray-400 hover:text-white text-sm transition-colors"
              >
                대시보드
              </a>
              <a
                href="/quant"
                className="text-gray-400 hover:text-white text-sm transition-colors"
              >
                퀀트
              </a>
              <a
                href="/signals"
                className="text-gray-400 hover:text-white text-sm transition-colors"
              >
                시그널
              </a>
            </div>
          </div>
        </nav>

        <main className="max-w-5xl mx-auto px-4 py-6">{children}</main>

        {/* 푸터 */}
        <footer className="border-t border-gray-800 mt-12 py-6 text-center text-gray-600 text-xs">
          <p>FLOWX | AI 기반 주식 시그널 &middot; 투자 판단은 본인 책임</p>
        </footer>
      </body>
    </html>
  );
}
