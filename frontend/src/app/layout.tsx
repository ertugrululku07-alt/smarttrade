import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "SmartTrade AI | Professional Crypto Trading Platform",
  description: "AI-powered crypto trading platform with StrategistAI, SupervisorAI, and LearnerAI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg-primary)' }}>
        <Sidebar />
        <main style={{ flex: 1, overflow: 'auto', height: '100vh' }}>
          {children}
        </main>
      </body>
    </html>
  );
}
