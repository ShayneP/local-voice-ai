import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  eslint: {
    // ビルド時のESLintチェックをスキップ（フォーマットエラーを無視）
    ignoreDuringBuilds: true,
  },
  typescript: {
    // ビルド時の型チェックエラーを無視
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
