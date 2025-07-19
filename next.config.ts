import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  webpack: (config: any) => {
    // Ignore .bin files in the model_web directory to prevent webpack from trying to process them
    config.module.rules.push({
      test: /\.bin$/,
      use: "ignore-loader",
    });
    return config;
  },
};

export default nextConfig;
