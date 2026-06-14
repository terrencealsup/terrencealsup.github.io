// @ts-check

import mdx from '@astrojs/mdx';
import preact from '@astrojs/preact';
import sitemap from '@astrojs/sitemap';
import { defineConfig, fontProviders } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
	// Served at the root of the user site: https://terrencealsup.github.io/
	// (no `base` — this repo is named <user>.github.io, so Pages serves it at /).
	site: 'https://terrencealsup.github.io',
	integrations: [mdx(), sitemap(), preact()],
	markdown: {
		// Renders $…$ / $$…$$ to static KaTeX HTML at build time (no client JS).
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	fonts: [
		{
			// Inter, self-hosted: Astro downloads and optimizes it at build time.
			provider: fontProviders.google(),
			name: 'Inter',
			cssVariable: '--font-inter',
			weights: [400, 500, 600, 700],
			styles: ['normal'],
			subsets: ['latin'],
			fallbacks: ['system-ui', '-apple-system', 'sans-serif'],
		},
	],
});
