// @ts-check

import mdx from '@astrojs/mdx';
import preact from '@astrojs/preact';
import sitemap from '@astrojs/sitemap';
import { defineConfig, fontProviders } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
	// `site` is the production origin; `base` is the sub-path this project page
	// is served from (https://terrencealsup.github.io/home/). To later promote
	// this to the root URL or a custom domain, drop `base` and repoint `site`.
	site: 'https://terrencealsup.github.io',
	base: '/home',
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
