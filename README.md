# terrencealsup — personal site

Personal website and blog built with [Astro](https://astro.build). Minimal,
ships zero JS by default, renders LaTeX math at build time, and supports
occasional interactive widgets (Preact islands).

## Develop

```sh
npm install
npm run dev      # http://localhost:4321/home/  (note the /home base path)
npm run build    # static build into dist/ (also the deploy gate — fails on bad LaTeX)
npm run preview  # serve the built dist/ at the base path
```

## Structure

- `src/pages/index.astro` — home page (headshot, bio, links).
- `src/pages/publications.astro` — publications list.
- `src/pages/blog/` — blog listing + `[...slug]` post route.
- `src/content/blog/*.{md,mdx}` — posts. Use `.md` for math-only posts and
  `.mdx` when embedding an interactive component.
- `src/components/` — layout components; `widgets/` holds interactive islands.
- `src/consts.ts` — site title, description, and contact links.
- `src/utils.ts` — `withBase()` for base-path-aware links to `public/` assets.
- `public/` — stable-URL assets (CV, notebooks, PDFs).

## Writing posts

Front matter: `title`, `pubDate` (required), optional `description`, `tags`,
`heroImage`, `draft`. Math uses `$…$` (inline) and `$$…$$` (display); a bad
expression fails the build. Reference post figures with **relative** paths
(`../../assets/...`) so Astro optimizes them; link `public/` downloads with the
base path (e.g. `/home/posts/...`).

To add an interactive widget, author the post as `.mdx`, import a component from
`src/components/widgets/`, and hydrate it with a `client:visible` directive.

## Deploy

Pushing to `main` runs `.github/workflows/deploy.yml`, which builds and deploys
to GitHub Pages. One-time setup in the repo: **Settings → Pages → Source =
"GitHub Actions"**. Served at `https://terrencealsup.github.io/home/`.

The `base: '/home'` in `astro.config.mjs` reflects the project-page sub-path. To
promote this to the root URL or a custom domain later, drop `base` and repoint
`site`.
