// Prefix an internal path with the configured base (e.g. `/home`) so links and
// public/ assets resolve correctly when the site is served from a sub-path.
// Imported assets (src/assets via <Image>) are already base-aware and do NOT
// need this — use it for hardcoded hrefs and files in public/.
export function withBase(path: string): string {
	const base = import.meta.env.BASE_URL.replace(/\/$/, ''); // '/home/' -> '/home'
	return path.startsWith('/') ? `${base}${path}` : `${base}/${path}`;
}
