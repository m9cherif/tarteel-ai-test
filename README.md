# Tasnim Cherif — Portfolio

A single-page portfolio for Tasnim Cherif, business data analyst. One
self-contained `index.html`: no build step, no dependencies, no external
requests. Open the file, or serve the folder.

```bash
python -m http.server 8000   # then open http://localhost:8000
```

## Publishing on GitHub Pages

Settings → Pages → Build and deployment → Deploy from a branch → pick the
branch and `/ (root)`. The site is live at
`https://<username>.github.io/<repo>/` within a minute or two.

## What still needs filling in

The page is deliberately honest about what it doesn't know yet. Anything
unfilled renders as a dashed slot, so it reads as intentionally blank rather
than broken — but it should not stay that way before you share the link.

| Section          | What to add                                        |
| ---------------- | -------------------------------------------------- |
| Masthead         | Location, and whether you're open to work           |
| Profile          | Rewrite in your own voice — it describes an approach, not your history |
| Capabilities     | Change the notes to what *you* have done with each tool |
| Selected work    | Three real projects, replacing the three dashed slots |
| Contact          | Email, LinkedIn, GitHub — then delete `class="unset"` |

Search `to add` and `Slot` in `index.html` to find every one of them.

## Design notes

- **Matcha and sakura** — a tonal green built around `#5F8340` on a rice-paper
  ground, with pink threaded through: the surname, the mark closing each
  heading, emphasised phrases, the whisk-stroke, and the end of the scroll bar.
  The two colours also carry meaning rather than only decorating: green marks
  the data and technical capabilities, pink marks the commercial ones, and the
  hero's two key phrases wash in the colour of the side they belong to.
  `Learning` is the only tag in neither — faint, dashed, honest about being in
  progress.
- **Light only.** The page commits to a single palette; there is no dark mode
  and no toggle. It stays light even when the visitor's OS prefers dark, so
  every colour is painted explicitly rather than inherited.
- **Falling petals** — a canvas layer above the content. Petals sway, rotate,
  and turn edge-on as they flutter; they shy away from the cursor within 140px,
  and a click shakes five more loose (capped, so it can't be spammed into a
  blizzard). None of them render under reduced motion.
- **Sticky index rail** — tracks the section you're reading and draws a short
  whisk-stroke under its number. Collapses to a numbered strip under 860px.
- **Reactive detail**: a canvas wash of slow matcha blooms that drift toward the
  cursor, a "steeping" scroll-progress bar that ends in pink, rows that steep
  green on hover, scroll-triggered reveals, and a theme toggle that persists.
- **Skill levels are labels, not invented percentages.** Everything reads
  `Core` except AI engineering, which reads `Learning`.
- **Contrast is measured, not eyeballed.** Every piece of text clears WCAG AA
  (4.5:1) against the ground — small labels use the deeper `-deep` / `-ink`
  variants for exactly this reason, while the lighter `--matcha` and `--sakura`
  are reserved for decoration that carries no words. The one exception is the
  121px surname at 4.17:1, which is large text and only needs 3:1.
- Both canvases resolve their colours from the CSS tokens, so the palette stays
  defined in one place.
- Fonts are system stacks, so nothing is fetched at load and nothing silently
  falls back to a different face.
- Every animation is disabled under `prefers-reduced-motion`.
