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
  ground, with baby pink `#C97B90` as the counterpoint. The two colours carry
  meaning rather than decorating: green marks the data and technical
  capabilities, pink marks the commercial ones, and the hero's two key phrases
  wash in the colour of the side they belong to. `Learning` is the only tag in
  neither colour — faint, dashed, honest about being in progress.
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
- **Light and dark** are both defined at token level, including the default
  "system" state where no theme attribute is set. Both canvases resolve their
  colours from the live tokens and repaint on a theme change, so neither the
  wash nor the petals ever fight the ground behind them.
- Fonts are system stacks, so nothing is fetched at load and nothing silently
  falls back to a different face.
- Every animation is disabled under `prefers-reduced-motion`.
