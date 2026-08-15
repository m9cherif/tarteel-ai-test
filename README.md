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

- **Ledger layout** — hairline rules and a monospace left rail, echoing the
  query results and spreadsheets the work actually happens in.
- **Two accents that mean something** — teal marks data and technical
  capabilities, oxblood marks commercial ones.
- **Skill levels are labels, not invented percentages.** Everything reads
  `Core` except AI engineering, which reads `Learning`.
- **Light and dark** are both defined at token level, including the default
  "system" state where no theme attribute is set.
- Fonts are system stacks, so nothing is fetched at load and nothing silently
  falls back to a different face.
