# P8 Phase E — Art Asset Catalog, Scaffolding & Sourcing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Sized for a **Sonnet implementer**.

**Goal:** Every place the dashboard currently fakes art with vector stand-ins or blanks gets
(a) a catalog entry the owner can act on, (b) a wired slot with an honest token-gradient
fallback (the scar), and (c) where the license is clean and free, the actual asset. Two concrete
deliverables land now: a **favicon** (interim generated mark) and the **commissioned-logo
pipeline** (artist brief + reserved `st.logo` slot) for the guru/genie wordmark.

**Architecture:** One catalog doc (`docs/art_assets.md`) is the single source of truth for every
art slot. The DESIGN §3 ambient-manifest infrastructure gets built for real
(`data/assets/ambient/ambient_manifest.json` + a loader in `dashboard/assets.py` — both paths
pre-reserved in handoff §5): slots without art render token gradients, so shipping the
scaffolding changes nothing visually until files arrive. Free/licensed sourcing is a research
pass whose output is manifest entries + downloaded clean-license files, never hot-linked URLs.

**Prereqs:** Phase C merged. May run parallel to Phase D with one rule: **`games.py` and
`constellation*.py` belong to D** — this phase catalogs their slots but does not edit them.

**Hard rules (DESIGN.md §3/§6, binding):** no AI-generated imagery, ever; ambient opacity
≤ 20%; never behind dense tables/stat grids; WCAG AA for text above art; every file's
license/attribution recorded in the manifest; stock or commissioned only.

**Branch:** `feature/dashboard-ux`. Gates + refactoring-specialist per task (Phase 0 plan,
Context section).

---

### Task E1: The catalog — `docs/art_assets.md`

**Files:**
- Create: `docs/art_assets.md`

Sweep every surface/component and record each art slot as one table row:
`slot id · surface/component · current placeholder · desired asset · source
(commissioned / licensed / free-stock / generated-SVG-keep) · license requirement · format+size
spec · priority (P1 ship-blocker for the brand, P2 upgrade, P3 nice)`.

Seed rows (verify each in code, then extend — the sweep must also grep for
`:material/person:`-style icon stand-ins, `_TEAM_PALETTE`-era comments, `scar` captions, and
`radial-gradient` hero washes):

| slot | surface | today | want | source |
|---|---|---|---|---|
| `logo_wordmark` | sidebar via `st.logo` (E4) | none (text title only) | guru/genie amid mystical sports paraphernalia + wordmark | **commissioned** (E4 brief) |
| `favicon` | browser tab (`st.set_page_config`, app.py:22 has no `page_icon`) | Streamlit default | logo-derived mark; interim generated ◈ comet (E3) | generated now → commissioned later |
| `ambient_tonight` | Tonight card washes | CSS radial gradients (mockup-derived) | faint night-sky field | free-stock (NASA/Unsplash class) |
| `ambient_receipts_hero` | Receipts verdict card (C7) | CSS nebula gradient | licensed nebula/night-sky texture | free-stock |
| `ambient_gutters` | app-level starfield | generated SVG (A1) | keep — generated is the design | generated-SVG-keep |
| `glyphs_game_shape` | Tonight/Games (C3) | generated inline SVG set | keep; optional licensed upgrade later | generated-SVG-keep (P3) |
| `astrolabe_engraving` | Games astrolabe (C5) | generated SVG bezel/ticks | optional licensed engraving texture | P3 |
| `constellation_silhouettes` | Games map (Phase D) | generated SVG paths | keep; optional artist pass | catalog-only here (D owns files) |
| `team_marks` | anywhere a team renders | colors only (`team_assets.json`) | official marks are league IP | **owner decision** — licensed or skip; colors suffice |
| `player_headshots` | offer cards/deep-dive | initials disc scar (P4.5) | league CDN headshots | **owner decision** — CDN ToS unverified (handoff §3) |

The catalog ends with a **"what the owner does next"** section: the commissioned-logo brief
pointer, the two owner-decision rows, and the free-stock shortlist from E5 awaiting approval.

- [ ] Sweep, write, commit `docs(p8-e): art asset catalog — every slot, source, license`

---

### Task E2: Ambient manifest infrastructure (the scar mechanism)

**Files:**
- Create: `src/sportstradamus/data/assets/ambient/ambient_manifest.json`
- Create: `src/sportstradamus/dashboard/assets.py` (path pre-reserved in handoff §5)
- Modify: `src/sportstradamus/dashboard/surfaces/tonight.py`, `receipts.py` (slot lookups
  replacing the hard-coded hero gradients; **not** `games.py` — Phase D territory)
- Test: `tests/golden/test_ambient_assets.py`

Manifest shape (DESIGN §3 verbatim contract — slot → file, opacity, placement, license):

```json
{
  "version": 1,
  "slots": {
    "ambient_tonight": {"file": null, "opacity": 0.14, "placement": "card-background",
                         "license": null, "attribution": null, "source_url": null},
    "ambient_receipts_hero": {"file": null, "opacity": 0.16, "placement": "hero-background",
                               "license": null, "attribution": null, "source_url": null}
  }
}
```

`assets.py` loader:

```python
@cache
def _manifest() -> dict: ...

def ambient_css(slot: str, fallback_gradient: str) -> str:
    """CSS background for a slot: the manifest file (at its recorded opacity,
    via a linear-gradient overlay that keeps text WCAG-AA) when present and
    licensed, else fallback_gradient unchanged. A file with a null license is
    treated as absent — unlicensed art never renders."""
```

Golden pins: null-file slot → fallback byte-identical to today's gradient; opacity ceiling
`<= 0.20` asserted for every slot; **a file entry with `license: null` renders the fallback**
(the license gate is code, not policy); manifest schema validated.

- [ ] Tests first → build → wire the two surfaces → gates → commit
  `feat(p8-e): ambient manifest + license-gated slot loader (gradient scars)`

---

### Task E3: Favicon — interim generated mark, wired now

**Files:**
- Create: `src/sportstradamus/dashboard/assets/favicon.svg`
- Modify: `src/sportstradamus/dashboard/app.py` (line 22)
- Test: extend `tests/golden/test_app_injections.py`

Hand-author a small SVG: the ◈ oracle diamond with a comet tail, gold `#C9A227` on the dark
`#0E1117` rounded square (radius 4px-equivalent — DESIGN small). Generated vector = exempt from
the licensing rule (starfield precedent, spec §2). Keep it legible at 16×16: one glyph, no text.

```python
st.set_page_config(
    page_title="Sportstradamus Dashboard",
    page_icon=str(Path(__file__).parent / "assets/favicon.svg"),
    layout="wide",
)
```

Verify in a live run the tab shows the mark; if Streamlit won't serve the SVG as favicon,
export it 64×64 PNG (one-time, any editor — record the export in the commit body) and point
`page_icon` at the PNG instead. Catalog row `favicon` stays open: the commissioned logo's square
mark replaces this file later — same path, zero code change.

- [ ] Pin (`page_icon` present + file exists) → author → wire → live check → gates → commit
  `feat(p8-e): interim favicon — gold oracle mark`

---

### Task E4: Commissioned logo — brief + reserved slot

**Files:**
- Create: `docs/art_briefs/logo_guru.md`
- Modify: `src/sportstradamus/dashboard/app.py` (`st.logo` with wordmark fallback)
- Test: extend the app-injections golden

The brief is the deliverable the owner hands an artist — complete enough to commission from
directly:

```markdown
# Sportstradamus logo — commission brief

**Subject.** A guru/genie figure — turbaned mystic, warm and knowing, not a caricature —
surrounded by mystical sports paraphernalia: a crystal basketball, orbiting baseball/football/
hockey-puck "moons", zodiac-style sport glyphs, a curl of lamp smoke resolving into a stat line.
Mystic-meets-sports; credible, not cartoonish (DESIGN.md §1 brand voice).

**Palette.** Dark-first: background #0E1117; gold #C9A227 the hero accent; electric blue
#2E6BE6 secondary; text/linework #E6E9EF. No purple gradients, no red as accent (DESIGN §6).

**Deliverables.** (1) horizontal lockup: mark + "SPORTSTRADAMUS" wordmark (wordmark in IBM
Plex Sans or hand-lettering harmonizing with Cinzel); (2) square mark alone (favicon source,
reads at 16px); (3) monochrome gold-on-dark variant of each. Vector SVG masters + PNG @1x/@2x
(wordmark ~600×160, mark 512×512).

**Usage + rights.** Web app sidebar (~24px tall render), browser favicon, future social. Full
exclusive commercial rights assigned; artist credited in the repo manifest. Human-made only —
no AI generation (house rule, non-negotiable).
```

App slot, live now with an honest fallback (`st.logo` renders ~24px tall; `icon_image` serves
the collapsed sidebar):

```python
_LOGO = Path(__file__).parent / "assets/logo_wordmark.png"
_MARK = Path(__file__).parent / "assets/logo_mark.png"
if _LOGO.exists():
    st.logo(str(_LOGO), icon_image=str(_MARK) if _MARK.exists() else None)
```

(no file committed yet → nothing renders → today's text title stands; when the commission
lands, the owner drops two files in and the logo appears with zero code change. Record both
paths in the E1 catalog + the manifest's attribution convention.)

- [ ] Brief → slot wiring + golden (logo block present, guarded on existence) → gates → commit
  `feat(p8-e): commissioned-logo brief + zero-code-change logo slot`

---

### Task E5: Free/licensed sourcing pass

**Files:**
- Modify: `docs/art_assets.md` (shortlist per slot)
- Possibly add: files under `src/sportstradamus/data/assets/ambient/` + manifest entries

Research pass (WebSearch/WebFetch) for the free-stock rows — target sources whose licenses are
machine-checkable and commercial-clean:

- **NASA / ESA imagery** (nebulas, star fields) — public domain / CC-attribution; ideal for
  `ambient_receipts_hero`.
- **Unsplash / Pexels** night-sky + sports-equipment-silhouette photography — their standard
  licenses permit commercial use without attribution (record it anyway).
- Skip anything with editorial-only, NC, or unclear terms — the manifest's license gate (E2)
  will refuse to render it regardless.

For each candidate: download, verify pixel dimensions ≥ 2× the render slot, convert/crop to a
web-weight JPEG/WebP (< 300 KB), place under `data/assets/ambient/`, fill the manifest entry
(`file`, `license`, `attribution`, `source_url`), and check the slot live (opacity ceiling +
text contrast above it). **Present the shortlist + rendered screenshots to the owner before
committing binaries** — art is an owner-taste call; the code path is already safe either way.

- [ ] Shortlist per slot in the catalog → owner picks → download + manifest + live check →
  gates → commit `feat(p8-e): first licensed ambient assets (owner-approved)`

---

## Exit criteria

- `docs/art_assets.md` covers every placeholder slot in the app (sweep re-run clean at the
  end: no un-cataloged `scar`/gradient/icon stand-in remains).
- Manifest + loader live with license-gated rendering; both wired surfaces byte-identical to
  today when slots are empty.
- Favicon shows in the tab; logo slot renders nothing today and needs zero code when art lands;
  the artist brief is commission-ready.
- Any committed asset has a filled manifest entry (license + attribution + source URL) and owner
  approval on record (ledger line).
- Three gates green; refactoring-specialist on every touched `.py`.
