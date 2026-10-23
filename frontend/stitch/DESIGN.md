---
name: Canine Vision AI
colors:
  surface: '#f9f9ff'
  surface-dim: '#d3daea'
  surface-bright: '#f9f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f0f3ff'
  surface-container: '#e7eefe'
  surface-container-high: '#e2e8f8'
  surface-container-highest: '#dce2f3'
  on-surface: '#151c27'
  on-surface-variant: '#514538'
  inverse-surface: '#2a313c'
  inverse-on-surface: '#ebf1ff'
  outline: '#867461'
  outline-variant: '#d8c3ad'
  surface-tint: '#855300'
  primary: '#653e00'
  on-primary: '#ffffff'
  primary-container: '#f59e0b'
  on-primary-container: '#613b00'
  inverse-primary: '#fdb965'
  secondary: '#085ac0'
  on-secondary: '#ffffff'
  secondary-container: '#2170e4'
  on-secondary-container: '#fefcff'
  tertiary: '#004c6a'
  on-tertiary: '#ffffff'
  tertiary-container: '#1abdff'
  on-tertiary-container: '#004966'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffddb8'
  primary-fixed-dim: '#fdb965'
  on-primary-fixed: '#2a1700'
  on-primary-fixed-variant: '#653e00'
  secondary-fixed: '#d8e2ff'
  secondary-fixed-dim: '#adc6ff'
  on-secondary-fixed: '#001a42'
  on-secondary-fixed-variant: '#004395'
  tertiary-fixed: '#c5e7ff'
  tertiary-fixed-dim: '#88cffa'
  on-tertiary-fixed: '#001e2d'
  on-tertiary-fixed-variant: '#004c6a'
  background: '#f9f9ff'
  on-background: '#151c27'
  surface-variant: '#dce3f2'
typography:
  display:
    fontFamily: Plus Jakarta Sans
    fontSize: 44px
    fontWeight: '800'
    lineHeight: 52px
    letterSpacing: -0.02em
  display-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 34px
    fontWeight: '800'
    lineHeight: 42px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 26px
    fontWeight: '700'
    lineHeight: 34px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
  title-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-desktop: 1.5rem
  margin: 1.5rem
  margin-desktop: 2.5rem
  space-xs: 0.25rem
  space-sm: 0.75rem
  space-md: 1.5rem
  space-lg: 2.5rem
  space-xl: 4rem
---

## Brand & Style

This design system establishes a welcoming yet technologically rigorous environment for automated canine image retrieval and breed classification. It bridges the emotional connection of companion animals with the high-reliability precision expected of modern computer vision platforms.

The aesthetic blends **Modern Tech** with organic warmth:
- **Clean Precision:** Open whitespace, crystalline surface hierarchies, and crisp data readouts communicate algorithmic accuracy.
- **Canine Warmth:** Softened geometry and amber-tinted focus cues temper the coldness of raw computer vision models, making complex inference understandable and approachable.
- **Visual Clarity:** Interfaces prioritize high-fidelity canine imagery, scanning previews, and multi-label confidence indicators without visual clutter.

## Colors

The color architecture is built around functional role separation:

- **Primary (`#855300` / `#f59e0b`):** Evokes the rich coats of golden retrievers. Reserved strictly for forward momentum, primary camera/upload activations, primary CTAs, and active engagement states.
- **Secondary (`#0058be` / `#2170e4`):** Represents algorithmic precision, neural network confidence metrics, matching percentages, and analytical diagnostics.
- **Tertiary (`#00658b` / `#1abdff`):** Used for auxiliary breed traits, phenotypic attributes, taxonomy tags, and secondary data chips.
- **Neutral Surface Ecosystem (`#f9f9ff` to `#151c27`):** Cool, slightly tinted slate-blue surfaces maintain an airy, sterile optical canvas where canine photography remains uncorrupted by background color cast. High-contrast ink text (`#151c27`) provides effortless readability.

## Typography

Typography relies entirely on **Plus Jakarta Sans**, chosen for its crisp geometric body, open counters, and subtle humanist curves.

- **Display & Headlines:** Used for breed title reveals, confidence hero summaries, and scan outcomes. The heavy weights (`700` and `800`) ground user focus upon visual classification results.
- **Body:** Scaled with generous line heights (`24px` on `16px` base; `28px` on `18px` large) to ensure detailed breed histories, temperament summaries, and care profiles read like high-end editorial publications.
- **Labels:** Compact, medium-to-semibold weights with slight positive tracking guarantee clarity on telemetry bars, model thresholds, and attribute pills.

## Layout & Spacing

The layout employs a responsive structured grid with strict whitespace discipline:

- **Desktop (>= 1024px):** A balanced dual-pane layout. The left viewport anchors the ingestion pipeline (image dropzone, bounding box overlay, cropped feature focus), while the right viewport presents progressive scan outputs, ranking lists, and morphological metrics. Max content container is constrained to 1280px.
- **Mobile (< 1024px):** Single-column stacked layout transitioning sequentially from capture/upload zone to top-ranked breed candidate cards.
- **Rhythm:** Spacing derives from a clean base unit (`8px`), preventing visual crowding around high-density AI diagnostic summaries.

## Elevation & Depth

Visual hierarchy is constructed through **Tonal Surface Layering** reinforced by chromatic ambient illumination:

- **Surface Tiers:** 
  - Canvas base rests at `surface` (`#f9f9ff`).
  - Dropzones and analytical panels nest at `surface-container-low` (`#f0f3ff`).
  - Active detection cards and elevated inspection sheets sit at `surface-container-lowest` (`#ffffff`).
- **Ambient Shadows:** Standard shadows are eschewed in favor of diffuse, low-opacity blurs tinted with brand spectrums:
  - Default card lift: `0 8px 30px -4px rgba(21, 28, 39, 0.06)`.
  - Focused inspection card: `0 20px 40px -8px rgba(0, 88, 190, 0.08)`.
  - Active upload state: `0 12px 32px -4px rgba(245, 158, 11, 0.12)`.
- **Ghost Boundaries:** High-density components utilize hairline borders (`1px solid #d8c3ad` or `1px solid #dce2f3`) rather than harsh dividers.

## Shapes

The shape system employs roundedness level **2**, striking a balance between machine precision and animal-friendly warmth:

- **Interactive Inputs & Buttons:** Sized with `rounded-lg` (`1rem`) to provide inviting, tactile touch targets.
- **Image Canvases & Results Containers:** Framed in `rounded-xl` (`1.5rem`) to create soft windowing for biological photography.
- **Metadata Badges & Analytical Gauges:** Bound by `rounded-full` (`9999px`) to visually delineate categorical data from content cards.

## Components

### Buttons
- **Primary Action (Scan / Upload):** Background `#855300`, text `#ffffff`, typography `label-md`, radius `1rem`, vertical padding `0.75rem`, horizontal padding `1.5rem`. On hover, transitions to `#f59e0b` with text `#613b00`.
- **Secondary (Telemetry / Alternative Candidates):** Background `#0058be`, text `#ffffff`, typography `label-md`, radius `1rem`. On hover, transitions to `#2170e4`.
- **Ghost / Outlined:** Border `1px solid #867461`, background transparent, text `#151c27`, radius `1rem`.

### Image Upload Zone
- Container background `#f0f3ff`, border `2px dashed #d8c3ad`, radius `1.5rem`, padding `2.5rem`.
- Active drag state shifts border to solid `#f59e0b` with a soft glow `0 0 0 4px rgba(245, 158, 11, 0.15)` and background to `#ffffff`.

### Confidence Indicators
- **Track:** Background `#e2e8f8`, height `8px`, radius `9999px`.
- **Fill Bar:** Dynamic width bound to confidence percentage, background `#2170e4`, radius `9999px`. For predictions exceeding 90% certainty, an amber highlight shimmer (`#f59e0b`) accentuates the terminal edge.

### Trait & Phenotype Tags
- Height `28px`, background `#1abdff` at 15% opacity or solid `#c5e7ff`, text `#004966`, font `label-sm`, horizontal padding `0.75rem`, radius `9999px`.

### Breed Result Card
- Background `#ffffff`, border `1px solid #dce2f3`, radius `1.5rem`, padding `1.5rem`, ambient shadow `0 8px 30px -4px rgba(21, 28, 39, 0.06)`. Houses rank indicator, primary breed name in `headline-md`, confidence gauge, and morphological trait badges.

### Bounding Box & Keypoint Overlays
- Canine detection frame rendered in `2px solid #2170e4` with corner accents in `#f59e0b`. Floating label displays inferred breed taxonomy in `label-sm` with background `#151c27` and text `#ffffff`.