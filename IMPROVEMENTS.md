# Suggested Improvements: Expanding SafeSF with More SF Data

This doc outlines improvements if you expand the tool by integrating more San Francisco open data stores (DataSF / data.sfgov.org). The app currently uses **311 Cases**, **Our415 Events**, and **Temporary Street Closures**.

---

## 1. Additional DataSF Datasets to Integrate

### High impact (strong 311 correlation or resource planning)

| Dataset | DataSF ID / Link | Why integrate |
|--------|------------------|----------------|
| **Police Department Incident Reports** | `wg3w-h783` | Correlate 311 (e.g. encampment, noise, graffiti) with incident reports by district/date; joint hotspot maps; predict 311 spikes where incidents are high. |
| **Building Permits** | `i98e-djp9` | Construction → more 311 (noise, debris, street damage). Use permit start/end dates as features for predictions and “event” annotations. |
| **DPW Street / Sidewalk Evaluation** | `83ki-hu3p` | Street condition scores by area; correlate with 311 (potholes, sidewalk, graffiti). Prioritize 311 response in low-score areas. |
| **Street Tree List** | (search “street tree” on data.sfgov.org) | Tree maintenance 311 vs tree locations; predict tree-related 311 by neighborhood. |

### Medium impact (context and UX)

| Dataset | Notes |
|--------|--------|
| **Weather** | Not DataSF—use NOAA or Open-Meteo API. Rain → sewer/street-flood 311; use as feature in models and “rain days” in the dashboard. |
| **Muni / Transit** | SFMTA or GTFS. High-ridership corridors vs 311 (e.g. street cleaning, litter); show transit overlay on map. |
| **Homelessness / Encampment** | Any city or HSH dataset on data.sfgov.org. Directly aligns with encampment 311; joint trends and district views. |
| **Parking citations / SFPark** | Parking 311 vs citation volume by area; optimize parking enforcement and 311 routing. |

### Lower effort, high value

- **311 Cases – other views**: If the portal has filtered views (e.g. by department, status), add one more “source” in `fetch_data` for a different slice (e.g. open-only, by origin).
- **More event types**: Any DataSF “events” or “closures” datasets (parades, filming, farmers’ markets) as additional “event” layers like Our415/Street Closures.

---

## 2. Architecture / Pipeline Improvements

- **Unified “SF data” fetcher**: One script (e.g. `scripts/fetch_sf_data.py`) that:
  - Takes a small config (dataset_id, date column, optional filters) and pulls all DataSF sources into a standard schema (date, geo, type, id).
  - Outputs to `data/` or GCS (e.g. `311_raw`, `crime`, `permits`, `events_our415`, `events_closures`) so `build_aggregates` and the API can stay generic.
- **Event table schema**: Standardize “events” as `(date, event_type, name, district_or_geo, source_dataset)`. Then:
  - `fetch_events.py` (or the unified fetcher) fills this for Our415, Street Closures, and optionally permits (e.g. “construction start”).
  - `build_aggregates` builds `events_agg` / `events_by_date` from this single table so adding a new event source = one more fetch + same pipeline.
- **Correlation API**: Generalize `/api/analytics/event-correlation` to “any secondary dataset vs 311” (e.g. crime vs 311, permits vs 311) with a `dataset` or `layer` query param and shared date/geo aggregation.

---

## 3. Modeling / Predictions

- **External features**: Add weather (rain, temp), “event count” by type, and optionally crime/permit counts by district/date as model features.
- **Category-specific models**: You already have category-level insights; train separate small models (or one model with strong category interaction terms) for top categories (e.g. encampment, graffiti, street cleaning) so predictions are category-aware.
- **Spatial features**: Use grid/district from `grid_agg` and district from 311; optionally add “distance to event” or “events in same grid” for event-driven spikes.

---

## 4. Frontend / Dashboard

- **Layers / toggles**: Add map and trend toggles for “311 only” vs “311 + events” vs “311 + crime” (or permits) so users can see overlapping layers.
- **Second dataset dropdown**: In Insights or a new “Compare” tab, let users pick a second metric (e.g. crime incidents, permits, event count) and show a time-series or district bar chart vs 311.
- **Weather**: Show “Rain days” or “Heavy rain” as an annotation on the main trend chart and in event correlation.
- **Export**: CSV/Excel export for the current view (filters + date range) for reports and external analysis.

---

## 5. Operational / Data Quality

- **Incremental sync**: For 311 and other large datasets, support “since last run” or “last N days” in fetch scripts to avoid full re-pulls and stay within API limits.
- **Data freshness**: Dashboard badge or footer: “311 data through YYYY-MM-DD; events through YYYY-MM-DD” from aggregate metadata.
- **Validation**: Light checks after fetch (e.g. row count, date range, nulls in key columns); log or block build_aggregates if validation fails.

---

## 6. Quick Wins (minimal new data)

1. **Weather**: Call Open-Meteo (or similar) for SF for the date range of your aggregates; store `date, precip, temp` and use in correlation API and as chart annotation.
2. **Crime**: Add `scripts/fetch_crime.py` for `wg3w-h783` (same Socrata pattern as `fetch_events`), aggregate by date and district, and add a “Crime vs 311” correlation view.
3. **Building permits**: Fetch `i98e-djp9` by permit issue/start date; add as “construction events” in `events_by_date` and re-run `build_aggregates` so existing event-correlation UI shows them.

---

## 7. Dataset IDs Summary (Socrata / DataSF)

- **311 Cases**: `vw6y-z8j6` (already used)
- **Our415 Events**: `8i3s-ih2a` (already used)
- **Temporary Street Closures**: `8x25-yybr` (already used)
- **Police Incident Reports**: `wg3w-h783`
- **Building Permits**: `i98e-djp9`
- **DPW Street/Sidewalk Evaluation**: `83ki-hu3p`

Use the same pattern as `fetch_events.py`: `sodapy` or Socrata API, optional `SF_OPEN_DATA_APP_TOKEN`, filter by date columns, write parquet or CSV, then plug into your aggregate pipeline.

---

If you want to implement one path first, the highest-leverage order is: **weather** (fast, no new DataSF signup), then **crime** (same API as events), then **building permits** as events, then **street/sidewalk** for quality and prioritization.
