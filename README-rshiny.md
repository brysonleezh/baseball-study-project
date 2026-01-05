# Statcast-Based Batting Analysis Dashboard (R Shiny)

This project is an interactive baseball analytics dashboard built on MLB Statcast data.  
It analyzes **plate-appearance–level outcomes** to study how hitters make decisions, generate contact, and produce damage.

The application is designed to reflect a real baseball analytics workflow, with clear separation between data loading, filtering logic, and visualization.

---

## Project Structure

The project is organized around two main components:

1. **Statcast data ingestion and preparation**
2. **Interactive R Shiny application for hitter analysis**

---

## 1. Data Source and Granularity

- Data is sourced from **MLB Statcast**.
- The unit of analysis is the **final pitch of each plate appearance**.
- Each row represents one completed plate appearance and its outcome.

This design aligns with how hitting results are evaluated in practice:
- Swing decisions
- Contact quality
- Plate appearance outcomes

---

## 2. Data Loading and Handling

- Statcast data is loaded once at app startup via a dedicated data-loading function.
- All downstream analysis operates on a standardized in-memory table.
- The Shiny app logic is fully decoupled from the data source, allowing the backend to be easily swapped (API, SQL Server, or flat files).

Key benefits:
- Consistent analysis across all views
- Efficient reactivity
- Reproducible metrics

---

## 3. Filtering and Context Controls

All visualizations share a single filtered dataset, ensuring analytical consistency.

Supported filters include:
- Batter
- Season range
- Pitch family (bucketed pitch types)
- Pitcher handedness
- Count context (Ahead / Even / Behind)
- Zone (In-zone / Out-of-zone)
- Pitch velocity range
- Ball-in-play (BIP) only toggle

Pitch types are grouped into high-level **pitch families** (Four-Seam, Sinker/Cutter, Breaking Ball, Offspeed) to reduce noise and improve interpretability.

---

## 4. Analytical Framework

The dashboard decomposes hitting into three conceptual components:

### Decision
- Swing behavior approximated using BIP rate
- Outcome distributions by pitch family
- Terminal pitch location heatmaps

### Contact
- Exit velocity distributions by season
- Launch angle distributions and stability
- Sweet spot and hard-hit rates

### Damage
- Identification of high-damage contact windows
- Density-based peak detection for damaging contact locations
- Comparison of on-base vs out outcomes in pitch location space

For spatial analysis, kernel density estimation is used to identify peak damage zones, with median-based fallbacks for small sample sizes.

---

## 5. Player Summary Card

A player-level summary card provides context for each analysis view, including:
- Plate appearances and balls in play
- Exit velocity and launch angle metrics
- Hard-hit, sweet spot, and barrel rates
- Home run efficiency

All metrics dynamically update based on the active filters.

---

## 6. R Shiny Application

🔗 **Live App:**  
https://bowenlizh.shinyapps.io/dbacks-batting/

The application is built with:
- `shiny` and `bslib` for UI
- `dplyr`, `ggplot2`, and related tidyverse tools for analysis and visualization

---

## Notes

- All metrics are calculated at the **plate appearance (play) level**, using the final pitch only.
- The app is designed for exploratory analysis rather than static reporting.

---

## Author

Bowen Li  
Baseball Analytics / Data Science
