# Statcast-Based Batting Analysis Dashboard (R Shiny)

An interactive **R Shiny application** for exploratory hitter analysis using **MLB Statcast** data.  
The dashboard is designed to mirror a professional baseball analytics workflow, breaking hitting performance into **Decision**, **Contact**, and **Damage** views with shared contextual filters.

🔗 **Live App**: https://bowenlizh.shinyapps.io/dbacks-batting/

![Decision View][app_overview.png]

---

## Overview

This app enables rapid, context-aware evaluation of hitters by analyzing **plate-appearance–level outcomes** using the **final pitch of each PA**.  
All views operate on a single filtered dataset to ensure analytical consistency across metrics and visualizations.

---

## Data Source & Granularity

- Source: **MLB Statcast**
- Unit of analysis: **Completed plate appearance**
- Data used: **Terminal pitch only**
- Rationale: Aligns with how swing decisions, contact quality, and outcomes are evaluated in practice

---

## Application Layout

### Sidebar (Global Filters)

- Batter
- Season range
- Pitch family (Four-Seam, Sinker/Cutter, Breaking, Offspeed)
- Pitcher handedness
- Count context (Ahead / Even / Behind)
- Zone (In / Out)
- Pitch velocity range
- Ball-in-play (BIP) only toggle

All filters apply globally across tabs.

---

### Main Tabs

#### 1. Decision
- Swing behavior approximated via BIP rate
- Outcome distributions by pitch family
- Terminal pitch location heatmaps

#### 2. Contact
- Exit velocity distributions by season
- Launch angle distributions and spread
- Sweet spot and hard-hit rates

#### 3. Damage
- High-damage contact windows
- Density-based peak detection for damage locations
- On-base vs out comparisons in pitch-location space

*KDE is used for spatial peak detection, with median-based fallbacks for small samples.*

---

## Player Summary Card

A dynamic player card provides context for all views:
- Plate appearances and balls in play
- Exit velocity and launch angle metrics
- Hard-hit, sweet spot, and barrel rates
- Home run efficiency

All metrics update based on active filters.

---

## Technical Design

- Data loaded once at app startup via a dedicated loader
- Standardized in-memory table for all analysis
- Clear separation of data, filtering logic, and visualization layers
- Backend-agnostic (flat files, API, or SQL)

---

## Intended Use

Designed for **exploratory analysis and hypothesis testing**, not static reporting.  
Supports rapid iteration on hitter tendencies, contact quality, and damage profiles.

---

## Tech Stack

- R Shiny, bslib
- dplyr, ggplot2, tidyverse

---

## Author

**Bowen Li**  
Baseball Analytics · Data Science
