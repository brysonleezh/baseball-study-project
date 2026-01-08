# Catcher Framing Analysis (Statcast)


::contentReference[oaicite:0]{index=0}


## Overview

This project analyzes **MLB catcher framing value** using pitch-level Statcast data.  
The objective is to **quantify a catcher’s ability to convert pitches into called strikes**, while properly controlling for pitch location, count, pitcher, batter, and umpire context.

The workflow mirrors professional baseball analytics practice, emphasizing **context adjustment, hierarchical shrinkage, and run-value interpretation**.

---

## Analytical Framework

Catcher framing is modeled as a probability problem:

> *Given a taken pitch, what is the probability it is called a strike?*

Framing value is defined as the **difference between observed and expected strike probability**, attributed to the catcher.

### Key Design Choices
- **Unit of analysis:** Taken pitches only (no swings)
- **Target variable:** Called strike (1) vs ball (0)
- **Primary metric:** Extra strikes attributable to framing
- **Evaluation scale:** Runs saved via run expectancy

---

## Data

- **Source:** MLB Statcast
- **Granularity:** Pitch-level
- **Key fields:**
  - Pitch location (`plate_x`, `plate_z`)
  - Count
  - Pitch type
  - Pitcher & batter handedness
  - Catcher ID
  - Umpire ID
  - Season context

### Preprocessing
- Normalized strike zone by batter height
- Filtered to competitive pitch locations
- Removed pitchouts and intentional balls
- Restricted to taken pitches only

---

## Modeling Approach

### 1. Baseline Strike Probability Model

A logistic / generalized additive framework estimates **league-average strike probability** using:
- Smoothed pitch location
- Count
- Pitch type
- Batter handedness
- Pitcher handedness

This model defines the expected strike zone absent catcher effects.

---

### 2. Catcher Framing Effects (Hierarchical)

Catchers are incorporated as **random effects** in a **Bayesian hierarchical model**, enabling:
- Partial pooling across catchers
- Stable estimates for low-sample players
- Separation of skill from noise

**Optional random effects:**
- Umpire
- Pitcher
- Season

---

### 3. Run Value Conversion

Extra strikes are converted to **runs saved** using:
- Count-specific run expectancy values
- Aggregation across seasons
- Normalization to runs per 1,000 taken pitches

---

## Outputs & Metrics

For each catcher:
- Extra strikes (raw and shrinkage-adjusted)
- Runs saved from framing
- Framing runs per 1,000 taken pitches
- Season-level summaries

---

## Validation & Diagnostics

- Posterior shrinkage inspection
- Year-to-year stability analysis
- League-wide distribution checks
- Sensitivity to zone boundary definitions

---

## Project Structure

