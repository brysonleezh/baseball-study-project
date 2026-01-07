# Corbin Carroll vs. Ketel Marte — Statcast Home Run Analysis

This project analyzes **pitch-level Statcast data** to compare the **home run profiles** of Corbin Carroll and Ketel Marte, focusing on contact quality, pitch velocity, and pitch location.

## Project Goals
- Compare **home-run contact characteristics** (launch speed, launch angle)
- Evaluate **pitch velocities** that result in home runs
- Analyze **where in the strike zone** each hitter generates damage
- Identify differences in **power consistency vs. flexibility**

## Data
- Source: MLB Statcast / StatAPI exports
- Granularity: **Pitch-level**
- Seasons: **2023–2025**
- Each row represents one pitch thrown to the batter

## Methods
- Event filtering to isolate **home runs**
- Density estimation of **launch speed × launch angle**
- Boxplots of **pitch velocity on HRs**
- Strike-zone visualizations with **3×3 grid counts**
- Summary statistics for exit velocity and launch angle

## Key Findings
- **Corbin Carroll** shows a **tight, repeatable home-run window**, with power concentrated in the middle–lower strike zone and lower variance in contact quality.
- **Ketel Marte** generates home runs across a **broader range of pitch velocities, launch conditions, and locations**, indicating greater flexibility but higher variance.
- Both hitters share a similar optimal HR launch window (~105 mph, ~27–29°).

