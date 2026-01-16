# Player Development Pitching Dashboard (R Shiny)

This Shiny application is designed for **Player Development staff and coaches** to quickly evaluate pitcher performance by comparing **recent performance** against **season baselines**.

The dashboard emphasizes **clarity, speed, and coach relevance**, focusing on four core metrics:

- **BB%** (Control)
- **K%** (Miss bats)
- **SLG% allowed** (Contact quality / damage)
- **FPS%** (First Pitch Strike %)

The goal is to minimize clicks and allow users to reach insight within seconds.

---

## How to Run the App

### Requirements

- R (>= 4.2)
- RStudio (optional but recommended)

### Install Required Packages

```r
install.packages(c(
  "shiny",
  "bslib",
  "dplyr",
  "lubridate",
  "ggplot2",
  "DT",
  "plotly",
  "grid",
  "png",
  "htmltools",
  "webshot2"
))
```

### Run the App
```r
shiny::runApp()
```


## Architecture Overview

- app.R
  - Entry point of the application
- global.R
    - Loads CSV files once at startup
    - Avoids repeated file I/O during interaction
- ui.R
    - Defines layout, navigation, and styling
    - Contains custom CSS for metric cards, goal podiums, and A4 report preview
    - Organizes content into four main tabs:
      - Player Overview
      - Trends View
      - Goals View
      - Report
- server.R
    - Data cleaning and flexible date parsing
    - Metric calculations (BB%, K%, SLG, FPS%, FPinZ%)
    - Rolling trend computation
    - Goal parsing and evaluation
    - PDF report generation using grid


## Performance Considerations

- Data is loaded once to avoid repeated disk reads
- Reactive expressions are scoped to pitcher-level filters only
- Metics are aggregated before visualizations
- Report generation uses base graphics (grid) instead of LaTeX for speed
- No database connections or external APIs are required

The app is optimized for fast load times and smooth interaction during coach meetings.

## Three example Coach Workflows

### Quick Performance Check

- Question: “How has this pitcher performed recently compared to his season?”
  - Select a pitcher ID
  - Review the metric cards in Player Overview
  - Compare Recent vs Season values
  - Scan the Most Recent Outing table


### Trend and Stability Review

- Question: “Is this change a real trend or short-term noise?”
  - Open the Trends View
  - Select a metric (e.g., K% or FPS%)
  - Adjust the rolling window size
  - Compare rolling trends to the season baseline
  
### Player Development Meeting

- Question: “What should we emphasize next?”
  - Open the Goals View
  - Review Primary, Secondary, and Tertiary goals
  - Check Season vs Recent progress indicators
  - Generate the 1-page A4 report
  - Share the PDF during a player or coach meeting