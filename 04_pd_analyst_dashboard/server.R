# server.R
library(shiny)
library(dplyr)
library(lubridate)
library(ggplot2)
library(DT)
library(bslib)
library(plotly)
library(png)
library(grid)
library(htmltools)
library(webshot2)
library(rsvg)
# Fast load times; clear labels/defaults; minimal clicks to insight

# PDF report design
# 

app_server <- function(input, output, session) {
  
  # ----------------------------
  # 1) Basic cleaning
  # ----------------------------
  events <- data$events %>%
    mutate(sched_date = as.Date(sched_date, format = "%m/%d/%Y")) %>%
    filter(!is.na(sched_date)) %>%
    arrange(pitcher_id, sched_date)
  
  pitches <- data$pitches %>%
    mutate(sched_date = as.Date(sched_date, format = "%m/%d/%Y")) %>%
    filter(!is.na(sched_date)) %>%
    arrange(pitcher_id, sched_date)
  
  
  goals <- data$goals
  
  # ----------------------------
  # 2) Populate pitcher_id choices
  # ----------------------------
  observe({
    ids <- sort(unique(events$pitcher_id))
    updateSelectInput(session, "pitcher_id", choices = ids, selected = "93229")
  })
  
  # ----------------------------
  # 3) Metric helpers (your schema)
  # ----------------------------
  safe_div <- function(num, den) ifelse(den > 0, num / den, NA_real_)
  
  calc_event_metrics <- function(ev) {
    # ev is filtered subset of events rows (PA-level rows)
    PA <- sum(ev$pa, na.rm = TRUE)
    AB <- sum(ev$ab, na.rm = TRUE)
    
    BB <- sum(ev$bb, na.rm = TRUE)
    SO <- sum(ev$so, na.rm = TRUE)
    
    TB <- 1 * sum(ev$X1b, na.rm = TRUE) +
      2 * sum(ev$X2b, na.rm = TRUE) +
      3 * sum(ev$X3b, na.rm = TRUE) +
      4 * sum(ev$hr,  na.rm = TRUE)
    
    list(
      pa = PA,
      ab = AB,
      bb_rate = safe_div(BB, PA),
      k_rate  = safe_div(SO, PA),
      slg     = safe_div(TB, AB)
    )
  }
  
  is_first_pitch_strike <- function(x) {
    # [1] "ball"                    "foul"                    "hit_into_play"           "hit_into_play_no_out"   
    # [5] "foul_tip"                "called_strike"           "swinging_strike"         "hit_into_play_score"    
    # [9] "swinging_strike_blocked" "blocked_ball"            "hit_by_pitch"            "automatic_ball"         
    # [13] "automatic_strike"        "foul_bunt"               "missed_bunt"             "bunt_foul_tip"  
    strike_set <- c(
      "called_strike", "swinging_strike", "swinging_strike_blocked",
      "foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "missed_bunt",
      "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
      "automatic_strike"
    )
    tolower(as.character(x)) %in% strike_set
  }
  
  calc_fps <- function(pit) {
    # First pitch defined by count before pitch = 0-0
    fp <- pit %>% filter(balls_before == 0, strikes_before == 0)
    n_pa <- nrow(fp)
    n_strikes <- sum(vapply(fp$pitch_result, is_first_pitch_strike, logical(1)), na.rm = TRUE)
    list(fps = safe_div(n_strikes, n_pa), n_pa = n_pa)
  }
  
  calc_fpinz <- function(pit) {
    # First pitch defined by count before pitch = 0-0
    fp <- pit %>% dplyr::filter(balls_before == 0, strikes_before == 0)
    n_pa <- nrow(fp)
    
    # Fixed strike-zone approximation (since sz_top/sz_bot not available)
    in_zone_vec <- with(fp,
                        plate_x >= -0.83 & plate_x <= 0.83 &
                          plate_z >= 1.5  & plate_z <= 3.5
    )
    
    n_in_zone <- sum(in_zone_vec, na.rm = TRUE)
    
    list(
      fpinz = safe_div(n_in_zone, n_pa),
      n_pa  = n_pa
    )
  }
  
  fmt_pct <- function(x) ifelse(is.na(x), "—", sprintf("%.1f%%", 100 * x))
  fmt_num <- function(x) ifelse(is.na(x), "—", sprintf("%.3f", x))
  
  # ----------------------------
  # 4) Filtered reactives (by pitcher)
  # ----------------------------
  pitcher_events <- reactive({
    req(input$pitcher_id)
    events %>% filter(pitcher_id == input$pitcher_id)
  }) %>% bindCache(input$pitcher_id)
  
  pitcher_pitches <- reactive({
    req(input$pitcher_id)
    pitches %>% filter(pitcher_id == input$pitcher_id)
  }) %>% bindCache(input$pitcher_id)
  
  date_bounds <- reactive({
    ev <- pitcher_events()
    if (nrow(ev) == 0) return(list(end = Sys.Date(), start = Sys.Date() - 13))
    end <- max(ev$sched_date, na.rm = TRUE)
    list(end = end, start = end - (as.integer(input$window) - 1))
  })
  
  recent_events <- reactive({
    b <- date_bounds()
    pitcher_events() %>% filter(sched_date >= b$start, sched_date <= b$end)
  })
  
  season_events <- reactive({
    pitcher_events()
  })
  
  recent_pitches <- reactive({
    b <- date_bounds()
    pitcher_pitches() %>% filter(sched_date >= b$start, sched_date <= b$end)
  })
  
  season_pitches <- reactive({
    pitcher_pitches()
  })
  
  # ----------------------------
  # 5) Recent vs Season metrics
  # ----------------------------
  metrics_recent <- reactive({
    evm <- calc_event_metrics(recent_events())
    fps <- calc_fps(recent_pitches())
    fpinz <- calc_fpinz(recent_pitches())
    list(
      pa = evm$pa,
      bb_rate = evm$bb_rate,
      k_rate  = evm$k_rate,
      slg     = evm$slg,
      fps     = fps$fps,
      fpinz   = fpinz$fpinz
    )
  })
  
  metrics_season <- reactive({
    evm <- calc_event_metrics(season_events())
    fps <- calc_fps(season_pitches())
    fpinz <- calc_fpinz(season_pitches())
    list(
      pa = evm$pa,
      bb_rate = evm$bb_rate,
      k_rate  = evm$k_rate,
      slg     = evm$slg,
      fps     = fps$fps,
      fpinz   = fpinz$fpinz
    )
  })
  
  fmt_pp <- function(delta) {
    ifelse(is.na(delta), "—", sprintf("%+.1f pp", 100 * delta))
  }
  
  fmt_diff <- function(delta) {
    ifelse(is.na(delta), "—", sprintf("%+.3f", delta))
  }
  
  make_metric_card <- function(title, recent_val, season_val, formatter,
                               higher_is_better = TRUE) {
    
    delta <- recent_val - season_val
    
    is_good <- dplyr::case_when(
      is.na(delta) ~ NA,
      higher_is_better ~ delta > 0,
      TRUE ~ delta < 0
    )
    
    pill_color <- dplyr::case_when(
      is.na(delta) ~ "color:#6c757d;background:rgba(108,117,125,.12);",
      delta == 0   ~ "color:#6c757d;background:rgba(108,117,125,.12);",
      is_good      ~ "color:#198754;background:rgba(25,135,84,.12);",
      TRUE         ~ "color:#dc3545;background:rgba(220,53,69,.12);"
    )
    
    
    pill_txt <- dplyr::case_when(
      is.na(delta) ~ "—",
      delta == 0   ~ "At Season",
      is_good      ~ "Better vs Season",
      TRUE         ~ "Worse vs Season"
    )
    
    
    value_box(
      title = NULL,
      value = tags$div(
        # 标题
        tags$div(
          style="display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:8px;",
          tags$div(style="font-weight:800;font-size:14px;", title),
          tags$span(
            style="font-size:11px;font-weight:900;padding:3px 8px;border-radius:999px;
                 background:rgba(235,110,31,0.12);color:#6a2f00;border:1px solid rgba(235,110,31,0.20);",
            "Recent (14d)"
          )
        ),
        
        # 主数值（唯一视觉中心）
        tags$div(
          style="font-size:36px;font-weight:900;line-height:1;",
          formatter(recent_val)
        ),
        
        # pill（结论）
        tags$div(
          style=paste0(
            "display:inline-block;margin-top:8px;",
            "font-size:12px;font-weight:800;padding:4px 10px;",
            "border-radius:999px;", pill_color
          ),
          pill_txt
        ),
        
        # season（弱信息）
        tags$div(
          style="margin-top:6px;font-size:11px;color:#6c757d;",
          "Season ", formatter(season_val)
        )
      )
    )
  }
  
  output$card_fps <- renderUI({
    r <- metrics_recent(); s <- metrics_season()
    make_metric_card("FPS%", r$fps, s$fps, fmt_pct, higher_is_better = TRUE)
  })
  
  output$card_bb <- renderUI({
    r <- metrics_recent(); s <- metrics_season()
    make_metric_card("BB%", r$bb_rate, s$bb_rate, fmt_pct, higher_is_better = FALSE)
  })
  
  output$card_k <- renderUI({
    r <- metrics_recent(); s <- metrics_season()
    make_metric_card("K%", r$k_rate, s$k_rate, fmt_pct, higher_is_better = TRUE)
  })
  
  output$card_slg <- renderUI({
    r <- metrics_recent(); s <- metrics_season()
    make_metric_card("SLG", r$slg, s$slg, fmt_num, higher_is_better = FALSE)
  })
  
  
  # ----------------------------
  # 6) Goals text (goals.csv)
  # ----------------------------
  
  # Help
  
  parse_goal <- function(txt) {
    txt <- trimws(as.character(txt))
    if (is.na(txt) || txt == "") return(NULL)
    
    dir <- dplyr::case_when(
      grepl("^increase\\b", tolower(txt)) ~ "increase",
      grepl("^decrease\\b", tolower(txt)) ~ "decrease",
      TRUE ~ "other"
    )
    
    metric <- NA_character_
    m <- regmatches(txt, regexec("(?i)(BB%|K%|SLG%|SLG|FPS%|FPinZ%)", txt, perl = TRUE))[[1]]
    if (length(m) > 1) metric <- m[2]
    
    target <- NA_real_
    tmatch <- regmatches(txt, regexec("([0-9]+\\.?[0-9]*)\\s*%?", txt, perl = TRUE))[[1]]
    if (length(tmatch) > 1) target <- as.numeric(tmatch[2])
    
    comp <- dplyr::case_when(
      grepl("or less|≤|<=", txt, ignore.case = TRUE) ~ "≤",
      grepl("or more|≥|>=", txt, ignore.case = TRUE) ~ "≥",
      TRUE ~ ""
    )
    
    list(dir = dir, metric = metric, target = target, comp = comp, raw = txt)
  }
  
  metric_to_values <- function(metric_label) {
    r <- metrics_recent(); s <- metrics_season()
    
    if (identical(metric_label, "BB%"))    return(list(recent = 100*r$bb_rate, season = 100*s$bb_rate, is_pct = TRUE))
    if (identical(metric_label, "K%"))     return(list(recent = 100*r$k_rate,  season = 100*s$k_rate,  is_pct = TRUE))
    if (metric_label %in% c("SLG","SLG%")) return(list(recent = r$slg,         season = s$slg,         is_pct = FALSE))
    if (identical(metric_label, "FPS%"))   return(list(recent = 100*r$fps,     season = 100*s$fps,     is_pct = TRUE))
    if (identical(metric_label, "FPinZ%")) return(list(recent = 100*r$fpinz,   season = 100*s$fpinz,   is_pct = TRUE))
    
    list(recent = NA_real_, season = NA_real_, is_pct = TRUE)
  }
  
  output$goals_text <- renderUI({
    req(input$pitcher_id)
    
    g1 <- goals %>% dplyr::filter(player_id == input$pitcher_id)
    if (nrow(g1) == 0) return(tags$em("No goals found for this pitcher."))
    
    css <- "
    .podium-stage{ width:90%; max-width:1040px; margin:8px auto 0 auto; }
    .podium-steps{
      display:grid;
      grid-template-columns: 1fr 1fr 1fr; 
      gap:18px;
      align-items:end;
      height:360px;
    }
    
    .step{
      position:relative;
      border-radius:22px;
      overflow:hidden;
      box-shadow: 0 18px 40px rgba(0,0,0,0.14), inset 0 2px 0 rgba(255,255,255,0.55);
    }
    .step::before{
      content:'';
      position:absolute; inset:0;
      background: linear-gradient(180deg, rgba(255,255,255,0.62) 0%, rgba(255,255,255,0.18) 35%, rgba(0,0,0,0.08) 100%);
      pointer-events:none;
    }
    .step::after{
      content:'';
      position:absolute; left:-22%; top:-38%;
      width:145%; height:62%;
      transform: skewY(-10deg);
      background: linear-gradient(90deg, rgba(255,255,255,0.58), rgba(255,255,255,0.18), rgba(255,255,255,0));
      opacity:0.95; pointer-events:none;
    }
    
    .step-1{ height:380px; background: linear-gradient(180deg,#f5d76e,#d4a017); }
    .step-2{ height:340px; background: linear-gradient(180deg,#e7e9ee,#aeb4bf); }
    .step-3{ height:300px; background: linear-gradient(180deg,#d7a27a,#9a5b2e); }
    
    .step-inner{
      position:absolute; inset:0;
      display:flex; flex-direction:column;
      justify-content:space-between;
      padding:14px;
      gap:12px;
    }
    
    /* Panel */
    .goal-panel{
      border-radius:18px;
      padding:14px;
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(0,0,0,0.06);
      box-shadow: 0 12px 26px rgba(0,0,0,0.12);
      display:grid;
      grid-template-rows:auto auto auto auto; /* title | metric | desc | bar */
      row-gap:10px;
    }
    .goal-panel.primary{ background: rgba(255,255,255,0.92); box-shadow: 0 16px 32px rgba(0,0,0,0.14); }
    
    .goal-rankline{ font-weight:900; color: rgba(0,0,0,0.70); margin:0; }
    .goal-metric{
      font-weight:950; letter-spacing:-0.2px; margin:0;
      display:flex; align-items:baseline; gap:10px;
      color: rgba(0,0,0,0.92);
    }
    .goal-sub{
      font-size: 11px !important;
      line-height: 1.15 !important;
      -webkit-line-clamp: 1 !important;
    }
    
    /* neutral arrow only */
    .arrow-neutral{ color: rgba(0,0,0,0.45); font-weight: 900; }
    
    /* ===== Clean compact bar ===== */
    .bar-wrap{ display:grid; row-gap:6px; }
    
    .bar-track{
      position:relative;
      height:12px;
      border-radius:999px;
      background: rgba(0,0,0,0.10);  /* one clean grey bar */
      overflow: visible;             /* allow labels above/below */
    }
    
    /* marker anchor */
    .mk{
      position:absolute;
      top:50%;
      left:0;                 
      transform: translate(-50%, -50%);
      width: 1px;
      height: 1px;
      pointer-events:none;
    }
    
    /* marker symbol: force true center */
    .mk::before{
      content:'';
      position:absolute;
      left:50%;
      top:50%;
      transform: translate(-50%, -50%);
      line-height:1;
      font-weight:950;
      text-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }

    
    /* Goal star (top) */
    .mk.goal::before{
      content:'★';
      font-size:14px;
      color: rgba(0,0,0,0.65);
    }
    
    /* Season asterisk (bottom) */
    .mk.season::before{
      content:'●';
      font-size:14px;
      color: rgba(0,0,0,0.55);
    }
    
    /* label bubble */
    .mk .mk-label{
      position:absolute;
      left:50%;
      transform: translateX(-50%);
      font-size:11px;
      font-weight: 800;
      color: rgba(0,0,0,0.62);
      white-space:nowrap;
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(0,0,0,0.08);
      padding: 3px 8px;
      border-radius: 999px;
      box-shadow: 0 8px 18px rgba(0,0,0,0.10);
    }
    
    /* place labels */
    .mk.goal  .mk-label{ top:-30px; }  /* above bar */
    .mk.season .mk-label{ top:14px; }  /* below bar */
    
    /* Optional: color star by tier (comment out if you want all neutral)
    .tier-primary   .mk.goal::before{ color:#d4a017; }
    .tier-secondary .mk.goal::before{ color:#aeb4bf; }
    .tier-tertiary  .mk.goal::before{ color:#9a5b2e; }
    */
    
    /* footer */
    .step-footer{ display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .step-label{
      font-weight:950; font-size:14px; line-height:1;
      background: rgba(255,255,255,0.42);
      border: 1px solid rgba(255,255,255,0.55);
      padding:10px 14px; border-radius:999px;
      box-shadow:0 10px 18px rgba(0,0,0,0.12);
    }
    .step-rank{
      width:46px; height:46px; border-radius:999px;
      display:flex; align-items:center; justify-content:center;
      font-weight:950; font-size:16px; line-height:1;
      background: rgba(255,255,255,0.55);
      border:1px solid rgba(255,255,255,0.65);
      box-shadow:0 12px 22px rgba(0,0,0,0.14);
    }
    
    /* Tier scaling (gentle) */
    .tier-primary   .goal-panel{ min-height: 260px; }
    .tier-secondary .goal-panel{ min-height: 240px; }
    .tier-tertiary  .goal-panel{ min-height: 210px; }

    
    .tier-primary .goal-rankline{ font-size:18px; }
    .tier-primary .goal-metric{ font-size:30px; }
    
    .tier-secondary .goal-rankline{ font-size:16px; }
    .tier-secondary .goal-metric{ font-size:24px; }
    
    .tier-tertiary .goal-rankline{ font-size:14px; }
    .tier-tertiary .goal-metric{ font-size:20px; }
    "
    
    get_col <- function(df, dot_name, space_name) {
      if (dot_name %in% names(df)) return(df[[dot_name]][1])
      if (space_name %in% names(df)) return(df[[space_name]][1])
      NA
    }
    
    
    
    fmtv <- function(x, is_pct) {
      if (is.na(x)) return("NA")
      if (is_pct) return(paste0(sprintf("%.1f", x), "%"))
      sprintf("%.3f", x)
    }
    
    bar_max <- function(metric_label, recent, season, goal, is_pct) {
      m <- max(c(recent, season, goal), na.rm = TRUE)
      if (!is.finite(m)) return(if (is_pct) 40 else 1.2)
      if (is_pct) return(max(25, min(100, m * 1.25)))
      max(1.0, m * 1.2)
    }
    
    compact_bar <- function(recent, season, goal, is_pct, dir, comp, metric_label) {
      maxv <- bar_max(metric_label, recent, season, goal, is_pct)
      clamp <- function(x) pmax(0, pmin(maxv, x))
      pos <- function(x) if (is.na(x)) NA_real_ else 100 * clamp(x)/maxv
      
      sp <- pos(season)
      gp <- pos(goal)
      
      season_txt <- fmtv(season, is_pct)
      goal_txt   <- if (is.na(goal)) "NA"
      else if (is_pct) paste0(comp, sprintf("%.1f", goal), "%")
      else paste0(comp, sprintf("%.3f", goal))
      
      tags$div(
        class = "bar-wrap",
        tags$div(
          class = "bar-track",
          
          if (!is.na(gp)) tags$div(
            class = "mk goal",
            style = paste0("left:", gp, "%;"),
            tags$div(class = "mk-label", HTML(paste0("★ Goal&nbsp;<b>", goal_txt, "</b>")))
          ),
          
          if (!is.na(sp)) tags$div(
            class = "mk season",
            style = paste0("left:", sp, "%;"),
            tags$div(class = "mk-label", HTML(paste0("● Season&nbsp;<b>", season_txt, "</b>")))
          )
        )
      )
    }
    
    goal_panel <- function(rank, label, goal_text, primary = FALSE) {
      p <- parse_goal(goal_text)
      if (is.null(p)) {
        return(tags$div(
          class = paste("goal-panel", if (primary) "primary" else ""),
          tags$div(class = "goal-rankline", paste0(rank, " ", label)),
          tags$div(class = "goal-sub", "Goal text missing or unparseable.")
        ))
      }
      
      arrow <- if (p$dir %in% c("increase", "decrease")) {
        tags$span(class = "arrow-neutral", ifelse(p$dir == "increase", "↑", "↓"))
      } else {
        tags$span("•")
      }
      
      metric_label <- if (!is.na(p$metric)) p$metric else "Metric"
      v <- metric_to_values(metric_label)
      
      tags$div(
        class = paste("goal-panel", if (primary) "primary" else ""),
        tags$div(class = "goal-rankline", paste0(rank, " ", label)),
        tags$div(class = "goal-metric", arrow, metric_label),
        tags$div(class = "goal-sub", p$raw),
        compact_bar(
          recent = v$recent,
          season = v$season,
          goal   = p$target,
          is_pct = v$is_pct,
          dir    = p$dir,
          comp   = p$comp,
          metric_label = metric_label
        )
      )
    }
    
    primary_txt   <- get_col(g1, "Primary.Goal",   "Primary Goal")
    secondary_txt <- get_col(g1, "Secondary.Goal", "Secondary Goal")
    tertiary_txt  <- get_col(g1, "Tertiary.Goal",  "Tertiary Goal")
    
    tags$div(
      tags$style(HTML(css)),
      tags$div(
        class = "podium-stage",
        tags$div(
          class = "podium-steps",
          tags$div(
            class = "step step-1 tier-primary",
            tags$div(
              class = "step-inner",
              goal_panel("1st", "Primary", primary_txt, primary = TRUE),
              tags$div(
                class = "step-footer",
                tags$div(class = "step-label", "Primary"),
                tags$div(class = "step-rank", "1")
              )
            )
          ),
          tags$div(
            class = "step step-2 tier-secondary",
            tags$div(
              class = "step-inner",
              goal_panel("2nd", "Secondary", secondary_txt, primary = FALSE),
              tags$div(
                class = "step-footer",
                tags$div(class = "step-label", "Secondary"),
                tags$div(class = "step-rank", "2")
              )
            )
          ),
          tags$div(
            class = "step step-3 tier-tertiary",
            tags$div(
              class = "step-inner",
              goal_panel("3rd", "Tertiary", tertiary_txt, primary = FALSE),
              tags$div(
                class = "step-footer",
                tags$div(class = "step-label", "Tertiary"),
                tags$div(class = "step-rank", "3")
              )
            )
          )
        )
      )
    )
  })
  
  # ----------------------------
  # 7) Most recent outing (group by sched_id)
  # ----------------------------
  
  recent_outing_summary <- reactive({
    ev <- pitcher_events()
    if (nrow(ev) == 0) return(NULL)
    
    last_date <- max(ev$sched_date, na.rm = TRUE)
    
    ev %>%
      filter(sched_date == last_date) %>%
      mutate(sched_date = as.Date(sched_date)) %>%   
      group_by(sched_id, Level_Code, sched_date) %>%
      summarise(
        PA = sum(pa, na.rm = TRUE),
        AB = sum(ab, na.rm = TRUE),
        BB = sum(bb, na.rm = TRUE),
        SO = sum(so, na.rm = TRUE),
        `BB%` = safe_div(sum(bb, na.rm = TRUE), sum(pa, na.rm = TRUE)),
        `K%`  = safe_div(sum(so, na.rm = TRUE), sum(pa, na.rm = TRUE)),
        SLG = safe_div(
          1 * sum(`X1b`, na.rm = TRUE) +
            2 * sum(`X2b`, na.rm = TRUE) +
            3 * sum(`X3b`, na.rm = TRUE) +
            4 * sum(hr,  na.rm = TRUE),
          sum(ab, na.rm = TRUE)
        ),
        .groups = "drop"
      ) %>%
      arrange(desc(PA))
  })
  
  output$recent_outing_tbl <- renderDT({
    out <- recent_outing_summary()
    if (is.null(out) || nrow(out) == 0) {
      return(datatable(data.frame(Message = "No outing found."), options = list(dom = "t")))
    }
    
    show <- out %>%
      mutate(
        `BB%` = sprintf("%.1f%%", 100 * `BB%`),
        `K%`  = sprintf("%.1f%%", 100 * `K%`),
        SLG   = sprintf("%.3f", SLG)
      )
    
    datatable(show, options = list(dom = "t", pageLength = 5))
  })
  
  # ----------------------------
  # 8) Summary table (recent vs season)
  # ----------------------------
  
  output$summary_tbl <- renderDT({
    r <- metrics_recent()
    s <- metrics_season()
    
    df <- data.frame(
      Window = c("Recent (Last 14d)", "Season"),
      PA = c(r$pa, s$pa),
      `BB%` = c(fmt_pct(r$bb_rate), fmt_pct(s$bb_rate)),
      `K%`  = c(fmt_pct(r$k_rate),  fmt_pct(s$k_rate)),
      SLG   = c(fmt_num(r$slg),     fmt_num(s$slg)),
      `FPS%`= c(fmt_pct(r$fps),     fmt_pct(s$fps)),
      check.names = FALSE
    )
    
    datatable(df, options = list(dom = "t"))
  })
  
  # ----------------------------
  # 9) Trends view (per game day, rolling by games)
  # ----------------------------
  daily_metrics <- reactive({
    req(input$pitcher_id)
    ev  <- pitcher_events()
    pit <- pitcher_pitches()
    if (nrow(ev) == 0) return(NULL)
    
    ev_day <- ev %>%
      group_by(sched_date) %>%
      summarise(
        PA = sum(pa, na.rm = TRUE),
        AB = sum(ab, na.rm = TRUE),
        BB = sum(bb, na.rm = TRUE),
        SO = sum(so, na.rm = TRUE),
        bb_rate = safe_div(sum(bb, na.rm = TRUE), sum(pa, na.rm = TRUE)),
        k_rate  = safe_div(sum(so, na.rm = TRUE), sum(pa, na.rm = TRUE)),
        slg = safe_div(
          1 * sum(X1b, na.rm = TRUE) +
            2 * sum(X2b, na.rm = TRUE) +
            3 * sum(X3b, na.rm = TRUE) +
            4 * sum(hr,  na.rm = TRUE),
          sum(ab, na.rm = TRUE)
        ),
        .groups = "drop"
      )
    
    fp_day <- pit %>%
      filter(balls_before == 0, strikes_before == 0) %>%
      group_by(sched_date) %>%
      summarise(
        fps = safe_div(sum(vapply(pitch_result, is_first_pitch_strike, logical(1)), na.rm = TRUE), n()),
        .groups = "drop"
      )
    
    ev_day %>%
      left_join(fp_day, by = "sched_date") %>%
      mutate(sched_date = as.Date(sched_date)) %>%
      arrange(sched_date)
  }) %>% bindCache(input$pitcher_id)
  
  roll_mean <- function(x, n) {
    # rolling mean with NA-safe behavior
    if (length(x) < n) return(rep(NA_real_, length(x)))
    as.numeric(stats::filter(x, rep(1 / n, n), sides = 1))
  }
  
  roll_sd <- function(x, n) {
    # rolling SD aligned with roll_mean() (sides = 1)
    if (length(x) < n) return(rep(NA_real_, length(x)))
    x <- as.numeric(x)
    mu <- as.numeric(stats::filter(x, rep(1 / n, n), sides = 1))
    mu2 <- as.numeric(stats::filter(x^2, rep(1 / n, n), sides = 1))
    var <- mu2 - mu^2
    var[var < 0] <- 0  # numerical safety
    sqrt(var)
  }
  
  as_num_vec <- function(x) {
    if (is.null(x)) return(numeric(0))
    
    if (is.data.frame(x)) x <- x[[1]]
    
    if (is.list(x) && !is.atomic(x)) {
      x <- unlist(x, use.names = FALSE)
    }
    
    suppressWarnings(as.numeric(x))
  }
  
  output$trend_plot <- renderPlotly({
    df <- daily_metrics()
    req(!is.null(df))
    
    metric <- input$trend_metric
    n      <- input$trend_roll_games
    
    shiny::validate(
      shiny::need(metric %in% names(df),
                  paste0("Metric column not found: ", as.character(metric)[1]))
    )
    
    # ---- force sched_date to real Date + stable ordering (prevents ggplotly/order issues)
    df <- df %>%
      dplyr::mutate(sched_date = as.Date(sched_date)) %>%
      dplyr::arrange(sched_date)
    
    # ---- raw metric -> numeric vector (robust to list/df columns)
    y_raw <- as_num_vec(df[[metric]])
    
    shiny::validate(
      shiny::need(length(y_raw) == nrow(df),
                  "Metric column is not a valid numeric vector (length mismatch)."),
      shiny::need(any(!is.na(y_raw)),
                  "No valid values for this metric under current filters.")
    )
    
    df$y_raw <- y_raw
    
    # ---- rolling mean + sd band
    df$y_roll <- roll_mean(df$y_raw, n)
    df$y_sd   <- roll_sd(df$y_raw, n)
    
    band_k <- 1
    df$y_lo <- df$y_roll - band_k * df$y_sd
    df$y_hi <- df$y_roll + band_k * df$y_sd
    
    season_baseline <- mean(df$y_raw, na.rm = TRUE)
    
    df$delta_prev    <- df$y_roll - dplyr::lag(df$y_roll)
    df$delta_vs_base <- df$y_roll - season_baseline
    
    # only keep rows where rolling exists
    df <- df %>% dplyr::filter(!is.na(y_roll))
    
    shiny::validate(
      shiny::need(nrow(df) > 0,
                  paste0("Not enough games to compute rolling window (n = ", n, "). Try a smaller n."))
    )
    
    # ---- format helpers
    is_pct <- metric %in% c("bb_rate", "k_rate", "fps")
    
    fmt_val <- function(x) {
      ifelse(is.na(x), "—",
             if (is_pct) sprintf("%.1f%%", 100 * x) else sprintf("%.3f", x))
    }
    fmt_delta <- function(x) {
      ifelse(is.na(x), "—",
             if (is_pct) sprintf("%+.1f pp", 100 * x) else sprintf("%+.3f", x))
    }
    
    baseline_label <- if (is_pct) sprintf("%.1f%%", 100 * season_baseline) else sprintf("%.3f", season_baseline)
    
    df$hover <- paste0(
      "<b>Date:</b> ", as.character(df$sched_date), "<br>",
      "<b>Raw:</b> ", fmt_val(df$y_raw), "<br>",
      "<b>Rolling (", n, "):</b> ", fmt_val(df$y_roll), "<br>",
      "<b>Δ vs prev rolling:</b> ", fmt_delta(df$delta_prev), "<br>",
      "<b>Season baseline:</b> ", baseline_label, "<br>",
      "<b>Δ vs baseline:</b> ", fmt_delta(df$delta_vs_base)
    )
    
    ylab_txt <- dplyr::case_when(
      metric == "bb_rate" ~ paste0("BB% (rolling ", n, " games)"),
      metric == "k_rate"  ~ paste0("K% (rolling ", n, " games)"),
      metric == "fps"     ~ paste0("FPS% (rolling ", n, " games)"),
      metric == "slg"     ~ paste0("SLG (rolling ", n, " games)"),
      TRUE ~ paste0(metric, " (rolling ", n, " games)")
    )
    
    # ---- colors
    col_line   <- "#1f77b4"
    col_point  <- "#1f77b4"
    col_ribbon <- "#1f77b4"
    col_base   <- "gray40"
    
    # ---- IMPORTANT: group = 1 + geom_path + size (ggplotly-safe) -> guarantees connected line
    g <- ggplot(df, aes(x = sched_date, group = 1)) +
      
      geom_ribbon(
        aes(ymin = y_lo, ymax = y_hi),
        fill  = col_ribbon,
        alpha = 0.15
      ) +
      
      geom_path(
        aes(y = y_roll, text = hover),
        color   = col_line,
        size    = 1.2,
        lineend = "round"
      ) +
      
      geom_point(
        aes(y = y_roll, text = hover),
        shape  = 21,
        size   = 3.0,
        stroke = 0.9,
        fill   = col_point,
        color  = "white"
      ) +
      
      geom_hline(
        yintercept = season_baseline,
        linetype   = "dashed",
        size       = 0.9,
        color      = col_base
      ) +
      # ===== baseline label =====
      annotate(
        "text",
        x = max(df$sched_date),
        y = season_baseline - 0.08 * diff(range(df$y_roll, na.rm = TRUE)),
        label = paste0("Season Avg: ", baseline_label),
        hjust = 1.05,
        vjust = -0.4,
        size  = 3.6,
        color = col_base
      ) +
      
      labs(x = NULL, y = ylab_txt) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.minor   = element_blank(),
        panel.grid.major.x = element_blank()
      )
    
    plotly::ggplotly(g, tooltip = "text") %>%
      layout(
        hovermode = "closest",
        margin = list(l = 50, r = 20, b = 40, t = 20)
      )
  })
  
  # ----------------------------
  # 10) Goals view: Show Progress
  # ----------------------------
  
  # Help
  
  status_and_gap <- function(value, goal, metric_label, goal_text_raw, is_pct, label = "Recent") {
    if (is.na(value) || is.na(goal)) {
      return(list(pill_class="ontrack", pill_txt=paste0(label, ": No Target"), gap_txt="(—)"))
    }
    higher_better <- is_higher_better(metric_label, goal_text_raw)
    gap <- value - goal
    gap_good <- if (higher_better) gap else -gap
    tol <- if (is_pct) 0.5 else 0.010
    
    pill_class <- dplyr::case_when(
      gap_good >=  tol  ~ "ahead",
      gap_good <= -tol  ~ "behind",
      TRUE              ~ "ontrack"
    )
    
    pill_txt <- dplyr::case_when(
      pill_class == "ahead"  ~ paste0(label, ": Ahead"),
      pill_class == "behind" ~ paste0(label, ": Behind"),
      TRUE                   ~ paste0(label, ": On Track")
    )
    
    gap_disp <- if (is_pct) sprintf("%+.1f%%", gap) else sprintf("%+.3f", gap)
    list(pill_class = pill_class, pill_txt = pill_txt, gap_txt = paste0("(", gap_disp, ")"))
  }
  
  is_higher_better <- function(metric_label, goal_text_raw) {
    txt <- tolower(as.character(goal_text_raw))
    if (grepl("^decrease\\b", txt)) return(FALSE)
    if (grepl("^increase\\b", txt)) return(TRUE)
    if (metric_label %in% c("BB%")) return(FALSE)
    if (metric_label %in% c("K%","FPS%","FPinZ%")) return(TRUE)
    if (metric_label %in% c("SLG","SLG%")) return(FALSE)
    TRUE
  }
  
  fmtv <- function(x, is_pct) {
    if (is.na(x)) return("NA")
    if (is_pct) return(paste0(sprintf("%.1f", x), "%"))
    sprintf("%.3f", x)
  }
  
  bar_max <- function(recent, season, goal, is_pct) {
    m <- max(c(recent, season, goal), na.rm = TRUE)
    if (!is.finite(m)) return(if (is_pct) 40 else 1.2)
    if (is_pct) return(max(25, min(100, m * 1.25)))
    max(1.0, m * 1.2)
  }
  
  output$goals_progress <- renderUI({
    req(input$pitcher_id)
    
    g1 <- goals %>% dplyr::filter(player_id == input$pitcher_id)
    if (nrow(g1) == 0) return(tags$em("No goals found for this pitcher."))
    
    css <- "
  /* ===== Goal Progress (3 stacked cards, 1 bar per card) ===== */
  .gp-wrap{
    width:100%;
    max-width: 1280px;
    margin: 12px auto 0 auto;
    padding: 8px 14px;
    display:grid;
    gap: 20px;
  }

  .gp-card{
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 12px 28px rgba(0,0,0,0.10);
  }

  .gp-card::before{
    content:'';
    position:absolute; inset:0;
    background: linear-gradient(180deg,
      rgba(255,255,255,0.65) 0%,
      rgba(255,255,255,0.22) 35%,
      rgba(0,0,0,0.06) 100%
    );
    pointer-events:none;
    z-index:0;
  }
  .gp-card::after{
    content:'';
    position:absolute; left:-22%; top:-38%;
    width:145%; height:62%;
    transform: skewY(-10deg);
    background: linear-gradient(90deg,
      rgba(255,255,255,0.58),
      rgba(255,255,255,0.18),
      rgba(255,255,255,0)
    );
    opacity:0.70;
    pointer-events:none;
    z-index:0;
  }

  .gp-card.primary   { background: linear-gradient(180deg,#f5d76e,#d4a017); }
  .gp-card.secondary { background: linear-gradient(180deg,#e7e9ee,#aeb4bf); }
  .gp-card.tertiary  { background: linear-gradient(180deg,#d7a27a,#9a5b2e); }

  .gp-head, .gp-body{ position: relative; z-index: 1; }

  .gp-head{
    background: rgba(255,255,255,0.72);
    border-bottom: 1px solid rgba(255,255,255,0.55);
    padding: 12px 16px;
    display:grid;
    grid-template-columns: 1fr auto;
    column-gap: 14px;
    row-gap: 8px;
    align-items:start;
  }

  .gp-left{ min-width: 0; }

  .gp-right{
    display:flex;
    flex-direction:column;
    align-items:flex-end;
    gap: 6px;
    text-align:right;
  }

  .gp-title{
    display:flex;
    align-items:center;
    gap:10px;
    font-weight:950;
    letter-spacing:-0.2px;
    color: rgba(0,0,0,0.78);
    font-size:18px;
  }

  .gp-badge{
    width:34px; height:34px; border-radius:999px;
    display:flex; align-items:center; justify-content:center;
    font-weight:950;
    background: rgba(255,255,255,0.7);
    border:1px solid rgba(0,0,0,0.08);
    box-shadow: 0 10px 18px rgba(0,0,0,0.08);
    flex: 0 0 auto;
  }

  .gp-raw{
    font-size:12px;
    color: rgba(0,0,0,0.55);
    white-space: nowrap;
    overflow:hidden;
    text-overflow: ellipsis;
    max-width: 520px;
  }

  .gp-sub{
    margin-top:6px;
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    align-items:center;
  }

  .gp-pill{
    display:inline-flex;
    align-items:center;
    padding: 3px 10px;
    border-radius:999px;
    font-size:11px;
    font-weight:900;
    border: 1px solid rgba(0,0,0,0.10);
    background: rgba(255,255,255,0.78);
    box-shadow: 0 8px 18px rgba(0,0,0,0.10);
    white-space:nowrap;
  }
  .gp-pill.ahead   { color: rgba(14,122,67,0.95);  border-color: rgba(14,122,67,0.25); }
  .gp-pill.ontrack { color: rgba(176,110,0,0.95);  border-color: rgba(176,110,0,0.25); }
  .gp-pill.behind  { color: rgba(176,36,36,0.95);  border-color: rgba(176,36,36,0.25); }

  /* ===== legend chips (scoped) ===== */
  .gp-legend-mini{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    justify-content:flex-end;
  }

  .gp-legend-item{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size:11px;
    font-weight:850;
    border: 1px solid rgba(0,0,0,0.10);
    background: rgba(255,255,255,0.70);
    color: rgba(0,0,0,0.62);
    white-space:nowrap;
  }

  .gp-legend-item .gp-lmk{
    position:static !important;
    transform:none !important;
    display:inline-block;
    font-size:12px;
    line-height:1;
    font-weight:950;
  }

  /* match your bar markers: Season=*, Recent=●, Goal=★ */
  .gp-legend-item.recent .gp-lmk{ color: rgba(0,0,0,0.88); }
  .gp-legend-item.season .gp-lmk{ color: rgba(13,110,253,0.85); opacity:0.75; }
  .gp-legend-item.goal   .gp-lmk{ color: rgba(0,0,0,0.60); }

  .gp-card.primary   .gp-legend-item.goal .gp-lmk{ color:#d4a017; }
  .gp-card.secondary .gp-legend-item.goal .gp-lmk{ color:#aeb4bf; }
  .gp-card.tertiary  .gp-legend-item.goal .gp-lmk{ color:#9a5b2e; }

  .gp-body{
    background: rgba(255,255,255,0.42);
    padding: 16px 18px 18px 18px;
  }

  .gp-row{ width:100%; display:grid; gap:8px; }

  .gp-bar{
    width:100%;
    position:relative;
    height:16px;
    border-radius:999px;
    background: rgba(0,0,0,0.12);
    overflow: visible;
  }

  .gp-mk{
    position:absolute;
    top:50%;
    transform: translate(-50%, -50%);
    width:1px; height:1px;
    pointer-events:none;
  }
  .gp-mk::before{
    position:absolute;
    left:50%;
    top:50%;
    transform: translate(-50%, -50%);
    line-height:1;
    font-weight:950;
    text-shadow: 0 2px 8px rgba(0,0,0,0.18);
  }
  .gp-mk.season::before{ content:'●'; font-size:12px; color: rgba(13,110,253,0.95); opacity:0.55; }
  .gp-mk.recent::before{ content:'♦'; font-size:13px; color: rgba(0,0,0,0.85);  }
  .gp-mk.goal::before  { content:'★'; font-size:14px; color: rgba(0,0,0,0.60); }

  .gp-card.primary   .gp-mk.goal::before{ color:#d4a017; }
  .gp-card.secondary .gp-mk.goal::before{ color:#aeb4bf; }
  .gp-card.tertiary  .gp-mk.goal::before{ color:#9a5b2e; }

  @media (max-width: 900px){
    .gp-head{ grid-template-columns: 1fr; }
    .gp-right{ align-items:flex-start; text-align:left; }
    .gp-raw{ max-width:100%; }
  }
  "
    
    # helpers ----
    get_col <- function(df, dot_name, space_name) {
      if (dot_name %in% names(df)) return(df[[dot_name]][1])
      if (space_name %in% names(df)) return(df[[space_name]][1])
      NA
    }
    
    parse_goal <- function(txt) {
      txt <- trimws(as.character(txt))
      if (is.na(txt) || txt == "") return(NULL)
      
      dir <- dplyr::case_when(
        grepl("^increase\\b", tolower(txt)) ~ "increase",
        grepl("^decrease\\b", tolower(txt)) ~ "decrease",
        TRUE ~ "other"
      )
      
      metric <- NA_character_
      m <- regmatches(txt, regexec("(?i)(BB%|K%|SLG%|SLG|FPS%|FPinZ%)", txt, perl = TRUE))[[1]]
      if (length(m) > 1) metric <- m[2]
      
      target <- NA_real_
      tmatch <- regmatches(txt, regexec("([0-9]+\\.?[0-9]*)\\s*%?", txt, perl = TRUE))[[1]]
      if (length(tmatch) > 1) target <- as.numeric(tmatch[2])
      
      comp <- dplyr::case_when(
        grepl("or less|≤|<=", txt, ignore.case = TRUE) ~ "≤",
        grepl("or more|≥|>=", txt, ignore.case = TRUE) ~ "≥",
        TRUE ~ ""
      )
      
      list(dir = dir, metric = metric, target = target, comp = comp, raw = txt)
    }
    
    metric_to_values <- function(metric_label) {
      r <- metrics_recent(); s <- metrics_season()
      if (identical(metric_label, "BB%"))    return(list(recent = 100*r$bb_rate, season = 100*s$bb_rate, is_pct = TRUE))
      if (identical(metric_label, "K%"))     return(list(recent = 100*r$k_rate,  season = 100*s$k_rate,  is_pct = TRUE))
      if (metric_label %in% c("SLG","SLG%")) return(list(recent = r$slg,         season = s$slg,         is_pct = FALSE))
      if (identical(metric_label, "FPS%"))   return(list(recent = 100*r$fps,     season = 100*s$fps,     is_pct = TRUE))
      if (identical(metric_label, "FPinZ%")) return(list(recent = 100*r$fpinz,   season = 100*s$fpinz,   is_pct = TRUE))
      list(recent = NA_real_, season = NA_real_, is_pct = TRUE)
    }
    

    
    progress_bar <- function(season, recent, goal, is_pct) {
      maxv <- bar_max(recent, season, goal, is_pct)
      clamp <- function(x) pmax(0, pmin(maxv, x))
      pos <- function(x) if (is.na(x)) NA_real_ else 100 * clamp(x)/maxv
      
      sp <- pos(season); rp <- pos(recent); gp <- pos(goal)
      
      tags$div(
        class = "gp-row",
        tags$div(
          class = "gp-bar",
          if (!is.na(gp)) tags$div(class="gp-mk goal",   style=paste0("left:", gp, "%;"), title=paste0("Goal: ",   fmtv(goal,   is_pct))),
          if (!is.na(sp)) tags$div(class="gp-mk season", style=paste0("left:", sp, "%;"), title=paste0("Season: ", fmtv(season, is_pct))),
          if (!is.na(rp)) tags$div(class="gp-mk recent", style=paste0("left:", rp, "%;"), title=paste0("Recent: ", fmtv(recent, is_pct)))
        )
      )
    }
    
    mk_goal_card <- function(rank_num, label, goal_text, tier_class) {
      p <- parse_goal(goal_text)
      if (is.null(p)) {
        return(
          tags$div(
            class = paste("gp-card", tier_class),
            tags$div(class="gp-head",
                     tags$div(class="gp-title", tags$div(class="gp-badge", as.character(rank_num)), paste0(label, " Goal")),
                     tags$div(class="gp-raw", "Goal text missing/unparseable.")
            ),
            tags$div(class="gp-body", tags$em("No goal to display."))
          )
        )
      }
      
      metric_label <- if (!is.na(p$metric)) p$metric else "Metric"
      v <- metric_to_values(metric_label)
      
      sg_season <- status_and_gap(v$season, p$target, metric_label, p$raw, v$is_pct, "Season")
      sg_recent <- status_and_gap(v$recent, p$target, metric_label, p$raw, v$is_pct, "Recent(Last 14 days)")
      
      tags$div(
        class = paste("gp-card", tier_class),
        tags$div(
          class = "gp-head",
          tags$div(
            class="gp-left",
            tags$div(class="gp-badge", as.character(rank_num)),
            tags$div(
              tags$div(class="gp-title", paste0(label, " (", metric_label, ")")),
              tags$div(
                class="gp-sub",
                tags$span(class=paste("gp-pill", sg_season$pill_class), paste0(sg_season$pill_txt, " ", sg_season$gap_txt)),
                tags$span(class=paste("gp-pill", sg_recent$pill_class), paste0(sg_recent$pill_txt, " ", sg_recent$gap_txt))
              )
            )
          ),
          tags$div(
            class="gp-right",
            
            # ✅ legend with values (use v$season/v$recent/p$target)
            tags$div(
              class="gp-legend-mini",
              tags$span(class="gp-legend-item recent",
                        tags$span(class="gp-lmk", "♦"),
                        tags$span(class="txt", paste0("Recent: ", fmtv(v$recent, v$is_pct)))),
              tags$span(class="gp-legend-item season",
                        tags$span(class="gp-lmk", "●"),
                        tags$span(class="txt", paste0("Season: ", fmtv(v$season, v$is_pct)))),
              tags$span(class="gp-legend-item goal",
                        tags$span(class="gp-lmk", "★"),
                        tags$span(class="txt", paste0("Goal: ",   fmtv(p$target, v$is_pct))))
            ),
            
            tags$div(class="gp-raw", p$raw)
          )
        ),
        tags$div(
          class="gp-body",
          progress_bar(season=v$season, recent=v$recent, goal=p$target, is_pct=v$is_pct)
        )
      )
    }
    
    primary_txt   <- get_col(g1, "Primary.Goal",   "Primary Goal")
    secondary_txt <- get_col(g1, "Secondary.Goal", "Secondary Goal")
    tertiary_txt  <- get_col(g1, "Tertiary.Goal",  "Tertiary Goal")
    
    tags$div(
      tags$style(HTML(css)),
      tags$div(
        class="gp-wrap",
        mk_goal_card(1, "Primary",   primary_txt,   "primary"),
        mk_goal_card(2, "Secondary", secondary_txt, "secondary"),
        mk_goal_card(3, "Tertiary",  tertiary_txt,  "tertiary")
      )
    )
  })
  
  # ----------------------------
  # 11) Report PNG
  # ----------------------------

  # Prep helper for plot
  
  # ---- helper: draw round profile pic (or placeholder) ----
  draw_profile_pic <- function(img_path = NULL, x=0.35, y=0.5, r=0.10, label="P") {
    # circle clip via alpha mask approach (simple + robust):
    # 1) draw circle bg
    grid::grid.circle(x=x, y=y, r=r, gp=grid::gpar(fill=rgb(1,1,1,0.85), col=rgb(0,0,0,0.10), lwd=2))
    
    if (!is.null(img_path) && file.exists(img_path)) {
      img <- png::readPNG(img_path)
      
      pad <- 0.95 
      grid::pushViewport(grid::viewport(
        x=x, y=y,
        width = 3*r*pad,
        height= 3*r*pad,
        clip="on"
      ))
      grid::grid.raster(img, width=0.95, height=1, interpolate=FALSE)
      grid::popViewport()
      
      # soft ring on top to "sell" the avatar
      # grid::grid.circle(x=0.5, y=0.5, r=0.52, gp=grid::gpar(fill=NA, col=rgb(0,0,0,0.12), lwd=3))
      grid::popViewport()
    } else {
      # placeholder initial
      grid::grid.text(label, x=x, y=y, gp=grid::gpar(fontsize=26, fontface="bold", col=rgb(0,0,0,0.55)))
    }
  }
  
  # ---- helper: simple podium (3 columns, different heights) ----
  draw_goals_podium_grid <- function(goals_df, recent, season, compact = TRUE){
    # tolerate both "Primary.Goal" and "Primary Goal" etc
    get_col <- function(df, dot_name, space_name){
      if (dot_name %in% names(df))  return(as.character(df[[dot_name]][1]))
      if (space_name %in% names(df)) return(as.character(df[[space_name]][1]))
      NA_character_
    }
    
    primary   <- get_col(goals_df,"Primary.Goal","Primary Goal")
    secondary <- get_col(goals_df,"Secondary.Goal","Secondary Goal")
    tertiary  <- get_col(goals_df,"Tertiary.Goal","Tertiary Goal")
    
    # parse goal text like your UI (simplified)
    parse_goal <- function(txt){
      txt <- trimws(as.character(txt))
      if (is.na(txt) || txt == "") return(NULL)
      
      dir <- dplyr::case_when(
        grepl("^increase\\b", tolower(txt)) ~ "increase",
        grepl("^decrease\\b", tolower(txt)) ~ "decrease",
        TRUE ~ "other"
      )
      
      m <- regmatches(txt, regexec("(?i)(BB%|K%|SLG%|SLG|FPS%|FPinZ%)", txt, perl = TRUE))[[1]]
      metric <- if (length(m) > 1) m[2] else NA_character_
      
      tmatch <- regmatches(txt, regexec("([0-9]+\\.?[0-9]*)\\s*%?", txt, perl = TRUE))[[1]]
      target <- if (length(tmatch) > 1) suppressWarnings(as.numeric(tmatch[2])) else NA_real_
      
      comp <- dplyr::case_when(
        grepl("or less|≤|<=", txt, ignore.case = TRUE) ~ "≤",
        grepl("or more|≥|>=", txt, ignore.case = TRUE) ~ "≥",
        TRUE ~ ""
      )
      
      list(dir=dir, metric=metric, target=target, comp=comp, raw=txt)
    }
    
    metric_to_values <- function(metric_label){
      # recent/season already computed in report
      if (identical(metric_label, "BB%"))    return(list(recent = 100*recent$bb_rate, season = 100*season$bb_rate, is_pct = TRUE))
      if (identical(metric_label, "K%"))     return(list(recent = 100*recent$k_rate,  season = 100*season$k_rate,  is_pct = TRUE))
      if (metric_label %in% c("SLG","SLG%")) return(list(recent = recent$slg,         season = season$slg,         is_pct = FALSE))
      if (identical(metric_label, "FPS%"))   return(list(recent = 100*recent$fps,     season = 100*season$fps,     is_pct = TRUE))
      if (identical(metric_label, "FPinZ%")) return(list(recent = 100*recent$fpinz,   season = 100*season$fpinz,   is_pct = TRUE))
      list(recent = NA_real_, season = NA_real_, is_pct = TRUE)
    }
    
    fmtv <- function(x, is_pct){
      if (is.na(x)) return("—")
      if (is_pct) return(sprintf("%.1f%%", x))
      sprintf("%.3f", x)
    }
    
    # compute bar scaling
    bar_max <- function(recent_v, season_v, goal_v, is_pct){
      m <- max(c(recent_v, season_v, goal_v), na.rm = TRUE)
      if (!is.finite(m)) return(if (is_pct) 40 else 1.2)
      if (is_pct) return(max(25, min(100, m * 1.25)))
      max(1.0, m * 1.2)
    }
    pos <- function(x, maxv){
      if (is.na(x)) return(NA_real_)
      0.02 + 0.96 * pmax(0, pmin(maxv, x)) / maxv   # map to [0.02,0.98] in npc
    }
    
    # ---- draw one card in a viewport ----
    draw_card <- function(rank, label, goal_text, fill_col){
      p <- parse_goal(goal_text)
      
      # card background
      grid::grid.roundrect(x=.5, y=.5, width=.98, height=.98, r=grid::unit(10,"pt"),
                           gp=grid::gpar(fill=fill_col, col=NA, alpha=0.22))
      grid::grid.roundrect(x=.5, y=.5, width=.98, height=.98, r=grid::unit(10,"pt"),
                           gp=grid::gpar(fill=rgb(1,1,1,0.80), col=rgb(0,0,0,0.06)))
      
      if (is.null(p)){
        grid::grid.text(paste0(rank, "  ", label), x=.06, y=.92, just=c("left","top"),
                        gp=grid::gpar(fontsize=11, fontface="bold", col=rgb(0,0,0,0.70)))
        grid::grid.text("Goal text missing / unparseable", x=.06, y=.82, just=c("left","top"),
                        gp=grid::gpar(fontsize=9, col=rgb(0,0,0,0.60)))
        return(invisible())
      }
      
      metric_label <- if (!is.na(p$metric)) p$metric else "Metric"
      v <- metric_to_values(metric_label)
      maxv <- bar_max(v$recent, v$season, p$target, v$is_pct)
      
      # header texts
      grid::grid.text(paste0(rank, "  ", label), x=.10, y=.88, just=c("left","top"),
                      gp=grid::gpar(fontsize=11, fontface="bold", col=rgb(0,0,0,0.70)))
      
      arrow <- if (p$dir=="increase") "↑" else if (p$dir=="decrease") "↓" else "•"
      grid::grid.text(paste0(arrow, "  ", metric_label), x=.10, y=.70, just=c("left","top"),
                      gp=grid::gpar(fontsize=14, fontface="bold", col=rgb(0,0,0,0.90)))
      
      # raw goal line
      txt_wrapped <- paste(strwrap(p$raw, width = 22), collapse = "\n")
      
      grid::grid.text(
        txt_wrapped,
        x = .10, y = .40,
        just = c("left","top"),
        gp = grid::gpar(fontsize = 7, col = rgb(0,0,0,0.60)),
      )
      
      
    }
    
    # ---- layout 3 columns in one row ----
    # Put the whole podium into a fixed-height area; caller decides where.
    fills <- c("#d4a017", "#aeb4bf", "#9a5b2e")
    texts <- list(primary, secondary, tertiary)
    labs  <- c("Primary","Secondary","Tertiary")
    
    for (i in 1:3){
      x_left <- (i-1)/3
      vp <- grid::viewport(x = x_left + 1/6, y = .5, width = 0.32, height = 0.98)
      grid::pushViewport(vp)
      draw_card(rank=i, label=labs[i], goal_text=texts[[i]], fill_col=fills[i])
      grid::popViewport()
    }
  }
  
  render_report_png <- function(file, pitcher_id, goals_df, recent_out, recent, season, dpi = 200){
    
    # A4 size in inches
    w_in <- 8.27
    h_in <- 11.69
    
    png(filename = file, width = w_in*dpi, height = h_in*dpi, res = dpi)
    on.exit(dev.off(), add = TRUE)
    
    grid::grid.newpage()
    # ===== Astros theme background =====
    grid::pushViewport(grid::viewport(x=0.5, y=0.5, width=1, height=1))
    
    grid::grid.rect(
      gp = grid::gpar(
        fill = "#F2F4F7",  # 接近打印纸
        col  = NA
      )
    )
    
    
    grid::popViewport()
    
    # page margin box
    vp <- grid::viewport(x=0.5, y=0.5, width=0.95, height=0.95)
    grid::pushViewport(vp)
    y_top <- 1
    
    # helpers
    fmt_pct <- function(x) ifelse(is.na(x), "—", sprintf("%.1f%%", 100*x))
    fmt_num <- function(x) ifelse(is.na(x), "—", sprintf("%.3f", x))
    
    y <- 0.98
    line <- function(step=0.03) { y <<- y - step }
    
    title <- function(txt){
      grid::grid.text(txt, x=0, y=y, just=c("left","top"),
                      gp=grid::gpar(fontsize=14, fontface="bold", col="#111111"))
      line(0.05)
    }
    h2 <- function(txt){
      grid::grid.text(txt, x=0, y=y, just=c("left","top"),
                      gp=grid::gpar(fontsize=12, fontface="bold", col="#111111"))
      line(0.035)
    }
    p <- function(txt){
      grid::grid.text(txt, x=0, y=y, just=c("left","top"),
                      gp=grid::gpar(fontsize=10, col="#222222"))
      line(0.03)
    }
    png_path <- 
    rsvg_png("astros_logo.svg", file = "astros_logo.png", width = 1600, height = 1600)
    # ===== Astros watermark =====
    if (file.exists("astros_logo.png")) {
      logo <- png::readPNG("astros_logo.png")
      grid::grid.raster(
        logo,
        x = 0.04, y = 0.96,
        width = 0.050,               # 水印大小
        just = c("center","center"),
        interpolate = TRUE,
        gp = grid::gpar(alpha = 0.04)  # 0.03~0.06 最稳
      )
    }
    grid::grid.text(
      paste0(
        "Internal Use Only  |  Generated ",
        format(Sys.Date(), "%Y-%m-%d"),
        "  |  Bowen Li"
      ),
      x = 0.5, y = 0.015,
      just = "center",
      gp = grid::gpar(
        fontsize = 8,
        col = rgb(0,0,0,0.45)
      )
    )
    
    
    # ===== Row 1: Header + (Left meta | Right goals) =====
    row1_h <- 0.24   # ✅ 这里可以 0.24~0.26 之间调
    vp1 <- grid::viewport(x=0.5, y=y_top - row1_h/2, width=1, height=row1_h, clip="on")
    grid::pushViewport(vp1)
    
    # full background card
    grid::grid.roundrect(
      x=0.5, y=0.5, width=1.00, height=1.00,
      r=grid::unit(18,"pt"),
      gp=grid::gpar(fill=rgb(1,1,1,0.55), col=rgb(0,0,0,0.08), lwd=1)
    )
    
    # ✅ 2-row layout: header / content (no notes row)
    lay1 <- grid::grid.layout(
      nrow=2, ncol=2,
      heights = grid::unit.c(grid::unit(0.26, "null"), grid::unit(0.74, "null")),
      widths  = grid::unit.c(grid::unit(0.42, "npc"), grid::unit(0.58, "npc"))
    )
    grid::pushViewport(grid::viewport(layout=lay1))
    
    # ---------- Header (span 2 cols) ----------
    grid::pushViewport(grid::viewport(layout.pos.row=1, layout.pos.col=1:2, clip="on"))
    grid::grid.text(
      "Stated Performance Goals",
      x=0.42, y=0.40, just=c("center","center"),
      gp=grid::gpar(fontsize=12, fontface="bold", col=rgb(0,0,0,0.82))
    )
    grid::popViewport()
    
    # ---------- Content: LEFT ----------
    grid::pushViewport(grid::viewport(layout.pos.row=2, layout.pos.col=1, clip="on"))
    
    draw_profile_pic(
      img_path = "profile_img.png",
      x = 0.38, y = 0.75, r = 0.17,
      label = "P"
    )
    
    grid::grid.text(
      paste0("Pitcher ID: ", pitcher_id),
      x = 0.10, y = 0.28, just=c("left","center"),
      gp = grid::gpar(fontsize=9, col=rgb(0,0,0,0.78))
    )
    
    grid::popViewport()
    
    # ---------- Content: RIGHT (in row2 col2, right-aligned) ----------
    grid::pushViewport(grid::viewport(layout.pos.row=2, layout.pos.col=2, clip="on"))
    
    grid::pushViewport(
      grid::viewport(
        x = 0.98, y = 0.45,
        width  = 0.70,
        height = 0.60,
        just = c("right","center"),
        clip = "on"
      )
    )
    # grid::grid.rect(gp=grid::gpar(col="blue", fill=NA, lwd=2))
    
    
    if (is.null(goals_df) || nrow(goals_df) == 0) {
      grid::grid.text(
        "No goals found for this pitcher.",
        x=0.03, y=0.80, just=c("left","top"),
        gp=grid::gpar(fontsize=11, col=rgb(0,0,0,0.65))
      )
    } else {
      draw_goals_podium_grid(goals_df, recent, season, compact=TRUE)
    }
    
    grid::popViewport()  # inner
    grid::popViewport()  # cell (right)
    
    grid::popViewport()  # layout
    grid::popViewport()  # vp1
    
    # advance y_top for Row2
    y_top <- y_top - row1_h*1.05
    
    # ===== Row 2: Most Recent Outing (card + padding + 2 stacked tables) =====
    row2_h <- 0.28
    vp2 <- grid::viewport(x=0.5, y=y_top - row2_h/2, width=1, height=row2_h, clip="on")
    grid::pushViewport(vp2)
    
    # card background（外框）
    grid::grid.roundrect(
      x=0.5, y=0.5, width=0.95, height=0.95,
      r=grid::unit(16,"pt"),
      gp=grid::gpar(fill=rgb(1,1,1,0.55), col=rgb(0,0,0,0.08), lwd=1)
    )
    
    # ---------- padding: inner viewport ----------
    pad_x <- 0.04   # 左右 padding
    pad_y <- 0.12   # 上下 padding（稍微减小，给内容更多空间）
    grid::pushViewport(grid::viewport(
      x=0.5, y=0.5,
      width  = 0.95 - 2*pad_x,
      height = 0.95 - 2*pad_y,
      clip="on"
    ))
    
    # --- Row2 typography tokens (统一字体/颜色/字重) ---
    col_title <- rgb(0,0,0,0.78)
    col_hdr   <- rgb(0,0,0,0.62)
    col_cell  <- rgb(0,0,0,0.72)
    col_grid  <- rgb(0,0,0,0.06)
    
    gp_title <- grid::gpar(fontsize=11.5, fontface="bold",  col=col_title)
    gp_sub   <- grid::gpar(fontsize=10.2, fontface="bold",  col=col_title)
    gp_hdr   <- grid::gpar(fontsize=9.0,  fontface="bold",  col=col_hdr)
    gp_cell  <- grid::gpar(fontsize=9.0,  fontface="plain", col=col_cell)
    
    # ---------- ✅ Shared table geometry (核心：统一列宽 + 行高) ----------
    # 你要对齐的列边界就靠它（两个表都从这套列宽裁剪）
    row2_col_w_named <- c(
      Date  = 0.18,
      Level = 0.14,
      Game  = 0.16,
      PA    = 0.10,
      AB    = 0.09,
      BB    = 0.09,
      SO    = 0.09,
      `BB%` = 0.12,
      `K%`  = 0.12,
      SLG   = 0.11,
      Window= 0.26,   # summary 用（Window + 其它列）
      `FPS%`= 0.12
    )
    # 统一行高（header 和每一行都用它）
    row2_row_h <- 0.18
    
    # title
    grid::grid.text(
      "Most Recent Outing",
      x=0.00, y=1.00, just=c("left","top"),
      gp=gp_title
    )
    
    # ---------------- helpers ----------------
    
    # safer date formatting (fixes 18490 issue)
    fmt_date_safe <- function(x) {
      if (is.null(x) || length(x) == 0 || is.na(x)) return("")
      if (inherits(x, "Date")) return(format(x, "%Y-%m-%d"))
      if (is.numeric(x)) {
        d <- try(as.Date(x, origin="1970-01-01"), silent=TRUE)
        if (!inherits(d, "try-error")) return(format(d, "%Y-%m-%d"))
      }
      s <- as.character(x)
      d2 <- try(as.Date(s), silent=TRUE)
      if (!inherits(d2, "try-error") && !is.na(d2)) return(format(d2, "%Y-%m-%d"))
      s
    }
    
    # ---- small helper: right align numeric cols ----
    is_num_col <- function(name){
      name %in% c("PA","AB","BB","SO","BB%","K%","SLG","FPS%","FPinZ%")
    }
    
    draw_mini_table <- function(df, x=0, y_top=0.85, w=1.0, n_show=1) {
      
      cols <- c("sched_date","Level_Code","sched_id","PA","AB","BB","SO","BB%","K%","SLG")
      have <- intersect(cols, names(df))
      d <- df[, have, drop=FALSE]
      if (nrow(d) > n_show) d <- d[1:n_show, , drop=FALSE]
      
      # format columns
      if ("sched_date" %in% names(d)) {
        # d[["sched_date"]] <- as.Date(as.character(d[["sched_date"]]), format = "%m/%d/%Y")
        d[["sched_date"]] <- ifelse(is.na(d[["sched_date"]]), as.character(d[["sched_date"]]), format(d[["sched_date"]], "%Y-%m-%d"))
      }
      if ("BB%" %in% names(d)) d[["BB%"]] <- sprintf("%.1f%%", 100 * as.numeric(d[["BB%"]]))
      if ("K%"  %in% names(d)) d[["K%"]]  <- sprintf("%.1f%%", 100 * as.numeric(d[["K%"]]))
      if ("SLG" %in% names(d)) d[["SLG"]] <- sprintf("%.3f", as.numeric(d[["SLG"]]))
      
      nice_names <- c(
        sched_date="Date", Level_Code="Level", sched_id="Game",
        PA="PA", AB="AB", BB="BB", SO="SO", "BB%"="BB%", "K%"="K%", SLG="SLG"
      )
      cn0 <- names(d)
      hdr <- unname(ifelse(cn0 %in% names(nice_names), nice_names[cn0], cn0))
      
      # ✅ shared column widths (subset + renormalize)
      col_w <- row2_col_w_named[hdr]
      col_w <- col_w / sum(col_w)
      
      nrows <- nrow(d)
      row_h <- row2_row_h
      h_total <- row_h * (nrows + 1)
      
      # header background
      grid::grid.roundrect(
        x = x + w/2, y = y_top - row_h/2,
        width = w, height = row_h,
        r = grid::unit(10,"pt"),
        gp = grid::gpar(fill=rgb(0,0,0,0.06), col=col_grid, lwd=1)
      )
      
      # header + vertical lines
      x_cursor <- x
      for (j in seq_along(hdr)) {
        cw <- w * col_w[j]
        
        grid::grid.text(
          hdr[j],
          x = x_cursor + 0.06*cw, y = y_top - row_h*0.55,
          just=c("left","center"),
          gp=gp_hdr
        )
        
        if (j > 1) {
          grid::grid.lines(
            x = grid::unit(c(x_cursor, x_cursor), "npc"),
            y = grid::unit(c(y_top - row_h, y_top - h_total), "npc"),
            gp = grid::gpar(col=col_grid, lwd=1)
          )
        }
        x_cursor <- x_cursor + cw
      }
      
      # body rows
      for (i in seq_len(nrows)) {
        y_i_top <- y_top - i*row_h
        
        # zebra stripe
        if (i %% 2 == 0) {
          grid::grid.roundrect(
            x = x + w/2, y = y_i_top - row_h/2,
            width=w, height=row_h,
            r=grid::unit(8,"pt"),
            gp=grid::gpar(fill=rgb(1,1,1,0.55), col=NA)
          )
        }
        
        x_cursor <- x
        for (j in seq_along(hdr)) {
          cw <- w * col_w[j]
          val <- as.character(d[i, j])
          
          if (is_num_col(hdr[j])) {
            grid::grid.text(
              val,
              x = x_cursor + 0.94*cw, y = y_i_top - row_h*0.55,
              just=c("right","center"),
              gp=gp_cell
            )
          } else {
            grid::grid.text(
              val,
              x = x_cursor + 0.06*cw, y = y_i_top - row_h*0.55,
              just=c("left","center"),
              gp=gp_cell
            )
          }
          x_cursor <- x_cursor + cw
        }
        
        # row separator
        grid::grid.lines(
          x = grid::unit(c(x, x+w), "npc"),
          y = grid::unit(c(y_i_top - row_h, y_i_top - row_h), "npc"),
          gp = grid::gpar(col=col_grid, lwd=1)
        )
      }
      
      invisible(h_total)
    }
    
    draw_summary_table <- function(r, s, x=0, y_top=0.78, w=1.0) {
      
      df <- data.frame(
        Window = c("Recent (Last 14d)", "Season"),
        PA     = c(r$pa, s$pa),
        `BB%`  = c(fmt_pct(r$bb_rate), fmt_pct(s$bb_rate)),
        `K%`   = c(fmt_pct(r$k_rate),  fmt_pct(s$k_rate)),
        SLG    = c(fmt_num(r$slg),     fmt_num(s$slg)),
        `FPS%` = c(fmt_pct(r$fps),     fmt_pct(s$fps)),
        check.names = FALSE
      )
      
      cn <- names(df)
      
      # ✅ shared column widths (subset + renormalize)
      # 这里用同一套 row2_col_w_named 的列宽
      col_w <- row2_col_w_named[cn]
      col_w <- col_w / sum(col_w)
      
      nrows <- nrow(df)
      row_h <- row2_row_h
      h_total <- row_h * (nrows + 1)
      
      # header bg
      grid::grid.roundrect(
        x = x + w/2, y = y_top - row_h/2,
        width = w, height = row_h,
        r = grid::unit(10,"pt"),
        gp = grid::gpar(fill=rgb(0,0,0,0.06), col=col_grid, lwd=1)
      )
      
      # header + vertical lines
      x_cursor <- x
      for (j in seq_along(cn)) {
        cw <- w * col_w[j]
        
        grid::grid.text(
          cn[j],
          x = x_cursor + 0.06*cw, y = y_top - row_h*0.55,
          just=c("left","center"),
          gp = gp_hdr
        )
        
        if (j > 1) {
          grid::grid.lines(
            x = grid::unit(c(x_cursor, x_cursor), "npc"),
            y = grid::unit(c(y_top - row_h, y_top - h_total), "npc"),
            gp = grid::gpar(col=col_grid, lwd=1)
          )
        }
        x_cursor <- x_cursor + cw
      }
      
      # body rows
      for (i in seq_len(nrows)) {
        y_i_top <- y_top - i*row_h
        
        if (i %% 2 == 0) {
          grid::grid.roundrect(
            x = x + w/2, y = y_i_top - row_h/2,
            width=w, height=row_h,
            r=grid::unit(8,"pt"),
            gp=grid::gpar(fill=rgb(1,1,1,0.55), col=NA)
          )
        }
        
        x_cursor <- x
        for (j in seq_along(cn)) {
          cw <- w * col_w[j]
          val <- as.character(df[i, j])
          
          if (is_num_col(cn[j])) {
            grid::grid.text(
              val,
              x = x_cursor + 0.94*cw, y = y_i_top - row_h*0.55,
              just=c("right","center"),
              gp = gp_cell
            )
          } else {
            grid::grid.text(
              val,
              x = x_cursor + 0.06*cw, y = y_i_top - row_h*0.55,
              just=c("left","center"),
              gp = gp_cell
            )
          }
          x_cursor <- x_cursor + cw
        }
        
        grid::grid.lines(
          x = grid::unit(c(x, x+w), "npc"),
          y = grid::unit(c(y_i_top - row_h, y_i_top - row_h), "npc"),
          gp = grid::gpar(col=col_grid, lwd=1)
        )
      }
      
      invisible(h_total)
    }
    
    # ---------- content: stacked (Recent Outing / Summary) ----------
    lay2 <- grid::grid.layout(
      nrow = 2, ncol = 1,
      heights = grid::unit.c(
        grid::unit(0.60, "npc"),
        grid::unit(0.40, "npc")
      )
    )
    grid::pushViewport(grid::viewport(layout = lay2))
    
    # TOP
    grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 1, clip = "on"))
    if (is.null(recent_out) || nrow(recent_out) == 0) {
      grid::grid.text(
        "No recent outing found.",
        x = 0.00, y = 0.78, just = c("left","top"),
        gp = grid::gpar(fontsize = 10, col = rgb(0,0,0,0.65))
      )
    } else {
      ro_tbl <- recent_out %>% dplyr::arrange(dplyr::desc(PA))
      
      # ✅ 让表高度完全由 row2_row_h 决定（1 行数据：header + 1 row）
      draw_mini_table(ro_tbl, x=0.00, y_top=0.78, w=1.00, n_show=1)
    }
    grid::popViewport()
    
    # BOTTOM
    grid::pushViewport(grid::viewport(layout.pos.row = 2, layout.pos.col = 1, clip = "on"))
    grid::grid.text(
      "Recent vs Season Summary",
      x = 0.00, y = 0.98, just = c("left","top"),
      gp = gp_title
    )
    
    r <- metrics_recent()
    s <- metrics_season()
    
    # ✅ 同样由 row2_row_h 决定（2 行数据：header + 2 rows）
    draw_summary_table(r, s, x=0.00, y_top=0.76, w=1.00)
    
    grid::popViewport()
    grid::popViewport()  # lay2
    
    # end Row2
    grid::popViewport()  # inner padding vp
    grid::popViewport()  # vp2
    y_top <- y_top - row2_h
    
    # ===== Row 3: Goal Progress (clean 3 cards via grid.layout) =====
    row3_h <- 0.40
    vp3 <- grid::viewport(x=0.5, y=y_top - row3_h/2, width=1, height=row3_h, clip="on")
    grid::pushViewport(vp3)
    
    # outer container card
    grid::grid.roundrect(
      x=0.5, y=0.5, width=0.95, height=0.95,
      r=grid::unit(16,"pt"),
      gp=grid::gpar(fill=rgb(1,1,1,0.45), col=rgb(0,0,0,0.08), lwd=1)
    )
    
    # ---------- padding viewport (关键：真正的内边距) ----------
    pad_x <- 0.04
    pad_y <- 0.08
    grid::pushViewport(grid::viewport(
      x=0.5, y=0.5,
      width  = 0.95 - 2*pad_x,
      height = 0.95 - 2*pad_y,
      clip="on"
    ))
    
    # title inside padding
    grid::grid.text(
      "Season vs Recent Performance",
      x=0.00, y=1.00, just=c("left","top"),
      gp=grid::gpar(fontsize=12, fontface="bold", col=rgb(0,0,0,0.80))
    )
    
    # ---------- layout: 1 header row spacer + 3 cards ----------
    lay3 <- grid::grid.layout(
      nrow=4, ncol=1,
      heights = grid::unit.c(
        grid::unit(0.12, "npc"),  # space under title
        grid::unit(1, "null"),
        grid::unit(1, "null"),
        grid::unit(1, "null")
      )
    )
    grid::pushViewport(grid::viewport(layout=lay3))
    
    # ===== helpers (grid version) =====
    
    # pills colors (same logic as you had)
    pill_fill <- function(cls){
      if (cls == "ahead")  return(rgb(25/255,135/255,84/255,0.18))
      if (cls == "behind") return(rgb(220/255,53/255,69/255,0.18))
      rgb(176/255,110/255,0/255,0.16)
    }
    pill_col <- function(cls){
      if (cls == "ahead")  return(rgb(25/255,135/255,84/255,0.95))
      if (cls == "behind") return(rgb(220/255,53/255,69/255,0.95))
      rgb(176/255,110/255,0/255,0.95)
    }
    
    # progress bar (track + markers), in current card viewport
    progress_bar_grid <- function(season, recent_v, goal_v, is_pct,
                                  x0=0.06, x1=0.94, y=0.22) {
      maxv <- bar_max(recent_v, season, goal_v, is_pct)
      clamp <- function(x) pmax(0, pmin(maxv, x))
      pos <- function(x) {
        if (is.na(x) || !is.finite(x)) return(NA_real_)
        x0 + (x1-x0) * clamp(x) / maxv
      }
      xs <- pos(season)
      xr <- pos(recent_v)
      xg <- pos(goal_v)
      
      # track
      grid::grid.roundrect(
        x=(x0+x1)/2, y=y, width=(x1-x0), height=0.12,
        r=grid::unit(10,"pt"),
        gp=grid::gpar(fill=rgb(0,0,0,0.10), col=NA)
      )
      
      # markers (match your UI meaning)
      if (!is.na(xs)) grid::grid.text("●", x=xs, y=y, gp=grid::gpar(fontsize=12, col=rgb(13/255,110/255,253/255,0.70)))
      if (!is.na(xr)) grid::grid.text("♦", x=xr, y=y, gp=grid::gpar(fontsize=13, col=rgb(0,0,0,0.85)))
      if (!is.na(xg)) grid::grid.text("+", x=xg, y=y, gp=grid::gpar(fontsize=13, col=rgb(0,0,0,0.55)))
    }
    
    # draw one card in a given layout row
    draw_goal_card_grid <- function(layout_row, rank_num, label, goal_text) {
      grid::pushViewport(grid::viewport(layout.pos.row=layout_row, layout.pos.col=1, clip="on"))
      
      # card background
      grid::grid.roundrect(
        x=0.5, y=0.5, width=1.00, height=1.00,
        r=grid::unit(16,"pt"),
        gp=grid::gpar(fill=rgb(1,1,1,0.72), col=rgb(0,0,0,0.08), lwd=1)
      )
      
      # inner padding for card
      grid::pushViewport(grid::viewport(x=0.5, y=0.5, width=0.96, height=0.96, clip="on"))
      
      p <- parse_goal(goal_text)
      if (is.null(p)) {
        grid::grid.text(
          paste0(label, ": goal missing/unparseable"),
          x=0.02, y=0.92, just=c("left","top"),
          gp=grid::gpar(fontsize=10.5, col=rgb(0,0,0,0.70))
        )
        grid::popViewport()
        grid::popViewport()
        return(invisible())
      }
      
      metric_label <- if (!is.na(p$metric)) p$metric else "Metric"
      v <- metric_to_values(metric_label)
      
      # statuses
      sg_season <- status_and_gap(v$season, p$target, metric_label, p$raw, v$is_pct, "Season")
      sg_recent <- status_and_gap(v$recent, p$target, metric_label, p$raw, v$is_pct, "Recent")
      
      # header left: "1. Primary (K%)"
      grid::grid.text(
        paste0(rank_num, ". ", label, " (", metric_label, ")"),
        x=0.02, y=0.95, just=c("left","top"),
        gp=grid::gpar(fontsize=11.2, fontface="bold", col=rgb(0,0,0,0.80))
      )
      
      # raw goal text (wrap by limiting width)
      # ✅ grid.text 的 wrap：用宽度受限的 viewport + just=left
      grid::pushViewport(grid::viewport(x=0.02, y=0.77, width=0.70, height=0.20, just=c("left","top"), clip="on"))
      grid::grid.text(
        p$raw,
        x=0, y=1, just=c("left","top"),
        gp=grid::gpar(fontsize=9.0, col=rgb(0,0,0,0.58)),
        default.units="npc"
      )
      grid::popViewport()
      
      draw_legend_one_line <- function(x_right=0.98, y=0.92, v, p){
        
        # 三段：每段符号+文字（同一gp）
        t_recent <- paste0("♦ Recent ", fmtv(v$recent, v$is_pct))
        t_season <- paste0("  |  ● Season ", fmtv(v$season, v$is_pct))
        t_goal   <- paste0("  |  + Goal ",   fmtv(p$target, v$is_pct))
        
        # 对应你的 marker style
        gp_recent <- grid::gpar(fontsize=9.0, col=rgb(0,0,0,0.85))                      # ♦ 黑
        gp_season <- grid::gpar(fontsize=9.0, col=rgb(13/255,110/255,253/255,0.70))     # ● 蓝
        gp_goal   <- grid::gpar(fontsize=9.0, col=rgb(0,0,0,0.55))                      # + 灰
        
        # 用 grobWidth 计算总宽度，实现稳定右对齐
        g1 <- grid::textGrob(t_recent, gp=gp_recent)
        g2 <- grid::textGrob(t_season, gp=gp_season)
        g3 <- grid::textGrob(t_goal,   gp=gp_goal)
        
        w1 <- grid::convertWidth(grid::grobWidth(g1), "npc", valueOnly=TRUE)
        w2 <- grid::convertWidth(grid::grobWidth(g2), "npc", valueOnly=TRUE)
        w3 <- grid::convertWidth(grid::grobWidth(g3), "npc", valueOnly=TRUE)
        
        x_left <- x_right - (w1 + w2 + w3)
        
        # 依次画三段
        grid::grid.text(t_recent, x=x_left,       y=y, just=c("left","top"), gp=gp_recent)
        grid::grid.text(t_season, x=x_left + w1,  y=y, just=c("left","top"), gp=gp_season)
        grid::grid.text(t_goal,   x=x_left + w1 + w2, y=y, just=c("left","top"), gp=gp_goal)
      }
      
      # right legend values (single line)
      draw_legend_one_line(x_right=0.98, y=0.92, v=v, p=p)
      
      # pills row (two pills)
      # Season pill
      grid::grid.roundrect(
        x=0.20, y=0.52, width=0.34, height=0.18,
        r=grid::unit(10,"pt"),
        gp=grid::gpar(fill=pill_fill(sg_season$pill_class), col=rgb(0,0,0,0.06))
      )
      grid::grid.text(
        paste0(sg_season$pill_txt, " ", sg_season$gap_txt),
        x=0.20, y=0.52,
        gp=grid::gpar(fontsize=9.0, col=pill_col(sg_season$pill_class))
      )
      
      # Recent pill
      grid::grid.roundrect(
        x=0.58, y=0.52, width=0.34, height=0.18,
        r=grid::unit(10,"pt"),
        gp=grid::gpar(fill=pill_fill(sg_recent$pill_class), col=rgb(0,0,0,0.06))
      )
      grid::grid.text(
        paste0(sg_recent$pill_txt, " ", sg_recent$gap_txt),
        x=0.58, y=0.52,
        gp=grid::gpar(fontsize=9.0, col=pill_col(sg_recent$pill_class))
      )
      
      # progress bar + markers
      progress_bar_grid(
        season   = v$season,
        recent_v = v$recent,
        goal_v   = p$target,
        is_pct   = v$is_pct,
        x0=0.06, x1=0.94, y=0.22
      )
      
      grid::popViewport()  # card inner pad
      grid::popViewport()  # card cell
    }
    
    # ---------- content: draw 3 cards ----------
    get_col_local <- function(df, dot_name, space_name) {
      if (dot_name %in% names(df)) return(df[[dot_name]][1])
      if (space_name %in% names(df)) return(df[[space_name]][1])
      NA
    }
    primary_txt   <- get_col_local(goals_df, "Primary.Goal",   "Primary Goal")
    secondary_txt <- get_col_local(goals_df, "Secondary.Goal", "Secondary Goal")
    tertiary_txt  <- get_col_local(goals_df, "Tertiary.Goal",  "Tertiary Goal")
    
    if (is.null(goals_df) || nrow(goals_df) == 0) {
      grid::pushViewport(grid::viewport(layout.pos.row=2, layout.pos.col=1))
      grid::grid.text(
        "No goals found for this pitcher.",
        x=0.00, y=0.80, just=c("left","top"),
        gp=grid::gpar(fontsize=10.5, col=rgb(0,0,0,0.65))
      )
      grid::popViewport()
    } else {
      draw_goal_card_grid(2, 1, "Primary",   primary_txt)
      draw_goal_card_grid(3, 2, "Secondary", secondary_txt)
      draw_goal_card_grid(4, 3, "Tertiary",  tertiary_txt)
    }
    
    grid::popViewport()  # lay3
    grid::popViewport()  # padding vp
    grid::popViewport()  # vp3
    y_top <- y_top - row3_h
  }
  
  # ----------------------------
  # 12) Report Preview (PNG) + Download (PDF)
  # ----------------------------
  
  preview_png_path <- reactiveVal(NULL)
  
  # preview: click button
  observeEvent(input$make_report_preview, {
    req(input$pitcher_id)
    
    tmp_png <- tempfile(fileext = ".png")
    
    render_report_png(
      file       = tmp_png,
      pitcher_id = input$pitcher_id,
      goals_df   = goals %>% dplyr::filter(player_id == input$pitcher_id),
      recent_out = recent_outing_summary(),
      recent     = metrics_recent(),
      season     = metrics_season(),
      dpi        = 200
    )
    
    preview_png_path(tmp_png)
  }, ignoreInit = TRUE)
  
  # UI
  output$report_preview_ui <- renderUI({
    if (is.null(preview_png_path())) {
      return(tags$div(
        class = "a4-frame",
        tags$div(
          class = "a4-paper",
          tags$div(
            style="padding:14px;color:rgba(0,0,0,0.55);font-size:13px;text-align:center;",
            "Click “Generate / Refresh Preview” to render the latest A4 preview."
          )
        )
      ))
    }
    
    tags$div(
      class = "a4-frame",
      tags$div(
        class = "a4-paper",
        imageOutput("report_preview_img", width = "100%", height = "100%")
      )
    )
  })
  
  output$report_preview_img <- renderImage({
    req(preview_png_path())
    list(src = preview_png_path(), contentType = "image/png", alt = "1-page report preview")
  }, deleteFile = FALSE)
  
  # Download PDF: convert the same A4 PNG -> 1-page PDF (no LaTeX)
  output$download_report_pdf <- downloadHandler(
    filename = function() paste0("pitcher_", input$pitcher_id, "_report.pdf"),
    content = function(file) {
      req(input$pitcher_id)
      
      tmp_png <- tempfile(fileext = ".png")
      
      render_report_png(
        file       = tmp_png,
        pitcher_id = input$pitcher_id,
        goals_df   = goals %>% dplyr::filter(player_id == input$pitcher_id),
        recent_out = recent_outing_summary(),
        recent     = metrics_recent(),
        season     = metrics_season(),
        dpi        = 200
      )
      
      # base R can write a single-page PDF and draw the PNG onto it
      img <- png::readPNG(tmp_png)
      grDevices::pdf(file, width = 8.27, height = 11.69, onefile = TRUE)
      grid::grid.newpage()
      grid::grid.raster(img, width = unit(1,"npc"), height = unit(1,"npc"))
      grDevices::dev.off()
    }
  )
  
}
