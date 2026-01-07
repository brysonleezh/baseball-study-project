# server.R
library(shiny)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggnewscale)
library(httr2)
library(jsonlite)
library(bslib)
library(scales)
library(RColorBrewer)
library(MASS)
library(tibble)
# library(reticulate)
# py_config()
# py_module_available("numpy")
# py_module_available("transformers")
# py_module_available("torch")

# use_condaenv("gemma", required = TRUE)
# transformers <- import("transformers")
# torch <- import("torch")
source("load_data.R", local = TRUE)

# load gemma model
# gemma <- local({
#   hf_token <- Sys.getenv("HF_TOKEN")  # 可选：有 token 就用，没也能跑（公开模型）
#   
#   processor <- transformers$AutoTokenizer$from_pretrained(
#     "google/gemma-2-2b-it",
#     token = if (hf_token == "") NULL else hf_token
#   )
#   
#   model <- transformers$AutoModelForCausalLM$from_pretrained(
#     "google/gemma-2-2b-it",
#     token = if (hf_token == "") NULL else hf_token
#   )
#   
#   # 如果你是 Mac M 系列，通常是 mps；否则 cpu
#   if (torch$backends$mps$is_available()) {
#     model <- model$to("mps")
#   } else {
#     model <- model$to("cpu")
#   }
#   
#   list(processor = processor, model = model)
# })

server <- function(input, output, session) {
  
  # provide `%||%` operator (define EARLY)
  `%||%` <- function(a, b) if (!is.null(a)) a else b
  
  # ---- global guards ----
  MIN_PA_OVERALL <- 25   
  MIN_BIP_OVERALL <- 10    
  
  blank_plot <- function(msg = "Not enough data for this view.", sub = NULL) {
    ggplot() +
      annotate("text", x = 0, y = 0.6, label = msg, size = 6, fontface = "bold") +
      annotate("text", x = 0, y = 0.4, label = sub %||% "", size = 4, color = "grey40") +
      xlim(-1, 1) + ylim(0, 1) +
      theme_void()
  }
  
  safe_weighted_mean <- function(x, w) {
    x <- as.numeric(x); w <- as.numeric(w)
    ok <- is.finite(x) & is.finite(w) & !is.na(x) & !is.na(w)
    if (!any(ok)) return(NA_real_)
    sum(x[ok] * w[ok]) / sum(w[ok])
  }
  
  
  # ---- local Gemma generation (NO HTTP) ----
  gemma_generate <- function(prompt, max_new_tokens = 220) {
    tok <- gemma$processor
    model <- gemma$model
    
    # 直接拼 instruct prompt（Gemma2-it 很吃这个格式）
    full_prompt <- paste0(
      "<start_of_turn>user\n", prompt, "<end_of_turn>\n",
      "<start_of_turn>model\n"
    )
    
    enc <- tok(
      full_prompt,
      return_tensors = "pt"
    )
    
    # move to model device
    enc <- enc$to(model$device)
    
    out <- model$generate(
      input_ids = enc$input_ids,
      attention_mask = enc$attention_mask,
      max_new_tokens = as.integer(max_new_tokens),
      do_sample = FALSE
    )
    
    # decode full then strip prompt
    txt <- tok$decode(out[[1]], skip_special_tokens = TRUE)
    
    # 简单去掉 prompt（不完美但可用）
    sub(".*<start_of_turn>model\\n", "", txt)
  }
  
  # summary of prompt
  make_ai_summary <- function(df, player, season_rng, filters = list()) {
    bip <- df %>% dplyr::filter(!is.na(launch_speed), !is.na(launch_angle))
    
    list(
      player = player,
      season_range = season_rng,
      n_PA = nrow(df),
      n_BIP = nrow(bip),
      
      ev_mean = if (nrow(bip) > 0) round(mean(bip$launch_speed, na.rm=TRUE), 2) else NA,
      ev_median = if (nrow(bip) > 0) round(median(bip$launch_speed, na.rm=TRUE), 2) else NA,
      ev_sd = if (nrow(bip) > 1) round(sd(bip$launch_speed, na.rm=TRUE), 2) else NA,
      
      la_mean = if (nrow(bip) > 0) round(mean(bip$launch_angle, na.rm=TRUE), 2) else NA,
      la_median = if (nrow(bip) > 0) round(median(bip$launch_angle, na.rm=TRUE), 2) else NA,
      la_sd = if (nrow(bip) > 1) round(sd(bip$launch_angle, na.rm=TRUE), 2) else NA,
      
      hardhit_rate = if (nrow(bip) > 0) round(mean(bip$launch_speed >= 95, na.rm=TRUE), 3) else NA,
      sweetspot_rate = if (nrow(bip) > 0) round(mean(bip$launch_angle >= 8 & bip$launch_angle <= 32, na.rm=TRUE), 3) else NA,
      
      filters = filters
    )
  }
  
  
  # ---- helper: safely pick an existing column name ----
  pick_col <- function(df, candidates) {
    hit <- candidates[candidates %in% names(df)]
    if (length(hit) == 0) return(NULL)
    hit[1]
  }
  
  # ---- pitch type collapsing (5 buckets) ----
  pitch_bucket <- function(pt) {
    pt <- toupper(trimws(as.character(pt)))
    dplyr::case_when(
      pt == "FF" ~ "Four-Seam",
      pt %in% c("SI", "FC") ~ "Sinker/Cutter",
      pt %in% c("SL", "ST", "CU", "KC", "CS", "SV", "SC") ~ "Breaking-Ball",
      pt %in% c("CH", "FS", "FO") ~ "Offspeed",
      TRUE ~ "Other"
    )
  }
  
  # ---- zone classifier (In-zone / Out-of-zone) ----
  add_zone_flag <- function(df) {
    n <- nrow(df)
    
    # If empty, add empty column safely
    if (n == 0) {
      df$in_zone <- logical(0)
      return(df)
    }
    
    # If columns missing, add NA with correct length
    if (!all(c("plate_x_last","plate_z_last") %in% names(df))) {
      df$in_zone <- rep(NA, n)
      return(df)
    }
    
    z_top <- if ("top_zone" %in% names(df)) df$top_zone else rep(NA, n)
    z_bot <- if ("bot_zone" %in% names(df)) df$bot_zone else rep(NA, n)
    
    z_top2 <- ifelse(is.na(z_top), 3.5, z_top)
    z_bot2 <- ifelse(is.na(z_bot), 1.5, z_bot)
    
    x_left  <- -0.83
    x_right <-  0.83
    
    df$in_zone <- with(df,
                       is.finite(plate_x_last) & is.finite(plate_z_last) &
                         plate_x_last >= x_left & plate_x_last <= x_right &
                         plate_z_last >= z_bot2 & plate_z_last <= z_top2
    )
    
    df
  }
  
  # ---- count helpers ----
  add_count_cols <- function(df) {
    n <- nrow(df)
    
    # Empty df: add empty vectors safely
    if (n == 0) {
      df$balls_ <- integer(0)
      df$strikes_ <- integer(0)
      df$count_str <- character(0)
      df$count_group <- character(0)
      return(df)
    }
    
    balls_col   <- pick_col(df, c("balls", "BALLS"))
    strikes_col <- pick_col(df, c("strikes", "STRIKES"))
    
    if (is.null(balls_col) || is.null(strikes_col)) {
      df$balls_ <- rep(NA_integer_, n)
      df$strikes_ <- rep(NA_integer_, n)
      df$count_str <- rep(NA_character_, n)
      df$count_group <- rep(NA_character_, n)
      return(df)
    }
    
    df$balls_ <- as.integer(df[[balls_col]])
    df$strikes_ <- as.integer(df[[strikes_col]])
    df$count_str <- paste0(df$balls_, "-", df$strikes_)
    
    df$count_group <- dplyr::case_when(
      is.na(df$balls_) | is.na(df$strikes_) ~ NA_character_,
      df$balls_ > df$strikes_ ~ "Ahead",
      df$balls_ == df$strikes_ ~ "Even",
      df$balls_ < df$strikes_ ~ "Behind"
    )
    
    df
  }
  
  
  plays <- reactiveVal(NULL)
  
  # -----------------------
  # Load Data
  # -----------------------
  
  observeEvent(TRUE, {
    df <- load_statcast_plays()
    plays(df)
  }, once = TRUE)
  
  # -----------------------
  # UI inputs (dynamic)
  # -----------------------
  
  output$player_ui <- renderUI({
    df <- plays(); req(df)
    choices <- sort(unique(na.omit(df$batter_name)))
    
    default_player <- "Ketel Marte"
    
    selectInput(
      "player",
      "Select batter",
      choices = choices,
      selected = if (default_player %in% choices) default_player else choices[1]
    )
  })
  
  output$pitch_ui <- renderUI({
    df <- plays(); req(df)
    buckets <- df %>%
      mutate(pitch_bucket = pitch_bucket(pitch_type_last)) %>%
      pull(pitch_bucket) %>%
      unique() %>%
      sort()
    
    selectInput(
      "pitch_type", "Select pitch type (bucket)",
      choices = c("ALL", buckets),
      selected = "ALL"
    )
  })
  
  output$season_ui <- renderUI({
    df <- plays(); req(df)
  
    seasons <- sort(unique(na.omit(df$season)))
  
    # 想要的默认区间
    default_min <- 2023
    default_max <- 2025
  
    # 兜底：确保默认值在数据范围内
    sel_min <- max(min(seasons), default_min)
    sel_max <- min(max(seasons), default_max)
  
    sliderInput(
      "season",
      "Season",
      min   = min(seasons),
      max   = max(seasons),
      value = c(sel_min, sel_max),
      step  = 1,
      sep   = ""
    )
  })

  # Pitcher handedness UI (if exists)
  output$hand_ui <- renderUI({
    df <- plays(); req(df)
    hand_col <- pick_col(df, c("pitch_hand","pitcher_throws","P_THROWS"))
    if (is.null(hand_col)) return(NULL)
    
    hands <- sort(unique(na.omit(as.character(df[[hand_col]]))))
    hands <- hands[hands %in% c("R","L","S")]
    selectInput("pitch_hand", "Pitcher Handness", choices = c("ALL", hands), selected = "ALL")
  })
  
  # Count filter mode
  output$count_ui <- renderUI({
    df <- plays(); req(df)
    
    balls_col   <- pick_col(df, c("balls","BALLS"))
    strikes_col <- pick_col(df, c("strikes","STRIKES"))
    if (is.null(balls_col) || is.null(strikes_col)) return(NULL)
    
    selectInput(
      inputId = "count_group",
      label   = "Count filter",
      choices = c("ALL", "Ahead", "Even", "Behind"),
      selected = "ALL"
    )     
  })
  
  # Zone filter UI (needs plate_x_last / plate_z_last)
  output$zone_ui <- renderUI({
    df <- plays(); req(df)
    if (!all(c("plate_x_last","plate_z_last") %in% names(df))) return(NULL)
    
    selectInput("zone_filter", "Zone", choices = c("ALL","In-zone","Out-of-zone"), selected = "ALL")
  })
  
  # Velo slider UI (tries to find a velo column)
  output$velo_ui <- renderUI({
    df <- plays(); req(df)
    velo_col <- pick_col(df, c("start_speed_last","relspeed","release_speed","release_speed_last","release_speed_pitch","RELSPEED"))
    if (is.null(velo_col)) return(NULL)
    
    v <- suppressWarnings(as.numeric(df[[velo_col]]))
    v <- v[is.finite(v)]
    if (length(v) < 10) return(NULL)
    
    sliderInput("velo", "Pitch Speed (mph)", min = floor(min(v)), max = ceiling(max(v)),
                value = c(floor(min(v)), ceiling(max(v))), step = 1)
  })
  
  # Min PA for plotting stability
  output$minpa_ui <- renderUI({
    numericInput("min_pa", "Min PA per group (for plots)", value = 30, min = 1, step = 1)
  })
  
  # -----------------------
  # filtered data
  # -----------------------
  filtered <- reactive({
    df <- plays()
    req(df, input$player, input$season)
    
    out <- df %>%
      filter(
        batter_name == input$player,
        season >= input$season[1],
        season <= input$season[2]
      ) %>%
      mutate(pitch_bucket = pitch_bucket(pitch_type_last))
    
    # pitch bucket filter
    if (!is.null(input$pitch_type) && input$pitch_type != "ALL") {
      out <- out %>% filter(pitch_bucket == input$pitch_type)
    }
    
    # add helper cols for further filters
    out <- add_count_cols(out)
    out <- add_zone_flag(out)
    
    # handedness filter
    hand_col <- pick_col(out, c("p_throws","pitcher_throws","P_THROWS","pitch_hand"))
    if (!is.null(hand_col) && !is.null(input$pitch_hand) && input$pitch_hand != "ALL") {
      out <- out %>% filter(as.character(.data[[hand_col]]) == input$pitch_hand)
    }
    
    # count filter (Ahead/Even/Behind)
    if (!is.null(input$count_group) && input$count_group != "ALL") {
      out <- out %>% filter(count_group == input$count_group)
    }
    
    
    # zone filter
    if (!is.null(input$zone_filter) && input$zone_filter != "ALL") {
      if (input$zone_filter == "In-zone")  out <- out %>% filter(in_zone %in% TRUE)
      if (input$zone_filter == "Out-of-zone") out <- out %>% filter(in_zone %in% FALSE)
    }
    
    # velo filter
    velo_col <- pick_col(out, c("relspeed","release_speed_last","start_speed_last","RELSPEED"))
    if (!is.null(velo_col) && !is.null(input$velo)) {
      v <- suppressWarnings(as.numeric(out[[velo_col]]))
      out <- out %>% mutate(.velo = v) %>%
        filter(is.na(.velo) | (.velo >= input$velo[1] & .velo <= input$velo[2])) %>%
        dplyr::select(-.velo)
    }
    
    # Only BIP filter
    if (isTRUE(input$only_bip)) {
      out <- out %>% filter(!is.na(launch_speed), !is.na(launch_angle))
    }
    
    out
  })
  
  # Player Cards
  # ---- Player card metrics (from filtered play-level data) ----
  player_card_metrics <- reactive({
    df <- filtered(); req(df)
    
    bip <- df %>% filter(!is.na(launch_speed), !is.na(launch_angle))
    PA <- nrow(df)
    BIP <- nrow(bip)
    
    hardhit <- if (BIP > 0) mean(bip$launch_speed >= 95, na.rm = TRUE) else NA_real_
    sweet   <- if (BIP > 0) mean(bip$launch_angle >= 8 & bip$launch_angle <= 32, na.rm = TRUE) else NA_real_
    barrel  <- if (BIP > 0) mean(bip$launch_speed >= 98 & bip$launch_angle >= 26 & bip$launch_angle <= 30, na.rm = TRUE) else NA_real_
    
    # HR logic (keep consistent with your other section)
    is_hr <- tolower(df$eventType %||% "") == "home_run" | tolower(df$event %||% "") == "home_run"
    HR <- sum(is_hr, na.rm = TRUE)
    
    ev_mean <- if (BIP > 0) mean(bip$launch_speed, na.rm = TRUE) else NA_real_
    ev_p90  <- if (BIP >= 10) as.numeric(quantile(bip$launch_speed, 0.90, na.rm = TRUE)) else NA_real_
    la_mean <- if (BIP > 0) mean(bip$launch_angle, na.rm = TRUE) else NA_real_
    
    list(
      player = input$player,
      season_min = input$season[1],
      season_max = input$season[2],
      PA = PA,
      BIP = BIP,
      BIP_rate = if (PA > 0) BIP / PA else NA_real_,
      HR = HR,
      HR_per_100_PA = if (PA > 0) 100 * HR / PA else NA_real_,
      ev_mean = ev_mean,
      ev_p90 = ev_p90,
      la_mean = la_mean,
      hardhit = hardhit,
      sweet = sweet,
      barrel = barrel
    )
  })
  
  # ---- (Optional) load player profiles from csv/json if exists ----
  player_profiles <- reactiveVal(NULL)
  
  observeEvent(TRUE, {
    # If you create a file later, this will auto-pick it up.
    # Preferred: data/player_profiles.csv
    if (file.exists("player_profiles.csv")) {
      prof <- read.csv("player_profiles.csv", stringsAsFactors = FALSE)
      player_profiles(prof)
    } else if (file.exists("player_profiles.json")) {
      prof <- jsonlite::fromJSON("player_profiles.json")
      player_profiles(as.data.frame(prof))
    }
  }, once = TRUE)
  
  get_player_profile <- function(name) {
    prof <- player_profiles()
    if (is.null(prof) || nrow(prof) == 0) return(NULL)
    if (!"batter_name" %in% names(prof)) return(NULL)
    row <- prof[prof$batter_name == name, , drop = FALSE]
    if (nrow(row) == 0) return(NULL)
    row[1, , drop = FALSE]
  }
  
  # ---- UI: Player Card ----
  output$player_card_ui <- renderUI({
    m <- player_card_metrics(); req(m)
    prof <- get_player_profile(m$player)
    
    # safe profile pulls
    team <- if (!is.null(prof) && "team" %in% names(prof)) prof$team else "—"
    pos  <- if (!is.null(prof) && "position" %in% names(prof)) prof$position else "—"
    bats <- if (!is.null(prof) && "bats" %in% names(prof)) prof$bats else "—"
    thr  <- if (!is.null(prof) && "throws" %in% names(prof)) prof$throws else "—"
    bio  <- if (!is.null(prof) && "bio" %in% names(prof)) prof$bio else NULL
    
    height <- if (!is.null(prof) && "height" %in% names(prof)) prof$height else NA
    weight <- if (!is.null(prof) && "weight" %in% names(prof)) prof$weight else NA
    birth_date <- if (!is.null(prof) && "birth_date" %in% names(prof)) prof$birth_date else NA
    age <- if (!is.null(prof) && "age" %in% names(prof)) prof$age else NA
    
    # format helpers
    fmt_pct <- function(x) if (is.na(x)) "—" else scales::percent(x, accuracy = 0.1)
    fmt_num <- function(x, d=1) if (is.na(x)) "—" else format(round(x, d), nsmall=d)
    fmt_txt <- function(x) {
      if (is.null(x) || length(x)==0 || is.na(x) || trimws(as.character(x))=="") "—"
      else as.character(x)
    }
    
    # nicer formatting for bio fields
    height_txt <- fmt_txt(height)
    weight_txt <- fmt_txt(weight)
    birth_txt  <- fmt_txt(birth_date)
    age_txt    <- fmt_txt(age)
    
    headshot <- NULL
    if (!is.null(prof) && "headshot_path" %in% names(prof) && !is.na(prof$headshot_path) && prof$headshot_path != "") {
      headshot <- tags$img(src = prof$headshot_path, style = "width:110px;height:110px;border-radius:14px;object-fit:cover;")
    } else if (!is.null(prof) && "headshot_url" %in% names(prof) && !is.na(prof$headshot_url) && prof$headshot_url != "") {
      headshot <- tags$img(src = prof$headshot_url, style = "width:110px;height:110px;border-radius:14px;object-fit:cover;")
    } else {
      headshot <- tags$div(
        style="width:110px;height:110px;border-radius:14px;background:#eef2f7;display:flex;align-items:center;justify-content:center;font-weight:700;",
        "No Photo"
      )
    }
    
    bslib::card(
      full_screen = FALSE,
      bslib::card_header(
        div(style="display:flex;gap:16px;align-items:center;",
            headshot,
            div(
              tags$div(style="font-size:22px;font-weight:800;", m$player),
              tags$div(style="color:#5b6776;",
                       paste0(team, " • ", pos, " • Bats: ", bats, " • Throws: ", thr)),
              # ✅ 新增：身高体重生日年龄
              tags$div(style="color:#5b6776;",
                       paste0("HT: ", height_txt,
                              " • WT: ", weight_txt,
                              " • DOB: ", birth_txt,
                              " • Age: ", age_txt)),
              tags$div(style="color:#5b6776;",
                       paste0("Seasons: ", m$season_min, "–", m$season_max,
                              "  |  Filters: Pitch=", (input$pitch_type %||% "ALL"),
                              ", Zone=", (input$zone_filter %||% "ALL"),
                              ", Count=", (input$count_group %||% "ALL")))
            )
        )
      ),
      bslib::card_body(
        bslib::layout_column_wrap(
          width = 1/4,
          bslib::value_box("PA", m$PA),
          bslib::value_box("BIP Rate", fmt_pct(m$BIP_rate)),
          bslib::value_box("Avg EV", paste0(fmt_num(m$ev_mean, 1), " mph")),
          bslib::value_box("90th Percentile EV", paste0(fmt_num(m$ev_p90, 1), " mph")),
          bslib::value_box("Avg Launch Angle", paste0(fmt_num(m$la_mean, 1), "°")),
          bslib::value_box("HardHit%", fmt_pct(m$hardhit)),
          bslib::value_box("SweetSpot%", fmt_pct(m$sweet)),
          bslib::value_box("Barrel%", fmt_pct(m$barrel)),
          bslib::value_box("HR Rate (per 100 BIP)", fmt_num(m$HR_per_100_PA, 2)),
        ),
        if (!is.null(bio)) tags$div(style="margin-top:10px;color:#3a4656;", bio)
      )
    )
  })
  
  # -----------------------
  # metrics (PA-level)
  # -----------------------
  pa_metrics_by_season_pitch <- reactive({
    df <- filtered(); req(nrow(df) > 0)
    min_pa <- input$min_pa %||% 1
    
    m <- df %>%
      mutate(
        is_bip = !is.na(launch_speed) & !is.na(launch_angle),
        hardhit = is_bip & launch_speed >= 95,
        sweetspot = is_bip & launch_angle >= 8 & launch_angle <= 32,
        barrel = is_bip & launch_speed >= 98 & launch_angle >= 26 & launch_angle <= 30,
        is_hr = tolower(eventType %||% "") == "home_run" | tolower(event %||% "") == "home_run"
      ) %>%
      group_by(season, pitch_bucket) %>%
      summarise(
        PA = n(),
        BIP = sum(is_bip, na.rm = TRUE),
        BIP_rate = mean(is_bip, na.rm = TRUE),
        HardHit_rate = ifelse(BIP == 0, NA_real_, mean(hardhit[is_bip], na.rm = TRUE)),
        SweetSpot_rate = ifelse(BIP == 0, NA_real_, mean(sweetspot[is_bip], na.rm = TRUE)),
        Barrel_rate = ifelse(BIP == 0, NA_real_, mean(barrel[is_bip], na.rm = TRUE)),
        HR = sum(is_hr, na.rm = TRUE),
        HR_per_100_PA = 100 * HR / PA,
        .groups = "drop"
      ) %>%
      filter(PA >= min_pa)
    
    m
  })
  
  # =========================
  # 1) Decision
  # =========================
  
  output$dec_swing_bar <- renderPlot({
    df <- filtered(); req(nrow(df) > 0)
    m <- pa_metrics_by_season_pitch(); req(nrow(m) > 0)
    
    m_all <- m %>%
      group_by(season) %>%
      summarise(
        BIP_rate = weighted.mean(BIP_rate, w = PA, na.rm = TRUE),
        PA = sum(PA, na.rm = TRUE),
        pitch_bucket = "ALL",
        .groups = "drop"
      )
    
    m_plot <- bind_rows(m, m_all)
    
    ggplot(m_plot, aes(x = factor(season), y = BIP_rate, fill = pitch_bucket)) +
      geom_col(position = position_dodge(width = 0.9)) +
      geom_text(
        aes(label = scales::percent(BIP_rate, accuracy = 1)),
        position = position_dodge(width = 0.9),
        vjust = -0.25,
        size = 4
      ) +
      scale_y_continuous(
        labels = scales::percent_format(accuracy = 1),
        limits = c(0, max(m_plot$BIP_rate, na.rm = TRUE) * 1.15)
      ) +
      labs(
        title = paste0(input$player, " — Swing% (proxy = BIP rate)"),
        subtitle = paste0(
          "Filters applied: ", input$pitch_type,
          ", Zone=", input$zone_filter %||% "NA",
          ", Count=", input$count_group %||% "ALL"
        ),
        x = "Season",
        y = "BIP rate (proxy)",
        fill = "Pitch bucket"
      ) +
      scale_fill_manual(
        values = c(
          "ALL" = "red",
          setNames(RColorBrewer::brewer.pal(max(3, length(unique(m$pitch_bucket))), "Set2")[1:length(unique(m$pitch_bucket))],
                   unique(m$pitch_bucket))
        )
      ) +
      theme_minimal()
  })
  
  output$dec_zone_heat <- renderPlot({
    df <- filtered(); req(nrow(df) > 0)
    
    d <- df %>%
      filter(is.finite(plate_x_last), is.finite(plate_z_last))
    
    z_top <- if ("top_zone" %in% names(d)) median(d$top_zone, na.rm = TRUE) else 3.5
    z_bot <- if ("bot_zone" %in% names(d)) median(d$bot_zone, na.rm = TRUE) else 1.5
    
    x_left  <- -0.83
    x_right <-  0.83
    
    ggplot(d, aes(plate_x_last, plate_z_last)) +
      geom_bin2d(bins = 40) +
      stat_density_2d(aes(group = after_stat(level)),
                      bins = 8, color = "grey30", linewidth = 0.3, alpha = 0.6) +
      geom_rect(xmin = x_left, xmax = x_right, ymin = z_bot, ymax = z_top,
                fill = NA, color = "grey20", linewidth = 0.8, inherit.aes = FALSE) +
      coord_equal(xlim = c(-1.5, 1.5), ylim = c(1, 4)) +
      scale_fill_viridis_c(option = "C", name = "Count") +
      labs(
        title = paste0(input$player, " — Terminal pitch location distribution"),
        subtitle = "Heatmap shows where PA ends (filters: pitch/hand/count/zone/velo)",
        x = "Plate X (ft)", y = "Plate Z (ft)"
      ) +
      theme_minimal()
  })
  
  output$dec_pitch_outcome <- renderPlot({
    df <- filtered(); req(nrow(df) > 0)
    
    d <- df %>%
      mutate(
        outcome = case_when(
          tolower(eventType %||% "") == "strikeout" ~ "K",
          tolower(eventType %||% "") == "walk" ~ "BB",
          tolower(eventType %||% "") == "home_run" ~ "HR",
          !is.na(launch_speed) ~ "BIP",
          TRUE ~ "Other"
        )
      ) %>%
      count(pitch_bucket, outcome) %>%
      group_by(pitch_bucket) %>%
      mutate(prop = n / sum(n)) %>%
      ungroup()
    
    ggplot(d, aes(x = pitch_bucket, y = prop, fill = outcome)) +
      geom_col() +
      scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
      labs(
        title = paste0(input$player, " — Pitch bucket × Outcome (PA-level)"),
        x = "Pitch bucket (last pitch of PA)", y = "Share", fill = "Outcome"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 20, hjust = 1))
  })
  
  # =========================
  # 2) Contact (你的原图保持不动)
  # =========================
  output$con_ev_dist <- output$con_la_bins <- output$con_sweet_gauge <- output$con_terminal_sweet_peak <- NULL
  
  # ---- 2A EV dist ----
  output$con_ev_dist <- renderPlot({
    df <- filtered(); req(nrow(df) > 0)
    d <- df %>% filter(!is.na(launch_speed)) %>% mutate(season = as.factor(season))
    req(nrow(d) > 5)
    
    s <- d %>% group_by(season) %>%
      summarise(n=n(), ev_mean=mean(launch_speed, na.rm=TRUE), ev_median=median(launch_speed, na.rm=TRUE), .groups="drop")
    
    yl <- quantile(d$launch_speed, probs=c(0.02,0.98), na.rm=TRUE)
    y_min <- max(0, yl[[1]] - 2); y_max <- yl[[2]] + 4
    
    ggplot(d, aes(x=season, y=launch_speed, fill=season)) +
      geom_violin(trim=FALSE, alpha=0.55, color=NA) +
      geom_boxplot(width=0.14, outlier.alpha=0.15, fill="white", color="grey30") +
      geom_point(data=s, aes(x=season, y=ev_mean), inherit.aes=FALSE, size=2.8, color="black") +
      geom_text(data=s, aes(x=season, y=y_max,
                            label=paste0("Mean: ",round(ev_mean,1)," mph\n",
                                         "Median: ",round(ev_median,1)," mph\n",
                                         "n=",n)),
                inherit.aes=FALSE, vjust=1, size=4) +
      scale_y_continuous(limits=c(y_min,y_max)) +
      scale_fill_brewer(palette="Set2", guide="none") +
      labs(title=paste0(input$player," — Exit Velocity distribution"),
           subtitle="Violin = EV distribution | Box = IQR | Dot = mean EV",
           x="Season", y="Exit Velocity (mph)") +
      theme_minimal()
  })
  
  # ---- 2B LA bins ----
  output$con_la_bins <- renderPlot({
    df <- filtered(); req(nrow(df) > 0)
    d <- df %>% filter(!is.na(launch_angle), is.finite(launch_angle)) %>% mutate(season=factor(season))
    req(nrow(d) > 10)
    d <- d %>% filter(launch_angle >= -90, launch_angle <= 90)
    
    s <- d %>% group_by(season) %>% summarise(n=n(), la_mean=mean(launch_angle, na.rm=TRUE),
                                              la_median=median(launch_angle, na.rm=TRUE), .groups="drop")
    
    ggplot(d, aes(x=launch_angle, fill=season, color=season)) +
      geom_histogram(aes(y=after_stat(density)), bins=40, alpha=0.55, linewidth=0.2) +
      geom_density(linewidth=1.0, alpha=0.25) +
      geom_vline(data=s, aes(xintercept=la_median, color=season), linewidth=1.0) +
      geom_vline(data=s, aes(xintercept=la_mean, color=season), linewidth=1.0, linetype="dashed") +
      geom_text(data=s, aes(x=88, y=Inf,
                            label=paste0("n=",n,"\nmed=",round(la_median,1),"°\nmean=",round(la_mean,1),"°"),
                            color=season),
                hjust=1, vjust=1.1, size=3.6, show.legend=FALSE) +
      coord_cartesian(xlim=c(-90,90)) +
      facet_wrap(~season, ncol=1) +
      scale_fill_brewer(palette="Set2", guide="none") +
      scale_color_brewer(palette="Set2", guide="none") +
      labs(title=paste0(input$player," — Launch Angle distribution by season"),
           subtitle="Histogram as density; solid=median, dashed=mean",
           x="Launch Angle (°)", y="Density") +
      theme_minimal()
  })
  
  # ---- outcome classifier (keep yours) ----
  classify_outcome <- function(event) {
    on_base_hit <- c("Single","Double","Triple","Home Run")
    on_base_free <- c("Walk","Intent Walk","Hit By Pitch","Catcher Interference")
    on_base_error <- c("Field Error")
    outs <- c("Strikeout","Groundout","Flyout","Lineout","Pop Out","Forceout","Runner Out",
              "Fielders Choice Out","Double Play","Grounded Into DP","Strikeout Double Play",
              "Sac Fly","Sac Bunt","Sac Fly Double Play","Bunt Groundout","Bunt Lineout","Bunt Pop Out")
    other_running <- c("Stolen Base 2B","Caught Stealing 2B","Caught Stealing 3B","Caught Stealing Home",
                       "Pickoff 1B","Pickoff Error 3B","Pickoff Caught Stealing 2B")
    
    case_when(
      event %in% on_base_hit ~ "On-base (Hit)",
      event %in% on_base_free ~ "On-base (BB/HBP/CI)",
      event %in% on_base_error ~ "On-base (ROE)",
      event %in% outs ~ "Out",
      event %in% other_running ~ "Other (Running)",
      TRUE ~ "Other (Unmapped)"
    )
  }
  
  # ---- 2C SWEET peak overlay (keep yours unchanged) ----
  output$con_terminal_sweet_peak <- renderPlot({
    df <- filtered(); req(nrow(df) > 0)
    
    d <- df %>% filter(is.finite(plate_x_last), is.finite(plate_z_last))
    req(nrow(d) > 20)
    
    bip <- d %>%
      filter(!is.na(launch_speed), !is.na(launch_angle)) %>%
      mutate(la_clip = pmin(pmax(launch_angle, -10), 45))
    
    sweet <- bip %>%
      filter(launch_speed >= 95, launch_angle >= 20, launch_angle <= 30)
    
    find_peak_or_median <- function(df_group, xlim=c(-1.5,1.5), ylim=c(1,4), n_grid=80, min_n_kde=25) {
      x <- df_group$plate_x_last; z <- df_group$plate_z_last
      n <- length(x)
      fallback <- list(x=median(x,na.rm=TRUE), z=median(z,na.rm=TRUE), method="median")
      if (n < min_n_kde) return(fallback)
      kde <- MASS::kde2d(x, z, n=n_grid, lims=c(xlim, ylim))
      idx <- which.max(kde$z)
      ij <- arrayInd(idx, dim(kde$z))
      list(x=kde$x[ij[1]], z=kde$y[ij[2]], method="kde")
    }
    
    p_sweet <- if (nrow(sweet) > 0) find_peak_or_median(sweet) else NULL
    peak_df <- dplyr::bind_rows(
      if (!is.null(p_sweet)) tibble::tibble(x=p_sweet$x, z=p_sweet$z, method=p_sweet$method)
    )
    
    # ---- label text for outside annotation ----
    peak_label <- if (nrow(peak_df) == 0) {
      "Damage Zone Peak\n(No sweet samples)"
    } else {
      paste0(
        "Damage Zone Peak\n",
        "(", round(peak_df$x[1], 2), ", ", round(peak_df$z[1], 2), ")\n",
        "EV ≥ 95 & LA 20–30\n",
        "peak via ", peak_df$method[1]
      )
    }
    
    ggplot() +
      geom_point(data=d, aes(x=plate_x_last, y=plate_z_last),
                 alpha=0.25, size=1.2, color="grey45") +
      
      stat_density_2d(
        data=sweet,
        aes(x=plate_x_last, y=plate_z_last, alpha=after_stat(level)),
        geom="polygon", fill="#F1C40F", color=NA, contour_var="ndensity"
      ) +
      scale_alpha(range=c(0.05,0.35), guide="none") +
      
      geom_point(data=bip, aes(x=plate_x_last, y=plate_z_last, color=launch_speed, size=la_clip),
                 alpha=0.85) +
      scale_color_viridis_c(option="C", name="Exit Velocity (mph)") +
      scale_size_continuous(
        name="Launch Angle (°)",
        range=c(1.8,6),
        breaks=c(0,10,20,30,40),
        labels=c("0","10","20","30","40")
      ) +
      
      geom_rect(
        data=data.frame(xmin=-0.83,xmax=0.83,ymin=1.5,ymax=3.5),
        aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax),
        inherit.aes=FALSE, fill=NA, color="black", linewidth=1.2
      ) +
      
      # ✅ Peak point: special color + outline + star shape
      geom_point(
        data=peak_df, aes(x=x, y=z),
        inherit.aes=FALSE,
        shape=8, size=6, stroke=1.3, color="#00BFFF"  # deep sky blue
      ) +
      geom_point(
        data=peak_df, aes(x=x, y=z),
        inherit.aes=FALSE,
        shape=21, size=5.2, stroke=1.2, fill=NA, color="black"
      ) +
      
      # ✅ Put label OUTSIDE (top-right)
      annotate(
        "label",
        x = Inf, y = Inf,
        hjust = 1.02, vjust = 1.02,
        label = peak_label,
        size = 4,
        fontface = "bold",
        label.size = 0.4,
        color = "black",
        fill = "white",
        alpha = 0.95
      ) +
      
      coord_fixed(xlim=c(-1.5,1.5), ylim=c(1,4), clip="off") +
      
      labs(
        title=paste0(input$player," — Damage Zone peak (EV ≥ 95, LA 20–30)"),
        subtitle="Color = Exit Velocity (mph) | Size = Launch Angle (°)",
        x="Plate X (ft)", y="Plate Z (ft)"
      ) +
      
      theme_minimal() +
      theme(
        # ✅ give extra right/top margin so outside label shows
        plot.margin = margin(t = 10, r = 140, b = 10, l = 10)
      )
  })
  
  # ---- 2D On-base vs Out density peak (yours, unchanged except bugfix for color mapping) ----
  output$con_sweet_gauge <- renderPlot({
    df <- filtered(); req(nrow(df) > 0)
    
    d <- df %>%
      filter(is.finite(plate_x_last), is.finite(plate_z_last)) %>%
      mutate(outcome = classify_outcome(event)) %>%
      filter(outcome %in% c("On-base (Hit)", "On-base (BB/HBP/CI)", "On-base (ROE)", "Out")) %>%
      mutate(
        outcome2 = ifelse(outcome == "Out", "Out", "On-base"),
        outcome2 = factor(outcome2, levels = c("On-base", "Out"))
      )
    
    req(nrow(d) > 20)
    
    don  <- d %>% filter(outcome2 == "On-base")
    dout <- d %>% filter(outcome2 == "Out")
    
    find_peak_or_median <- function(df_group, xlim=c(-1.5,1.5), ylim=c(1,4), n_grid=80, min_n_kde=25) {
      x <- df_group$plate_x_last; z <- df_group$plate_z_last
      n <- length(x)
      fallback <- list(x=median(x,na.rm=TRUE), z=median(z,na.rm=TRUE), method="median")
      if (n < min_n_kde) return(fallback)
      kde <- MASS::kde2d(x, z, n=n_grid, lims=c(xlim, ylim))
      idx <- which(kde$z == max(kde$z, na.rm=TRUE), arr.ind=TRUE)[1,]
      list(x=kde$x[idx[1]], z=kde$y[idx[2]], method="kde")
    }
    
    p_on  <- if (nrow(don)  > 0) find_peak_or_median(don)  else NULL
    p_out <- if (nrow(dout) > 0) find_peak_or_median(dout) else NULL
    
    peaks <- bind_rows(
      if (!is.null(p_on))  tibble::tibble(outcome2=factor("On-base", levels=c("On-base","Out")),
                                          x=p_on$x, z=p_on$z, method=p_on$method),
      if (!is.null(p_out)) tibble::tibble(outcome2=factor("Out", levels=c("On-base","Out")),
                                          x=p_out$x, z=p_out$z, method=p_out$method)
    ) %>%
      mutate(color = ifelse(outcome2=="On-base", "#1E8449", "#7D3C98"))
    
    ggplot() +
      geom_point(data=d, aes(x=plate_x_last, y=plate_z_last),
                 alpha=0.35, size=1.4, color="grey35") +
      
      stat_density_2d(data=don,
                      aes(x=plate_x_last, y=plate_z_last, alpha=after_stat(level)),
                      geom="polygon", fill="#1E8449", color=NA, contour_var="ndensity") +
      stat_density_2d(data=dout,
                      aes(x=plate_x_last, y=plate_z_last, alpha=after_stat(level)),
                      geom="polygon", fill="#7D3C98", color=NA, contour_var="ndensity") +
      scale_alpha(range=c(0.05,0.35), guide="none") +
      
      geom_rect(data=data.frame(xmin=-0.83,xmax=0.83,ymin=1.5,ymax=3.5),
                aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax),
                inherit.aes=FALSE, fill=NA, color="black", linewidth=0.75) +
      
      # ✅ Peak point: STAR + outline (per facet)
      geom_point(data=peaks, aes(x=x, y=z),
                 inherit.aes=FALSE,
                 shape=8, size=6, stroke=1.3, color="#00BFFF") +
      geom_point(data=peaks, aes(x=x, y=z),
                 inherit.aes=FALSE,
                 shape=21, size=5.2, stroke=1.1, fill=NA, color="black") +
      
      # ✅ Outside label per facet
      geom_label(
        data = peaks %>%
          mutate(lbl = paste0("Peak: (", round(x,2), ", ", round(z,2), ")\n", "Method: ", method)),
        aes(x = Inf, y = Inf, label = lbl),
        inherit.aes = FALSE,
        hjust = 1.02, vjust = 1.02,
        size = 4, fontface = "bold",
        label.size = 0.35,
        fill = "white", alpha = 0.95
      ) +
      
      facet_wrap(~ outcome2, ncol=2) +
      coord_fixed(xlim=c(-1.5,1.5), ylim=c(1,4), clip="off") +
      
      labs(
        title=paste0(input$player," — Terminal pitch location"),
        subtitle="Green=On-base density peak | Purple=Out density peak (kde or median fallback)",
        x="Plate X (ft)", y="Plate Z (ft)"
      ) +
      theme_minimal() +
      theme(
        strip.text=element_text(face="bold"),
        plot.title=element_text(face="bold"),
        # ✅ give room so outside labels are visible
        plot.margin = margin(t = 10, r = 160, b = 10, l = 10)
      )
    
  })
  
  # ---- AI Analysis

  # Preview
  output$tbl <- renderTable({
    df <- filtered()
    head(df, 20)
  })
}
