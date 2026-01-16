# ui.R
library(shiny)
library(bslib)
library(DT)
library(plotly)

app_ui <- function(request) {
  
  # Astros-inspired bslib theme (colors only; layout unchanged)
  theme_astros <- bs_theme(
    version = 5,
    bg = "#F6F7FB",
    fg = "#111111",
    primary = "#002D62",     # Astros Navy
    secondary = "#EB6E1F"    # Astros Orange
  )
  
  
  page_sidebar(
    theme = theme_astros,
    title = tags$div(
    style = "
        display:flex;
        align-items:center;
        gap:10px;
        ",
        tags$img(
          src = "https://upload.wikimedia.org/wikipedia/commons/6/6b/Houston-Astros-Logo.svg",
          style = "
        height:60px;
        width:auto;
        opacity:0.95;
      "
        ),
        tags$span(
          style = "
        font-weight: 900;
        font-size: 17px;
        letter-spacing: -0.3px;
        color: #002D62;          /* Astros Navy */
        line-height: 1;
      ",
          "Astros PD Dashboard (Anonymous Pitchers)"
        )
    ),
    
    
    tags$head(
      tags$style(HTML("
        /* ===== Global Tokens ===== */
        :root{
          --app-font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, 'Noto Sans', 'Apple Color Emoji','Segoe UI Emoji';

          /* Astros-ish palette */
          --astros-navy: #002D62;
          --astros-orange: #EB6E1F;
          --astros-orange2: #F4911E;

          /* Neutrals */
          --ink: rgba(0,0,0,0.78);
          --muted: rgba(0,0,0,0.55);

          /* Glass tokens (match your Goals cards vibe) */
          --r: 18px;
          --r2: 22px;
          --card-bg: rgba(255,255,255,0.86);
          --card-brd: rgba(0,0,0,0.06);
          --shadow: 0 14px 34px rgba(0,0,0,0.10);
          --shadow-soft: 0 10px 24px rgba(0,0,0,0.08);
        }

        /* ===== Page background ===== */
        html, body{
          font-family: var(--app-font) !important;
          font-size: 14px;
          color: var(--ink);
          background:
            radial-gradient(1200px 600px at 15% -10%, rgba(235,110,31,0.18), transparent 60%),
            radial-gradient(900px 500px at 85% 0%, rgba(0,45,98,0.16), transparent 55%),
            linear-gradient(180deg, #F7F8FC 0%, #F3F5FA 100%);
        }

        /* unify headings / card titles */
        h1,h2,h3,h4,h5,h6,
        .card-header,
        .bslib-card .card-header,
        .nav-link,
        .sidebar .form-label,
        .sidebar label{
          font-family: var(--app-font) !important;
        }

        /* ===== Cards: glass style ===== */
        .card, .bslib-card, .card-body, .card-header{
          border-radius: var(--r) !important;
        }
        .bslib-card, .card{
          background: var(--card-bg) !important;
          border: 1px solid var(--card-brd) !important;
          box-shadow: var(--shadow-soft);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
        }
        .card-header, .bslib-card .card-header{
          background: transparent !important;
          border-bottom: 1px solid rgba(0,0,0,0.06) !important;
          font-weight: 850 !important;
          letter-spacing: -0.2px;
          padding: 14px 16px;
        }
        .card-body{
          padding: 14px 16px;
        }

        /* ===== Sidebar: glass panel ===== */
        .bslib-sidebar, .sidebar{
          background: rgba(255,255,255,0.78) !important;
          border-right: 1px solid rgba(0,0,0,0.06);
          box-shadow: 0 12px 30px rgba(0,0,0,0.06);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          font-size: 13px;
        }
        .bslib-sidebar .form-control,
        .bslib-sidebar .btn{
          font-size: 13px;
        }
        .bslib-sidebar .form-label, .sidebar label{
          font-size: 12.5px;
          color: var(--muted);
          font-weight: 750;
        }
        .bslib-sidebar .form-control{
          border-radius: 14px;
          border: 1px solid rgba(0,0,0,0.10);
        }

        /* ===== Tabs: pill style with Astros accent ===== */
        .nav-tabs{
          border-bottom: 0 !important;
          gap: 8px;
          padding: 6px 6px 0 6px;
        }
        .nav-tabs .nav-link{
          border: 1px solid rgba(0,0,0,0.08) !important;
          border-radius: 999px !important;
          padding: 8px 14px !important;
          font-weight: 800;
          color: rgba(0,0,0,0.65) !important;
          background: rgba(255,255,255,0.55) !important;
        }
        .nav-tabs .nav-link.active{
          color: #fff !important;
          background: linear-gradient(135deg, var(--astros-navy), rgba(0,45,98,0.82)) !important;
          border-color: rgba(0,45,98,0.35) !important;
          box-shadow: 0 10px 22px rgba(0,45,98,0.18);
        }

        /* ===== Buttons ===== */
        .btn, .btn-primary{
          border-radius: 14px !important;
          font-weight: 800 !important;
        }
        .btn-primary{
          background: var(--astros-navy) !important;
          border-color: rgba(0,45,98,0.35) !important;
        }
        .btn-primary:hover{
          background: var(--astros-orange) !important;
          border-color: rgba(235,110,31,0.45) !important;
        }

        /* ===== DT tables ===== */
        table.dataTable, table.dataTable td, table.dataTable th{
          font-family: var(--app-font) !important;
          font-size: 12.5px;
        }
        table.dataTable thead th{
          color: rgba(0,0,0,0.65);
          background: rgba(255,255,255,0.70);
          border-bottom: 1px solid rgba(0,0,0,0.08) !important;
        }

        /* Plotly modebar / hover fonts */
        .plotly, .plotly div{
          font-family: var(--app-font) !important;
        }

        /* two metrics per group, equal spacing */
        .metric-grid{
          display:grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        .metric-grid .card,
        .metric-grid .bslib-card{
          height: 100%;
        }
        @media (max-width: 900px){
          .metric-grid{ grid-template-columns: 1fr; }
        }

        /* Small polish: hr softer */
        hr{
          border-top: 1px solid rgba(0,0,0,0.08);
          opacity: 1;
        }
        /* ===== Sidebar as glass card ===== */
        .bslib-sidebar, .sidebar{
          background: rgba(255,255,255,0.82) !important;
          border: 1px solid rgba(0,0,0,0.08) !important;
          border-radius: 22px !important;
          margin: 12px 10px 12px 12px; 
          padding: 12px 12px 14px 12px;  
          box-shadow: 0 16px 36px rgba(0,0,0,0.10);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
        }
        
        .bslib-sidebar hr{
          border-top: 1px solid rgba(0,0,0,0.08);
          margin: 14px 0;
        }
        
        /* Sidebar 表单标签 */
        .bslib-sidebar .form-label,
        .sidebar label{
          font-size: 12.5px;
          font-weight: 800;
          letter-spacing: -0.2px;
          color: rgba(0,0,0,0.65);
        }
        
        /* Sidebar 输入框 */
        .bslib-sidebar .form-control{
          border-radius: 14px;
          border: 1px solid rgba(0,0,0,0.12);
          background: rgba(255,255,255,0.90);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }
        
        /* Sidebar radio buttons spacing */
        .bslib-sidebar .form-check{
          margin-bottom: 6px;
        }
        
        /* Sidebar download button */
        .bslib-sidebar .btn{
          width: 100%;
          border-radius: 16px;
          font-weight: 900;
          letter-spacing: -0.2px;
        }
      /* ===== Move the default < sidebar toggle INSIDE the sidebar card ===== */

      /* sidebar becomes positioning context, and hides anything outside rounded corners */
      .bslib-sidebar, .sidebar{
        position: relative !important;
        overflow: hidden !important;
      }
      
      /* try all common toggle selectors across bslib versions */
      .bslib-sidebar-toggle,
      .sidebar-toggle,
      .bslib-sidebar .bslib-sidebar-toggle,
      .bslib-sidebar .sidebar-toggle,
      .bslib-page-sidebar .bslib-sidebar-toggle,
      .bslib-page-sidebar .sidebar-toggle,
      .bslib-sidebar-collapser,
      .bslib-sidebar .bslib-sidebar-collapser,
      .bslib-sidebar-collapse,
      .bslib-sidebar .bslib-sidebar-collapse{
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
        left: auto !important;
        bottom: auto !important;
        margin: 0 !important;
        z-index: 1000 !important;
      
        width: 34px !important;
        height: 34px !important;
        border-radius: 12px !important;
      
        background: rgba(255,255,255,0.75) !important;
        border: 1px solid rgba(0,0,0,0.10) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.10) !important;
      
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
      
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
      }
      
      /* make the chevron/icon match Astros navy */
      .bslib-sidebar-toggle svg,
      .sidebar-toggle svg,
      .bslib-sidebar-toggle i,
      .sidebar-toggle i,
      .bslib-sidebar-toggle span,
      .sidebar-toggle span{
        color: #002D62 !important;
        fill: #002D62 !important;
      }
      
      /* subtle hover like your theme */
      .bslib-sidebar-toggle:hover,
      .sidebar-toggle:hover,
      .bslib-sidebar-collapser:hover,
      .bslib-sidebar-collapse:hover{
        background: rgba(235,110,31,0.12) !important; /* Astros orange tint */
        border-color: rgba(235,110,31,0.25) !important;
      }
      /* ===== A4 responsive preview frame ===== */
      .a4-frame{
        width: 100%;
        max-width: 900px;        /* 桌面上不要太宽 */
        margin: 0 auto;
      }
      .a4-paper{
        width: 100%;
        aspect-ratio: 1 / 1.4142; /* A4: 210/297 */
        background: #fff;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .a4-paper img{
        width: 100%;
        height: 100%;
        object-fit: contain;     /* 关键：自动缩放适配屏幕 */
        display: flex;
      }


    /* make layout a positioning context */
    .bslib-page-sidebar .bslib-sidebar-layout{
      position: relative !important;
    }
    
    /* ===== Hide the page_sidebar collapse chevron (expanded state) ===== */
    .bslib-page-sidebar .collapse-toggle,
    .bslib-page-sidebar .bslib-sidebar-layout .collapse-toggle,
    .bslib-page-sidebar .bslib-sidebar-layout button.collapse-toggle,
    .bslib-page-sidebar button.collapse-toggle{
      display: none !important;
      visibility: hidden !important;
      opacity: 0 !important;
      pointer-events: none !important;
    }
    


      "))
    ),
    
    sidebar = sidebar(
      selectInput("pitcher_id", "Pitcher ID", choices = NULL),
      radioButtons(
        "window",
        "Compare Window",
        choices = c("Recent vs Season" = 14),
        selected = 14
      ),
      hr(),
      tags$em(
        "Recent window: last appearance date (YYYY-MM-DD) minus 14 days"
        )
      # downloadButton("download_pdf", "Export Player PDF")
    ),
    
    navset_card_tab(
      nav_panel(
        "Player Overview",
        # ===== Window banner (add this) =====
        card(
          class = "mb-2",
          card_body(
            tags$div(
              style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;",
              tags$span(
                style="font-weight:900;color:#002D62;",
                "Compare:"
              ),
              tags$span(
                style="display:inline-block;padding:4px 10px;border-radius:999px;font-weight:900;
                 background:rgba(235,110,31,0.14);color:#6a2f00;border:1px solid rgba(235,110,31,0.25);",
                "Recent (last 14 days before most recent game)"
              ),
              tags$span(style="color:rgba(0,0,0,0.45);font-weight:800;", "vs"),
              tags$span(
                style="display:inline-block;padding:4px 10px;border-radius:999px;font-weight:900;
                 background:rgba(0,45,98,0.10);color:#002D62;border:1px solid rgba(0,45,98,0.18);",
                "Season-to-date"
              )
            )
          )),
        
        fluidRow(
          column(
            6,
            card(
              full_screen = FALSE,
              card_header("Control (process & command)"),
              div(
                class = "metric-grid",
                uiOutput("card_fps"),
                uiOutput("card_bb")
              )
            )
          ),
          column(
            6,
            card(
              full_screen = FALSE,
              card_header("Outcomes (results & damage)"),
              div(
                class = "metric-grid",
                uiOutput("card_k"),
                uiOutput("card_slg")
              )
            )
          )
        ),
        fluidRow(
          column(
            12,
            card(card_header("Stated Performance Goals"), uiOutput("goals_text"))
          )
        ),
        fluidRow(
          column(
            6,
            card(card_header("Most Recent Outing (Game-Level Summary)"),
                 DTOutput("recent_outing_tbl"))
          ),
          column(
            6,
            card(card_header("Recent vs Season Summary"),
                 DTOutput("summary_tbl"))
          )
        )
      ),
      
      nav_panel(
        "Trends View",
        card(
          card_header("Metric Trends (Rolling)"),
          fluidRow(
            column(
              4,
              selectInput(
                "trend_metric",
                "Metric",
                choices = c("BB%" = "bb_rate", "K%" = "k_rate", "SLG" = "slg", "FPS%" = "fps"),
                selected = "k_rate"
              )
            ),
            column(4, sliderInput("trend_roll_games", "Rolling window (games)", min = 2, max = 20, value = 6))
          ),
          plotplotlyOutput <- plotlyOutput("trend_plot", height = 320)
        )
      ),
      
      nav_panel(
        "Goals View",
        card(
          card_header("Goals Progress"),
          uiOutput("goals_progress")
        )
      ),
      nav_panel(
        "Report",
        card(
          card_header("Player Report (1-page A4)"),
          fluidRow(
            column(
                    4,
                    div(
                      style = "
                          display: flex;
                          flex-direction: column;
                          align-items: center;
                          justify-content: flex-start;
                          padding-top: 48px;      
                          gap: 10px;
                          text-align: center;
                        "
                      ,
                      
                      # ===== 主按钮 =====
                      actionButton(
                        "make_report_preview",
                        "Generate / Refresh Preview",
                        class = "btn-primary",
                        style = "min-width: 230px;"
                      ),
                      
                      # 说明文字（按钮说明）
                      tags$div(
                        style = "
              font-size: 12px;
              color: rgba(0,0,0,0.55);
              font-style: italic;
              max-width: 260px;
            ",
                        "Regenerates the 1-page A4 report preview using the latest selected pitcher and window."
                      ),
                      
                      tags$div(style = "height: 6px;"),
                      
                      # ===== 下载按钮 =====
                      downloadButton(
                        "download_report_pdf",
                        "Download 1-page PDF",
                        style = "
              min-width: 230px;
              font-weight: 800;
            "
                      ),
                      
                      # 下载说明
                      tags$div(
                        style = "
              font-size: 12px;
              color: rgba(0,0,0,0.55);
              font-style: italic;
              max-width: 260px;
            ",
                        "Exports the current preview as a single-page A4 PDF (print-ready)."
                      )
                    )
                  )
            ,
            column(
              8,
              uiOutput("report_preview_ui")
            )
          ),
          hr(),
          tags$div(
            style="font-size:12px;color:rgba(0,0,0,0.60);",
            "Preview is rendered from a 1-page A4 canvas (fast, no LaTeX)."
          )
        )
      )
    )
  )
}
