# ui.R
library(shiny)
library(bslib)

ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  titlePanel("Diamondbacks Batting Statcast Dashboard (2020-2025)"),
  sidebarLayout(
    sidebarPanel(
      uiOutput("player_ui"),
      uiOutput("pitch_ui"),
      uiOutput("season_ui"),
      tags$hr(),
      uiOutput("hand_ui"),
      uiOutput("count_ui"),
      uiOutput("zone_ui"),
      uiOutput("velo_ui"),
      # uiOutput("minpa_ui"),
      tags$hr(),
      checkboxInput("only_bip", "Include Ball in Play (BIP) only", value = FALSE),
      tags$hr(),
      helpText(
        tags$i("Note: Metrics are calculated based on the final pitch of each plate appearance (Play Level)."),
        tags$br()
      )
    ),
    mainPanel(
      tabsetPanel(
        tabPanel(
          "Player Card",
          uiOutput("player_card_ui")
        ),
        tabPanel(
          "Visualization",
          tabsetPanel(
            tabPanel("Exit Velocity distribution", plotOutput("con_ev_dist", height = 520)),
            tabPanel("Launch Angle bins", plotOutput("con_la_bins", height = 420)),
            tabPanel("Damage Zone Peak", plotOutput("con_terminal_sweet_peak", height = 420)),
            tabPanel("On-base/Out Density Peak", plotOutput("con_sweet_gauge", height = 420))
          )
        )
      )
    )
  )
)
