# app.R
library(shiny)
source("ui.R")
source("server.R")
source("global.R")

shinyApp(ui = app_ui, server = app_server)

