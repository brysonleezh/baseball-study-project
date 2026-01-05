# app.R
library(shiny)

# 让 app.R 所在目录成为工作目录（关键！）
app_dir <- normalizePath(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd(app_dir)

# source 前先打印确认
message("Working dir: ", getwd())
message("Files: ", paste(list.files(), collapse = ", "))

source("load_data.R", local = TRUE)
source("ui.R",        local = TRUE)
source("server.R",    local = TRUE)

# 确认函数已加载
message("Has load_statcast_plays? ", exists("load_statcast_plays"))
message("Has get_db_conn? ", exists("get_db_conn"))

shinyApp(ui = ui, server = server)

# Three things change
# Add Player Card