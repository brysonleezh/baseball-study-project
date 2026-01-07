suppressPackageStartupMessages({
  library(DBI)
  library(odbc)
  library(dplyr)
  library(jsonlite)
})

DEFAULT_DB_CONFIG_PATH <- ".db_config.json"
DEFAULT_CSV_PATH <- "statcast_plays.csv"

# ---------------------------
# Load DB config (unchanged)
# ---------------------------
load_db_config <- function(path = DEFAULT_DB_CONFIG_PATH) {
  if (!file.exists(path)) {
    stop(
      "Missing DB config file: ", normalizePath(path, winslash = "/"),
      "\nCurrent working dir: ", getwd(),
      "\nTip: put .db_config.json in the same folder as app.R"
    )
  }
  
  cfg <- jsonlite::fromJSON(path)
  
  if (!is.null(cfg$username) && is.null(cfg$uid)) cfg$uid <- cfg$username
  if (!is.null(cfg$password) && is.null(cfg$pwd)) cfg$pwd <- cfg$password
  if (!is.null(cfg$user) && is.null(cfg$uid)) cfg$uid <- cfg$user
  if (!is.null(cfg$pw) && is.null(cfg$pwd)) cfg$pwd <- cfg$pw
  
  if (is.null(cfg$driver)) cfg$driver <- "ODBC Driver 18 for SQL Server"
  if (is.null(cfg$encrypt)) cfg$encrypt <- "yes"
  if (is.null(cfg$trust_server_certificate)) cfg$trust_server_certificate <- "yes"
  
  required <- c("server", "database", "uid", "pwd")
  missing <- required[!nzchar(as.character(cfg[required]))]
  if (length(missing) > 0) {
    stop("Missing fields in .db_config.json: ", paste(missing, collapse = ", "))
  }
  
  cfg
}

# ---------------------------
# DB connection (unchanged)
# ---------------------------
get_db_conn <- function(config_path = DEFAULT_DB_CONFIG_PATH, verbose = TRUE) {
  cfg <- load_db_config(config_path)
  
  if (verbose) {
    message("==== DB CONNECT DEBUG ====")
    message("WD: ", getwd())
    message("Config path: ", normalizePath(config_path, winslash = "/"))
    message("Driver: ", cfg$driver)
    message("Server: ", cfg$server)
    message("Database: ", cfg$database)
    message("==========================")
  }
  
  conn_str <- paste0(
    "Driver={", cfg$driver, "};",
    "Server=", cfg$server, ";",
    "Database=", cfg$database, ";",
    "Uid=", cfg$uid, ";",
    "Pwd=", cfg$pwd, ";",
    "Encrypt=", cfg$encrypt, ";",
    "TrustServerCertificate=", cfg$trust_server_certificate, ";"
  )
  
  DBI::dbConnect(odbc::odbc(), .connection_string = conn_str)
}

# =========================================================
# MAIN ENTRY: load_statcast_plays()
# Priority:
#   1) CSV snapshot (deploy-safe)
#   2) SQL Server fallback (local dev)
# =========================================================
load_statcast_plays <- function(
    csv_path = DEFAULT_CSV_PATH,
    use_db_fallback = TRUE
) {
  
  # ---- 1. Try CSV first (recommended for deployment)
  if (file.exists(csv_path)) {
    message("📄 Loading statcast data from CSV: ", csv_path)
    
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    
  } else if (use_db_fallback) {
    # ---- 2. Fallback to SQL Server (local dev)
    message("🗄️ CSV not found. Loading from SQL Server...")
    
    con <- get_db_conn(config_path = DEFAULT_DB_CONFIG_PATH)
    on.exit(DBI::dbDisconnect(con), add = TRUE)
    
    df <- DBI::dbGetQuery(con, "SELECT * FROM dbo.statcast_plays;")
    
  } else {
    stop(
      "No data source available:\n",
      "- CSV not found: ", normalizePath(csv_path, winslash = "/", mustWork = FALSE), "\n",
      "- DB fallback disabled"
    )
  }
  
  # ---- 3. Light post-processing (safe for both paths)
  df %>%
    mutate(
      startTime = suppressWarnings(as.POSIXct(startTime, tz = "UTC")),
      endTime   = suppressWarnings(as.POSIXct(endTime, tz = "UTC"))
    )
}
