#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(httr)
  library(jsonlite)
  library(chromote)
  library(bskyr)
})

api_base <- "https://statsapi.mlb.com"
team_id <- 615
sport_id <- 16
state_path <- "state.json"
prospects_path <- "prospects.json"
player_cache_path <- "player_cache.json"
tango_path <- "data/tango_we.csv"
template_path <- "templates/boxscore_card.Rhtml"
css_path <- "templates/boxscore_card.css"
out_image <- "artifacts/dsl_orange_boxscore.png"
max_chars <- 300

dir.create("artifacts", showWarnings = FALSE, recursive = TRUE)

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || (is.character(a) && !nzchar(a[1]))) b else a
as_int <- function(x, d = 0L) suppressWarnings(ifelse(is.na(as.integer(x)), d, as.integer(x)))
as_num <- function(x, d = NA_real_) suppressWarnings(ifelse(is.na(as.numeric(x)), d, as.numeric(x)))
html_escape <- function(x) {
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  x
}

load_json <- function(path, fallback) {
  if (!file.exists(path)) return(fallback)
  fromJSON(path, simplifyVector = FALSE)
}

save_json <- function(path, obj) {
  write_json(obj, path, pretty = TRUE, auto_unbox = TRUE, null = "null")
}

load_state <- function() {
  st <- load_json(state_path, list(bootstrapped = FALSE, posted_games = list(), recaps = list(), seen_transaction_ids = list()))
  if (is.null(st$posted_games)) st$posted_games <- list()
  st
}

fetch_schedule <- function(start_date, end_date, override_date = "") {
  q <- list(sportId = sport_id, teamId = team_id)
  if (nzchar(override_date)) {
    q$date <- override_date
  } else {
    q$startDate <- start_date
    q$endDate <- end_date
  }
  res <- GET(paste0(api_base, "/api/v1/schedule"), query = q, timeout(30))
  stop_for_status(res)
  payload <- content(res, as = "text", encoding = "UTF-8") |> fromJSON(simplifyVector = FALSE)
  games <- list()
  for (d in payload$dates %||% list()) games <- c(games, d$games %||% list())
  games
}

fetch_feed <- function(game_pk) {
  res <- GET(paste0(api_base, "/api/v1.1/game/", game_pk, "/feed/live"), query = list(language = "en"), timeout(30))
  stop_for_status(res)
  content(res, as = "text", encoding = "UTF-8") |> fromJSON(simplifyVector = FALSE)
}

status_class <- function(feed) {
  status <- feed$gameData$status %||% list()
  detailed <- tolower(status$detailedState %||% "")
  coded <- toupper(status$codedGameState %||% "")
  abstract <- tolower(status$abstractGameState %||% "")
  if (grepl("suspend", detailed)) return("Suspended")
  if (detailed %in% c("final", "game over") || coded == "F" || abstract == "final") return("Final")
  NULL
}

normalize_name <- function(x) {
  x <- tolower(trimws(x %||% ""))
  x <- iconv(x, to = "ASCII//TRANSLIT")
  gsub("\\s+", " ", x)
}

load_prospects <- function() {
  p <- load_json(prospects_path, list(prospects = list()))
  out <- list()
  for (pr in p$prospects %||% list()) {
    out[[normalize_name(pr$name %||% "")]] <- pr
    if (!is.null(pr$personId)) out[[paste0("id:", pr$personId)]] <- pr
  }
  out
}

load_cache <- function() {
  c <- load_json(player_cache_path, list(players = list()))
  if (is.null(c$players)) c$players <- list()
  c
}

person_details <- function(person_id, cache) {
  key <- as.character(person_id)
  if (!is.null(cache$players[[key]])) return(list(detail = cache$players[[key]], cache = cache))
  det <- list(pitchHand = "", seasonEra = "")
  url <- paste0(api_base, "/api/v1/people/", person_id)
  res <- try(GET(url, query = list(hydrate = "stats(group=[pitching],type=[season],sportId=16)"), timeout(30)), silent = TRUE)
  if (!inherits(res, "try-error") && status_code(res) < 300) {
    payload <- content(res, as = "text", encoding = "UTF-8") |> fromJSON(simplifyVector = FALSE)
    person <- (payload$people %||% list(list()))[[1]]
    det$pitchHand <- toupper(person$pitchHand$code %||% "")
    for (grp in person$stats %||% list()) {
      splits <- grp$splits %||% list()
      if (length(splits) > 0) {
        det$seasonEra <- splits[[1]]$stat$era %||% ""
        if (nzchar(det$seasonEra)) break
      }
    }
  }
  cache$players[[key]] <- det
  list(detail = det, cache = cache)
}

format_xbh <- function(h, d2, d3, hr) {
  h <- as_int(h)
  if (h <= 0) return("0")
  bits <- c()
  if (as_int(d2) > 0) bits <- c(bits, paste(as_int(d2), "2B"))
  if (as_int(d3) > 0) bits <- c(bits, paste(as_int(d3), "3B"))
  if (as_int(hr) > 0) bits <- c(bits, paste(as_int(hr), "HR"))
  if (length(bits) == 0) as.character(h) else paste0(h, " (", paste(bits, collapse = ", "), ")")
}

season_hitter <- function(stat) {
  pa <- stat$plateAppearances %||% stat$battersFaced %||% ""
  ops <- stat$ops %||% ""
  hr <- stat$homeRuns
  sb <- stat$stolenBases
  parts <- c()
  if (nzchar(as.character(pa))) parts <- c(parts, paste("PA", pa))
  if (nzchar(as.character(ops))) parts <- c(parts, paste("OPS", ops))
  if (!is.null(hr)) parts <- c(parts, paste("HR", hr))
  if (!is.null(sb)) parts <- c(parts, paste("SB", sb))
  paste(parts, collapse = " ")
}

season_pitcher <- function(stat) {
  parts <- c()
  era <- stat$era %||% ""
  bf <- as_num(stat$battersFaced, NA_real_)
  k <- as_num(stat$strikeOuts, NA_real_)
  bb <- as_num(stat$baseOnBalls, NA_real_)
  ip <- as.character(stat$inningsPitched %||% "")
  ip_num <- suppressWarnings(as.numeric(ip))
  if (is.na(ip_num) && grepl("\\.", ip)) {
    x <- strsplit(ip, "\\.")[[1]]
    ip_num <- as.numeric(x[1]) + as.numeric(x[2]) / 3
  }
  if (nzchar(era)) parts <- c(parts, paste("ERA", era))
  if (!is.na(bf) && bf > 0) {
    if (!is.na(k)) parts <- c(parts, sprintf("K%% %.1f", 100 * k / bf))
    if (!is.na(bb)) parts <- c(parts, sprintf("BB%% %.1f", 100 * bb / bf))
  }
  if (!is.na(ip_num) && ip_num > 0) {
    if (!is.na(k)) parts <- c(parts, sprintf("k9 %.1f", 9 * k / ip_num))
    if (!is.na(bb)) parts <- c(parts, sprintf("bb9 %.1f", 9 * bb / ip_num))
  }
  paste(parts, collapse = " ")
}

extract_teams <- function(feed) {
  home <- feed$gameData$teams$home$id == team_id
  box <- feed$liveData$boxscore$teams
  orange <- if (home) box$home else box$away
  opp <- if (home) box$away else box$home
  list(orange = orange, opp = opp, orange_is_home = home)
}

extract_hitters <- function(players, prospects) {
  rows <- list()
  for (pl in players) {
    st <- pl$stats$batting %||% list()
    pa <- as_int(st$plateAppearances %||% 0)
    pos <- pl$position$abbreviation %||% ""
    bo <- as_int(pl$battingOrder %||% NA)
    appeared <- !is.na(bo) || pa > 0 || as_int(st$atBats %||% 0) > 0 || as_int(st$baseOnBalls %||% 0) > 0 || as_int(st$runs %||% 0) > 0
    if (!appeared) next
    if (pos == "P" && pa == 0) next
    slot <- ifelse(is.na(bo), 99L, bo %/% 100)
    seq <- ifelse(is.na(bo), 0L, bo %% 100)
    name <- pl$person$fullName %||% ""
    pr <- prospects[[paste0("id:", pl$person$id)]] %||% prospects[[normalize_name(name)]]
    show_name <- paste(trimws(pos), html_escape(name))
    if (!is.null(pr)) {
      show_name <- paste(trimws(pos), paste0("<span class='prospect-name'>", html_escape(name), "</span>"))
    }
    rows[[length(rows) + 1]] <- list(
      slot = slot, seq = seq, person_id = pl$person$id, is_prospect = !is.null(pr),
      display_name = show_name,
      plain_name = paste(trimws(pos), name),
      pa = pa,
      ab = as_int(st$atBats %||% 0), r = as_int(st$runs %||% 0),
      h = format_xbh(st$hits %||% 0, st$doubles %||% 0, st$triples %||% 0, st$homeRuns %||% 0),
      bb = as_int(st$baseOnBalls %||% 0), k = as_int(st$strikeOuts %||% 0), sb = as_int(st$stolenBases %||% 0),
      season = season_hitter(pl$seasonStats$batting %||% list()),
      game_line = sprintf("%s %d-%d, %d R, %d BB, %d K", paste(trimws(pos), name), as_int(st$hits %||% 0), as_int(st$atBats %||% 0), as_int(st$runs %||% 0), as_int(st$baseOnBalls %||% 0), as_int(st$strikeOuts %||% 0))
    )
  }
  if (length(rows) == 0) return(rows)
  slot_vec <- vapply(rows, function(r) as_int(r[["slot"]], 99L), integer(1))
  seq_vec <- vapply(rows, function(r) as_int(r[["seq"]], 0L), integer(1))
  rows[order(slot_vec, seq_vec)]
}

extract_pitchers <- function(players, prospects, cache) {
  rows <- list()
  for (pl in players) {
    st <- pl$stats$pitching %||% list()
    if (length(st) == 0 || is.null(st$inningsPitched)) next
    hand_res <- person_details(pl$person$id, cache)
    cache <- hand_res$cache
    hand <- hand_res$detail$pitchHand
    pref <- if (hand == "L") "LHP" else if (hand == "R") "RHP" else "P"
    name <- pl$person$fullName %||% ""
    pr <- prospects[[paste0("id:", pl$person$id)]] %||% prospects[[normalize_name(name)]]
    show_name <- paste(pref, html_escape(name))
    if (!is.null(pr)) show_name <- paste(pref, paste0("<span class='prospect-name'>", html_escape(name), "</span>"))

    swstr <- st$swingingStrikes %||% ""
    gb <- st$groundOuts %||% ""
    rows[[length(rows) + 1]] <- list(
      person_id = pl$person$id, is_prospect = !is.null(pr),
      display_name = show_name,
      plain_name = paste(pref, name),
      ip = st$inningsPitched %||% "0.0", h = as_int(st$hits %||% 0), r = as_int(st$runs %||% 0),
      er = as_int(st$earnedRuns %||% 0), bb = as_int(st$baseOnBalls %||% 0),
      k = as_int(st$strikeOuts %||% 0), bf = st$battersFaced %||% "", swstr = swstr, gb = gb,
      season = season_pitcher(pl$seasonStats$pitching %||% list()),
      game_line = sprintf("%s IP, %d H, %d ER, %d BB, %d K", st$inningsPitched %||% "0.0", as_int(st$hits %||% 0), as_int(st$earnedRuns %||% 0), as_int(st$baseOnBalls %||% 0), as_int(st$strikeOuts %||% 0))
    )
  }
  list(rows = rows, cache = cache)
}

tango_lookup <- local({
  map <- NULL
  function(inning, half, outs, base, diff) {
    if (is.null(map)) {
      df <- read.csv(tango_path, stringsAsFactors = FALSE)
      key <- paste(df$inning, df$half, df$outs, df$base, df$scoreDiffHome, sep = "|")
      map <<- setNames(df$weHome, key)
    }
    inning <- max(1L, min(9L, as_int(inning)))
    outs <- max(0L, min(2L, as_int(outs)))
    base <- max(0L, min(7L, as_int(base)))
    diff <- max(-10L, min(10L, as_int(diff)))
    k <- paste(inning, half, outs, base, diff, sep = "|")
    as_num(map[[k]], 0.5)
  }
})

base_mask_from_runners <- function(runners, phase = c("start", "end"), keep_out = FALSE) {
  phase <- match.arg(phase)
  mask <- 0L
  for (r in runners %||% list()) {
    mov <- r$movement %||% list()
    if (phase == "start") {
      b <- tolower(mov$start %||% "")
      if (b %in% c("1b", "first")) mask <- bitwOr(mask, 1L)
      if (b %in% c("2b", "second")) mask <- bitwOr(mask, 2L)
      if (b %in% c("3b", "third")) mask <- bitwOr(mask, 4L)
    } else {
      if (!keep_out && isTRUE(mov$isOut)) next
      b <- tolower(mov$end %||% "")
      if (b %in% c("1b", "first")) mask <- bitwOr(mask, 1L)
      if (b %in% c("2b", "second")) mask <- bitwOr(mask, 2L)
      if (b %in% c("3b", "third")) mask <- bitwOr(mask, 4L)
    }
  }
  mask
}

compute_wpa <- function(feed, orange_is_home, debug = FALSE) {
  plays <- feed$liveData$plays$allPlays %||% list()
  home_prev <- 0L
  away_prev <- 0L
  out <- list()

  for (pl in plays) {
    about <- pl$about %||% list()
    inn <- as_int(about$inning %||% 1)
    half <- tolower(about$halfInning %||% "top")
    outs_after <- as_int(pl$count$outs %||% 0)
    runners <- pl$runners %||% list()
    outs_on_play <- if (!is.null(pl$result$outs)) as_int(pl$result$outs) else sum(vapply(runners, function(x) isTRUE(x$movement$isOut), logical(1)))
    outs_before <- max(0L, outs_after - outs_on_play)

    home_after <- as_int(pl$result$homeScore %||% home_prev)
    away_after <- as_int(pl$result$awayScore %||% away_prev)

    base_before <- base_mask_from_runners(runners, "start")
    base_after <- base_mask_from_runners(runners, "end")

    diff_before <- home_prev - away_prev
    diff_after <- home_after - away_after

    we_home_before <- tango_lookup(inn, half, outs_before, base_before, diff_before)
    if (outs_after >= 3) {
      if (half == "top") {
        half2 <- "bottom"; inn2 <- inn
      } else {
        half2 <- "top"; inn2 <- inn + 1L
      }
      we_home_after <- tango_lookup(inn2, half2, 0, 0, diff_after)
    } else {
      we_home_after <- tango_lookup(inn, half, outs_after, base_after, diff_after)
    }

    we_o_before <- if (orange_is_home) we_home_before else 1 - we_home_before
    we_o_after <- if (orange_is_home) we_home_after else 1 - we_home_after

    batter <- pl$matchup$batter$fullName %||% "Unknown"
    event <- pl$result$event %||% pl$result$description %||% "play"
    moment <- sprintf("%s%d: %s %s — Orange WE %d%%→%d%% (%+.0f%%)",
                      toupper(substr(half, 1, 1)), inn, batter, event,
                      round(100 * we_o_before), round(100 * we_o_after), round(100 * (we_o_after - we_o_before)))

    item <- list(moment = moment, wpa_orange = we_o_after - we_o_before,
                 debug = list(inning = inn, half = half, outs_before = outs_before, outs_after = outs_after,
                              base_before = base_before, base_after = base_after,
                              score_before = c(home = home_prev, away = away_prev),
                              score_after = c(home = home_after, away = away_after)))
    out[[length(out) + 1]] <- item
    home_prev <- home_after; away_prev <- away_after
  }

  if (length(out) == 0) return(list())
  ord <- order(vapply(out, function(x) abs(x$wpa_orange), numeric(1)), decreasing = TRUE)
  best <- out[head(ord, 3)]
  if (debug) {
    for (b in best) cat("DEBUG_WPA:", b$moment, "| state=", toJSON(b$debug, auto_unbox = TRUE), "\n")
  }
  best
}

render_card <- function(feed, hitters, pitchers, moments, outfile) {
  teams <- feed$gameData$teams
  linescore <- feed$liveData$linescore$teams
  home_name <- teams$home$name %||% "Home"
  away_name <- teams$away$name %||% "Away"
  home_runs <- as_int(linescore$home$runs %||% 0)
  away_runs <- as_int(linescore$away$runs %||% 0)
  status <- feed$gameData$status$detailedState %||% "Final"
  venue <- feed$gameData$venue$name %||% ""

  headline <- sprintf("%s %d, %s %d", away_name, away_runs, home_name, home_runs)
  meta <- sprintf("%s • %s", status, venue)

  hitter_rows_html <- paste(vapply(hitters, function(h) {
    cls <- if (h$seq > 0) "player-name indent" else "player-name"
    sprintf("<tr><td class='left %s'>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td class='left season'>%s</td></tr>",
            cls, h$display_name, h$pa, h$ab, h$r, h$h, h$bb, h$k, h$sb, html_escape(h$season %||% ""))
  }, character(1)), collapse = "")

  pitcher_rows_html <- paste(vapply(pitchers, function(p) {
    sprintf("<tr><td class='left'>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td class='left season'>%s</td></tr>",
            p$display_name, p$ip, p$h, p$r, p$er, p$bb, p$k, p$bf %||% "", p$swstr %||% "", p$gb %||% "", html_escape(p$season %||% ""))
  }, character(1)), collapse = "")

  moments_rows_html <- paste(vapply(moments, function(m) sprintf("<li>%s</li>", html_escape(m$moment)), character(1)), collapse = "")
  tpl <- paste(readLines(template_path, warn = FALSE), collapse = "\n")
  tpl <- sub("\\{\\{headline\\}\\}", html_escape(headline), tpl)
  tpl <- sub("\\{\\{meta_line\\}\\}", html_escape(meta), tpl)
  tpl <- sub("\\{\\{hitters_rows\\}\\}", hitter_rows_html, tpl)
  tpl <- sub("\\{\\{pitchers_rows\\}\\}", pitcher_rows_html, tpl)
  tpl <- sub("\\{\\{moments_rows\\}\\}", moments_rows_html, tpl)

  html_file <- tempfile(fileext = ".html")
  writeLines(tpl, html_file)

  file.copy(css_path, file.path(dirname(html_file), "boxscore_card.css"), overwrite = TRUE)

  chrome_bin <- Sys.getenv("CHROMOTE_CHROME", "")
  if (nzchar(chrome_bin)) cat(sprintf("Chromote browser: %s\n", chrome_bin))
  chrome_args <- Sys.getenv("CHROMOTE_CHROME_ARGS", "")
  if (nzchar(chrome_args)) cat(sprintf("Chromote args: %s\n", chrome_args))

  b <- ChromoteSession$new()
  on.exit(try(b$close(), silent = TRUE), add = TRUE)

  # Load page robustly: Page.loadEventFired can be missed if listener attaches too late.
  b$Page$enable()
  b$Runtime$enable()
  b$Page$navigate(paste0("file://", normalizePath(html_file)))

  ready <- FALSE
  for (i in 1:50) {
    st <- try(
      b$Runtime$evaluate(
        "(() => document.readyState === 'complete' && !!document.querySelector('#card'))()",
        returnByValue = TRUE
      )$result$value,
      silent = TRUE
    )
    if (!inherits(st, "try-error") && isTRUE(st)) {
      ready <- TRUE
      break
    }
    Sys.sleep(0.1)
  }
  if (!ready) {
    stop("Chromote: timed out waiting for #card to be ready")
  }

  rect <- b$Runtime$evaluate("(() => { const el=document.querySelector('#card'); const r=el.getBoundingClientRect(); return {x:r.x,y:r.y,width:Math.max(1,r.width),height:Math.max(1,r.height),scale:window.devicePixelRatio||1}; })()", returnByValue = TRUE)$result$value
  clip <- list(x = rect$x, y = rect$y, width = rect$width, height = rect$height, scale = rect$scale)
  png <- b$Page$captureScreenshot(format = "png", clip = clip, fromSurface = TRUE)$data
  writeBin(jsonlite::base64_dec(png), outfile)
}

build_prospect_lines <- function(hitters, pitchers) {
  lines <- list()
  for (h in hitters) {
    if (!isTRUE(h$is_prospect)) next
    lines[[length(lines) + 1]] <- list(game = h$game_line, season = h$season %||% "")
  }
  for (p in pitchers) {
    if (!isTRUE(p$is_prospect)) next
    lines[[length(lines) + 1]] <- list(game = paste(p$plain_name, p$game_line), season = p$season %||% "")
  }
  lines
}

build_post_text <- function(feed, lines, orange_is_home) {
  ls <- feed$liveData$linescore$teams
  status <- feed$gameData$status$detailedState %||% "Final"
  orange_runs <- if (orange_is_home) as_int(ls$home$runs %||% 0) else as_int(ls$away$runs %||% 0)
  opp_runs <- if (orange_is_home) as_int(ls$away$runs %||% 0) else as_int(ls$home$runs %||% 0)
  first <- sprintf("%s: DSL Giants Orange %d, Opp %d", status, orange_runs, opp_runs)
  fmt <- function(items, with_season = TRUE) {
    body <- vapply(items, function(x) if (with_season && nzchar(x$season)) paste(x$game, "|", x$season) else x$game, character(1))
    paste(c(first, body), collapse = "\n")
  }
  items <- lines
  txt <- fmt(items, TRUE)
  while (nchar(txt, type = "chars") > max_chars && length(items) > 0) {
    items <- items[-length(items)]
    txt <- fmt(items, TRUE)
  }
  if (nchar(txt, type = "chars") > max_chars) txt <- fmt(items, FALSE)
  txt
}

main <- function() {
  override_gamepk <- Sys.getenv("OVERRIDE_GAMEPK", "")
  override_date <- Sys.getenv("OVERRIDE_DATE", "")
  dry_run <- Sys.getenv("DRY_RUN", "0") == "1"
  force_repost <- Sys.getenv("FORCE_REPOST", "0") == "1"
  force_post <- Sys.getenv("FORCE_POST", "0") == "1"
  debug_wpa <- Sys.getenv("DEBUG_WPA", "0") == "1"
  force_any_post <- force_repost || force_post

  state <- load_state()
  prospects <- load_prospects()
  cache <- load_cache()

  today <- Sys.Date()
  if (nzchar(override_gamepk)) {
    cat(sprintf("Using OVERRIDE_GAMEPK=%s\n", override_gamepk))
    games <- list(list(gamePk = override_gamepk, gameDate = ""))
  } else {
    games <- fetch_schedule(as.character(today - 2), as.character(today), override_date)
  }

  if (length(games) == 0) {
    cat("No target games found.\n")
    state$last_run_iso <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    save_json(state_path, state)
    save_json(player_cache_path, cache)
    return(invisible(NULL))
  }

  games <- games[order(vapply(games, function(g) g$gameDate %||% "", character(1)), decreasing = TRUE)]

  posted_count <- 0L
  terminal_seen <- FALSE

  for (target in games) {
    game_pk <- as.character(target$gamePk %||% "")
    if (!nzchar(game_pk)) next

    feed <- tryCatch(fetch_feed(game_pk), error = function(e) {
      cat(sprintf("Skipping game %s after feed error: %s\n", game_pk, conditionMessage(e)))
      NULL
    })
    if (is.null(feed)) next

    terminal <- status_class(feed)
    if (is.null(terminal)) {
      cat(sprintf("Game %s not terminal yet; skipping.\n", game_pk))
      next
    }
    terminal_seen <- TRUE

    if (is.null(state$posted_games[[game_pk]])) {
      state$posted_games[[game_pk]] <- list(posted_final = FALSE, posted_suspended = FALSE, last_status = terminal)
    }
    gstate <- state$posted_games[[game_pk]]

    already <- if (terminal == "Final") isTRUE(gstate$posted_final) else isTRUE(gstate$posted_suspended)
    if (already && !force_any_post) {
      cat(sprintf("Already posted for game %s (%s); set FORCE_REPOST=1 or FORCE_POST=1 to override.\n", game_pk, terminal))
      gstate$last_status <- terminal
      gstate$last_seen_iso <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
      state$posted_games[[game_pk]] <- gstate
      next
    }

    tms <- extract_teams(feed)
    hitters <- extract_hitters(tms$orange$players %||% list(), prospects)
    pitch_res <- extract_pitchers(tms$orange$players %||% list(), prospects, cache)
    pitchers <- pitch_res$rows
    cache <- pitch_res$cache
    moments <- compute_wpa(feed, tms$orange_is_home, debug_wpa)

    render_card(feed, hitters, pitchers, moments, out_image)

    prospect_lines <- build_prospect_lines(hitters, pitchers)
    post_text <- build_post_text(feed, prospect_lines, tms$orange_is_home)
    alt <- "DSL Giants Orange game box score with batting, pitching, and key win-probability moments."

    if (!dry_run) {
      auth <- bs_auth(Sys.getenv("BSKY_HANDLE"), Sys.getenv("BSKY_APP_PASSWORD"))
      bs_post(text = post_text, images = c(out_image), images_alt = c(alt), auth = auth)
      cat(sprintf("Posted game %s (%s) to Bluesky.\n", game_pk, terminal))
    } else {
      cat(sprintf("DRY_RUN enabled for game %s; skipping Bluesky post.\n", game_pk))
      cat(post_text, "\n")
    }

    gstate$last_status <- terminal
    gstate$last_seen_iso <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    if (terminal == "Final") gstate$posted_final <- TRUE
    if (terminal == "Suspended") gstate$posted_suspended <- TRUE
    state$posted_games[[game_pk]] <- gstate
    posted_count <- posted_count + 1L
  }

  if (!terminal_seen) cat("No Final/Suspended games found in current window.\n")
  cat(sprintf("Processed %d games; posted %d updates.\n", length(games), posted_count))

  state$last_run_iso <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  save_json(state_path, state)
  save_json(player_cache_path, cache)
}

main()
