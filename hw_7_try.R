library(tidyverse)
library(gutenbergr)
titles <- c(
  "BTFV",
  "BURG"
)
crimetype <- gutenberg_works(title %in% titles) %>%
  gutenberg_download(meta_fields = "title") %>%
  mutate(document = row_number())
