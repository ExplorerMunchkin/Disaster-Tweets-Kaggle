setwd("SET YOUR WORKING DIRECTORY")
getwd()


if (!("rvest" %in% installed.packages())) {
  install.packages("rvest")
}
library(rvest)


url <- "https://www.noslang.com/dictionary/"
df <- data.frame()

my_letters <- c('1',letters)

for (letter in my_letters){
  overview_url <- paste(url, letter, sep = "")
  slang_rows <-  (read_html(overview_url) %>% html_nodes('div.dictionary-word'))
  for (r in slang_rows){
    key <- html_nodes(r, 'dt') %>% html_text()
    key <- substr(key,1,nchar(key)-2)
    value <-  html_nodes(r, 'dd') %>% html_text()
    value <- tolower(value)
    temp_df <- data.frame(Key = key,
                          Value = value)
    
    if (identical(df, data.frame())) {
      df <- temp_df
    } else {
      df <- rbind(df, temp_df)
    }
  }
}

# df <- tibble::rowid_to_column(df, "ID")
write.csv(df,"abbreviations.csv", row.names = TRUE, eol = "\r\n", fileEncoding = "UTF-8") 




