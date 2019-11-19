Crawl filings from https://www.sec.gov/Archives/edgar/full-index/

- Per year and quarter, get the master.ixs (or a zipped version)
- For each filing (one per line), check if the form is in ['10-K', '10-Q', '8-K']
- If yes, get the URL, e.g. edgar/data/1001115/0001193125-09-251202.txt, then construct the filing index URL, e.g. 
https://www.sec.gov/Archives/edgar/data/1001115/000119312509251202/0001193125-09-251202-index.html
- get the file, parse the table <table class="tableFile" summary="Document Format Files">; for each tr, check if the Type matches "EX-10.*"
- if yes, get the html file as the contract!
