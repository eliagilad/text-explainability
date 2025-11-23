import sys

sys.argv = ["myscript.py", "--foo", "bar"]

import external_repos.HEDGE.bert.hedge_main_bert_imdb_debug

args = parser.parse_args(["--data", "myfile.txt"])