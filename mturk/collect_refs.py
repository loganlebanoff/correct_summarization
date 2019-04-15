import hashlib

import pandas as pd

source_articles = "data/article/articles.txt"
summarizers = ["abs-rl-rerank", "bottom-up", "dca", "novel", "pg", "reference"]


def filter_str(strn: str):
    strip = strn.rstrip()
    dquot = strip.replace('"', "&quot;")
    squot = dquot.replace("'", "&apos;")
    backt = squot.replace("`", "&apos;")
    sep_t = backt.replace("\t", "||")
    
    return sep_t


def _parse_articles_to_df():
    d = []
    with open(source_articles, "r") as f:
        for line in f.readlines():
            hashed_line = hashlib.sha256(line.encode("utf-8")).hexdigest()
            d.append({
                "article_id": hashed_line,
                "article_content": filter_str(line),
            })
    
    article_df = pd.DataFrame(columns=["article_id", "article_content"], data=d)
    return article_df
    
    
def _parse_summaries_to_df(article_df):
    summaries_ls = []
    for major_idx, summarizer in enumerate(summarizers):
        with open(f"data/{summarizer}/source_indices.txt", "r") as sources, \
             open(f"data/{summarizer}/summaries.txt", "r") as summaries:
            summarizer = summarizer.replace("-", "_")
            for idx, (srcs, summ) in enumerate(zip(sources, summaries)):
                if major_idx == 0: summaries_ls.append({})
                
                summaries_ls[idx].update({
                    f"{summarizer}_srcs": filter_str(srcs),
                    f"{summarizer}_summ": filter_str(summ),
                })

    summary_df = pd.DataFrame(summaries_ls)
    article_df = article_df.join(summary_df)
    
    return article_df
    
    
def main():
    article_df = _parse_articles_to_df()
    article_df = _parse_summaries_to_df(article_df)
    
    import csv
    
    article_df.set_index(keys=["article_id"], inplace=True, drop=True)
    article_df.to_csv("mturk/data/articles.csv", quoting=csv.QUOTE_ALL)
    article_df.to_hdf("mturk/data/articles.hdf", key="articles")
    

if __name__ == "__main__":
    main()
