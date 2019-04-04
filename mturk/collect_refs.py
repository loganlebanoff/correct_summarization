import hashlib

import pandas as pd

source_articles = "data/article/articles.txt"
summarizers = ["abs-rl-rerank", "bottom-up", "dca", "novel", "pg", "reference"]


def _parse_articles_to_df():
    d = []
    with open(source_articles, "r") as f:
        for line in f.readlines():
            hashed_line = hashlib.sha256(line.encode("utf-8")).hexdigest()
            d.append({
                "article_id": hashed_line,
                "article_content": line.rstrip(),
            })
    
    article_df = pd.DataFrame(columns=["article_id", "article_content"], data=d)
    article_df.set_index(keys=["article_id"], inplace=True, drop=True)
    article_df.to_csv("mturk/data/articles.csv")
    article_df.to_hdf("mturk/data/articles.hdf", key="articles")
    
    
def _parse_summaries_to_df():
    for summarizer in summarizers:
        d = []
        with open(f"data/{summarizer}/source_indices.txt", "r") as src, \
             open(f"data/{summarizer}/summaries.txt", "r") as summaries:
            pass


def main():
    _parse_articles_to_df()
    pass


if __name__ == "__main__":
    main()
