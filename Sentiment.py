import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("dataset/test_file.csv")

analyzer = SentimentIntensityAnalyzer()

IDLink = []
SentimentTitle = []
SentimentHeadline = []

for n in range(df.shape[0]):
    title = df.iloc[n,1]
    headline = df.iloc[n,2]
    title_analyzed = analyzer.polarity_scores(title)
    headline_analyzed = analyzer.polarity_scores(headline)
    IDLink.append(df.iloc[n,0])
    SentimentTitle.append(title_analyzed['compound'])
    SentimentHeadline.append(headline_analyzed['compound'])
df["IDLink"] = IDLink
df["SentimentTitle"] = SentimentTitle
df["SentimentHeadline"] = SentimentHeadline
df[['IDLink','SentimentTitle','SentimentHeadline']].to_csv('final_submission.csv',index=False)