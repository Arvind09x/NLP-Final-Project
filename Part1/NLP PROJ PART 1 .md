1 Interactive Application (7 \+ 10 \+ 8 \+ 8\)  
Choose a Subreddit of your choice on a socially relevant theme (e.g. geopolitics, public health, AI, etc.). You  
need to scrape at least 15,000 posts, covering a span of minimum 6 months from this subreddit. Use Reddit’s

official API (via PRAW) and store your data in a local database. The Arctic-Shift project also provides data-  
dumps for several archived subreddits. Then, create an interactive application with the following features-  
1.1 . Show aggregate properties of the scraped content’s database, for instance, number of users, number

of posts, number of comments, etc.

1.2 . Identify between 5-20 key topics of conversation in this subreddit and present these in a compre-  
hensible format to the end-users. Each identified topic should have a short descriptive label, its top 5-10

keywords, and its share of total posts; You are free to add new attributes as you see fit.  
1.3 . Distinguish between trending topics and persistent topics. You can do this one of many ways:  
tagging each extracted topic with a label, using different routing for present and persistent pages entirely,  
etc.  
1.4 . For each major topic, identify whether there is agreement / disagreement among users. Your system  
should (i) classify each comment’s stance as broadly support or opposing the dominant position on the topic  
(ii) group users by said stance (iii) provide a short summary of the key arguments made by each side- this  
can be extractive or generative.  
Note: Feel free to make design choices which are unspecified in this spec independently and justify them  
in your report. For instance, the data’s listing type (hot/top/new), cleaning filters applied, definitions of  
persistent vs trending topics, architecture design for the application etc.  
