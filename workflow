UNLEASHING CUSTOMER SENTIMENTS

## PROBLEM STATEMENT ##
Develop a solution to analyze unstructured customer feedback from various channels (e.g., emails, social media, surveys) to extract key insights, sentiment, and recurring issues. 
The goal is to transform this feedback into structured data that can inform service improvements and product offerings.

## ACCEPTANCE CRITERIA ##
Extract sentiment (positive, negative, neutral) from customer comments.
Identify common themes or issues mentioned in the feedback (TBC)
Generate actionable insights for improving services.

## STEPS ##

STEP1: DEFINE DATA SOURCES (Channels): 
Social media comments (e.g., Twitter, Facebook)
Survey responses
Call Transcripts
FOR THIS HACKATHON THE SCOPE IS TWITTER
https://www.kaggle.com/datasets/slythe/twitter-scrape-of-the-top-5-banks-in-south-africa?select=full_2021.csv

STEP2: DATA INGESTION
Unstructured Data: 
This could include customer reviews, survey responses, call transcripts, social media comments, or chat logs, usually in the form of text.
Use APIs for social media platforms (e.g., Twitter API) to fetch feedback.
Use tools like BeautifulSoup or Scrapy for gathering data from websites.
Storage: Amazon S3: Store your raw unstructured data (e.g., text files, PDFs, audio files) in S3 buckets.
Amazon Kinesis: For real-time streaming data like live feedback from social media or customer service chatbots.

STEP3: DATA PREPROCESSING AND CLEANISNG
Convert all feedback into a consistent text format:
Remove Noise: duplicates, special characters, whitespace
Normalize text: lowercase
Tokenization: Split text into individual words or phrases and removal of stop words, if necessary.
This can be handled using AWS Lambda for simple event-driven data preprocessing tasks (e.g., removing noise, converting formats).

STEP4: DATA STORAGE INTO STRUCTURED (PARQUET) FORMAT FOR FURTHER ANALYSIS
Parquet is a structured data format. It is a columnar storage file format designed for efficient data storage and retrieval, especially with large datasets.

STEP5: ANALYSIS ON DATA (using BEDROCK)
(a) Sentiment Analysis
Use Bedrock's pre-built large language models (LLMs) to perform sentiment analysis on customer feedback.
This will classify each piece of feedback as positive, negative, or neutral and give it a sentiment score.
(b) Entity Recognition (NER)
Leverage Bedrock’s ability to recognize entities in feedback, such as product names, service mentions, and customer-specific keywords (e.g., "credit card," "loan," "customer support").
(c) Summarization
Use text summarization models in Bedrock to create concise summaries for long feedback responses. This is helpful for reducing large feedback into more manageable insights.
(d) Topic Modeling
Bedrock’s models can also extract common topics or themes from the feedback, giving you structured categories like "customer service," "loan processing time," or "mobile banking issues."

STEP6: STRUCTURE THE OUTPUT:
Once the Bedrock model returns insights (sentiment, entities, summaries, topics), organize this data into a structured format such as a CSV or JSON file.
Each row/entry can contain:
Feedback_ID (unique identifier)
Sentiment Score (positive, neutral, negative)
Entities Identified (e.g., "loan", "customer service")
Summary (shortened version of feedback)
Topic (e.g., "mobile app," "customer service")

STEP7: STORE AND QUERY STRUCTURED DATA:
Storage Options:
Store the structured data in Amazon DynamoDB (NoSQL) or Amazon RDS (Relational database) for efficient querying and reporting.
Querying and Insights:
Use Amazon Athena to query the structured data directly from S3 if you choose to store the results in CSV or JSON format.
This can be used to perform analytics and generate reports on customer feedback trends.

STEP8: Visualization and Reporting:
Use Amazon QuickSight to create visualizations like:
Sentiment trends over time (positive vs. negative feedback).
Most common topics or issues reported by customers.
Entity-based insights (e.g., which products or services are most frequently mentioned).
This can help generate actionable insights for improving customer service or product offerings.

Example Workflow:
Store Feedback: Upload customer feedback data to Amazon S3.
Preprocessing: Clean and normalize the data using AWS Lambda.
Analysis with Bedrock:
Run sentiment analysis, entity recognition, and topic modeling using Bedrock models.
Summarize long-form feedback into concise points.
Structured Output: Export the results (e.g., sentiment, topics, summaries) to a structured format (CSV, JSON).
Store & Query: Store structured data in Amazon DynamoDB or query it directly from S3 using Athena.
Visualization: Use Amazon QuickSight to build visual reports showing key insights.
This workflow is streamlined and allows you to quickly extract valuable insights from unstructured feedback with minimal setup and customization, leveraging the power of pre-built Bedrock models.

----------------------------------------------------------------------------------------------

Sentiment Analysis, 
Topic Modeling, 
Named Entity Recognition (NER), 
Trend Analysis, 
Recommendation System, 

1. Sentiment Analysis:
Goal: Determine whether the tweets express positive, negative, or neutral sentiments.
Bedrock Integration: You can use pre-trained models like those offered by Amazon Titan, Cohere, or Anthropic Claude to classify the sentiment of each tweet.
Approach: Pass the cleaned base_tweet text through a sentiment analysis model, and aggregate results to understand overall customer sentiment.
Example:
Percentage of positive, negative, or neutral tweets.
Sentiment trend over time (e.g., customer satisfaction based on tweets over different days).

### 1. Sentiment Analysis
df['sentiment'] = df['cleaned_tweet'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
sentiment_counts = df['sentiment'].value_counts()

# Plot Sentiment Distribution
plt.figure(figsize=(6,4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution')
plt.show()


def analyze_sentiment(tweet_text):
    prompt = f"Analyze the sentiment of the following tweet:\nTweet: {tweet_text}\nAnswer as 'positive', 'negative', or 'neutral'."
    
    response = bedrock_client.invoke_model(
        modelId='amazon.titan',  # Assuming Titan is used for sentiment analysis
        body={
            "input_text": prompt
        },
        contentType='application/json'
    )
    
    # Extract the model's response
    result = response['results'][0]['output_text']
    return result.strip().lower()  # 'positive', 'negative', or 'neutral'

df_non_spam['sentiment'] = df_non_spam['cleaned_tweet'].apply(analyze_sentiment)
df_non_spam[['id', 'cleaned_tweet', 'sentiment']].to_csv('non_spam_tweets_with_sentiment.csv', index=False)



2. Topic Modeling:
Goal: Identify common themes or topics discussed in the tweets.
Bedrock Integration: Use a model capable of performing unsupervised learning for clustering or a topic modeling approach like LDA (Latent Dirichlet Allocation).
Approach: Automatically categorize tweets into common topics (e.g., "customer service", "online banking", "fraud issues").
Example:

Show which topics are generating the most conversation among users.
Identify new emerging topics over time.

### 2. Topic Modeling (Using LDA)

# Vectorize the tweets
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.95, min_df=2)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_tweet'])

# Fit LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Display Topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(lda, tfidf_feature_names, 5)


3. Named Entity Recognition (NER):
Goal: Extract important entities such as locations, people, organizations, or dates mentioned in the tweets.
Bedrock Integration: Use a language model that can extract entities from text, such as Amazon Titan or Cohere models.
Approach: Recognize entities like banks, services, or locations in customer tweets.
Example:

Identify the most frequently mentioned bank products or services.
Extract complaints tied to specific regions or locations.

### 3. Named Entity Recognition (NER)
df['entities'] = df['cleaned_tweet'].apply(lambda x: ner_model(x))

# Display NER results
for index, row in df[['cleaned_tweet', 'entities']].head(5).iterrows():
    print(f"Tweet: {row['cleaned_tweet']}")
    print(f"Entities: {row['entities']}")
    print()


4. Trend Analysis:
Goal: Monitor how customer sentiment or tweet activity changes over time.
Bedrock Integration: Use a model to analyze the progression of sentiments, keywords, or topics over different time periods.
Approach: Analyze spikes in activity or sentiment to correlate with events like product launches, outages, or promotions.
Example:

Track sentiment trends before and after a major banking event (e.g., a new feature launch).
Identify seasonal trends in customer complaints or praises.
### 4. Trend Analysis
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['sentiment_score'] = df['sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else -1 if x == 'NEGATIVE' else 0)

# Sentiment trend over time
trend_df = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.lineplot(x='date', y='sentiment_score', data=trend_df)
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=45)
plt.show()

5. Recommendation System:
Goal: Provide recommendations to customers based on their tweets (e.g., services or products they might benefit from).
Bedrock Integration: Use machine learning models to recommend products or services based on tweet content.
Approach: Based on the sentiment or topic of a tweet, recommend relevant banking services or solutions.
Example:

If a customer complains about high transaction fees, recommend low-fee accounts.
If a customer tweets about fraud concerns, suggest fraud prevention services.

### 5. Recommendation System (Simple Keyword Matching)
recommendations = {
    "high fees": "We recommend switching to our low-fee accounts.",
    "fraud": "Check out our advanced fraud detection services.",
    "customer service": "Contact our premium support for better service."
}

def recommend_service(tweet):
    for keyword, recommendation in recommendations.items():
        if keyword in tweet:
            return recommendation
    return "No specific recommendation"

df['recommendation'] = df['cleaned_tweet'].apply(recommend_service)

# Display a few recommendations
df[['cleaned_tweet', 'recommendation']].head(5)
