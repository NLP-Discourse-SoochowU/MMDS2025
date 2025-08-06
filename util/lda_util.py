# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

def lda_modeling(documents, num_topics):
    # Preprocess the data: Tokenization and vectorization
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(documents)

    # Fit the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Display the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    num_top_words = 10

    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

    # # Print the perplexity of the model
    # print(f"Perplexity: {lda.perplexity(X)}")
    perplexity = lda.perplexity(X)
    return lda, perplexity, vectorizer
