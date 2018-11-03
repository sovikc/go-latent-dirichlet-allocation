# Latent dirichlet allocation
An LDA model for discovering topics that occur in a collection of documents

Topic Modeling is a technique to extract the hidden topics from large volumes of text. Latent Dirichlet Allocation is a popular algorithm for topic modeling. The challenge, however, is how to extract good quality of topics that are clear, segregated and meaningful. This depends heavily on the quality of text preprocessing and the strategy of finding the optimal number of topics.

LDA’s approach to topic modeling is, it considers each document a collection of topics in a certain proportion. And each topic a collection of keywords, again, in a certain proportion. In LDA, each document may be viewed as a mixture of various topics where each document is considered to have a set of topics that are assigned to it via LDA.

Once you provide the algorithm with the number of topics, all it does it to rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.

A topic is a collection of dominant keywords that are typical representatives. Just by looking at the keywords, one can identify what the topic is all about.

The following are key factors to obtaining good segregation topics:
1. Quality of text processing.
2. Variety of topics the text talks about.
3. Choice of topic modeling algorithm.
4. Number of topics fed to the algorithm.
5. Algorithm's tuning parameters.
