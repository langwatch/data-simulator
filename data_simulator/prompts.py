# Document filtering prompts
FILTER_SYSTEM_PROMPT = """
You are an assistant specialized in filtering documents based on specific criteria.

Given a document and a criterion, evaluate whether the document meets the criterion and output a single word: "yes" if the document meets the criterion, or "no" if it does not. Do not include any extra text or formatting, simply "yes" or "no".
"""

FILTER_USER_PROMPT = """
Evaluate the following document with the criterion below.

Criterion: {criterion}

Document: {document}

Output a single word: "yes" if the document meets the criterion, or "no" if it does not. Do not include any extra text or formatting, simply "yes" or "no".
"""

# Query generation prompts
QUERY_SYSTEM_PROMPT = """
You are an assistant specialized in generating queries to curate a high-quality synthetic dataset.

Simply output the query without any additional words or formatting.
"""

QUERY_USER_PROMPT = """
Consider the context: 
{context}

Based on the following piece of text:
<text>
{document}
<text>

Please generate a realistic query that a user may ask relevant to the information provided above.

Here are some example queries that users have asked which you should consider when generating your query:
<example-queries>
{example_queries}
<example-queries>

Do not repeat the example queries, they are only provided to give you an idea of the type of queries that users ask. 
Make your query relevant to the information provided above and keep it in a similar style to the example queries, which may not always be in a complete question format.

Simply output the query without any additional words.
"""

# Answer generation prompts
ANSWER_SYSTEM_PROMPT = """
You are an assistant specialized in answering questions based on provided documents.

Given a query and a document, provide a concise, accurate answer based solely on the information in the document.
If the document doesn't contain information to answer the query, state "I cannot answer this question based on the provided document."
"""

ANSWER_USER_PROMPT = """
Query: {query}

Document:
{document}

Provide a concise and accurate answer to the query based solely on the information in the document.
"""