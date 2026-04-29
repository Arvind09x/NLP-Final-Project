1 Conversation System 25  
Build a question-answering system on top of the Reddit content repository you collected in Part 1 of this  
project. The system must use the Retrieval-Augmented Generation (RAG) framework: given a user query,  
it should retrieve relevant posts and comments from your database, and pass them as context to a language  
model to generate an answer.  
You are free to choose your chunking strategy, embedding model, and vector store. Some free and open  
options include ChromaDB or FAISS for the vector store, and sentence-transformers/all-MiniLM-L6-v2 for  
embeddings. For the language models, connect to at least two different LLM endpoints — free-tier options  
include the Groq API, Google AI Studio, and Together AI.  
For evaluating your system, construct a ground-truth evaluation set of at least 15 question-answer pairs,  
written by you based on your reading of the subreddit. Your questions should cover a range of types: (i)  
factual questions about the community (ii) opinion-summary questions (e.g. ”what do users think about  
X?”) (iii) at least two adversarial questions whose answers are not present in the corpus. Use this set to  
produce a comparative performance report across your chosen LLMs.  
Your evaluation must include at minimum: ROUGE-L, BERTScore, and manual scores / binary flags for  
faithfulness (report this as a percentage across your test set). Present your results in a table comparing  
models across these metrics, and include a short qualitative analysis of where each model succeeds and fails  
on your specific corpus.  
2 Indian Langauge Translation Task 25  
Evaluate the capability of your chosen LLMs to work in at least one Indian language. You may choose any  
scheduled Indian language (Hindi, Tamil, Bengali, Marathi, etc.). Clearly state your chosen language at the  
start of this section. Design a translation or generation task of your choice. Three acceptable task formats

are-  
1\. Translation: Ask the model to translate a set of English Reddit posts or summaries into the target

language

2\. Cross-lingual QA: Pose questions in the Indian language and evaluate whether the model answers cor-  
rectly using the English corpus as context

3\. Summarisation: Ask the model to produce a summary of a topic discussions directly in the Indian language  
These tasks are suggested, but you are not limited to these; any task designed to evaluate translation  
capabilities will be accepted. Whichever format you choose, prepare a reference output set (ground truth)  
of at least 20 examples. References may be produced using a combination of human judgment, Google  
Translate as a baseline, or a native speaker if available.  
Suggested metrics for this task include chrF (character n-gram F-score), BERTScore with a multilingual

model, manual scores of 5–10 outputs on a scale for grammatical correctness (fluency) and meaning preser-  
vation (adequacy).

Deliberately include difficult test cases — posts containing code-mixed English/Hindi/language of choice,  
Reddit-specific slang (e.g. ”AITA”, ”NTA”, community in-jokes), and named entities. Analyse how each  
model handles these edge cases specifically.

1

3 Note on Bias Detection 9  
Prepare a note on bias detection capability of the LLM using your own probes. You are expected to design  
your own probes and report findings with evidence drawn from your corpus, examples include: Is there  
bias in the data? Is the model deliberately smudging the bias? Are the model answers getting biased by the  
inherent bias of Reddit demographics?  
4 Note on Ethics 8

Prepare a reflective note on the ethical dimensions of collecting, storing, and querying Reddit data, address-  
ing the following-

(i) Did you encounter situations where personal information was compromised despite anonymization? Re-  
flect on scenarios from your own dataset where combining username, posting history, and post content could

allow re-identification of a real individual  
(ii) Does your system violate the ”Right to be Forgotten” – i.e. what happens if a user deletes a post  
that is in your training set? Is full compliance realistic for a production RAG system?  
Note: Feel free to make design choices which are unspecified  
