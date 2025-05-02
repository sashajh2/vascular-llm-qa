from sentence_transformers import SentenceTransformer

from retriever.retrieval import retrieve_docs

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
question = "A 65-year old male with a history of hypertension, hyperlipidemia, and tobacco abuse presents to the vascular surgery clinic with complaints of pain in his legs. Patient states that the pain is crampy in quality, mostly in his calf muscles, it starts after he walks 4 blocks, and is worse on the right as compared to the left. He notices that when he stops walking, the pain resolves. This pain is keeping him from being able to participate in social events and spend quality time with his family.\n\nWhat is the name of the symptom the patient is experiencing?"

relevant_docs = retrieve_docs(question, include_relevant=True, top_k=4)
print(f"Relevant Document {relevant_docs}")
irrelevant_docs = retrieve_docs(question, include_relevant=False, top_k=4)
print(f"Irrelevant Document {irrelevant_docs}")