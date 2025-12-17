### Rule-Based NLP Agent: Sentence Reading Problem

**Problem Overview**  
The Sentence Reading problem tests an agent’s ability to **understand natural language at a symbolic level**. Given a simple English sentence and a related question, the agent must extract the correct answer by interpreting grammatical structure, semantic roles, and question intent (e.g., *who*, *what*, *where*, *when*, *how*).

Unlike statistical NLP models, this task emphasizes **explicit reasoning and linguistic structure** rather than learned embeddings.

**Approach**  
I implemented a **rule-based sentence reading agent** using a production-system architecture. The agent processes each sentence–question pair and answers questions by applying deterministic If–Then rules grounded in common-sense linguistic reasoning.

The agent uses:
- **spaCy preprocessing** to tokenize and categorize words (verbs, nouns, pronouns, time expressions, locations, quantities, etc.)
- A **working memory** that stores linguistic features and expands as new patterns are encountered
- **Frame-based representations**, where each sentence is represented with slots such as subject, verb, object, time, and location

**Question Handling**
Questions are classified by interrogative type:
- Who
- What
- Where
- When
- How (with subtypes such as *how many*, *how far*, *how long*)

Production rules retrieve the appropriate frame slot based on the question type.  
For example, in sentences with both direct and indirect objects, specialized rules determine which entity to return depending on the question.

**Key Properties**
- Fully deterministic and interpretable reasoning
- Linear-time complexity in sentence length
- High efficiency for known sentence patterns
- Strong performance on structured, unambiguous language
- Limited generalization beyond the predefined rule catalog

**Human Comparison**
The agent mirrors human reasoning for simple, well-structured sentences but lacks the flexibility humans have when encountering unfamiliar linguistic constructions.

**Files**
- `SentenceReadingAgent.py` — Rule-based NLP agent implementation (**omitted for confidentiality; available upon request**)
- `Mini-Project 3 Journal.pdf` — Detailed design, rule examples, performance analysis, and human comparison

---

## Disclaimer
The content (code, report, analyses, writeup, etc.) in this folder are part of the coursework at **Georgia Tech** and are for **demonstration purposes only**. 
Any unauthorized use, reproduction, or distribution may result in a violation of copyright laws and will be subject to appropriate actions.

_**By accessing this folder, you agree to adhere to all copyright policies.**_
