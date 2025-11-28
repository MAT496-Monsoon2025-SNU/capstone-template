Template for creating and submitting MAT496 capstone project.

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show your creativity. Think about all the problems which you can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, We can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, we can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs. Let your imagination run free.


-------------------------

# Project report Template

## Title: [BizMentor AI: AI Business Consultant]

## Overview

[BizScale AI is an intelligent business consulting agent designed to analyse a business, identify the bottlenecks in its pipeline, and generate personalized, step-by-step recommendations using the combined knowledge of three top business coaches: Dan Martell, Sam Ovens, and Alex Hormozi.

The project collects expert knowledge (YouTube transcripts, blogs, case studies, etc.) from the three coaches, preprocesses it, and stores it in a searchable knowledge base. Using RAG (Retrieval-Augmented Generation), the assistant retrieves relevant insights from these experts and applies them to the user’s business situation.

The system is implemented using LangGraph, where each coach acts as an independent analysis node. The results from all three nodes are merged to produce a final, structured consulting report. This report includes:

  -Key bottlenecks
  -Root causes
  -Action plan
  -Strategic recommendations
  -KPIs to track

BizScale AI essentially simulates what a founder would experience if they had access to three top-tier business consultants advising them at once.]

## Reason for picking up this project

This project uses LLMs to solve a real problem: business owners need clear, personalized advice without hiring expensive consultants. Building an AI Business Consultant allows me to apply almost every concept taught in MAT496 in a practical and meaningful way.

How it Aligns With Course Topics:

• Prompting:
Used to create distinct personas for Dan Martell, Sam Ovens, and Alex Hormozi, and to guide the assistant to give structured, consultant-style answers.

• Structured Output:
Final reports follow clear sections like Snapshot, Bottlenecks, Root Causes, Action Plan, and KPIs. Ensures predictable and professional output.

• Semantic Search:
Used to prepare and organize expert content from blogs/transcripts for retrieval (even in simple form).

• Retrieval-Augmented Generation (RAG):
Each coach’s analysis uses both user input + retrieved expert knowledge to give more accurate recommendations.

• Tool Calling / MCP:
Retrieval functions act as “tools” called inside the graph to fetch relevant expert insights.

• LangGraph (State, Nodes, Graph):
Core architecture of the system:

  -State holds business info, KPIs, and analyses.

  -Nodes represent each coach’s analysis.

  -Graph merges all outputs into a final consulting report.

## Plan

Plan  
I plan to execute these steps to complete my project.

[DONE] Step 1: Set up project structure and environment  
- Create virtual environment and install required dependencies.  
- Fork the MAT496 repository and create a new project folder.  
- Add initial folder structure: src/, data/, notebooks/, docs/.  
- Create an empty main Python file (e.g., src/business_consultant_graph.py).

[TODO] Step 2: Implement basic LangGraph with a single coach (no RAG)  
→ Uses: Prompting, LangGraph (State, Nodes, Graph)  
- Define the LangGraph state (business_description, goal, analyses, final_report).  
- Implement one node for a single business coach persona using a system prompt.  
- Build minimal StateGraph: START → coach analysis → final_report → END.  
- Add a simple CLI runner to take input and produce output.

[TODO] Step 3: Test and refine the single-coach flow  
→ Uses: Prompting, Structured Output  
- Test the graph with sample business descriptions.  
- Improve the quality of responses by refining system & human prompts.  
- Add basic structure to the output (diagnosis + recommendations).  

[TODO] Step 4: Extend the graph to all three coaches  
→ Uses: Prompting, Structured Output, LangGraph (parallel multi-node flow)  
- Add persona prompts for Dan Martell, Sam Ovens, Alex Hormozi.  
- Implement individual nodes for each coach.  
- Modify the graph to run all three analyses and store results in state.  
- Update final_report node to merge their perspectives into one structured report.

[TODO] Step 5: Collect expert content for each coach  
→ Uses: Semantic Search (preparation step for RAG)  
- Identify data sources (YouTube transcripts, blogs, articles, case studies).  
- Download text content into the data/ directory.  
- Organize content by coach for efficient retrieval later.

[TODO] Step 6: Preprocess and chunk the expert content  
→ Uses: Semantic Search (chunking enables vector indexing)  
- Clean the raw transcripts (remove timestamps, noise, filler text).  
- Chunk into meaning-preserving segments.  
- Save as JSON or .txt for embedding.

[TODO] Step 7: Build a simple RAG pipeline  
→ Uses: Semantic Search, Retrieval Augmented Generation (RAG)  
- Generate embeddings for all chunks using an embedding model.  
- Store embeddings in a vector store (e.g., FAISS).  
- Implement a retrieval module that returns top-k relevant chunks per coach.  
- Test retrieval independently with sample queries.

[TODO] Step 8: Integrate RAG into the LangGraph  
→ Uses: RAG, Prompting, LangGraph (State + Node updates), Tool Calling (if retrieval tool exposed)  
- Update each coach node to include retrieved expert knowledge in prompts.  
- Combine user context + retrieved text → coach-specific analysis.  
- Ensure the state stores enriched, evidence-based analyses.  
- If using MCP or tool-calling module, integrate retrieval as a tool call.

[TODO] Step 9: Polish the final report structure and formatting  
→ Uses: Structured Output  
- Define a consistent multi-section consulting report (Snapshot, Bottlenecks, Root Causes, Action Plan, KPIs).  
- Enforce structured output using templates or schema-guided prompting.  
- Ensure clarity, professionalism, and consultant-like tone.

[TODO] Step 10: Prepare scenario tests and evaluate the pipeline  
→ Uses: Prompting, RAG, LangGraph Debugging (with LangSmith optionally)  
- Create multiple business examples for testing.  
- Run full workflow and save outputs in docs/.  
- Adjust prompts, retrieval parameters, or node order based on results.






## Conclusion:

I had planned to achieve {this this}. I think I have/have-not achieved the conclusion satisfactorily. The reason for your satisfaction/unsatisfaction.

----------

# Added instructions:

- This is a `solo assignment`. Each of you will work alone. You are free to talk, discuss with chatgpt, but you are responsible for what you submit. Some students may be called for viva. You should be able to each and every line of work submitted by you.

- `commit` History maintenance.
  - Fork this respository and build on top of that.
  - For every step in your plan, there has to be a commit.
  - Change [TODO] to [DONE] in the plan, before you commit after that step. 
  - The commit history should show decent amount of work spread into minimum two dates. 
  - **All the commits done in one day will be rejected**. Even if you are capable of doing the whole thing in one day, refine it in two days.  
 
 - Deadline: Nov 30, Sunday 11:59 pm


# Grading: total 25 marks

- Coverage of most of topics in this class: 20
- Creativity: 5
  