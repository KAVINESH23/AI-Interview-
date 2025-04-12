# INTELLIGENT INTERVIEW AUTOMATION USING  PROMPT- BASED LLM FRAMEWORK FOR ADAPTIVE CANDIDATE EVALUATION
   An AI-powered interview automation system that dynamically generates skill-based questions using LLMs (via Groq API) and evaluates candidate responses with structured prompt engineering and binary scoring, enabling scalable and intelligent assessments.

# PURPOSE
   Uses LLMs for real-time generation, not a fixed question bank
  
# TECHNICAL STACK
  1.**LARGE LANGUAGE MODEL(LLMs)** - DeepSeek, Gemma2-9B-IT, LLaMA-3.3-70B hosted via Groq API
  
  2.**LANGCHAIN**                  - PromptTemplates
  
  3.**PROMPT ENGINEERING**         - Prompt engineering involves crafting specific instructions to guide LLMs to generate relevant, domain-specific questions without hints or explanations.
  
  4.**STREAMLIT**                  - Frontend  UI
  
  5.**PYTHON**                     - Core logic

# WORKFLOW 


# EVALUATION
 
Evaluation Strategy

Candidate responses are evaluated in two phases:

1. **Rubric-Based Scoring (1–4)**  
   - Correctness, Depth, and Relevance  
   - Each criterion scored out of 4 (Max score: 12)

2. **Binary Scoring**  
   - Final evaluation based on threshold (≥9/12 = Pass, else Fail)
   - Results are stored and displayed after each response


