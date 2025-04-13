# INTELLIGENT INTERVIEW AUTOMATION USING  PROMPT- BASED LLM FRAMEWORK FOR ADAPTIVE CANDIDATE EVALUATION
   An AI-powered interview automation system that dynamically generates skill-based questions using LLMs (via Groq API) and evaluates candidate responses with structured prompt engineering and binary scoring, enabling scalable and intelligent assessments.

# PURPOSE
   The aim of this initiative is to transform conventional hiring methods by using AI-powered automation to remove human bias and inefficiencies. By employing prompt-based LLMs, the system creates customized interview questions on the spot, adjusting to various job positions and candidate answers without depending on set question lists. This method guarantees equitable, scalable, and smart assessments of candidates while greatly minimizing the manual work needed in hiring.  
   
# TECHNICAL STACK
  1.**LARGE LANGUAGE MODEL(LLMs)** - DeepSeek, Gemma2-9B-IT, LLaMA-3.3-70B hosted via Groq API
  
  2.**LANGCHAIN**                  - PromptTemplates
  
  3.**PROMPT ENGINEERING**         - Prompt engineering involves crafting specific instructions to guide LLMs to generate relevant, domain-specific questions without hints or explanations.
  
  4.**STREAMLIT**                  - Frontend  UI
  
  5.**PYTHON**                     - Core logic
  
# PROMPTING TECHNIQUES
  **Few-Shot Prompting**        : Aptitude and Logical reasoning 
  
  **Chain-of-Thought**          : Coding questions

  **Knowledge-Based**           : Technical stage

  **Adaptive Graph of Thought** : Behavioral assessments

# WORKFLOW 
   Overall view of the project and here i have attached the workflow image
https://github.com/KAVINESH23/AI-Interview-/blob/main/images/Workflow%20Diagram.png?raw=true
# EVALUATION
 
Evaluation Strategy

Candidate responses are evaluated in two phases:

1. **Rubric-Based Scoring (1–4)**  
   - Correctness, Depth, and Relevance  
   - Each criterion scored out of 4 (Max score: 12)

2. **Binary Scoring**  
   - Final evaluation based on threshold (≥9/12 = Pass, else Fail)
   - Results are stored and displayed after each response


# FUTURE ENHANCEMENT

   1.**Candidate Feedback Reports**: Generate AI-powered personalized improvement insights post-interview.

   2.**Integration with HR Tools**: Seamless ATS/CRM connectivity for end-to-end hiring automation.

   3.**LLM Based Evaluation**     : To avoid hallucinations of LLM.

# STREAMLIT 
   Output of Gemma Model 
   1. **Question Generation**  https://github.com/KAVINESH23/AI-Interview-/blob/main/images/g1.png?raw=true
   2. **Candidate Response**   https://github.com/KAVINESH23/AI-Interview-/blob/main/images/g2.png?raw=true

 
# PROJECT FILES
   app.py               -         Core Streamlit app integrating Groq API, LangChain orchestration, prompt engineering workflows, interview 
                                  automation logic, and UI components.
                           
   requirements.txt     -         Python dependencies
   
   .env                 -        It use the Groq API to host the LLMs 
   
   images               -        Contains visual documentation of workflow ,streamlit ouput g1 for Question Generation and g2 for Candidate Response
                                              

