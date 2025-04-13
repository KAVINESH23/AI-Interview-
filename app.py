import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import base64

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configurations
MODEL_CONFIG = {
    "deepseek-r1-distill-qwen-32b": {
        "temperature": 0.3,
        "max_retries": 5,
        "question_prefix": "SQL question:"
    },
    "gemma2-9b-it": {
        "temperature": 0.5,
        "max_retries": 4,
        "question_prefix": ""
    },
    "llama-3.3-70b-versatile": {
        "temperature": 0.4,
        "max_retries": 3,
        "question_prefix": ""
    }
}
# Define enhanced prompt templates with specified techniques
APTITUDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in generating aptitude and logical reasoning questions. Use this process to create unique and diverse questions:

    **Rules**:
    1. Generate questions only from verified mathematical or logical principles.
    2. Ensure there is a single correct answer for each question.
    3. Avoid subjective interpretations, ambiguous phrasing, or overly complex scenarios.
    4. Include a mix of quantitative and logical reasoning topics.
    5. Do not use a default topic; generate questions from various stages/topics such as:
       - Simple interest/Clocks/Calendars/Blood relations/Time and work/Speed and distance/Percentage/yllogisms/Arrangements (e.g., seating, order)/Logical deductions

    **Instructions**:
    - Focus on the given topic if provided. If no topic is specified, generate questions from different stages/topics listed above.
    - Use clear and concise language for each question.
    - Provide at least one example for clarity.

    **Few-Shot Examples**:

    ** Time and Work**
    Question: "A can complete a task in 10 days, and B can complete the same task in 15 days. How long will it take for both A and B to complete the task together?"
    Answer: 6 days (1 / (1/10 + 1/15) = 6)

    **Output Rules**:
    - Single question per output.
    - No markdown formatting.
    - Avoid using special characters like `**` at the start or end of questions.
    - Include at least one example for clarity.
    """),
])

CODING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert coder. Use the following Chain-of-Thought process to generate a coding problem:
    
    **Process:**
    1. **Understand the Skill**: Identify the core concept or skill ({skill}) the problem should focus on.
    2. **Define the Problem Context**: Create a real-world or abstract scenario where the skill is applied.
    3. **Break Down the Problem**:
       - Step 1: Clearly state the input(s) required for the problem.
       - Step 2: Specify the expected output(s) or goal.
       - Step 3: Outline any constraints or edge cases.
    4. **Provide Examples**: Include at least one example with inputs and outputs to clarify the problem.
    
    **Rules**:
    - Use clear and concise language.
    - Avoid ambiguous terms or overly complex scenarios.
    - Ensure the problem is solvable within a reasonable time frame.
    - Include a mix of basic and advanced concepts to challenge the candidate.
    
    **Example**:
    Skill: Arrays
    Problem Statement: "Given an array of integers, find the two numbers that add up to a specific target value."
    Input: An array of integers and a target value (e.g., [2, 7, 11, 15], target = 9).
    Output: Indices of the two numbers that sum to the target (e.g., [0, 1]).
    Constraints: You may assume there is exactly one solution, and you cannot use the same element twice.
    Example: 
      Input: [2, 7, 11, 15], target = 9
      Output: [0, 1]
    
    **Output Rules**:
    - Single problem per output.
    - No markdown formatting."""),
    ("human", """Generate a coding problem about {skill} with:""")
])

TECHNICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical interviewer for {role} positions. Structure questions using this framework:

For {skill}, analyze through 5 lenses:
1. Core Theory: Fundamental principles, mathematical foundations, domain laws
2. System Dynamics: Interactions between components, failure cascades, feedback loops
3. Implementation Reality: Tradeoffs, debugging nightmares, "it works on my machine" gaps
4. Evolution: Historical solutions → current best practices → emerging alternatives
5. Paradoxes: Situations where standard patterns contradict (e.g., CAP theorem)

**Question Design Protocol**
1. Select 2 knowledge layers → Identify tension/conflict
2. Frame as concrete scenario requiring prioritization
3. Demand explanation of ripple effects

**Examples**

1. Skill: Database Engineering  
   Question: "When would you prioritize ACID compliance over horizontal scalability in a financial system? How does this choice impact disaster recovery strategies?"

**Output Rules**
1. Force candidate to choose between competing priorities
2. Require explanation of second-order consequences
3. Single question (<2 sentences)
4. Ends with '?'
5. No markdown"""),
    
    ("human", """Create a {skill} question using knowledge layers. {question_prefix}""")
])
BEHAVIORAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a behavioral expert. Use the following structured process to generate behavioral questions:
    
    **Process:**
    1. Start with the core skill ({skill}) and identify potential workplace challenges.
    2. Use Chain-of-Thought reasoning to break down the scenario into logical steps:
       - Step 1: Identify the challenge.
       - Step 2: Frame the challenge as a real-world experience.
       - Step 3: Tailor the question to the role ({role}).
    3. Use few-shot learning with diverse examples to guide the generation of questions.
    
    **Few-Shot Examples:**
    Example 1:
    Core Skill: Communication
    Role: Junior Developer
    Question: "Describe a time when you had to explain a technical concept to a non-technical team member. How did you ensure they understood?"
    
    
    **Rules:**
    - Avoid hypothetical scenarios ("Imagine...").
    - Use "Describe a time..." or "Tell me..." format.
    - Adapt the complexity and focus based on the role (e.g., junior vs. senior).
    - Ensure the question is concise (<2 sentences) and ends with '?'.
    
    **Output Rules:**
    - Single question per output.
    - No markdown formatting."""),
    ("human", """Create a behavioral question about {skill} for {role}.
    {question_prefix}""")
])


EVALUATION_PROMPT_TEMPLATE = """
Evaluate the interview answer using these STRICT criteria (1-4 scale):
1. Correctness: 
   - 4: Fully accurate with no technical errors, covers all aspects
   - 3: Mostly correct but missing minor details
   - 2: Partially correct with significant gaps
   - 0: Fundamentally incorrect
2. Depth: 
   - 4: Advanced concepts + examples + trade-offs + edge cases
   - 3: Good detail but missing some aspects
   - 2: Basic explanation only
   - 1: Superficial treatment
3. Relevance: 
   - 4: Directly addresses all parts of question
   - 3: Mostly relevant with minor tangents
   - 2: Partially relevant
   - 1: Off-topic
Checklist for Depth:
- [ ] Includes concrete examples/code snippets
- [ ] Discusses performance implications
- [ ] Mentions trade-offs/limitations
- [ ] Addresses edge cases
Question: {question}
Answer: {answer}
JSON Output Format:
{{
  "rubric": {{
    "correctness": {{"score": 1-4, "reason": "detailed technical evaluation"}},
    "depth": {{"score": 1-4, "reason": "depth analysis with checklist items"}},
    "relevance": {{"score": 1-4, "reason": "relevance check"}}
  }},
  "strengths": ["key strengths"],
  "suggestions": ["specific improvements based on checklist"],
  "total_score": "sum of all scores (3-12)"
}}
"""
EVALUATION_PROMPT = ChatPromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)

def generate_question(role, skill, model_name, existing_questions, stage):
    config = MODEL_CONFIG.get(model_name, {})
    max_retries = config.get("max_retries", 3)
    prefix = config.get("question_prefix", "")
    
    # Select appropriate prompt template based on stage
    if stage == 'aptitude':
        prompt = APTITUDE_PROMPT
    elif stage == 'coding':
        prompt = CODING_PROMPT
    elif stage == 'technical':
        prompt = TECHNICAL_PROMPT
    else:  # behavioral
        prompt = BEHAVIORAL_PROMPT
    
    for attempt in range(max_retries):
        try:
            model = ChatGroq(
                temperature=config.get("temperature", 0.5),
                groq_api_key=GROQ_API_KEY,
                model_name=model_name
            )
            chain = prompt | model
            
            existing_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(existing_questions[-5:])])
            existing_display = f"Existing questions:\n{existing_list}" if existing_questions else "No existing questions"
            
            response = chain.invoke({
                "role": role,
                "skill": skill,
                "existing_questions": existing_display,
                "question_prefix": prefix
            })
            
            raw_question = response.content.strip()
            question = re.sub(r'\<.*?\>', '', raw_question, flags=re.DOTALL)
            question = re.sub(r'\(.*?\)|Note:.*|```.*```', '', question)
            question = re.split(r'\?|```', question)[0].strip() + '?'
            question = question.replace('`', '').strip()
            
            # Prevent duplicates
            question_core = re.sub(r'[^a-zA-Z0-9]', '', question.lower())
            existing_cores = [re.sub(r'[^a-zA-Z0-9]', '', q.lower()) for q in existing_questions]
            if not question.endswith('?'):
                raise ValueError("Missing question mark")
            if question_core in existing_cores:
                raise ValueError("Duplicate question core detected")
            if any(q in question for q in existing_questions):
                raise ValueError("Partial duplicate detected")
                
            return question
        except Exception as e:
            st.toast(f"Retry {attempt+1}/{max_retries}: {str(e)}")
            continue
    return "❌ Failed to generate valid question after multiple attempts"

def evaluate_answer(question, answer, model_name):
    try:
        model = ChatGroq(temperature=0.2, groq_api_key=GROQ_API_KEY, model_name=model_name)
        chain = EVALUATION_PROMPT | model | JsonOutputParser()
        response = chain.invoke({"question": question, "answer": answer})
        
        # Convert scores to integers and calculate totals
        rubric_scores = {
            "correctness": int(response["rubric"]["correctness"]["score"]),
            "depth": int(response["rubric"]["depth"]["score"]),
            "relevance": int(response["rubric"]["relevance"]["score"])
        }
        total_score = sum(rubric_scores.values())
        binary_score = 1 if total_score >= 9 else 0  # Threshold at 9/12
        
        return {
            "rubric_scores": rubric_scores,
            "total_score": total_score,
            "binary_score": binary_score,
            "feedback": "\n".join([
                f"**{k.title()}** ({v}/4): {response['rubric'][k]['reason']}"
                for k, v in rubric_scores.items()
            ]),
            "strengths": response.get("strengths", []),
            "suggestions": response.get("suggestions", [])
        }
    except Exception as e:
        return {
            "error": str(e),
            "rubric_scores": {"correctness": 0, "depth": 0, "relevance": 0},
            "total_score": 0,
            "binary_score": 0,
            "feedback": "Evaluation failed",
            "strengths": [],
            "suggestions": []
        }

def get_text_download_link(content, filename):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Results</a>'

# Session State Initialization
if "job_role" not in st.session_state:
    st.session_state.update({
        "job_role": "Select",
        "skill": "Aptitude",
        "questions": [],
        "current_index": 0,
        "answers": [],
        "evaluations": [],
        "rubric_totals": [],
        "binary_scores": [],
        "num_questions": 10,
        "ready_next": False,
        "selected_model": "gemma2-9b-it",
        "prev_config": ("", "", "Aptitude", 10, "aptitude"),
        "stage": "aptitude"
    })

with st.sidebar:
    st.header("🛠 Interview Configuration")
    
    # Question range selector
    st.session_state.num_questions = st.number_input(
        "Number of Questions:",
        min_value=1,
        max_value=15,
        value=10,
        step=1
    )
    
    model_options = list(MODEL_CONFIG.keys())
    st.session_state.selected_model = st.selectbox(
        "🤖 AI Model:",
        options=model_options,
        index=model_options.index(st.session_state.selected_model)
    )
    job_role = st.selectbox(
        "💼 Job Role:",
        options=["Select", "Data Analyst", "Machine Learning Engineer"])
    
    stage = st.selectbox(
        "Stage:",
        options=["aptitude", "coding", "technical", "behavioral"])
    
    # Skill selection logic
    skill = "Aptitude"
    if stage == 'coding':
        coding_skills = ["Python", "SQL", "DSA", "R"]
        skill = st.selectbox(
            "📚 Coding Skill:",
            options=["Select"] + coding_skills,
            index=0 if st.session_state.skill not in coding_skills 
                   else coding_skills.index(st.session_state.skill)+1)
    elif stage != 'aptitude':
        skills_map = {
            "Data Analyst": ["Python", "SQL", "Excel", "Data Visualization"],
            "Machine Learning Engineer": ["Python","R","SQL","Machine Learning", "Deep Learning"]
        }
        skills = ["Select"] + skills_map.get(job_role, [])
        skill = st.selectbox(
            "📚 Skill:",
            options=skills,
            index=0 if st.session_state.skill not in skills 
                   else skills.index(st.session_state.skill))

    # Reset session on config change
    current_config = (st.session_state.selected_model, job_role, skill, 
                     st.session_state.num_questions, stage)
    if current_config != st.session_state.prev_config:
        st.session_state.update({
            "questions": [],
            "current_index": 0,
            "answers": [''] * st.session_state.num_questions,
            "evaluations": [{} for _ in range(st.session_state.num_questions)],
            "rubric_totals": [0] * st.session_state.num_questions,
            "binary_scores": [0] * st.session_state.num_questions,
            "ready_next": False,
            "prev_config": current_config,
            "skill": skill,
            "stage": stage
        })

st.title("🧑‍💻 Technical Interview Assistant")

if st.session_state.skill != "Select" or st.session_state.stage == 'aptitude':
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"Question {st.session_state.current_index + 1} of {st.session_state.num_questions}")
    with col2:
        st.caption(f"Model: {st.session_state.selected_model.split('-')[0]}")

    # Question generation
    if len(st.session_state.questions) < st.session_state.num_questions:
        while len(st.session_state.questions) <= st.session_state.current_index:
            new_q = generate_question(
                st.session_state.job_role,
                st.session_state.skill if st.session_state.stage != 'aptitude' else "Aptitude",
                st.session_state.selected_model,
                st.session_state.questions,
                st.session_state.stage)
            st.session_state.questions.append(new_q)

    current_q = st.session_state.questions[st.session_state.current_index]
    st.markdown(f"<h3 style='font-size: 24px;'>{current_q}</h3>", 
                unsafe_allow_html=True)  #    
    answer = st.text_area("Your Answer:", 
                        value=st.session_state.answers[st.session_state.current_index], 
                        height=150,
                        key=f"ans_{st.session_state.current_index}")

    if not st.session_state.ready_next:
        if st.button("Submit Answer"):
            if answer.strip():
                evaluation = evaluate_answer(current_q, answer.strip(), st.session_state.selected_model)
                st.session_state.answers[st.session_state.current_index] = answer.strip()
                st.session_state.evaluations[st.session_state.current_index] = evaluation
                st.session_state.rubric_totals[st.session_state.current_index] = evaluation["total_score"]
                st.session_state.binary_scores[st.session_state.current_index] = evaluation["binary_score"]
                st.session_state.ready_next = True
                st.rerun()

    if st.session_state.ready_next:
        # Calculate cumulative scores
        total_rubric = sum(st.session_state.rubric_totals)
        total_binary = sum(st.session_state.binary_scores)
        max_rubric = (st.session_state.current_index + 1) * 12
        
        st.progress((st.session_state.current_index + 1) / st.session_state.num_questions)
        st.subheader(f"Rubric Score: {total_rubric}/{max_rubric}")
        st.subheader(f"Passed Questions: {total_binary}/{st.session_state.current_index + 1}")
        
        eval_data = st.session_state.evaluations[st.session_state.current_index]
        with st.expander("Detailed Feedback"):
            st.markdown(eval_data["feedback"])
            st.markdown("**Strengths:**")
            for strength in eval_data.get("strengths", []):
                st.markdown(f"- {strength}")
            st.markdown("**Suggestions:**")
            for suggestion in eval_data.get("suggestions", []):
                st.markdown(f"- {suggestion}")

        if st.session_state.current_index < st.session_state.num_questions - 1:
            if st.button("Next Question ➡️"):
                st.session_state.current_index += 1
                st.session_state.ready_next = False
                st.rerun()
        else:
            st.success("🎉 Interview Complete!")
            report_content = f"Final Scores:\nRubric: {total_rubric}/{(st.session_state.num_questions)*12}"
            report_content += f"\nPassed Questions: {total_binary}/{st.session_state.num_questions}\n\n"
            
            for i, (q, a, e) in enumerate(zip(st.session_state.questions, 
                                            st.session_state.answers,
                                            st.session_state.evaluations)):
                report_content += f"Question {i+1}:\n{q}\n\nAnswer:\n{a}\n\n"
                report_content += f"Correctness: {e['rubric_scores']['correctness']}/4\n"
                report_content += f"Depth: {e['rubric_scores']['depth']}/4\n"
                report_content += f"Relevance: {e['rubric_scores']['relevance']}/4\n"
                report_content += f"Total: {e['total_score']}/12 | Passed: {'Yes' if e['binary_score'] else 'No'}\n"
                report_content += "-"*50 + "\n\n"
            
            st.markdown(get_text_download_link(report_content, "interview_report.txt"), 
                       unsafe_allow_html=True)
