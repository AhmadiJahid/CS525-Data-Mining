from neo4j import GraphDatabase
from openai import OpenAI
import re

# Setup
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "12345678"

client = OpenAI(api_key="sk-097bd9e031674cab87bef5f05e49ed19", base_url="https://api.deepseek.com")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# 1. Use DeepSeek to extract diagnosis name from user input
def extract_diagnosis_name(user_input):
    system_prompt = "Extract only the diagnosis name from the following sentence. Return it as plain text, no explanation, no formatting."
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()

# 2. Query Neo4j for procedures
def get_fuzzy_procedures_for_diagnosis(diagnosis, driver):
    cypher = f"""
    WITH toLower('{diagnosis}') AS input
    MATCH (d:Diagnosis)
    WITH d, apoc.text.sorensenDiceSimilarity(toLower(d.name), input) AS sim
    WHERE sim > 0.7
    WITH d ORDER BY sim DESC LIMIT 1
    MATCH (d)-[:TRIGGERS]->(g:RuleGroup)-[r:RECOMMENDS]->(p:Procedure)
    RETURN d.name AS diagnosis, p.name AS procedure, r.confidence AS confidence
    ORDER BY r.confidence DESC
    """
    with driver.session() as session:
        result = session.run(cypher)
        return result.data()


# 3. Format context string
def format_context(diagnosis, procedure_info_list):
    context_lines = [f"Diagnosis: {diagnosis}"]
    if procedure_info_list:
        context_lines.append("Recommended Procedures:")
        for item in procedure_info_list:
            context_lines.append(f"- {item['procedure']} (confidence: {round(item['confidence'], 2)})")
    else:
        context_lines.append("No procedures found.")
    return "\n".join(context_lines)

# 4. Get LLM output explanation
def generate_llm_response(context, user_prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "The following context includes a medical diagnosis and procedures associated with it. Based only on this context, explain to the user what the procedures are and why they might be recommended."
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n\nContext:\n{context}"
            }
        ],
        stream=False
    )
    return response.choices[0].message.content

# 5. Main interface function
def get_chatbot_response(user_input):
    print(f"💬 User input: {user_input}")

    # Step 1: Extract diagnosis
    diagnosis = extract_diagnosis_name(user_input)
    print(f"🔍 Extracted diagnosis: {diagnosis}")

    # Step 2: Get procedures
    procedures = get_fuzzy_procedures_for_diagnosis(diagnosis, driver)

    # Step 3: Format context
    context = format_context(diagnosis, procedures)
    print(f"\n📄 Context:\n{context}")

    # Step 4: Generate final LLM explanation
    final_answer = generate_llm_response(context, user_input)
    print(f"\n📝 LLM Answer:\n{final_answer}")
    return final_answer

get_chatbot_response("I was diagnosed with malignant neoplasm of ovary. What would you suggest me as a procedure?")
