from neo4j import GraphDatabase
from openai import OpenAI


# Replace with your Aura connection details
URI = "neo4j+s://104abb61.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "uSV92xXhnyT0B2nLnSTOaqna4FfQxJHHbdRmqdE___A"
CSV_PATH1 = "Mock_Rule_Set_1.csv"
CSV_PATH2 = "Mock_Rule_Set_2.csv"

# Connect to Neo4j
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
client = OpenAI(api_key="sk-097bd9e031674cab87bef5f05e49ed19", base_url="https://api.deepseek.com")

def extract_facts(prompt):
    system_prompt = """
    You extract structured patient info from a sentence. Output JSON with these keys:
    - gender: "Male" or "Female"
    - age: number
    - symptoms: list of strings
    - lab_tests: an OBJECT where each key is a lab test name and each value is the result (e.g., "Hemoglobin": "Low")
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def expand_symptoms(symptom):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a medical NLP assistant. Given a symptom, return 3 to 5 similar or related medical symptoms as a JSON list. Do not include definitions."
            },
            {
                "role": "user",
                "content": f"Expand this symptom: {symptom}"
            }
        ],
        stream=False
    )
    import re, json
    cleaned = re.sub(r"```json|```", "", response.choices[0].message.content).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return [symptom]  # fallback
    
def expand_all_symptoms(symptoms):
    all_expanded = set()
    for s in symptoms:
        expanded = expand_symptoms(s)
        all_expanded.update(expanded)
    return list(all_expanded)

def query_neo4j(cypher_string, driver):
    with driver.session() as session:
        result = session.run(cypher_string)
        return [record.data() for record in result]

def get_treatment_info(diagnosis_name, driver):
    cypher = f"""
    MATCH (d:Diagnosis {{name: '{diagnosis_name}'}})
    -[:TREATED_BY]->(r:Rule)
    
    OPTIONAL MATCH (r)-[:RECOMMENDS {{type: 'Prescription'}}]->(m:Prescription)
    OPTIONAL MATCH (r)-[:RECOMMENDS {{type: 'Procedure'}}]->(p:Procedure)
    
    RETURN
        collect(DISTINCT m.name) AS prescriptions,
        collect(DISTINCT p.name) AS procedures
    """
    return query_neo4j(cypher, driver)

def generate_fuzzy_cypher_query(gender, age, expanded_symptoms, lab_tests):
    symptom_list_str = ', '.join([f'"{s}"' for s in expanded_symptoms])

    symptom_match = f"""
    OPTIONAL MATCH (s:Symptom)-[:PART_OF]->(r)
    WHERE s.name IN [{symptom_list_str}]
    """

    lab_match = "\n".join([
        f"""OPTIONAL MATCH (l:LabTest {{name: '{name}'}})-[rel{j}:PART_OF {{result: '{res}'}}]->(r)"""
        for j, (name, res) in enumerate(lab_tests.items())
    ])

    patient_profile = ""
    if gender or age:
        profile_clauses = []
        if gender:
            profile_clauses.append(f"p.gender = '{gender}'")
        if age:
            if age >= 60:
                profile_clauses.append("p.age_group = '>60'")
            elif age >= 40:
                profile_clauses.append("p.age_group = '40-60'")
            else:
                profile_clauses.append("p.age_group = '<40'")
        patient_profile = f"""
        OPTIONAL MATCH (p:PatientProfile)-[:PART_OF]->(r)
        WHERE {' AND '.join(profile_clauses)}
        """

    return f"""
MATCH (r:Rule)
{symptom_match}
{lab_match}
{patient_profile}
OPTIONAL MATCH (r)-[:IMPLIES]->(d:Diagnosis)
WITH r, d, count(DISTINCT s) AS symptom_score, count(DISTINCT p) AS profile_score,
     {len(lab_tests)} AS expected_lab_count,
     size([rel IN [{', '.join(f'rel{j}' for j in range(len(lab_tests)))}] WHERE rel IS NOT NULL]) AS lab_score
WHERE symptom_score + lab_score + profile_score > 0
RETURN d.name AS diagnosis, r.rule_id AS rule, symptom_score, lab_score, profile_score,
       symptom_score + lab_score + profile_score AS total_score
ORDER BY total_score DESC
LIMIT 5
"""

def generate_final_response(context, prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "The following context contains a diagnosis based on the information patient provided, and prescriptions related to that diagnosis. Give a medical answer based on this context. Do not provide additional treatment outside of the context."},
            {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
        ],
        stream=False
    )
    return response.choices[0].message.content

# user_input = "I'm a 67-year-old male. I feel tired and dizzy. My hemoglobin is low. What might I have?"

def get_chatbot_response(user_input):
    print(user_input)
    # Step 1: Extract patient info
    parsed = extract_facts(user_input)
    print("🔍 Parsed:", parsed)

    import json, re
    cleaned = re.findall(r'({(?:.|\n)*})', parsed, re.DOTALL)[0].strip()
    print(cleaned)
    facts = json.loads(cleaned)

    gender = facts.get("gender")
    age = facts.get("age")
    symptoms = facts.get("symptoms", [])
    lab_tests = facts.get("lab_tests", {})

    # Step 2: Expand symptoms
    expanded_symptoms = expand_all_symptoms(symptoms)
    expanded_symptoms = [s.capitalize() for s in expanded_symptoms]
    print("🔁 Expanded symptoms:", expanded_symptoms)

    # Step 3: Generate fuzzy Cypher
    cypher_query = generate_fuzzy_cypher_query(gender, age, expanded_symptoms, lab_tests)
    print("\n📡 Cypher:\n", cypher_query)

    # Step 4: Query KG
    diagnosis_results = query_neo4j(cypher_query, driver)
    print("\n🧠 Diagnoses:", diagnosis_results)

    if diagnosis_results:
        top_diagnosis = diagnosis_results[0]['diagnosis']
        treatments = get_treatment_info(top_diagnosis, driver)
    else:
        top_diagnosis = "No strong match found"
        treatments = []

    # Step 5: Compose context + LLM summary
    context = f"Diagnosis: {top_diagnosis}\n Treatments: {[t for t in treatments]}"

    final_answer = generate_final_response(context, user_input)
    print("\n📝 Final Answer:\n", final_answer)
    return final_answer