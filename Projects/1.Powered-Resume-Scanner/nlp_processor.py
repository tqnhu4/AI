# nlp_processor.py
import spacy
import re
import json

# Load SpaCy model (only needs to be loaded once)
try:
    nlp = spacy.load("en_core_web_sm") # Use "en_core_web_md" for better accuracy
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

# Load skill list from JSON file (example)
# Create a data/skills.json file with content like: {"programming": ["Python", "Java", "C++"], "web_dev": ["HTML", "CSS", "JavaScript"]}
# Or define a simple list for demo purposes:
DEFAULT_SKILLS = {
    "programming": ["Python", "Java", "C++", "Go", "JavaScript", "SQL", "R"],
    "web_development": ["HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Streamlit"],
    "cloud": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes"],
    "data_science": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "R", "SQL"],
    "project_management": ["Agile", "Scrum", "Jira", "Trello"],
    "languages": ["English", "Spanish", "French", "Vietnamese"],
    "soft_skills": ["Communication", "Teamwork", "Problem-solving", "Leadership", "Adaptability", "Critical Thinking"]
}

# Load skills from a JSON file if it exists
try:
    with open('data/skills.json', 'r', encoding='utf-8') as f:
        ALL_SKILLS = json.load(f)
except FileNotFoundError:
    ALL_SKILLS = DEFAULT_SKILLS
    print("data/skills.json not found, using default skills.")
    
except json.JSONDecodeError:
    ALL_SKILLS = DEFAULT_SKILLS
    print("Error decoding data/skills.json, using default skills.")


def extract_skills(text):
    """
    Extracts skills from resume text by matching with a predefined skill list.
    """
    doc = nlp(text.lower()) # Convert to lowercase for easier comparison
    found_skills = set()
    for category, skills_list in ALL_SKILLS.items():
        for skill in skills_list:
            if skill.lower() in doc.text: # Find skill in text
                found_skills.add(skill)
    return list(found_skills)


def analyze_structure(text):
    """
    Analyzes the basic structure of the resume (e.g., presence of section headers).
    This is a very basic analysis; you can expand it.
    """
    sections = {
        "contact_info": False,
        "summary_objective": False,
        "experience": False,
        "education": False,
        "skills": False,
        "projects": False,
        "awards": False,
        "references": False
    }

    # Define common keywords for each section
    keywords = {
        "contact_info": ["contact", "email", "phone", "linkedin"],
        "summary_objective": ["summary", "objective", "profile", "about me"],
        "experience": ["experience", "work history", "employment"],
        "education": ["education", "academic", "university"],
        "skills": ["skills", "abilities", "competencies", "technical"],
        "projects": ["projects", "portfolio"],
        "awards": ["awards", "honors", "achievements"],
        "references": ["references available", "references upon request"]
    }

    text_lower = text.lower()
    for section, kws in keywords.items():
        for kw in kws:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                sections[section] = True
                break
    return sections

def get_readability_score(text):
    """
    Calculates a readability score (e.g., Flesch-Kincaid Readability Test).
    Use Textatistic or Textblob for more detail;
    here, we simplify by counting sentences and words.
    """
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    num_words = len([token for token in doc if token.is_alpha]) # Only count alphabetic words

    if num_words == 0:
        return 0, "No words found for readability analysis."

    # Flesch-Kincaid Reading Ease (simplified)
    # 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    # This is the full formula. For simplicity, we'll use average sentence length.
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else num_words

    # A simple rule of thumb:
    # Overly long sentences can decrease readability.
    # This is just a simple example, not a full F-K score calculation.
    if avg_sentence_length > 20:
        return 60, "Sentences might be too long." # Lower score
    elif avg_sentence_length > 15:
        return 75, "Good sentence length." # Medium score
    else:
        return 90, "Excellent sentence length." # High score

def generate_suggestions(text, job_description=None):
    """
    Analyzes the resume and generates improvement suggestions.
    """
    suggestions = []

    # 1. Check overall length
    if len(text.split()) < 200: # Example: resume is too short
        suggestions.append("Consider adding more details to your experience, projects, or education sections to provide a comprehensive view of your qualifications.")
    elif len(text.split()) > 800: # Example: resume is too long
        suggestions.append("Your resume might be too long. Try to condense information, focus on relevance, and use bullet points for conciseness. Aim for 1-2 pages.")

    # 2. Check resume structure
    structure = analyze_structure(text)
    missing_sections = [
        section.replace('_', ' ').title()
        for section, present in structure.items() if not present
        and section in ["experience", "education", "skills", "contact_info"] # Essential sections
    ]
    if missing_sections:
        suggestions.append(f"Essential sections appear to be missing or unclear: {', '.join(missing_sections)}. Ensure these sections are clearly defined.")
    if not structure["summary_objective"]:
        suggestions.append("Consider adding a concise 'Summary' or 'Objective' statement at the beginning to quickly highlight your value proposition.")

    # 3. Skill analysis
    extracted_skills = extract_skills(text)
    if not extracted_skills:
        suggestions.append("No specific skills were identified. Ensure your technical and soft skills are clearly listed, preferably in a dedicated 'Skills' section.")
    else:
        suggestions.append(f"Identified skills: {', '.join(extracted_skills)}.")
        # Example: suggest missing skills if a JD is provided (advanced part)
        if job_description:
            job_doc = nlp(job_description.lower())
            required_skills = extract_skills(job_description) # Extract skills from JD
            missing_job_skills = [s for s in required_skills if s not in extracted_skills]
            if missing_job_skills:
                suggestions.append(f"Consider adding or emphasizing the following skills relevant to the job description: {', '.join(missing_job_skills)}.")

    # 4. Readability check
    readability_score, readability_msg = get_readability_score(text)
    if readability_score < 70: # Low score
        suggestions.append(f"Readability might be an issue. {readability_msg} Use simpler language, shorter sentences, and active voice.")
    else:
        suggestions.append(f"Readability looks good. {readability_msg}")

    # 5. Check for action verbs (Advanced)
    # Requires a list of action verbs and more complex parsing
    action_verbs_found = False # Placeholder
    common_action_verbs = ["managed", "developed", "led", "implemented", "analyzed", "created", "optimized", "collaborated"]
    for verb in common_action_verbs:
        if re.search(r'\b' + re.escape(verb) + r'\b', text.lower()):
            action_verbs_found = True
            break
    if not action_verbs_found:
        suggestions.append("Use strong action verbs at the beginning of your bullet points to describe your achievements (e.g., 'Managed', 'Developed', 'Led').")

    # 6. Formatting and Consistency (Advanced)
    # Very difficult to check automatically without deeper parsing (e.g., font, size, bullet points)
    # suggestions.append("Ensure consistent formatting (fonts, sizes, spacing) throughout your resume.")

    return suggestions