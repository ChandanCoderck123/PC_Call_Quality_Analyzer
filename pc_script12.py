"""
pc_script12.py
--------------
PC Call Analyzer - Complete Script for Patient Counsellor Call Management

This script manages the entire workflow for PC (Patient Counsellor) call analysis:
1. Creates and initializes database tables
2. Loads canonical reference scripts (Script 1)
3. Generates multiple variants (v1-v5) for each script section using GPT
4. Provides a foundation for call analysis and quality assurance

USAGE:
  # 1) Ensure database tables exist
  python pc_script12.py init_db

  # 2) Insert/Update Script 1 (canonical reference)
  python pc_script12.py load_base_ref

  # 3) Generate Script 2 variants (v1..v5 for each section via GPT)
  python pc_script12.py gen_variants

ENVIRONMENT VARIABLES:
  DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
  OPENAI_API_KEY=sk-... (Required for GPT variant generation)
  GPT_MODEL=gpt-4o-mini (Optional override)
  EMBED_MODEL=text-embedding-3-large (Not used here but kept for parity)
"""

import os
import json
import time
import argparse
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Imports ---
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# --- OpenAI (GPT) Imports with Compatibility Handling ---
try:
    # Try importing the new OpenAI client (for newer versions)
    from openai import OpenAI
    _HAS_NEW_OPENAI = True
except ImportError:
    # Fallback to legacy OpenAI import
    _HAS_NEW_OPENAI = False
    import openai

# ---------------- CONFIGURATION SETTINGS ----------------

# Database connection URL - uses environment variable with fallback
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://qispineadmin:TrOpSnl1H1QdKAFsAWnY@qispine-db.cqjl02ffrczp.ap-south-1.rds.amazonaws.com:5432/qed_prod")

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# ---------------- DATABASE UTILITY FUNCTIONS ----------------

def get_engine() -> Engine:
    """
    Create and return a SQLAlchemy database engine with connection pooling.
    
    Returns:
        Engine: SQLAlchemy engine instance for database operations
    """
    return create_engine(
        DATABASE_URL, 
        pool_pre_ping=True,  # Verify connection before using
        pool_size=5,         # Number of persistent connections
        max_overflow=5       # Maximum number of connections beyond pool_size
    )

def run_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> None:
    """
    Execute a SQL statement without returning results (for INSERT, UPDATE, DELETE).
    
    Args:
        engine: Database engine instance
        sql: SQL query string to execute
        params: Optional parameters for parameterized queries
    """
    with engine.begin() as conn:
        # Use text() to safely handle SQL strings and prevent injection
        conn.execute(text(sql), params or {})

def fetch_all(engine: Engine, sql: str, params: Optional[dict] = None) -> list:
    """
    Execute a SQL query and return all results as a list of dictionaries.
    
    Args:
        engine: Database engine instance
        sql: SQL query string to execute
        params: Optional parameters for parameterized queries
        
    Returns:
        list: List of dictionaries representing each row
    """
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
        # Convert each row to dictionary for easy access
        return [dict(row._mapping) for row in result.fetchall()]

# --------------- DATABASE TABLE DEFINITIONS ------------------

# DDL for the main PC reference table that stores both Script 1 and Script 2 variants
DDL_PC_REF_TABLE = """
CREATE TABLE IF NOT EXISTS pc_ref_table (
    -- Primary identifier for the script version
    script_code TEXT PRIMARY KEY,
    
    -- Base Script 1 sections (canonical reference)
    og_pc_script TEXT,
    og_intro_rapport_building TEXT,
    og_purpose_reassurance TEXT,
    og_pre_consultation_information_gathering TEXT,
    og_conclusion_next_steps TEXT,
    og_pc_cd_handshake TEXT,
    og_cd_pc_handshake TEXT,
    og_onboarding_patient TEXT,

    -- Script 2 variants (v1..v5) for each section - generated via GPT
    -- Intro and Rapport Building variants
    og_intro_rapport_buildingv1 TEXT,
    og_intro_rapport_buildingv2 TEXT,
    og_intro_rapport_buildingv3 TEXT,
    og_intro_rapport_buildingv4 TEXT,
    og_intro_rapport_buildingv5 TEXT,

    -- Purpose and Reassurance variants
    og_purpose_reassurancev1 TEXT,
    og_purpose_reassurancev2 TEXT,
    og_purpose_reassurancev3 TEXT,
    og_purpose_reassurancev4 TEXT,
    og_purpose_reassurancev5 TEXT,

    -- Pre-consultation Information Gathering variants
    og_pre_consultation_information_gatheringv1 TEXT,
    og_pre_consultation_information_gatheringv2 TEXT,
    og_pre_consultation_information_gatheringv3 TEXT,
    og_pre_consultation_information_gatheringv4 TEXT,
    og_pre_consultation_information_gatheringv5 TEXT,

    -- Conclusion and Next Steps variants
    og_conclusion_next_stepsv1 TEXT,
    og_conclusion_next_stepsv2 TEXT,
    og_conclusion_next_stepsv3 TEXT,
    og_conclusion_next_stepsv4 TEXT,
    og_conclusion_next_stepsv5 TEXT,

    -- PC to CD Handshake variants
    og_pc_cd_handshakev1 TEXT,
    og_pc_cd_handshakev2 TEXT,
    og_pc_cd_handshakev3 TEXT,
    og_pc_cd_handshakev4 TEXT,
    og_pc_cd_handshakev5 TEXT,

    -- CD to PC Handshake variants
    og_cd_pc_handshakev1 TEXT,
    og_cd_pc_handshakev2 TEXT,
    og_cd_pc_handshakev3 TEXT,
    og_cd_pc_handshakev4 TEXT,
    og_cd_pc_handshakev5 TEXT,

    -- Patient Onboarding variants
    og_onboarding_patientv1 TEXT,
    og_onboarding_patientv2 TEXT,
    og_onboarding_patientv3 TEXT,
    og_onboarding_patientv4 TEXT,
    og_onboarding_patientv5 TEXT
);
"""

def init_db():
    """
    Initialize the database by creating the pc_ref_table if it doesn't exist.
    
    This function ensures the required table structure is available for storing
    both canonical scripts and their variants.
    """
    engine = get_engine()
    run_sql(engine, DDL_PC_REF_TABLE)
    print(" pc_ref_table ensured/created successfully.")

# --------------- OPENAI GPT HELPER FUNCTIONS ----------------

def _get_client():
    """
    Initialize and return the OpenAI client with proper API key configuration.
    
    Returns:
        OpenAI client instance or None for legacy versions
        
    Raises:
        RuntimeError: If OPENAI_API_KEY is not set in environment
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure your OpenAI API key.")
    
    if _HAS_NEW_OPENAI:
        # Initialize new OpenAI client (v1.0+)
        return OpenAI(api_key=OPENAI_API_KEY)
    else:
        # Set API key for legacy OpenAI version
        openai.api_key = OPENAI_API_KEY
        return None

def gpt_text(prompt: str, temperature: float = 0.6) -> str:
    """
    Generate text using OpenAI GPT model with specified parameters.
    
    Args:
        prompt: The input prompt for GPT
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        
    Returns:
        str: Generated text response from GPT
        
    Raises:
        Exception: If GPT API call fails
    """
    client = _get_client()
    
    # System message to guide GPT behavior - crucial for preserving placeholders
    system_message = "You rewrite text preserving placeholders like [Patient's Last Name], [Doctor's Full Name], etc."
    
    if _HAS_NEW_OPENAI:
        # New OpenAI client syntax (v1.0+)
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    else:
        # Legacy OpenAI client syntax
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message["content"].strip()

# --------------- SCRIPT 1: CANONICAL BASE TEXTS ----------------

# Complete PC Script - the master template for all patient interactions
BASE_PC_SCRIPT = """Goodmorning/Afternoon/Evening, Ma'am/Sir!
Welcome to Qi Spine Clinic! May I know your name, please?
Goodmorning/Afternoon/Evening, my name is [Patient'sName], I have an appointment at [Timeslot].
Yes, Mr./Ms. [Patient's Last Name], I can see your appointment is scheduled with Dr. [Doctor's Full Name] at [Time Slot]. Please, make yourself comfortable. Would you like a glass of water?
Yes, please, thank you. Mr./Ms. [Patient's Last Name], please allow me a minute to allocate a room for you, and I'll be right back. Mr./Ms. [Patient's Last Name], please follow me this way. Please make yourself at ease.
Hello Mr./Ms. [Patient's Last Name] – I'm [PC's Name], your patient counsellor for today. I will be assisting you throughout your journey. Before we begin the consultation, I just need some basic information about you and your condition. May I proceed with a quick pre-assessment?
Yes, sure. How much time will it take? Should not take more than 10 mins. I'll be using our AI Powered system, Dr. QI; that helps us keep everything organized and when the doctor joins you, all relevant information will be readily available to them. That way, you won't have to repeat everything, and it helps personalize your treatment as well as saves time.
Ok, sure. Great, how did you hear about us? I saw an ad on Instagram/facebook, and then someone called to schedule the appointment. Ok. Just to confirm, you weren't referred to us by a doctor, right?
I'll note that down. And may I know your profession? Well, is that really necessary? It helps us understand the kind of daily stresses your spine goes through. For example, if you sit all day, or if you're on your feet a lot. Oh, okay. I work in IT, so it's a lot of sitting. "Right" (PC continues with questions from the app, shows the screen to the patient while doing so, selects the option that comes closest to the patient's response and proceeds with the rest of the questions from Levels 1, 2, 3 & 8. Should the patient be unclear about any question asked, it is important that the PC explains it in simple words in order to make an accurate record in the app about the patient's history)
· Note: When asking the question in the app about medications, the PC also needs to ask about any previous investigations. This is not mentioned in the app and hence the PC needs to make a mental note about it.
(PC fills in the necessary information, maintaining a friendly and attentive demeanour.)
Alright, Mr./Ms. [Patient's Last Name], that's all the information we need. Dr. [Doctor's Full Name] will be with you in just a moment. I'll be right outside if you need anything and will join you again after your consultation to go over your treatment plan.
(Note: Concludes the pre-consultation smoothly, reassuring the patient and setting expectations.)
Dr. [CD's Name], Mr./Ms. [Patient's Last Name] is ready for her consultation. Here's a quick overview:
• Levels: She filled out Levels 2, 3, and 8 in the Dr. QI app. Please review them before entering.
• Case Summary: She's primarily experiencing [Patient's Main Symptom, e.g., lower back pain]. She has previously tried [Previous Treatments taken if any, e.g., over-the-counter pain relievers] with limited success. Her goals are [Patient's Goals, e.g., to reduce pain, improve posture, and return to her daily activities, lift weights etc.].
• Location: She resides at [location] and it takes her [time in minutes/hours] to travel to the clinic.
• Source: She found us through an Instagram/facebook advertisement [or any source as mentioned by the patient including patient referral].
• Referral: She was not referred by another doctor; she scheduled the appointment directly after seeing the online ad.
(Note: The posture correction aids are present in the room.)
(CD acknowledges and reviews the information and completes the consultation.)
Mention the case grade and readiness of patient to take the treatment.
Explain possible barriers, if any, which were tackled (could be clinical in nature)
or/and untackled by the Doctor (could be cost, distance and other elements which
the patient may have said or not said). If tackled, explain what has been
communicated to the patient).
(After the consult, the doctor can first give an informal brief to the PC in the
doctor's room about the patient's readiness for treatment. Following this, a quick
reiteration in front of the patient—highlighting the need to start treatment
today—can help create a sense of urgency and can give a natural flow to the PC to
start their conversation)
(Smiles warmly, maintains eye contact) "Hello again, Ms. [Patient's Last Name]. How did your consultation go, have you understood your diagnosis? Are all your questions about your back/neck pain addressed?"
(Note: Starts with a warm re-greeting, shows the PC is attentive and remembers the patient.)
Patient (Responds).
(If no) "I understand you still have some questions. Would you like me to call the doctor back in to clarify anything?" 
(Note: Proactive in offering further assistance, avoids assumptions.)
(If yes) "Great! Then let me walk you through your personalized treatment plan and the payment details. Please feel free to stop me at any point if you have any questions. Before we start, would like to inform you that as part of our quality assurance process, this conversation may be recorded. This helps us review and improve the support we provide. May I go ahead and record this conversation with your consent?"
(Note: Clear transition, invites patient participation.)
"I hope you received the digital prescription and service slip on your registered mobile number. Do you have it open?"
"Yes, I have received it."
"Perfect. If you see here, your provisional diagnosis is mechanically responsive back pain caused by muscle imbalances and your recovery would take between 8 to 10 weeks. You can also see that 90% of the cases similar to yours who have completed our treatment stages have gotten completely better (refer the prescription and adjust accordingly). That means your pain score, which is now [current pain score], will go down to [expected pain score], and your spine function score, which reflects your ability to do your daily activities and is currently [current spine function score], will increase to [expected spine function score]. The doctor is also aiming to achieve certain Functional Goals – (Mention Functional Goals here). Do you have any questions till here?"
Note: Slightly more conversational tone. Customise the above based on the provisional diagnosis, time duration, the success rate summary, pain score and spine function score mentioned in the prescription; and once done talking about the outcomes – Create Need and Urgency.
"Great. Now, let's talk about the stages of your treatment. For the first 2–3 weeks, you'll come in three times a week. During this phase, we'll focus on reducing your pain using Cell-Repair Therapy and Medical Movements. By the way, did the doctor explain what Cell-Repair Therapy involves?"
(If the patient says no):
"No? Okay, let me explain. In Cell-Repair Therapy, we gently attach electrodes to your pain areas. These electrodes deliver a very low-frequency current. You won't feel anything uncomfortable, but what it does is help repair damaged muscle tissues and reduce inflammation. This process helps significantly reduce pain and stiffness. So, for those initial appointments, you'll be receiving Cell-Repair Therapy." 
(Needs to re-emphasize the importance of the product Doc has prescribed.)
(If the patient says yes):
"Great, so you understand how that will help with your initial pain. And along with that, we will be doing medical movements — movements that work like medicine and will help you manage your pain."
"Once your pain has reduced, it's not the end as we have not yet identified and fixed the root cause. So the doctor will schedule your first DSA test, or the Digital Spine Analysis, which will provide a comprehensive assessment of your spine muscle health. It compares your current muscle function to the ideal range for someone of your height, weight, age, and gender. We'll generate a comprehensive report that clearly shows which specific muscles are strong and which ones need strengthening. (Talk about isolation, targeted strengthening, etc.) This report will be the foundation for planning the next phase of your treatment."
"After the DSA, you'll come in twice a week for targeted medical movements and training on the specialized equipment you saw outside. During these sessions, your doctor will focus on strengthening your spine muscles, directly addressing the root cause of your pain to prevent it from returning. You'll also receive a personalized home medical movement program to follow on the days you're not in the clinic."
"Towards the end of your treatment, we'll conduct another DSA test. This will allow us to compare your progress and visually demonstrate the improvements in your muscle function."
"Also, it's important to note that we have a panel of experts and your treatment would be led under the guidance of an orthopaedic doctor; if your treating doctor feels an orthopaedic consultation would be beneficial at any point during your treatment, we'll schedule an appointment with our Orthopaedic Doctor at no additional cost."
"Are you clear on everything so far, or do you have any questions?"
Patient (responds).
(If yes) Address questions raised. Refer Annexure.
PC mentions about QI Assurance for A, B and C and OS1 and OS2 cases. Recovery is a commitment; every patient is protected by QI Assurance: if recovery goals are not met despite full medical participation, we provide no extra cost until recovery is achieved.
Talk about Progress Reviews and Escalations. Talk about the Dos and Don'ts and swiftly pitch the product.
(If yes) "Great! Let me help you with the clinic functionality — we're open Monday to Saturday, from 7:00 AM to 9:00 PM, and Sundays 10:00 AM – 6:00 PM. We will be assigning you two doctors (Primary and Secondary) to ensure your treatment is always on track."
"Lastly, I'd like you to download our app. It's a great tool for providing feedback on your pain levels, which helps us track your recovery. You'll also find videos and pictures of your prescribed exercises, appointment timings, and you can even manage your appointments through the app."
(If no, proceed) "The total cost for the first four stages of your treatment is [mention cost]. This covers everything we've discussed, including Cell-Repair Therapy, DSA tests, Medical Movements, and any prescribed products. Should we proceed with the payment?" 
(Let's start the treatment right away to avoid sounding salesy.)
(Note: Clearly explain the app's benefits. Share patient app link. Refer Annexure.)
"""

# Individual section breakdowns for variant generation

BASE_INTRO_RAPPORT = """Goodmorning/Afternoon/Evening, Ma'am/Sir!
Welcome to Qi Spine Clinic! May I know your name, please?
Goodmorning/Afternoon/Evening, my name is [Patient'sName], I have an appointment at [Timeslot]."""

BASE_PURPOSE_REASSURANCE = """Yes, Mr./Ms. [Patient's Last Name], I can see your appointment is scheduled with Dr. [Doctor's Full Name] at [Time Slot]. Please, make yourself comfortable. Would you like a glass of water?
Yes, please, thank you. Mr./Ms. [Patient's Last Name], please allow me a minute to allocate a room for you, and I'll be right back. Mr./Ms. [Patient's Last Name], please follow me this way. Please make yourself at ease."""

BASE_PRE_CONSULT = """Hello Mr./Ms. [Patient's Last Name] – I'm [PC's Name], your patient counsellor for today. I will be assisting you throughout your journey. Before we begin the consultation, I just need some basic information about you and your condition. May I proceed with a quick pre-assessment?
Yes, sure. How much time will it take? Should not take more than 10 mins. I'll be using our AI Powered system, Dr. QI; that helps us keep everything organized and when the doctor joins you, all relevant information will be readily available to them. That way, you won't have to repeat everything, and it helps personalize your treatment as well as saves time.
Ok, sure. Great, how did you hear about us? I saw an ad on Instagram/facebook, and then someone called to schedule the appointment. Ok. Just to confirm, you weren't referred to us by a doctor, right?
I'll note that down. And may I know your profession? Well, is that really necessary? It helps us understand the kind of daily stresses your spine goes through. For example, if you sit all day, or if you're on your feet a lot. Oh, okay. I work in IT, so it's a lot of sitting. "Right" (PC continues with questions from the app, shows the screen to the patient while doing so, selects the option that comes closest to the patient's response and proceeds with the rest of the questions from Levels 1, 2, 3 & 8. Should the patient be unclear about any question asked, it is important that the PC explains it in simple words in order to make an accurate record in the app about the patient's history)
· Note: When asking the question in the app about medications, the PC also needs to ask about any previous investigations. This is not mentioned in the app and hence the PC needs to make a mental note about it.
(PC fills in the necessary information, maintaining a friendly and attentive demeanour.)"""

BASE_CONCLUSION_NEXT = """Alright, Mr./Ms. [Patient's Last Name], that's all the information we need. Dr. [Doctor's Full Name] will be with you in just a moment. I'll be right outside if you need anything and will join you again after your consultation to go over your treatment plan.
(Note: Concludes the pre-consultation smoothly, reassuring the patient and setting expectations.)"""

BASE_PC_CD = """Dr. [CD's Name], Mr./Ms. [Patient's Last Name] is ready for her consultation. Here's a quick overview:
• Levels: She filled out Levels 2, 3, and 8 in the Dr. QI app. Please review them before entering.
• Case Summary: She's primarily experiencing [Patient's Main Symptom, e.g., lower back pain]. She has previously tried [Previous Treatments taken if any, e.g., over-the-counter pain relievers] with limited success. Her goals are [Patient's Goals, e.g., to reduce pain, improve posture, and return to her daily activities, lift weights etc.].
• Location: She resides at [location] and it takes her [time in minutes/hours] to travel to the clinic.
• Source: She found us through an Instagram/facebook advertisement [or any source as mentioned by the patient including patient referral].
• Referral: She was not referred by another doctor; she scheduled the appointment directly after seeing the online ad.
(Note: The posture correction aids are present in the room.)
(CD acknowledges and reviews the information and completes the consultation.)"""

BASE_CD_PC = """Mention the case grade and readiness of patient to take the treatment.
Explain possible barriers, if any, which were tackled (could be clinical in nature)
or/and untackled by the Doctor (could be cost, distance and other elements which
the patient may have said or not said). If tackled, explain what has been
communicated to the patient).
(After the consult, the doctor can first give an informal brief to the PC in the
doctor's room about the patient's readiness for treatment. Following this, a quick
reiteration in front of the patient—highlighting the need to start treatment
today—can help create a sense of urgency and can give a natural flow to the PC to
start their conversation)"""

BASE_ONBOARDING = """(Smiles warmly, maintains eye contact) "Hello again, Ms. [Patient's Last Name]. How did your consultation go, have you understood your diagnosis? Are all your questions about your back/neck pain addressed?"
(Note: Starts with a warm re-greeting, shows the PC is attentive and remembers the patient.)
Patient (Responds).
(If no) "I understand you still have some questions. Would you like me to call the doctor back in to clarify anything?" 
(Note: Proactive in offering further assistance, avoids assumptions.)
(If yes) "Great! Then let me walk you through your personalized treatment plan and the payment details. Please feel free to stop me at any point if you have any questions. Before we start, would like to inform you that as part of our quality assurance process, this conversation may be recorded. This helps us review and improve the support we provide. May I go ahead and record this conversation with your consent?"
(Note: Clear transition, invites patient participation.)
"I hope you received the digital prescription and service slip on your registered mobile number. Do you have it open?"
"Yes, I have received it."
"Perfect. If you see here, your provisional diagnosis is mechanically responsive back pain caused by muscle imbalances and your recovery would take between 8 to 10 weeks. You can also see that 90% of the cases similar to yours who have completed our treatment stages have gotten completely better (refer the prescription and adjust accordingly). That means your pain score, which is now [current pain score], will go down to [expected pain score], and your spine function score, which reflects your ability to do your daily activities and is currently [current spine function score], will increase to [expected spine function score]. The doctor is also aiming to achieve certain Functional Goals – (Mention Functional Goals here). Do you have any questions till here?"
Note: Slightly more conversational tone. Customise the above based on the provisional diagnosis, time duration, the success rate summary, pain score and spine function score mentioned in the prescription; and once done talking about the outcomes – Create Need and Urgency.
"Great. Now, let's talk about the stages of your treatment. For the first 2–3 weeks, you'll come in three times a week. During this phase, we'll focus on reducing your pain using Cell-Repair Therapy and Medical Movements. By the way, did the doctor explain what Cell-Repair Therapy involves?"
(If the patient says no):
"No? Okay, let me explain. In Cell-Repair Therapy, we gently attach electrodes to your pain areas. These electrodes deliver a very low-frequency current. You won't feel anything uncomfortable, but what it does is help repair damaged muscle tissues and reduce inflammation. This process helps significantly reduce pain and stiffness. So, for those initial appointments, you'll be receiving Cell-Repair Therapy." 
(Needs to re-emphasize the importance of the product Doc has prescribed.)
(If the patient says yes):
"Great, so you understand how that will help with your initial pain. And along with that, we will be doing medical movements — movements that work like medicine and will help you manage your pain."
"Once your pain has reduced, it's not the end as we have not yet identified and fixed the root cause. So the doctor will schedule your first DSA test, or the Digital Spine Analysis, which will provide a comprehensive assessment of your spine muscle health. It compares your current muscle function to the ideal range for someone of your height, weight, age, and gender. We'll generate a comprehensive report that clearly shows which specific muscles are strong and which ones need strengthening. (Talk about isolation, targeted strengthening, etc.) This report will be the foundation for planning the next phase of your treatment."
"After the DSA, you'll come in twice a week for targeted medical movements and training on the specialized equipment you saw outside. During these sessions, your doctor will focus on strengthening your spine muscles, directly addressing the root cause of your pain to prevent it from returning. You'll also receive a personalized home medical movement program to follow on the days you're not in the clinic."
"Towards the end of your treatment, we'll conduct another DSA test. This will allow us to compare your progress and visually demonstrate the improvements in your muscle function."
"Also, it's important to note that we have a panel of experts and your treatment would be led under the guidance of an orthopaedic doctor; if your treating doctor feels an orthopaedic consultation would be beneficial at any point during your treatment, we'll schedule an appointment with our Orthopaedic Doctor at no additional cost."
"Are you clear on everything so far, or do you have any questions?"
Patient (responds).
(If yes) Address questions raised. Refer Annexure.
PC mentions about QI Assurance for A, B and C and OS1 and OS2 cases. Recovery is a commitment; every patient is protected by QI Assurance: if recovery goals are not met despite full medical participation, we provide no extra cost until recovery is achieved.
Talk about Progress Reviews and Escalations. Talk about the Dos and Don'ts and swiftly pitch the product.
(If yes) "Great! Let me help you with the clinic functionality — we're open Monday to Saturday, from 7:00 AM to 9:00 PM, and Sundays 10:00 AM – 6:00 PM. We will be assigning you two doctors (Primary and Secondary) to ensure your treatment is always on track."
"Lastly, I'd like you to download our app. It's a great tool for providing feedback on your pain levels, which helps us track your recovery. You'll also find videos and pictures of your prescribed exercises, appointment timings, and you can even manage your appointments through the app."
(If no, proceed) "The total cost for the first four stages of your treatment is [mention cost]. This covers everything we've discussed, including Cell-Repair Therapy, DSA tests, Medical Movements, and any prescribed products. Should we proceed with the payment?" 
(Let's start the treatment right away to avoid sounding salesy.)
(Note: Clearly explain the app's benefits. Share patient app link. Refer Annexure.)
"""

def load_base_ref():
    """
    Insert or update Script 1 (canonical reference) under script_code='PC_Script001'.
    
    This function:
    1. Ensures the database is initialized
    2. Upserts the canonical script sections into pc_ref_table
    3. Provides the foundation for variant generation
    """
    # Ensure database tables exist
    init_db()
    
    engine = get_engine()
    
    # UPSERT query - inserts new record or updates existing one
    upsert_query = """
    INSERT INTO pc_ref_table (
        script_code, og_pc_script,
        og_intro_rapport_building, og_purpose_reassurance,
        og_pre_consultation_information_gathering, og_conclusion_next_steps,
        og_pc_cd_handshake, og_cd_pc_handshake, og_onboarding_patient
    ) VALUES (
        :script_code, :full_script,
        :intro, :purpose,
        :pre_consult, :conclusion,
        :pc_cd, :cd_pc, :onboarding
    )
    ON CONFLICT (script_code) DO UPDATE SET
        og_pc_script = EXCLUDED.og_pc_script,
        og_intro_rapport_building = EXCLUDED.og_intro_rapport_building,
        og_purpose_reassurance = EXCLUDED.og_purpose_reassurance,
        og_pre_consultation_information_gathering = EXCLUDED.og_pre_consultation_information_gathering,
        og_conclusion_next_steps = EXCLUDED.og_conclusion_next_steps,
        og_pc_cd_handshake = EXCLUDED.og_pc_cd_handshake,
        og_cd_pc_handshake = EXCLUDED.og_cd_pc_handshake,
        og_onboarding_patient = EXCLUDED.og_onboarding_patient;
    """
    
    # Execute the upsert with all script sections
    run_sql(engine, upsert_query, {
        "script_code": "PC_Script001",
        "full_script": BASE_PC_SCRIPT,   
        "intro": BASE_INTRO_RAPPORT,
        "purpose": BASE_PURPOSE_REASSURANCE,
        "pre_consult": BASE_PRE_CONSULT,
        "conclusion": BASE_CONCLUSION_NEXT,
        "pc_cd": BASE_PC_CD,
        "cd_pc": BASE_CD_PC,
        "onboarding": BASE_ONBOARDING
    })
    print(" Script 1 (canonical reference) loaded/updated in pc_ref_table.")

# --------------- SCRIPT 2: VARIANT GENERATION ----------------

# Prompt template for GPT to generate variants while preserving placeholders
PARAPHRASE_PROMPT = """
Paraphrase the following script passage into {k} distinct versions that:
- keep the same intent and sequence,
- PRESERVE placeholders EXACTLY (e.g., [Patient's Last Name], [Doctor's Full Name], [PC's Name], [Timeslot]),
- maintain professional, empathetic tone,
- keep meaning & length roughly similar,
- avoid adding new facts or removing instructions.

Return as numbered paragraphs 1..{k} without commentary.

Original:
"""

def _make_variants(text_block: str, k: int = 5) -> List[str]:
    """
    Generate v1..v5 variants for a text block using GPT.
    
    This function:
    - Calls GPT to generate k distinct variants
    - Preserves all placeholders exactly
    - Includes robust error handling and fallback
    
    Args:
        text_block: The original text to generate variants for
        k: Number of variants to generate (default: 5)
        
    Returns:
        List[str]: List of k variant texts
    """
    try:
        # Generate variants using GPT with the paraphrase prompt
        raw_response = gpt_text(PARAPHRASE_PROMPT.format(k=k) + text_block, temperature=0.7)
    except Exception as e:
        # Fallback: use original text if GPT call fails
        print(f" [WARN] Variant generation failed: {e}. Using base text as fallback.")
        return [text_block] * k

    variants = []
    
    # Parse the response by looking for numbered lines (1., 2., etc.)
    for line in raw_response.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Check for numbered lines (1., 2., 3., 4., 5.)
        if (len(variants) < k and 
            (line.startswith("1.") or line.startswith("2.") or 
             line.startswith("3.") or line.startswith("4.") or 
             line.startswith("5."))):
            # Extract text after the number and period
            variant_text = line.split(".", 1)[1].strip()
            variants.append(variant_text)

    # Fallback: if numbered parsing didn't work, try splitting by blank lines
    if len(variants) < k:
        chunks = [chunk.strip() for chunk in raw_response.split("\n\n") if chunk.strip()]
        variants = chunks[:k]

    # Ensure we return exactly k variants, filling with original if needed
    while len(variants) < k:
        variants.append(text_block)

    return variants[:k]

def gen_variants():
    """
    Create v1..v5 variants for each base section and save to pc_ref_table.
    
    This function:
    1. Fetches the base Script 1 from database
    2. Generates 5 variants for each section using GPT
    3. Updates the database with all variants
    4. Implements rate limiting to avoid API limits
    """
    engine = get_engine()
    
    # Fetch the base script record
    rows = fetch_all(engine, "SELECT * FROM pc_ref_table WHERE script_code=:script_code", {"script_code": "PC_Script001"})
    if not rows:
        raise RuntimeError("Missing base reference. Run: python pc_script12.py load_base_ref first.")
    
    base_record = rows[0]

    # Define the sections to generate variants for
    sections_to_variant = {
        "og_intro_rapport_building": base_record["og_intro_rapport_building"],
        "og_purpose_reassurance": base_record["og_purpose_reassurance"],
        "og_pre_consultation_information_gathering": base_record["og_pre_consultation_information_gathering"],
        "og_conclusion_next_steps": base_record["og_conclusion_next_steps"],
        "og_pc_cd_handshake": base_record["og_pc_cd_handshake"],
        "og_cd_pc_handshake": base_record["og_cd_pc_handshake"],
        "og_onboarding_patient": base_record["og_onboarding_patient"],
    }

    variant_map: Dict[str, List[str]] = {}
    
    # Generate variants for each section
    for column_name, text_content in sections_to_variant.items():
        print(f"🛠 Generating variants for {column_name} ...")
        variant_map[column_name] = _make_variants(text_content, k=5)
        
        # Rate limiting to avoid hitting API limits
        time.sleep(0.2)

    # Build the massive UPDATE query with all variant columns
    update_query = """
    UPDATE pc_ref_table SET
      -- Intro and Rapport Building variants
      og_intro_rapport_buildingv1=:intro1, og_intro_rapport_buildingv2=:intro2, 
      og_intro_rapport_buildingv3=:intro3, og_intro_rapport_buildingv4=:intro4, 
      og_intro_rapport_buildingv5=:intro5,
      
      -- Purpose and Reassurance variants
      og_purpose_reassurancev1=:purp1, og_purpose_reassurancev2=:purp2, 
      og_purpose_reassurancev3=:purp3, og_purpose_reassurancev4=:purp4, 
      og_purpose_reassurancev5=:purp5,
      
      -- Pre-consultation Information Gathering variants
      og_pre_consultation_information_gatheringv1=:pre1, 
      og_pre_consultation_information_gatheringv2=:pre2, 
      og_pre_consultation_information_gatheringv3=:pre3, 
      og_pre_consultation_information_gatheringv4=:pre4, 
      og_pre_consultation_information_gatheringv5=:pre5,
      
      -- Conclusion and Next Steps variants
      og_conclusion_next_stepsv1=:conc1, og_conclusion_next_stepsv2=:conc2, 
      og_conclusion_next_stepsv3=:conc3, og_conclusion_next_stepsv4=:conc4, 
      og_conclusion_next_stepsv5=:conc5,
      
      -- PC to CD Handshake variants
      og_pc_cd_handshakev1=:pccd1, og_pc_cd_handshakev2=:pccd2, 
      og_pc_cd_handshakev3=:pccd3, og_pc_cd_handshakev4=:pccd4, 
      og_pc_cd_handshakev5=:pccd5,
      
      -- CD to PC Handshake variants
      og_cd_pc_handshakev1=:cdpc1, og_cd_pc_handshakev2=:cdpc2, 
      og_cd_pc_handshakev3=:cdpc3, og_cd_pc_handshakev4=:cdpc4, 
      og_cd_pc_handshakev5=:cdpc5,
      
      -- Patient Onboarding variants
      og_onboarding_patientv1=:onb1, og_onboarding_patientv2=:onb2, 
      og_onboarding_patientv3=:onb3, og_onboarding_patientv4=:onb4, 
      og_onboarding_patientv5=:onb5
    WHERE script_code=:script_code
    """
    
    # Prepare all parameters for the update
    update_parameters = {
        "script_code": "PC_Script001",
        # Intro variants
        "intro1": variant_map["og_intro_rapport_building"][0],
        "intro2": variant_map["og_intro_rapport_building"][1],
        "intro3": variant_map["og_intro_rapport_building"][2],
        "intro4": variant_map["og_intro_rapport_building"][3],
        "intro5": variant_map["og_intro_rapport_building"][4],
        
        # Purpose variants
        "purp1": variant_map["og_purpose_reassurance"][0],
        "purp2": variant_map["og_purpose_reassurance"][1],
        "purp3": variant_map["og_purpose_reassurance"][2],
        "purp4": variant_map["og_purpose_reassurance"][3],
        "purp5": variant_map["og_purpose_reassurance"][4],
        
        # Pre-consultation variants
        "pre1": variant_map["og_pre_consultation_information_gathering"][0],
        "pre2": variant_map["og_pre_consultation_information_gathering"][1],
        "pre3": variant_map["og_pre_consultation_information_gathering"][2],
        "pre4": variant_map["og_pre_consultation_information_gathering"][3],
        "pre5": variant_map["og_pre_consultation_information_gathering"][4],
        
        # Conclusion variants
        "conc1": variant_map["og_conclusion_next_steps"][0],
        "conc2": variant_map["og_conclusion_next_steps"][1],
        "conc3": variant_map["og_conclusion_next_steps"][2],
        "conc4": variant_map["og_conclusion_next_steps"][3],
        "conc5": variant_map["og_conclusion_next_steps"][4],
        
        # PC to CD handshake variants
        "pccd1": variant_map["og_pc_cd_handshake"][0],
        "pccd2": variant_map["og_pc_cd_handshake"][1],
        "pccd3": variant_map["og_pc_cd_handshake"][2],
        "pccd4": variant_map["og_pc_cd_handshake"][3],
        "pccd5": variant_map["og_pc_cd_handshake"][4],
        
        # CD to PC handshake variants
        "cdpc1": variant_map["og_cd_pc_handshake"][0],
        "cdpc2": variant_map["og_cd_pc_handshake"][1],
        "cdpc3": variant_map["og_cd_pc_handshake"][2],
        "cdpc4": variant_map["og_cd_pc_handshake"][3],
        "cdpc5": variant_map["og_cd_pc_handshake"][4],
        
        # Onboarding variants
        "onb1": variant_map["og_onboarding_patient"][0],
        "onb2": variant_map["og_onboarding_patient"][1],
        "onb3": variant_map["og_onboarding_patient"][2],
        "onb4": variant_map["og_onboarding_patient"][3],
        "onb5": variant_map["og_onboarding_patient"][4],
    }
    
    # Execute the update
    run_sql(engine, update_query, update_parameters)
    print(" Script 2 variants (v1-v5 for all sections) saved to pc_ref_table.")

# --------------- COMMAND LINE INTERFACE ----------------

def main():
    """
    Main CLI entry point for the PC Call Analyzer script.
    
    Handles command line arguments and executes appropriate functions.
    """
    parser = argparse.ArgumentParser(description="PC Call Analyzer - Script 1 + Script 2 Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Initialize database command
    subparsers.add_parser("init_db", help="Initialize database tables")
    
    # Load base reference command
    subparsers.add_parser("load_base_ref", help="Load Script 1 (canonical reference)")
    
    # Generate variants command
    subparsers.add_parser("gen_variants", help="Generate Script 2 variants (v1-v5)")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Execute appropriate function based on command
    if args.command == "init_db":
        init_db()
    elif args.command == "load_base_ref":
        load_base_ref()
    elif args.command == "gen_variants":
        gen_variants()
    else:
        # Show help if no valid command provided
        parser.print_help()

# Standard Python script entry point
if __name__ == "__main__":
    main()