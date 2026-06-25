import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

 
nltk.download('punkt', quiet=True)

 
st.set_page_config(page_title="Chatbot for FAQ", page_icon="🤖", layout="centered")
st.title(" Chatbot for FAQ")
st.write("Ask me anything about our training programs, placements, fees, or certifications!")
st.write("---")

 
faq_dataset = {
   
    "hello": "Hello! Welcome to our AI Support. How can I help you today?",
    "hi": "Hi there! How can I assist you with your learning journey today?",
    "hey": "Hey! Great to see you. What questions do you have for me?",
    "who are you": "I am an AI-powered FAQ Assistant here to help you clear your doubts regarding our courses and services.",
    "thank you": "You're very welcome! Let me know if you need help with anything else.",
 
    "what courses do you offer": "We offer comprehensive programs in Python Programming, Full-Stack Web Development, Data Science, Machine Learning, and Mobile App Development using Flutter.",
    "which is the best course for beginners": "If you are a beginner, we highly recommend starting with our Python Programming or Full-Stack Web Development course.",
    "do you teach machine learning": "Yes, we have an advanced Machine Learning & Deep Learning course covering NLP, Computer Vision, and Generative AI.",
    "what is the duration of the courses": "Most of our professional courses take 3 to 6 months to complete, depending on the schedule (regular or weekend batches).",
    
 
    "will i get a certificate": "Yes, upon completing all project submissions and modules, you will receive an industry-recognized Course Completion Certificate.",
    "is the certificate free": "Yes, the digital certificate is completely free and included in your course enrollment package.",
    "are there any prerequisites for learning": "No prior coding background is required for our foundation courses. Anyone with logical thinking can start learning!",
 
    "what is the course fee": "The fee varies from course to course (ranging from ₹5,000 to ₹15,000). Please check our syllabus brochure or contact support for exact pricing.",
    "do you offer discounts": "Yes, we offer early-bird discounts and special pricing structural discounts for college students with a valid ID card.",
    "can i pay in installments": "Yes, you can split your course fee payment into 2 or 3 easy monthly installments.",
    "is there a refund policy": "Yes, we provide a 7-day no-questions-asked full refund policy if you are not satisfied with the training.",
 
    "do you provide job placement support": "Yes, we provide 100% placement assistance, including resume building, mock interviews, and sharing your profile with our 100+ hiring partners.",
    "will there be real-time projects": "Absolutely! Every course includes at least 3-4 real-world projects so you can build a strong practical portfolio for your resume.",
    "can i do an internship here": "Yes, top-performing students get an opportunity to work on live industrial internships after course completion.",
    
 
    "how to contact customer support": "You can reach our dedicated support team via email at support@eduplatform.com or call/WhatsApp us directly at +91 98765 43210.",
    "what are the support timings": "Our technical and student support desks are active from 9:00 AM to 7:00 PM (Monday to Saturday)."
}

faq_questions = list(faq_dataset.keys())

 
def get_best_response(user_query):
 
    user_query_clean = user_query.lower().strip()
    
    
    if user_query_clean in faq_dataset:
        return faq_dataset[user_query_clean]
        
    all_texts = faq_questions + [user_query_clean]
    
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[0][best_match_idx]
    
    
    if highest_score < 0.15:
        return "I'm sorry, I couldn't find an exact answer for that specific question. Could you please rephrase it, or feel free to drop an email to support@eduplatform.com!"
    
    matched_question = faq_questions[best_match_idx]
    return faq_dataset[matched_question]
 
if "messages" not in st.session_state:
    st.session_state.messages = []

 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if user_input := st.chat_input("Ask me about fees, courses, placements, etc..."):
    
     
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    
    bot_response = get_best_response(user_input)
    
     
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})