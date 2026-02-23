import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from PIL import Image
import io
import json
from datetime import datetime
import easyocr
from collections import defaultdict

st.set_page_config(
    page_title="Full Exam Auto-Evaluator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        background-color: #f0f8ff;
    }
    .score-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .grade-a { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .grade-b { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .grade-c { background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%); color: white; }
    .grade-d { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    .grade-f { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); color: white; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader (cached for performance)"""
    return easyocr.Reader(['en'], gpu=False)

class QuestionExtractor:
    """Extract and parse questions from text"""
    
    def __init__(self):
        self.question_patterns = [
            r'(?:Question|Q\.?|Ques\.?)\s*(?:#)?\s*(\d+)[:\.\)]?\s*(.+?)(?=(?:Question|Q\.?|Ques\.?)\s*(?:#)?\s*\d+|$)',
            r'(\d+)[:\.\)]\s*(.+?)(?=\d+[:\.\)]|$)',
            r'Q(\d+)[:\.\)]?\s*(.+?)(?=Q\d+|$)',
        ]
    
    def extract_questions(self, text):
        """Extract questions and answers from text"""
        questions = {}
        
        # Try each pattern
        for pattern in self.question_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                q_num = int(match.group(1))
                q_content = match.group(2).strip()
                
                if q_content and len(q_content) > 10:
                    questions[q_num] = q_content
        if not questions:
            sections = text.split('\n\n')
            for i, section in enumerate(sections, 1):
                if section.strip() and len(section.strip()) > 20:
                    questions[i] = section.strip()
        
        return questions
    
    def parse_answer_key(self, text):
        """Parse answer key with question numbers and model answers"""
        answer_key = {}
        questions = self.extract_questions(text)
        
        for q_num, answer in questions.items():
            marks_match = re.search(r'\[(\d+)\s*marks?\]', answer, re.IGNORECASE)
            marks = int(marks_match.group(1)) if marks_match else 10  # Default 10 marks
            
            clean_answer = re.sub(r'\[\d+\s*marks?\]', '', answer, flags=re.IGNORECASE).strip()
            
            answer_key[q_num] = {
                'answer': clean_answer,
                'marks': marks
            }
        
        return answer_key

class ExamEvaluator:
    def __init__(self, use_bert=True):
        self.use_bert = use_bert
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        if use_bert:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        text = text.lower()
        text = ' '.join(text.split())
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def extract_keywords(self, text, top_n=10):
        clean_text = self.preprocess_text(text)
        words = word_tokenize(clean_text)
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:top_n]]
    
    def calculate_keyword_match(self, student_answer, model_answer):
        student_keywords = set(self.extract_keywords(student_answer, top_n=20))
        model_keywords = set(self.extract_keywords(model_answer, top_n=20))
        
        if not model_keywords:
            return 0.0
        
        common_keywords = student_keywords.intersection(model_keywords)
        score = len(common_keywords) / len(model_keywords)
        return min(score, 1.0)
    
    def calculate_tfidf_similarity(self, student_answer, model_answer):
        student_clean = self.preprocess_text(student_answer)
        model_clean = self.preprocess_text(model_answer)
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([model_clean, student_clean])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def calculate_bert_similarity(self, student_answer, model_answer):
        if not self.use_bert:
            return 0.0
        
        embeddings = self.bert_model.encode([model_answer, student_answer])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def evaluate_answer(self, student_answer, model_answer, max_marks=10):
        keyword_score = self.calculate_keyword_match(student_answer, model_answer)
        tfidf_score = self.calculate_tfidf_similarity(student_answer, model_answer)
        
        if self.use_bert:
            bert_score = self.calculate_bert_similarity(student_answer, model_answer)
            combined_score = (0.3 * keyword_score + 0.2 * tfidf_score + 0.5 * bert_score)
        else:
            combined_score = (0.4 * keyword_score + 0.6 * tfidf_score)
        
        awarded_marks = round(combined_score * max_marks, 2)
        
        return {
            'awarded_marks': awarded_marks,
            'max_marks': max_marks,
            'percentage': round(combined_score * 100, 2),
            'keyword_score': round(keyword_score, 3),
            'tfidf_score': round(tfidf_score, 3)
        }

def extract_text_from_image(image):
    """Extract text from uploaded image using OCR"""
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        img_array = np.array(image)
        reader = load_ocr_reader()
        results = reader.readtext(img_array)
        text = ' '.join([result[1] for result in results])
        
        return text.strip()
    except Exception as e:
        st.error(f"Error in OCR: {str(e)}")
        return None

def calculate_grade(percentage):
    """Calculate letter grade from percentage"""
    if percentage >= 90:
        return 'A+', 'grade-a'
    elif percentage >= 80:
        return 'A', 'grade-a'
    elif percentage >= 70:
        return 'B', 'grade-b'
    elif percentage >= 60:
        return 'C', 'grade-c'
    elif percentage >= 50:
        return 'D', 'grade-d'
    else:
        return 'F', 'grade-f'

if 'evaluator' not in st.session_state:
    with st.spinner("Loading AI models... This may take a moment on first run."):
        st.session_state.evaluator = ExamEvaluator(use_bert=True)

if 'question_extractor' not in st.session_state:
    st.session_state.question_extractor = QuestionExtractor()

if 'answer_key' not in st.session_state:
    st.session_state.answer_key = None

if 'student_answers' not in st.session_state:
    st.session_state.student_answers = None

if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

st.markdown('<h1 class="main-header">📋 Full Exam Auto-Evaluation System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Upload complete answer sheets and answer keys for automatic evaluation</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Settings")
    
    total_marks = st.number_input(
        "Total Exam Marks:",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    st.markdown("---")
    
    st.header("📊 Grading Scale")
    st.info("""
    **Grade Distribution:**
    - A+/A: 80-100% (Excellent)
    - B: 70-79% (Good)
    - C: 60-69% (Average)
    - D: 50-59% (Pass)
    - F: <50% (Fail)
    """)
    
    st.markdown("---")
    
    st.header("ℹ️ How It Works")
    st.success("""
    1. Upload answer key (with Q numbers)
    2. Upload student answer sheet
    3. AI extracts & matches questions
    4. Evaluates each answer
    5. Generates final score & report
    """)
    
    if st.button("🔄 Reset All"):
        st.session_state.answer_key = None
        st.session_state.student_answers = None
        st.session_state.evaluation_results = None
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["📚 Upload Answer Key", "📝 Upload Answer Sheet", "📊 View Results", "📈 Statistics"])

with tab1:
    st.header("📚 Step 1: Upload Answer Key")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Method")
        answer_key_mode = st.radio(
            "Choose input method:",
            ["📝 Text Input", "📷 Image Upload (OCR)"],
            key="answer_key_mode"
        )
        
        if answer_key_mode == "📝 Text Input":
            answer_key_text = st.text_area(
                "Enter answer key:",
                height=400,
                placeholder="""Example format:

Question 1: Machine learning is a subset of AI that enables systems to learn from data. [10 marks]

Question 2: Binary Search Tree is a hierarchical data structure with specific ordering properties. [15 marks]

Q3: Deadlock occurs when processes are waiting for resources held by each other. [10 marks]
""",
                key="answer_key_text_input"
            )
            
            if st.button("📥 Load Answer Key", key="load_text_key"):
                if answer_key_text:
                    answer_key = st.session_state.question_extractor.parse_answer_key(answer_key_text)
                    st.session_state.answer_key = answer_key
                    st.success(f"✅ Answer key loaded! Found {len(answer_key)} questions.")
                    st.rerun()
                else:
                    st.error("Please enter answer key text!")
        
        else:
            uploaded_key_image = st.file_uploader(
                "Upload answer key image:",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="answer_key_image"
            )
            
            if uploaded_key_image:
                image = Image.open(uploaded_key_image)
                st.image(image, caption="Answer Key Image", use_container_width=True)
                
                if st.button("🔍 Extract Answer Key", key="extract_key"):
                    with st.spinner("Extracting text from answer key..."):
                        extracted_text = extract_text_from_image(uploaded_key_image)
                        if extracted_text:
                            answer_key = st.session_state.question_extractor.parse_answer_key(extracted_text)
                            st.session_state.answer_key = answer_key
                            st.success(f"✅ Answer key extracted! Found {len(answer_key)} questions.")
                            st.rerun()
    
    with col2:
        st.subheader("Loaded Answer Key")
        
        if st.session_state.answer_key:
            st.success(f"✅ {len(st.session_state.answer_key)} questions loaded")
            
            for q_num, data in sorted(st.session_state.answer_key.items()):
                with st.expander(f"Question {q_num} - [{data['marks']} marks]"):
                    st.write(data['answer'])
        else:
            st.info("👈 Upload answer key to see preview here")

with tab2:
    st.header("📝 Step 2: Upload Student Answer Sheet")
    
    if not st.session_state.answer_key:
        st.warning("⚠️ Please upload answer key first (Step 1)")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Student Information")
            student_name = st.text_input("Student Name:", placeholder="Enter student name")
            student_id = st.text_input("Student ID:", placeholder="Enter student ID")
            
            st.markdown("---")
            
            answer_sheet_mode = st.radio(
                "Choose input method:",
                ["📝 Text Input", "📷 Image Upload (OCR)"],
                key="answer_sheet_mode"
            )
            
            if answer_sheet_mode == "📝 Text Input":
                answer_sheet_text = st.text_area(
                    "Enter student answers:",
                    height=400,
                    placeholder="""Example format:

Question 1: Machine learning helps computers learn from data automatically...

Question 2: BST is a tree where left child is smaller...

Q3: Deadlock happens when processes wait for each other...
""",
                    key="answer_sheet_text"
                )
                
                if st.button("📥 Load Student Answers", key="load_student_text"):
                    if answer_sheet_text and student_name:
                        student_answers = st.session_state.question_extractor.extract_questions(answer_sheet_text)
                        st.session_state.student_answers = student_answers
                        st.session_state.student_name = student_name
                        st.session_state.student_id = student_id
                        st.success(f"✅ Loaded {len(student_answers)} answers!")
                        st.rerun()
                    else:
                        st.error("Please enter student name and answers!")
            
            else:
                uploaded_answer_image = st.file_uploader(
                    "Upload student answer sheet:",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                    key="answer_sheet_image"
                )
                
                if uploaded_answer_image:
                    image = Image.open(uploaded_answer_image)
                    st.image(image, caption="Student Answer Sheet", use_container_width=True)
                    
                    if st.button("🔍 Extract Student Answers", key="extract_answers"):
                        if student_name:
                            with st.spinner("Extracting text from answer sheet..."):
                                extracted_text = extract_text_from_image(uploaded_answer_image)
                                if extracted_text:
                                    student_answers = st.session_state.question_extractor.extract_questions(extracted_text)
                                    st.session_state.student_answers = student_answers
                                    st.session_state.student_name = student_name
                                    st.session_state.student_id = student_id
                                    st.success(f"✅ Extracted {len(student_answers)} answers!")
                                    st.rerun()
                        else:
                            st.error("Please enter student name first!")
        
        with col2:
            st.subheader("Extracted Student Answers")
            
            if st.session_state.student_answers:
                st.success(f"✅ {len(st.session_state.student_answers)} answers extracted")
                
                for q_num, answer in sorted(st.session_state.student_answers.items()):
                    with st.expander(f"Question {q_num}"):
                        st.write(answer[:200] + "..." if len(answer) > 200 else answer)
            else:
                st.info("👈 Upload answer sheet to see preview here")
        if st.session_state.student_answers and st.session_state.answer_key:
            st.markdown("---")
            col_btn = st.columns([1, 2, 1])
            
            with col_btn[1]:
                if st.button("🎯 EVALUATE EXAM", type="primary", use_container_width=True):
                    with st.spinner("Evaluating exam... Please wait..."):
                        results = []
                        total_obtained = 0
                        total_max = 0
                        for q_num in st.session_state.answer_key.keys():
                            if q_num in st.session_state.student_answers:
                                model_answer = st.session_state.answer_key[q_num]['answer']
                                student_answer = st.session_state.student_answers[q_num]
                                max_marks = st.session_state.answer_key[q_num]['marks']
                                
                                result = st.session_state.evaluator.evaluate_answer(
                                    student_answer,
                                    model_answer,
                                    max_marks
                                )
                                results.append({
                                    'question_num': q_num,
                                    'obtained': result['awarded_marks'],
                                    'max': max_marks,
                                    'percentage': result['percentage'],
                                    'student_answer': student_answer
                                })
                                
                                total_obtained += result['awarded_marks']
                                total_max += max_marks
                            else:
                                results.append({
                                    'question_num': q_num,
                                    'obtained': 0,
                                    'max': st.session_state.answer_key[q_num]['marks'],
                                    'percentage': 0,
                                    'student_answer': "NOT ATTEMPTED"
                                })
                                total_max += st.session_state.answer_key[q_num]['marks']
                        
                        st.session_state.evaluation_results = {
                            'results': results,
                            'total_obtained': total_obtained,
                            'total_max': total_max,
                            'percentage': (total_obtained / total_max * 100) if total_max > 0 else 0,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.success("✅ Evaluation complete!")
                        st.rerun()

with tab3:
    st.header("📊 Evaluation Results")
    
    if not st.session_state.evaluation_results:
        st.info("📝 No results yet. Please complete Steps 1 and 2, then click 'Evaluate Exam'.")
    else:
        results = st.session_state.evaluation_results
        st.subheader("👤 Student Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Student Name", st.session_state.get('student_name', 'N/A'))
        with col2:
            st.metric("Student ID", st.session_state.get('student_id', 'N/A'))
        with col3:
            st.metric("Evaluation Date", results['timestamp'])
        
        st.markdown("---")
        st.subheader("🎯 Final Score")
        
        grade, grade_class = calculate_grade(results['percentage'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="score-card {grade_class}">
                    Grade<br>{grade}
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Total Marks",
                f"{results['total_obtained']:.2f} / {results['total_max']}"
            )
        
        with col3:
            st.metric(
                "Percentage",
                f"{results['percentage']:.2f}%"
            )
        
        with col4:
            attempted = sum(1 for r in results['results'] if r['student_answer'] != "NOT ATTEMPTED")
            st.metric(
                "Questions Attempted",
                f"{attempted} / {len(results['results'])}"
            )
        
        st.markdown("---")
        st.subheader("📝 Question-wise Breakdown")
        
        for result in results['results']:
            q_num = result['question_num']
            obtained = result['obtained']
            max_marks = result['max']
            percentage = result['percentage']
            if percentage >= 75:
                color = "#28a745"
                emoji = "✅"
            elif percentage >= 50:
                color = "#ffc107"
                emoji = "⚠️"
            else:
                color = "#dc3545"
                emoji = "❌"
            
            with st.expander(f"{emoji} Question {q_num}: {obtained}/{max_marks} marks ({percentage:.1f}%)"):
                col_q1, col_q2 = st.columns([1, 1])
                
                with col_q1:
                    st.markdown("**Model Answer:**")
                    if q_num in st.session_state.answer_key:
                        st.info(st.session_state.answer_key[q_num]['answer'])
                
                with col_q2:
                    st.markdown("**Student Answer:**")
                    if result['student_answer'] == "NOT ATTEMPTED":
                        st.error("NOT ATTEMPTED")
                    else:
                        st.write(result['student_answer'])
                
                st.progress(percentage / 100)        
        st.markdown("---")
        col_download = st.columns([1, 2, 1])
        
        with col_download[1]:
            report_data = {
                'Student Name': [st.session_state.get('student_name', 'N/A')],
                'Student ID': [st.session_state.get('student_id', 'N/A')],
                'Total Marks': [f"{results['total_obtained']:.2f}/{results['total_max']}"],
                'Percentage': [f"{results['percentage']:.2f}%"],
                'Grade': [grade],
                'Date': [results['timestamp']]
            }

            for result in results['results']:
                report_data[f"Q{result['question_num']}"] = [f"{result['obtained']}/{result['max']}"]
            
            df = pd.DataFrame(report_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Detailed Report (CSV)",
                data=csv,
                file_name=f"exam_report_{st.session_state.get('student_id', 'student')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

with tab4:
    st.header("📈 Performance Statistics")
    
    if not st.session_state.evaluation_results:
        st.info("📝 No statistics available yet.")
    else:
        results = st.session_state.evaluation_results

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Score Distribution")
            
            question_scores = [
                {
                    'Question': f"Q{r['question_num']}",
                    'Obtained': r['obtained'],
                    'Maximum': r['max'],
                    'Percentage': r['percentage']
                }
                for r in results['results']
            ]
            
            df_scores = pd.DataFrame(question_scores)
            st.dataframe(df_scores, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📉 Performance Analysis")

            percentages = [r['percentage'] for r in results['results']]
            
            stats = {
                'Metric': ['Average Score', 'Highest Score', 'Lowest Score', 'Questions Above 75%', 'Questions Below 50%'],
                'Value': [
                    f"{np.mean(percentages):.2f}%",
                    f"{np.max(percentages):.2f}%",
                    f"{np.min(percentages):.2f}%",
                    sum(1 for p in percentages if p >= 75),
                    sum(1 for p in percentages if p < 50)
                ]
            }
            
            df_stats = pd.DataFrame(stats)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("💪 Strengths & Weaknesses")
        
        col_sw1, col_sw2 = st.columns(2)
        
        with col_sw1:
            st.markdown("**✅ Strong Areas (≥75%)**")
            strong = [r for r in results['results'] if r['percentage'] >= 75]
            if strong:
                for r in strong:
                    st.success(f"Question {r['question_num']}: {r['percentage']:.1f}%")
            else:
                st.info("No questions scored above 75%")
        
        with col_sw2:
            st.markdown("**⚠️ Needs Improvement (<50%)**")
            weak = [r for r in results['results'] if r['percentage'] < 50]
            if weak:
                for r in weak:
                    st.error(f"Question {r['question_num']}: {r['percentage']:.1f}%")
            else:
                st.success("All questions scored above 50%!")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AI-Based Full Exam Evaluation System | Powered by BERT, NLP & OCR</p>
    </div>
""", unsafe_allow_html=True)