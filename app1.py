import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="Resume Screening System", layout="wide")

# -------------------------------
# 1. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Resume.csv")

df = load_data()

# -------------------------------
# 2. Clean Text Function
# -------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.lower()

df['cleaned_resume'] = df['Resume_str'].apply(clean_text)

# -------------------------------
# 3. UI Title
# -------------------------------
st.title("📄  Resume Screening System")
st.markdown("### 🔍 Find the Best Candidates Automatically")

# -------------------------------
# Sidebar Settings
# -------------------------------
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Select number of top candidates", 5, 20, 10)

# -------------------------------
# Input Job Description
# -------------------------------
job_desc = st.text_area("📝 Enter Job Description")

# -------------------------------
# Button Click
# -------------------------------
if st.button("🚀 Analyze Candidates"):

    if job_desc.strip() == "":
        st.warning("Please enter a job description")
    else:

        # Clean job description
        job_desc_clean = clean_text(job_desc)
        job_words = job_desc_clean.split()

        # -------------------------------
        # 4. Scoring Logic
        # -------------------------------
        def score_resume(resume):
            words = resume.split()
            return sum(1 for w in job_words if w in words)

        df['score'] = df['cleaned_resume'].apply(score_resume)

        # Keyword Boost
        keywords = ["python", "machine learning", "data", "nlp", "ai"]
        df['keyword_score'] = df['cleaned_resume'].apply(
            lambda x: sum(1 for k in keywords if k in x)
        )

        # Final Score
        df['final_score'] = (
            0.6 * df['score'] +
            0.4 * df['keyword_score']
        )

        # -------------------------------
        # 5. Create Correct Column (IMPORTANT FIX)
        # -------------------------------
        target_role = "information technology"

        df['Category_clean'] = df['Category'].str.lower().str.replace('-', ' ')
        df['correct'] = df['Category_clean'].apply(
            lambda x: 1 if target_role in x else 0
        )

        # -------------------------------
        # 6. Ranking
        # -------------------------------
        ranked_df = df.sort_values(by='final_score', ascending=False)
        top_df = ranked_df.head(top_k)

        # -------------------------------
        # 7. Layout
        # -------------------------------
        col1, col2 = st.columns([2, 1])

        # LEFT: Table
        with col1:
            st.subheader("🏆 Top Candidates")

            st.markdown("""
🔹 **Explanation:**
- `score` → how many job words match resume  
- `keyword_score` → important skills like Python, ML  
- `final_score` → combined score used for ranking  
""")

            st.dataframe(top_df[['Category', 'score', 'keyword_score', 'final_score']])

        # RIGHT: Chart
        with col2:
            st.subheader("📊 Score Chart")

            fig, ax = plt.subplots()
            ax.barh(range(len(top_df)), top_df['final_score'])
            ax.set_yticks(range(len(top_df)))
            ax.set_yticklabels(top_df['Category'])
            ax.invert_yaxis()
            st.pyplot(fig)

        # -------------------------------
        # 8. Evaluation (Beginner Friendly)
        # -------------------------------
        st.subheader("📈 Model Performance")

        correct_count = top_df['correct'].sum()
        precision = correct_count / top_k

        st.markdown(f"""
### ✅ What this means:

- Total Top Candidates = **{top_k}**
- Correct IT Resumes = **{correct_count}**
- Precision@{top_k} = **{precision:.2f}**

👉 Example:
If 1 correct out of 10 → Precision = 0.1  
If 5 correct out of 10 → Precision = 0.5  
""")

        st.success(f"🎯 Final Precision@{top_k}: {precision:.2f}")

        # -------------------------------
        # 9. Detailed Output
        # -------------------------------
        st.subheader("🔍 Detailed Result")

        st.write("Correct Labels (1 = IT, 0 = Not IT):")
        st.write(top_df['correct'].values)

        # -------------------------------
        # 10. Resume Preview
        # -------------------------------
        st.subheader("📄 Resume Preview")

        selected_index = st.selectbox(
            "Select candidate index to view resume",
            top_df.index
        )

        st.write(df.loc[selected_index, 'Resume_str'][:1000])

        # -------------------------------
        # 11. Model Explanation
        # -------------------------------
        st.subheader("🧠 How This Model Works")

        st.markdown("""
1. **Text Cleaning**
   - Removes symbols and converts text to lowercase

2. **Matching Score**
   - Counts how many job description words appear in resume

3. **Keyword Boost**
   - Gives extra points for important skills like Python, ML

4. **Final Score**
   - Combines both scores

5. **Ranking**
   - Higher score → better candidate

6. **Evaluation**
   - Checks how many IT resumes are in top results
""")