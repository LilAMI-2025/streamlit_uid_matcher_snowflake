
import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="UID Matcher", layout="wide")

@st.cache_resource
def get_snowflake_engine():
    sf = st.secrets["snowflake"]
    return create_engine(
        f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}?warehouse={sf.warehouse}&role={sf.role}"
    )

def run_snowflake_query():
    engine = get_snowflake_engine()
    query_mapped = """
        SELECT HEADING_0, MAX(UID) AS UID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL
        GROUP BY HEADING_0
    "
    query_unmapped = """
        SELECT DISTINCT HEADING_0
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NULL AND NOT LOWER(HEADING_0) LIKE 'our privacy policy%'
    "
    df_mapped = pd.read_sql(query_mapped, engine)
    df_unmapped = pd.read_sql(query_unmapped, engine)
    return df_mapped, df_unmapped

synonym_map = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

def apply_synonyms(text):
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return text

def enhanced_normalize(text):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = apply_synonyms(text)
    words = text.split()
    return ' '.join([w for w in words if w not in ENGLISH_STOP_WORDS])

st.title("📊 UID Matcher with Snowflake")

if st.button("🔁 Connect and Run Matching"):
    df_mapped, df_unmapped = run_snowflake_query()
    df_mapped = df_mapped[df_mapped["heading_0"].notna()].reset_index(drop=True)
    df_unmapped = df_unmapped[df_unmapped["heading_0"].notna()].reset_index(drop=True)

    df_mapped["norm_text"] = df_mapped["heading_0"].apply(enhanced_normalize)
    df_unmapped["norm_text"] = df_unmapped["heading_0"].apply(enhanced_normalize)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(df_mapped["norm_text"].tolist() + df_unmapped["norm_text"].tolist())

    mapped_vecs = vectorizer.transform(df_mapped["norm_text"])
    unmapped_vecs = vectorizer.transform(df_unmapped["norm_text"])
    similarity_matrix = cosine_similarity(unmapped_vecs, mapped_vecs)

    mapped_uids, mapped_to, similarities, match_confidences = [], [], [], []
    for i, sim_row in enumerate(similarity_matrix):
        best_idx = sim_row.argmax()
        best_score = sim_row[best_idx]
        if best_score >= 0.60:
            matched_uid = df_mapped.iloc[best_idx]["uid"]
            matched_q = df_mapped.iloc[best_idx]["heading_0"]
            confidence = "✅ High"
        elif best_score >= 0.50:
            matched_uid = df_mapped.iloc[best_idx]["uid"]
            matched_q = df_mapped.iloc[best_idx]["heading_0"]
            confidence = "⚠️ Low"
        else:
            matched_uid = None
            matched_q = None
            confidence = "❌ No match"

        mapped_uids.append(matched_uid)
        mapped_to.append(matched_q)
        similarities.append(round(best_score, 4))
        match_confidences.append(confidence)

    df_unmatched = df_unmapped.copy()
    df_unmatched["Suggested_UID"] = mapped_uids
    df_unmatched["Matched_Question"] = mapped_to
    df_unmatched["Similarity"] = similarities
    df_unmatched["Match_Confidence"] = match_confidences

    st.info("💡 Running semantic fallback matching...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_u = model.encode(df_unmatched["heading_0"].tolist(), convert_to_tensor=True)
    emb_m = model.encode(df_mapped["heading_0"].tolist(), convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_u, emb_m)

    sem_matches, sem_scores = [], []
    for i in range(len(df_unmatched)):
        best_idx = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_idx].item()
        if score >= 0.60:
            sem_matches.append(df_mapped.iloc[best_idx]["uid"])
            sem_scores.append(round(score, 4))
        else:
            sem_matches.append(None)
            sem_scores.append(None)

    df_unmatched["Semantic_UID"] = sem_matches
    df_unmatched["Semantic_Similarity"] = sem_scores

    df_unmatched["Final_UID"] = df_unmatched["Suggested_UID"].combine_first(df_unmatched["Semantic_UID"])
    df_unmatched["Final_Question"] = df_unmatched["Matched_Question"]
    df_unmatched["Final_Match_Type"] = df_unmatched.apply(
        lambda row: row["Match_Confidence"] if pd.notnull(row["Suggested_UID"]) else ("🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"),
        axis=1
    )

    uid_conflicts = df_unmatched.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_unmatched["UID_Conflict"] = df_unmatched["Final_UID"].apply(lambda x: "⚠️ Conflict" if x in duplicate_uids else "")

    st.success("✅ Matching complete. Results below:")
    st.dataframe(df_unmatched[[
        "heading_0", "Final_UID", "Final_Question", "Final_Match_Type",
        "Similarity", "Semantic_Similarity", "UID_Conflict"
    ]])

    st.download_button(
        "📥 Download Unified Matches",
        df_unmatched.to_csv(index=False),
        "unified_uid_matches.csv",
        "text/csv"
    )
