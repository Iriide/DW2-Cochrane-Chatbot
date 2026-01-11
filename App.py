import streamlit as st
from src.RAGmodel import RAGModel
import pandas as pd

def run_queries(all_queries, answers, model):

    for i, query in enumerate(all_queries, 1):
        with st.expander(f"Query {i}: {query}", expanded=False):
            response, simplified_prompt, used_chunks, tokens_used, custom_tokens_used, quality, trust = model.ask(query, answer=answers[i-1] if i <= len(answers) else None)
            print(f"Quality: {quality}")
            st.markdown(f"**Response:**\n\n{response}")
            st.markdown(f"**Simplified Prompt:**\n\n```{simplified_prompt}```")

            st.markdown("**Retrieved Chunks:**")
            for chunk in used_chunks:
                st.markdown(f"- **Section:** {chunk['section']}")
                st.markdown(f"  ```{chunk['content'][:300]}...```")

            quality_info_keys = [k for k in quality.keys() if k != "low_quality_answer_reason"]
            col = st.columns(len(quality_info_keys))
            for k, key in enumerate(quality_info_keys):
                col[k].metric(key.replace("_", " ").title(), str(quality[key]))
            col = st.columns(3)
            for k, (key, value) in enumerate(trust.items()):
                col[k].metric(f"Trust {key}", f"{value:.2f}")
            col = st.columns(3)
            for k, (key, value) in enumerate(custom_tokens_used.items()):
                col[k].metric(f"Custom Token {key}", f"{value:.2f}")
            col1, col2 = st.columns(2)
            col1.metric("Token Count", tokens_used)
            col2.metric("Low Quality Reason", "None" if not len(quality["low_quality_answer_reason"]) > 0 else
            quality["low_quality_answer_reason"][0])


def main():
    with open("src/questions/questions_cochrane(Human-made questions).csv", "r") as f:
        data = pd.read_csv(f, delimiter=';').to_dict(orient="records")

    queries = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    model = RAGModel()

    st.set_page_config(page_title="RAGModel UI", layout="wide")
    st.title("🎓 Cochrane Handbook Assistant")

    if "ask_custom" not in st.session_state:
        st.session_state.ask_custom = False

    if st.button("Run RAGModel on All Queries"):
        run_queries(queries, answers, model)

    if st.button("Ask a Custom Query"):
        st.session_state.ask_custom = True

    if st.session_state.ask_custom:
        custom_query = st.text_input("Enter your query:")
        if custom_query:
            run_queries([custom_query], [], model)
    else:
        st.info(
            f"Click one of the buttons above to run the model on {len(queries)} common queries or ask your own")

if __name__ == "__main__":
    main()