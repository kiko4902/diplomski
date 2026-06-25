import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.title("SMOTE Visualizer")

if st.button("Start"):
    # Testiraj jedan po jedan widget
    opt = st.selectbox("Dataset", ["A", "B", "C"])
    st.write(f"Selected: {opt}")

    k = st.slider("k", 1, 15, 5)
    st.write(f"k = {k}")

    n = st.number_input("Samples", 50, 500, 200)
    st.write(f"Samples: {n}")

    if st.button("Generate"):
        st.success(f"Done: {opt}, k={k}, n={n}")
