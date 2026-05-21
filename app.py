# =========================================================
# TECHNOLOGY COMPARISON TAB
# =========================================================

with tabs[5]:

    st.subheader("🚀 Proposed System vs Existing Technologies")

    comparison_data = {

        "Feature": [

            "Rule-Based Detection",

            "Machine Learning",

            "Reinforcement Learning",

            "Traffic Prediction",

            "Real-Time Dashboard",

            "Edge Computing",

            "Cloud Integration",

            "Adaptive Mitigation",

            "Live Monitoring",

            "Automatic Response"
        ],

        "Traditional IDS": [

            1, 0, 0, 0, 0, 0, 0, 0, 1, 0
        ],

        "ML-Based IDS": [

            0, 1, 0, 0, 1, 0, 0, 0, 1, 0
        ],

        "Proposed System": [

            1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
    }

    comparison_df = pd.DataFrame(
        comparison_data
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    # =====================================================
    # BAR CHART
    # =====================================================

    fig_compare = go.Figure()

    fig_compare.add_trace(

        go.Bar(

            name="Traditional IDS",

            x=comparison_df["Feature"],

            y=comparison_df["Traditional IDS"]
        )
    )

    fig_compare.add_trace(

        go.Bar(

            name="ML-Based IDS",

            x=comparison_df["Feature"],

            y=comparison_df["ML-Based IDS"]
        )
    )

    fig_compare.add_trace(

        go.Bar(

            name="Proposed System",

            x=comparison_df["Feature"],

            y=comparison_df["Proposed System"]
        )
    )

    fig_compare.update_layout(

        title="📊 Technology Comparison",

        barmode='group',

        height=500
    )

    st.plotly_chart(

        fig_compare,

        use_container_width=True
    )

    # =====================================================
    # SYSTEM DESCRIPTION
    # =====================================================

    st.markdown("""

    ## ✅ Advantages of Proposed System

    - Combines ML + Rule-Based Detection
    - Uses Reinforcement Learning
    - Supports Edge + Cloud Computing
    - Performs Traffic Prediction
    - Provides Real-Time Monitoring
    - Automatically Selects Mitigation Actions
    - Adaptive to Dynamic Traffic Conditions

    """)

    # =====================================================
    # SCORE CARD
    # =====================================================

    st.subheader("🏆 Innovation Score")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(

            "Traditional IDS",

            "20%"
        )

    with c2:

        st.metric(

            "ML IDS",

            "45%"
        )

    with c3:

        st.metric(

            "Your Proposed System",

            "95%"
        )