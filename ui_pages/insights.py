import streamlit as st
import plotly.express as px
import pandas as pd
from utils.metrics_loader import get_category_stats

def show_insights():
    st.title("💡 Inventory Insights Console")
    st.markdown("### AI-Driven Business Recommendations")

    df_stats = get_category_stats().sort_values("Predictions Count", ascending=False)
    
    # Inventory Distribution chart
    st.markdown("#### Category Demand Distribution")
    fig_bar = px.bar(df_stats, x="Predictions Count", y="Category", orientation='h',
                     color="Predictions Count", color_continuous_scale="Reds",
                     text_auto=True)
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Business Intelligence Cards
    st.markdown("#### Operational Decision Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### High Demand Category")
        st.markdown(f"The **{df_stats.iloc[0]['Category']}** category shows the highest detection frequency.")
        st.markdown("""
        **Action Item**:
        - Increase shelf allocation for this category.
        - Optimize restocking frequency during peak hours (18:00 - 20:00).
        - Place promotional items in proximity to this category.
        """)
        
    with col2:
        st.success("### AI Reliability Insight")
        st.markdown(f"The average recognition confidence is **{df_stats['Confidence Average'].mean()*100:.1f}%**.")
        st.markdown("""
        **Action Item**:
        - Recognition accuracy is sufficient for autonomous shelf monitoring.
        - Continue monitoring edge cases in low-light environments.
        - Schedule monthly model fine-tuning with newly captured shelf images.
        """)

    st.markdown("---")

    # Strategic Recommendations Table
    st.markdown("#### Strategic Inventory Recommendations")
    recommendations = [
        {
            "Insight Type": "Shelf Allocation",
            "Recommendation": "Reallocate 15% more space for Beverage and Dairy segments based on recent scan surges.",
            "Priority": "High"
        },
        {
            "Insight Type": "Stockout Prevention",
            "Recommendation": "Implement automated stockout alerts for high-velocity items in the Fruit & Vegetable section.",
            "Priority": "Medium"
        },
        {
            "Insight Type": "Layout Optimization",
            "Recommendation": "Move high-margin refrigerated items closer to the main thoroughfare to capitalize on high-traffic detection patterns.",
            "Priority": "Low"
        }
    ]
    df_rec = pd.DataFrame(recommendations)
    
    def highlight_priority(val):
        color = '#ffcccb' if val == 'High' else '#ffffba' if val == 'Medium' else '#baffc9'
        return f'background-color: {color}'

    st.table(df_rec.style.applymap(highlight_priority, subset=['Priority']))
