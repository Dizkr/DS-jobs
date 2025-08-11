
import streamlit as st
from src.utils import load_latest_file
import plotly.express as px
import pandas as pd

INPUT_DIR = "data/processed"

category_orders_dict = {
    'Experience': [
        'Junior', 'Mid', 'Senior', 'Manager/C-level'
    ],
}

def render():
    #st.header("💵 Salary Overview")

    # Load data
    salaries_df = load_latest_file("justjoinit_salaries", INPUT_DIR)

    categories = salaries_df['Category'].unique()
    categories = categories[categories!="Unclassified"]

    exp_levels = ['Mid', 'Senior']

    sal_types= ['Net per month - B2B', 'Net per hour - B2B', 'Gross per month - Permanent']

    sal_opt = st.segmented_control("Choose salary type", sal_types, selection_mode="single",
    default = 'Net per month - B2B')
    cat_opt = st.segmented_control("Choose category", categories, selection_mode="single")
    exp_opt = st.segmented_control("Choose experience", exp_levels, selection_mode="single")

    if sal_opt is None:
        sal_opt = 'Net per month - B2B'
    if cat_opt is None:
        cat_opt = 'All'
    if exp_opt is None:
        exp_opt = 'All'

    # Apply filters
    df = salaries_df.copy()

    df = df[df['type'] == sal_opt]
    if cat_opt != 'All':
        df = df[df['Category'] == cat_opt]

    if exp_opt != 'All':
        df = df[df['Experience'] == exp_opt]

    # What is the hue?
    if cat_opt == 'All' and exp_opt == 'All':
        hue = 'Experience'
    elif cat_opt != 'All' and exp_opt == 'All':
        hue = 'Experience'
    elif cat_opt == 'All' and exp_opt != 'All':
        hue = 'Category'
    else:  # both filters are applied
        hue = None

    # Sort by mean salary to make plot clearer
    df['mean']=(df['max']+df['min'])/2
    df = df.sort_values(['mean', 'min']).reset_index(drop=True)

    df['y_pos'] = df.index

    # Prepare long-format DataFrame for line plotting
    if hue:
        id_v = ['y_pos', hue]
    else:
        id_v = 'y_pos'

    df_long = pd.melt(
        df,
        id_vars=id_v,
        value_vars=['min', 'max'],
        var_name='bound',
        value_name='salary'
    )

    # Add a line group ID (each pair min-max for one job)
    df_long['line_id'] = df_long['y_pos']

    # Plotly Express line plot
    fig = px.line(
        df_long,
        y='salary',
        x='y_pos',
        color=hue,
        line_group='line_id',
        labels={'y_pis':' ', 'salary': 'Salary'},
        title='Salary Ranges per Job Listing',
        height=600
    )

    # Customize y-axis to hide ticks
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        margin=dict(l=40, r=200),
        showlegend=True
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)