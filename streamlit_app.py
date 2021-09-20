import streamlit as st
import numpy as np
import plotly.express as px

from darts.darts import Dartboard
st.set_page_config(layout="wide")
st.title('A Data Scientist Plays Darts')

page = st.sidebar.radio('Page:', ['Expected score map', 'Blackboard', 'Distribution calculator'])
if page == 'Expected score map':
    st.header('Expected Score Map')
    PIXELS = 401
    db = Dartboard(pixels=PIXELS)

    nonspherical = st.checkbox('Non-spherical distribution?')
    if nonspherical:
        tilted = st.checkbox('Tilted Gaussian distribution?')

    with st.form('Parameters:') as form:
        mean_cols = st.columns(2)
        with mean_cols[0]:
            mu_x = st.number_input('x mean', min_value=-200.0, max_value=200.0, value=0.0)
        with mean_cols[1]:
            mu_y = st.number_input('y mean', min_value=-200.0, max_value=200.0, value=0.0)
        mu = np.array([mu_x, mu_y])

        if not nonspherical:
            sigma = st.number_input('Throw variance', value=50.0)
            Sigma = sigma*np.eye(2)
        else:
            if not tilted:
                sigma_x = st.number_input('x variance', value=50.0)
                sigma_y = st.number_input('y variance', value=50.0)
                Sigma = np.array([[sigma_x, 0.0], [0.0, sigma_y]])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    sigma_11 = st.number_input('sigma_11')
                    sigma_21 = st.number_input('sigma_21')
                with col2:
                    sigma_12 = st.number_input('sigma_12', value=sigma_21)
                    sigma_22 = st.number_input('sigma_22')
                Sigma = np.array([[sigma_11, sigma_12], [sigma_21, sigma_22]])
        st.form_submit_button()

    fig = px.imshow(db.db_score_map, origin='lower', height=1000)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    aim_center = [int(PIXELS/2), int(PIXELS/2)]
    throw_stds = [s/db.mm_per_pixel for s in [np.sqrt(Sigma[0, 0]), np.sqrt(Sigma[1, 1])]]
    fig.add_shape(type='circle',
                  x0=aim_center[0]+throw_stds[0],
                  y0=aim_center[1]-throw_stds[1],
                  x1=aim_center[0]-throw_stds[0],
                  y1=aim_center[1]+throw_stds[1],
                  line_color='red',
                  fillcolor='red', opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    exp_score_map = db.exp_score_map(mu, Sigma, padding=100)
    exp_fig = px.imshow(db.exp_score_map(mu, Sigma, padding=100),
                        origin='lower', height=1000)
    exp_fig.update_xaxes(showticklabels=False)
    exp_fig.update_yaxes(showticklabels=False)

    st.plotly_chart(exp_fig, use_container_width=True)
