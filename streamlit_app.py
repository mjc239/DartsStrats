import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from darts.darts import Dartboard
st.set_page_config(page_title='Darts with Stats',
                   page_icon='🎯',
                   layout="wide")
st.title('🎯 A Data Scientist Plays Darts')

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

    fig = px.imshow(db.db_score_map, origin='lower')
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
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)

    exp_score_map = db.exp_score_map(mu, Sigma, padding=100)
    exp_fig = px.imshow(db.exp_score_map(mu, Sigma, padding=100), origin='lower')
    exp_fig.update_xaxes(showticklabels=False)
    exp_fig.update_yaxes(showticklabels=False)
    exp_fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))

    st.plotly_chart(exp_fig, use_container_width=True)

if page == 'Blackboard':
    st.header('Game Blackboard')

    def start_game(players, start_number):
        st.session_state['current_player'] = 0
        for i in range(len(players)):
            st.session_state[f'player_{i}_score'] = int(start_number)
        st.session_state['game_set_up'] = True

    def update_score(scores, player):
        score = int(sum(scores))
        if st.session_state[f'player_{player}_score'] >= score:
            st.session_state[f'player_{player}_score'] -= score
        st.session_state['current_player'] = (st.session_state['current_player'] + 1) % len(players)

    def skip_player():
        st.session_state['current_player'] = (st.session_state['current_player'] + 1) % len(players)

    def reset_game():
        del st.session_state['current_player']
        for i in range(len(players)):
            st.session_state[f'player_{i}_score'] = start_number

    with st.expander('Game setup'):
        players = [st.text_input(f'Player {i}') for i in range(1, 5)]
        players = [player for player in players if player!='']
        start_number = st.number_input('Start number', value=501)
        st.button('Set up game', on_click=start_game, args=(players, int(start_number)))

    if 'current_player' not in st.session_state:
        st.session_state['current_player'] = 0

    if 'game_set_up' in st.session_state:
        st.subheader('Current scores')
        scores_df = pd.DataFrame(data=[st.session_state[f"player_{i}_score"] for i in range(len(players))],
                                 index=players,
                                 columns=['Current score'])
        st.write(scores_df.to_markdown())

        current_player = st.session_state["current_player"]
        st.markdown('''---''')
        st.subheader(f'{players[current_player]}\'s turn')
        cols = st.columns(3)
        with cols[0]:
            score_1 = st.number_input('Dart 1 score', 0)
        with cols[1]:
            score_2 = st.number_input('Dart 2 score', 0)
        with cols[2]:
            score_3 = st.number_input('Dart 3 score', 0)
        st.button('Submit scores', on_click=update_score, args=([score_1, score_2, score_3], current_player))

        st.button('Skip to next player', on_click=skip_player)
        st.markdown('''---''')
        reset_game = st.button('Reset game', on_click=reset_game)
if page != 'Blackboard':
    if 'game_set_up' in  st.session_state:
        del st.session_state['game_set_up']