import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score


QREF = 1055


def get_data() -> io.StringIO:

    data = st.sidebar.text_area('Data').replace('\t', ',')
    with st.sidebar.beta_expander('Help', expanded=False):
        st.markdown('''
        Enter HRR Data. Do not include column names. Expects 2 columns: 
        
        * Time in seconds
        * HRR in kW
        
        If copied from excel, tabs will be automatically converted to commas.
        ''')

    return io.StringIO(data)


def graph_original(df):

    cht = alt.Chart(df).mark_line().encode(
        x=alt.X('Time', title='Time [sec]'),
        y=alt.Y('HRR', title='HRR [kW]'),
    )
    pts = cht.mark_point().encode(
        color='used',
        tooltip=['Time', 'HRR']
    )
    st.altair_chart(cht + pts, use_container_width=True)


def ramp_original(df):

    with st.beta_expander('FDS RAMP', expanded=False):
        peak = max(df.HRR)
        st.code('\n'.join([
            f"&RAMP ID='HRR_RAMP', T={row.Time:.3f}, F={row.HRR / peak:.3f}/"
            for row in df.itertuples()
        ]))


def t2_hrr(times: np.array, tv: int, tg: int, qref: int = QREF) -> np.array:
    hrrs = qref * ((times - tv) / tg) ** 2
    hrrs[np.where(times <= tv)] = 0
    return hrrs


def find_best(df: pd.DataFrame):

    times = df.loc[df.used, 'Time']

    tv = st.sidebar.slider(
        'Virtual Time',
        min_value=0,
        max_value=int(times.max()),
        value=0,
    )
    tg = st.sidebar.slider(
        'Growth Time',
        min_value=0,
        max_value=int(times.max()),
        value=int(times.max()),
    )

    guess = pd.DataFrame({
        'Time': times,
        'curve': 'guess',
        'HRR': t2_hrr(times.values, tv=tv, tg=tg),
    })
    score = r2_score(df.loc[df.used, 'HRR'], guess.HRR)

    st.latex(r'\dot{Q}=\dot{Q}_{ref}\left(\frac{t-t_v}{t_g}\right)^2')
    st.latex(
        f'\\dot{{Q}}={QREF} kW\\left(\\frac{{t-{tv} s}}{{{tg} s}}\\right)^2')
    st.latex(f'R^2={{{score:.3f}}}')

    cht = alt.Chart(
        pd.concat([df, guess])
    ).mark_line().encode(
        x='Time',
        y='HRR',
        color='curve'
    )
    st.altair_chart(cht, use_container_width=True)


def main():

    data = get_data()
    df = pd.read_csv(data, header=None, names=['Time', 'HRR'])
    df['curve'] = 'orig'
    df['used'] = False

    if df.empty:
        st.stop()

    times = st.sidebar.slider(
        't Max',
        min_value=0,
        max_value=int(df.Time.max()),
        value=(0, int(df.loc[df.HRR.idxmax(), 'Time'])),
    )

    df.loc[(df.Time > times[0]) & (df.Time < times[1]), 'used'] = True

    graph_original(df)
    ramp_original(df)
    find_best(df)


if __name__ == '__main__':
    main()
