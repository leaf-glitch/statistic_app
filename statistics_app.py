import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# title
st.set_page_config(page_title="statistics tool", layout="wide")
st.title("statistics tool")
st.markdown("Values are labeled at the top of the error bars and rounded to two decimal places")

# Session State 
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None

hatch_options = {
    "No Fill": "", "Diagonal Lines (//)": "//", "Anti-Diagonal Lines (\\\\)": "\\\\", 
    "Crossed Lines (xx)": "xx", "Dots (..)": "..", "Plus Signs (++)": "++", 
    "Vertical Lines (||)": "||", "Circles (OO)": "O"
}

def get_sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."

# upload
st.sidebar.header("analysis")
analysis_type = st.sidebar.selectbox("Statistics Test Type", ["one-way ANOVA", "T-test"])
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

if uploaded_file:
    # read data
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file, header=None)
    else:
        df_raw = pd.read_excel(uploaded_file, header=None)
    df_raw = df_raw.dropna(how='all')

    # preview data
    st.subheader(" Raw Data Preview")
    st.dataframe(df_raw, use_container_width=True)

    # 選擇分析範圍
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Select Analysis Range")
    
    selected_rows = []
    for i in range(len(df_raw)):
        row_preview = df_raw.iloc[i].dropna().astype(str).tolist()
        if not row_preview: continue
        preview_text = f"Row {i+1}: {' | '.join(row_preview[:3])}..."
        is_picked = st.sidebar.checkbox(preview_text, value=True, key=f"row_sel_{i}")
        if is_picked:
            selected_rows.append(i)

    if st.sidebar.button(" Execute/Reset Statistical Analysis"):
        data_list = []
        for i in selected_rows:
            row = df_raw.iloc[i].dropna()
            if len(row) < 2: continue
            g_name = str(row.iloc[0])
            for v in row.iloc[1:]:
                try:
                    num_value = float(v)
                    data_list.append({"group": g_name, "value": num_value})
                except (ValueError, TypeError):
                    continue
        
        if data_list:
            final_df = pd.DataFrame(data_list)
            st.session_state.df_final = final_df
            group_data = final_df.groupby("group")["value"].apply(list)
            
            results = {"type": analysis_type, "pairs": []}
            if analysis_type == "one-way ANOVA":
                f_stat, p_val = stats.f_oneway(*group_data)
                results["p_total"] = p_val
                if p_val < 0.05:
                    tukey = pairwise_tukeyhsd(final_df['value'], final_df['group'], 0.05)
                    tukey_res = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                    results["tukey_df"] = tukey_res
                    for _, r in tukey_res.iterrows():
                        results["pairs"].append({
                            "label": f"{r['group1']} vs {r['group2']}",
                            "g1": r['group1'], "g2": r['group2'], "p": r['p-adj'], "sig": r['reject']
                        })
            else: # T-test
                if len(group_data) == 2:
                    t_stat, p_val = stats.ttest_ind(group_data.iloc[0], group_data.iloc[1])
                    results["p_total"] = p_val
                    results["pairs"].append({
                        "label": f"{group_data.index[0]} vs {group_data.index[1]}",
                        "g1": group_data.index[0], "g2": group_data.index[1], "p": p_val, "sig": p_val < 0.05
                    })
            st.session_state.analysis_results = results

# plot
if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    df = st.session_state.df_final
    unique_groups = list(df['group'].unique())
    
    st.markdown("---")
    col_set, col_plot = st.columns([1, 2])

    with col_set:
        st.subheader("plot settings")
        
        group_styles = {}
        for g in unique_groups:
            with st.expander(f"plot settings: {g}"):
                c = st.color_picker(f"color", value="#D0D0D0", key=f"c_{g}")
                h = st.selectbox(f"hatch", list(hatch_options.keys()), key=f"h_{g}")
                group_styles[g] = {"color": c, "hatch": hatch_options[h]}
        
        st.write("Select Significance Lines to Display")
        selected_pairs = []
        for pair in res["pairs"]:
            is_checked = st.checkbox(f"{pair['label']} (p={pair['p']:.4f})", 
                                     value=False, 
                                     key=f"check_{pair['label']}")
            if is_checked:
                selected_pairs.append(pair)

    with col_plot:
        stats_summary = df.groupby("group")["value"].agg(['mean', 'std']).reindex(unique_groups).reset_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(unique_groups))
        max_total_y = (stats_summary['mean'] + stats_summary['std']).max()

        for i, row in stats_summary.iterrows():
            s = group_styles[row['group']]
            ax.bar(i, row['mean'], yerr=[[0], [row['std']]], 
                   color=s['color'], hatch=s['hatch'],
                   edgecolor='black', linewidth=2, capsize=6)
            
            label_y = row['mean'] + row['std'] + (max_total_y * 0.02)
            ax.text(i, label_y, f"{row['mean']:.2f}", ha='center', va='bottom', 
                    fontweight='bold', color='black', fontsize=10)

        if selected_pairs:
            selected_pairs.sort(key=lambda x: abs(unique_groups.index(str(x['g1'])) - unique_groups.index(str(x['g2']))))
            base_y = max_total_y * 1.15
            step_y = max_total_y * 0.15
            
            for i, p in enumerate(selected_pairs):
                idx1, idx2 = unique_groups.index(str(p['g1'])), unique_groups.index(str(p['g2']))
                y = base_y + (i * step_y)
                tick_h = step_y * 0.2
                ax.plot([idx1, idx1, idx2, idx2], [y-tick_h, y, y, y-tick_h], color='black', lw=1.5)
                ax.text((idx1+idx2)/2, y, get_sig_stars(p['p']), ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique_groups)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        
        st.pyplot(fig)
        
    with st.expander(" View Detailed Statistical Report"):
        st.write(f"Overall Test Result: p = {res['p_total']:.6f}")
        if "tukey_df" in res:
            st.dataframe(res["tukey_df"])
