from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import pandas as pd
import tempfile
import pickle
import base64

def get_download_link(df, filename):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
    return my_autopct
    
def processing(df):
    st.text("Step 1: Filtering data for 'PMaps Sales Orientation' section...")
    ques_id_to_drop = [25131, 25132, 25133, 25134, 25140, 25141, 25142, 25143, 25144]
    df = df[df['SectionName'] == 'PMaps Sales Orientation']
    
    st.text("Step 2: Selecting relevant columns...")
    df = df[['CandidateUniqueId', 'EmailAddress', 'OriginalQuestionId', 'OriginalOptionId']]
    
    st.text("Step 3: Dropping specified question IDs...")
    df = df[~df['OriginalQuestionId'].isin(ques_id_to_drop)]
    
    st.text("Step 4: Grouping by candidate and question...")
    idx = df.groupby(['EmailAddress', 'OriginalQuestionId'])['CandidateUniqueId'].idxmin()
    df = df.loc[idx].reset_index(drop=True)
    
    st.text("Step 5: Pivoting the DataFrame...")
    df_pivoted = df.pivot(index='EmailAddress', columns='OriginalQuestionId', values='OriginalOptionId')
    df_pivoted = df_pivoted.fillna(0).astype(int)
    
    st.text("Step 6: One-hot encoding the data...")
    encoded_df = pd.get_dummies(df_pivoted, prefix_sep='_', columns=df_pivoted.columns)
    encoded_df = encoded_df[[col for col in encoded_df.columns if not col.endswith('_0')]]
    
    st.text("Step 7: Creating a new DataFrame with encoded values...")
    new_df = pd.DataFrame(index=df_pivoted.index, columns=encoded_df.columns, dtype=int).fillna(0)
    new_df.update(encoded_df)

    return new_df.copy(deep=True)

def main():
    st.title("PMaps Prediction App")
    
    option = st.selectbox('Choose your Analysis Mode',('Joint Model', 'Performance Only', 'Attrition Only'))
    with st.expander(f'Traning Insights of : {option} are as follows:'):    
        graphs = Image.open(f"{option}/graph.png")
        model = pickle.load(open(f"{option}/model.pkl", 'rb'))
        st.image(graphs)
        with open(f"{option}/feature.csv", "rb") as file:
            st.download_button(label="Feature Importance.csv", data=file, 
            file_name="feature.csv", mime="text/csv",)

    st.sidebar.title("Upload Positonal Data")
    
    instructions = """
    Instructions:
    Please provide a CSV file containing the following mandatory columns:

    1. `CandidateUniqueId`: Unique identifier for each candidate.
    2. `EmailAddress`: Email address of the candidate.
    3. `SectionName`: Name of the section in the test.
    4. `OriginalQuestionId`: Identifier for the original question.
    5. `OriginalOptionId`: Identifier for the original option selected by the candidate.
    """
    st.sidebar.markdown(instructions)
    st.sidebar.markdown("Please confirm that you have read and understood the instructions below before proceeding.")
    instructions_confirmed = st.sidebar.checkbox("I confirm that I have read and understood the instructions")
    
    if instructions_confirmed:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            st.sidebar.text("Processing...")
            df = pd.read_csv(uploaded_file)
            processed_df = processing(df)
            
            st.sidebar.text("Making Predictions...")
            preds = model.predict(processed_df)
            probs = model.predict_proba(processed_df)
            probs = list(map(lambda x: str(round(100*max(x),2))+'%', probs))
    
            df_out = processed_df.copy(deep=True)
                        
            if option == 'Joint Model':
                df_out['Performance'] = list(map(lambda x: x.split('_')[0] , preds))
                df_out['Attrition'] = list(map(lambda x: x.split('_')[1] , preds))
                
                df_out['Confidence'] = probs
                df_out = df_out[['Performance', 'Attrition', 'Confidence']].astype(str)
                
            elif option == 'Performance Only':
                df_out['Performance'], df_out['Confidence'] = preds, probs
                df_out = df_out[['Performance', 'Confidence']].astype(str)
                
            elif option == 'Attrition Only':
                df_out['Attrition'], df_out['Confidence'] = preds, probs
                df_out = df_out[['Attrition', 'Confidence']].astype(str)

            st.sidebar.text("Saving Predictions...")

            st.sidebar.text("Making Graphs...")
            
            if 'Performance' in df_out.columns:
                frequency_perf = Counter(df_out['Performance'])
                labels_perf, sizes_perf = zip(*frequency_perf.items())
                labels_perf = [x + ' Performer' for x in labels_perf]
                min_size_index_perf = sizes_perf.index(min(sizes_perf))
                explode_perf = [0.05 if i == min_size_index_perf else 0 for i in range(len(sizes_perf))]

            if 'Attrition' in df_out.columns:
                frequency_attr = Counter(df_out['Attrition'])
                labels_attr, sizes_attr = zip(*frequency_attr.items())
                min_size_index_attr = sizes_attr.index(min(sizes_attr))
                explode_attr = [0.05 if i == min_size_index_attr else 0 for i in range(len(sizes_attr))]

            if option == 'Joint Model':
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
                axs[0].pie(sizes_perf, labels=labels_perf, explode=explode_perf, autopct=make_autopct(sizes_perf), startangle=10)
                axs[0].set_title('Performance Distribution')
                axs[0].axis('equal')
        
                axs[1].pie(sizes_attr, labels=labels_attr, explode=explode_attr, autopct=make_autopct(sizes_attr), startangle=10)
                axs[1].set_title('Attrition Distribution')
                axs[1].axis('equal')

            elif option == 'Performance Only':
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
                axs.pie(sizes_perf, labels=labels_perf, explode=explode_perf, autopct=make_autopct(sizes_perf), startangle=10)
                axs.set_title('Performance Distribution')
                axs.axis('equal')

            elif option == 'Attrition Only':
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
                axs.pie(sizes_attr, labels=labels_attr, explode=explode_attr, autopct=make_autopct(sizes_attr), startangle=10)
                axs.set_title('Attrition Distribution')
                axs.axis('equal')
            
            st.sidebar.text("Completed!")
            st.markdown(get_download_link(df_out, 'prediction_data.csv'), unsafe_allow_html=True)

            plt.tight_layout()
            with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as tmpfile:
                plt.savefig(tmpfile.name, format='png')
                st.image(tmpfile.name)
            plt.close(fig)
            
            st.write("### Predictions")
            st.write(df_out)

if __name__ == "__main__":
    main()
