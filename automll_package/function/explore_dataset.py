from streamlit_extras.dataframe_explorer import dataframe_explorer
import streamlit as st

cleaned_data = st.session_state["cleaned_data"]
data = st.session_state["data"]
file_name = st.session_state["file_name"]


@st.cache_data
def convert_df_to_csv(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def explore_datasets(data,cleaned_data):
    # ------Creating a sided bar
    DataFrame = st.selectbox("Select DataFrame",["Original DataFrame","cleaned DataFrame"])

    if DataFrame == "Original DataFrame":
        st.title("📊 Original DataFrame")

        # Add a checkbox to show all columns
        show_all_columns = st.checkbox("Show All Columns",value=True)

        if show_all_columns:
            filterd_df = dataframe_explorer(data)
            st.dataframe(filterd_df)
            st.download_button(
                    label="Download data as CSV",
                    data=convert_df_to_csv(filterd_df),
                    file_name=f"{file_name}_updated_by_yb.csv",
                    mime='text/csv',
                    )
            if st.button('Save Data'):
                    cleaned_data = filterd_df
                    st.success("Successfully Save")
        else:
            selected_column = st.multiselect("Select a columns",data.columns)
            
            if selected_column:
                st.subheader("cleaned DataFrame")
                filterd_df = dataframe_explorer(data[selected_column])
                st.dataframe(filterd_df)
                st.download_button(
                    label="Download data as CSV",
                    data=convert_df_to_csv(filterd_df),
                    file_name=f"{file_name}_updated_by_yb.csv",
                    mime='text/csv',
                    )
                if st.button('Save Data'):
                    cleaned_data = filterd_df
                    st.success("Successfully Save")
            else:
                st.warning("Please select at least one column.")


    else:

        st.title("🧹 cleaned DataFrame")

        # Add a checkbox to show all columns
        show_all_columns = st.checkbox("Show All Columns",value=True)

        # If show_all_columns is True, show all columns. Otherwise, use multi-select
        if show_all_columns:
            st.subheader("All Columns")
            filterd_df = dataframe_explorer(cleaned_data)
            st.dataframe(filterd_df)
            st.download_button(
                    label="Download data as CSV",
                    data=convert_df_to_csv(filterd_df),
                    file_name=f"{file_name}_updated_by_yb.csv",
                    mime='text/csv',
                    )
            if st.button('Save Data'):
                cleaned_data = filterd_df
                st.success("Successfully Save")
        else:
            # Select a particular column to show
            selected_column = st.multiselect("Select a Column", cleaned_data.columns)

            if selected_column:
                st.subheader("cleaned DataFrame")
                filterd_df = dataframe_explorer(cleaned_data[selected_column])
                st.dataframe(filterd_df)
                st.download_button(
                    label="Download data as CSV",
                    data=convert_df_to_csv(filterd_df),
                    file_name=f"{file_name}_updated_by_DataBot.csv",
                    mime='text/csv',
                    )
                if st.button('Save Data'):
                    cleaned_data = filterd_df
                    st.success("Successfully Save")
            else:
                st.warning("Please select at least one column.")

    st.session_state["cleaned_data"] = cleaned_data
# st.session_state["file_name"] = file_name
