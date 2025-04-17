import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic - Làm sạch & Trực quan", layout="centered")
st.title("Phân tích dữ liệu Titanic")
st.markdown("""
Ứng dụng này sử dụng dữ liệu từ Seaborn để hiển thị trực quan:
- Tỷ lệ sống sót trung bình theo hạng vé (Pclass)
- Phân phối độ tuổi hành khách trên tàu Titanic
""")

# Load và làm sạch dữ liệu
@st.cache_data
def load_and_clean_data():
    df = sns.load_dataset("titanic")

    # Kiểm tra dữ liệu thiếu
    missing_info = df.isnull().sum()

    # Tính median tuổi theo nhóm sex & pclass
    median_age_by_group = df.groupby(['sex', 'pclass'])['age'].median()

    # Hàm thay thế giá trị thiếu
    def fill_missing_age(row):
        if pd.isnull(row['age']):
            return median_age_by_group.loc[row['sex'], row['pclass']]
        else:
            return row['age']

    # Áp dụng điền tuổi
    df['age'] = df.apply(fill_missing_age, axis=1)

    return df, missing_info

df, missing_info = load_and_clean_data()

# Sidebar filters
st.sidebar.header("Lọc dữ liệu")
sex_filter = st.sidebar.multiselect("Chọn giới tính", options=df['sex'].unique(), default=df['sex'].unique())
pclass_filter = st.sidebar.multiselect("Chọn hạng vé (Pclass)", options=df['pclass'].unique(), default=df['pclass'].unique())

# Áp dụng bộ lọc
df_filtered = df[df['sex'].isin(sex_filter) & df['pclass'].isin(pclass_filter)]

# Hiển thị thông tin dữ liệu
with st.expander("Thông tin dữ liệu và giá trị thiếu"):
    st.write("### Thông tin thiếu ban đầu:")
    st.dataframe(missing_info[missing_info > 0])
    st.write("### Sau xử lý, các giá trị thiếu ở 'age' đã được thay thế theo nhóm `sex` + `pclass`.")

# Bar chart: tỷ lệ sống sót theo hạng vé
st.subheader("Tỷ lệ sống sót trung bình theo hạng vé (Pclass)")

survival_by_pclass = df_filtered.groupby("pclass")["survived"].mean().sort_index()

fig1, ax1 = plt.subplots()
ax1.bar(survival_by_pclass.index, survival_by_pclass.values, color='skyblue', edgecolor='black')
ax1.set_xlabel("Hạng vé (Pclass)")
ax1.set_ylabel("Tỷ lệ sống sót")
ax1.set_title("Tỷ lệ sống sót theo hạng vé")
ax1.set_ylim(0, 1)
st.pyplot(fig1)

# Histogram: phân phối độ tuổi
st.subheader("Phân phối độ tuổi hành khách (sau xử lý)")

fig2, ax2 = plt.subplots()
sns.histplot(df_filtered['age'], bins=20, kde=True, ax=ax2, color='orchid')
ax2.set_xlabel("Tuổi")
ax2.set_title("Phân phối độ tuổi hành khách")
st.pyplot(fig2)

# Tùy chọn xem dữ liệu
if st.checkbox("Xem bảng dữ liệu sau xử lý"):
    st.dataframe(df_filtered[['sex', 'pclass', 'age', 'survived']].head(1000))

# Footer
st.caption("Dữ liệu từ Seaborn - Web xây dựng bởi Ngô Đại Hiệp_2121051055")
