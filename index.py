import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def home_page():
    st.title("Machine Learning with Dataset")
    
    # อธิบายเกี่ยวกับ dataset
    st.write("""
    📊ในหน้านี้ เราจะใช้ dataset ที่เกี่ยวข้องกับข้อมูลของผู้ที่ทำงานในสายเทคโนโลยีและโปรแกรมมิ่ง 
    ซึ่งได้มาจากการสำรวจของ Stack Overflow โดยใช้ข้อมูลจากไฟล์ **survey_results_public.csv**.
    ข้อมูลนี้จะช่วยในการวิเคราะห์ความสัมพันธ์ระหว่างประสบการณ์การเขียนโปรแกรมและเงินเดือนของผู้ทำงานในสายนี้
    """)
    
    # เพิ่มรายละเอียดของ dataset
    st.write("""
    ข้อมูลนี้มีคอลัมน์ที่สำคัญเช่น:
    - **YearsCodePro**: จำนวนปีที่มีประสบการณ์ในการเขียนโปรแกรม
    - **ConvertedCompYearly**: เงินเดือนต่อปีที่ได้รับ (USD)
    
    เราจะทำการแปลงข้อมูลในคอลัมน์ "YearsCodePro" ให้เป็นตัวเลข และวิเคราะห์ข้อมูลเหล่านี้ในขั้นตอนต่าง ๆ
    """)
    st.markdown("<h3 style='text-align: center;'>ผมจะทำความสะอาดข้อมูลก่อน</h3>", unsafe_allow_html=True)
    st.image("mac1.jpg", width=800)

    st.markdown("<h4 style='text-align: center;'>ทำการสร้างกราฟกระจาย เพื่อแสดงความสัมพันธ์ระหว่าง จำนวนปีที่เขียนโปรแกรม กับเงินเดือนประจำปี</h4>", unsafe_allow_html=True)
    st.image("mac2.jpg", width=800)

    st.markdown("<h3 style='text-align: center;'>ทำการแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบและใช้โมเดล Linear Regression ในการทำนายเงินเดือนตามจำนวนปีที่เขียนโปรแกรม</h3>", unsafe_allow_html=True)
    st.image("mac3.jpg", width=800)

    st.markdown("<h3 style='text-align: center;'>ประเมินผลลัพธ์ของโมเดลที่ฝึกมาและแสดงกราฟที่มีเส้นการทำนาย (Regression Line) เพื่อให้เห็นว่าความสัมพันธ์ระหว่างประสบการณ์และเงินเดือนเป็นอย่างไร</h3>", unsafe_allow_html=True)
    st.image("mac4.jpg", width=800)
    
    st.write(""" โดยผมได้นำ data set มาจาก Stack Overflow """)
    st.markdown("Survey results public(https://survey.stackoverflow.co)")
    
# หน้าเกี่ยวกับ (Demo Machine Learning)
def about_page():
    st.title("Demo Machine Learning")
    st.write("Demo Machine Learning.") 


    # === 1. โหลดข้อมูล ===
    file_path = "survey_results_public.csv"  # เปลี่ยนเป็นที่อยู่ไฟล์ของคุณ

    # โหลดข้อมูลและแสดงบางส่วน
    @st.cache_data
    def load_data():
        return pd.read_csv(file_path)

    df = load_data()

    # === 2. ทำความสะอาดข้อมูล ===
    def convert_years_code(x):
        """แปลง YearsCodePro เป็นตัวเลข"""
        if pd.isnull(x):
            return np.nan
        elif x == "Less than 1 year":
            return 0.5
        elif x == "More than 50 years":
            return 51   
        else:
            try:
                return float(x)
            except ValueError:
                return np.nan

    # แปลงข้อมูลปีประสบการณ์
    df["YearsCodePro"] = df["YearsCodePro"].apply(convert_years_code)

    # === 3. กรองข้อมูล ===
    # กรองเฉพาะข้อมูลที่ไม่มี NaN และกำจัดค่าผิดปกติใน "ConvertedCompYearly"
    df_filtered = df[["YearsCodePro", "ConvertedCompYearly"]].dropna()

    # กรองเฉพาะข้อมูลที่มีเงินเดือนในช่วงที่เหมาะสม
    df_filtered = df_filtered[(df_filtered["ConvertedCompYearly"] > 1000) & (df_filtered["ConvertedCompYearly"] < 500000)]

    # === 4. สร้างอินเตอร์เฟซด้วย Streamlit ===
    st.title("Machine Learning with Dataset")

    st.write("""
    ในหน้านี้ คุณสามารถกรอกข้อมูลปีที่มีประสบการณ์ในการเขียนโปรแกรม เพื่อทำนายเงินเดือน
    และดูกราฟแสดงความสัมพันธ์ระหว่างประสบการณ์และเงินเดือน
    """)

    # รับข้อมูลจากผู้ใช้
    years_of_experience = st.slider("กรุณากรอกจำนวนปีที่มีประสบการณ์การเขียนโปรแกรม:", 0, 51, 1)

    # === 5. แบ่งข้อมูลสำหรับ Train & Test ===
    X = df_filtered[["YearsCodePro"]]
    y = df_filtered["ConvertedCompYearly"]

    # แบ่งข้อมูลเป็น train และ test
    X_train = X
    y_train = y

    # === 6. เทรนโมเดล Linear Regression ===
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ปุ่มเพื่ออัปเดตกราฟและการทำนาย
    if st.button('Update Graph'):
        # ทำนายค่าเงินเดือนจากข้อมูลที่กรอก
        predicted_salary = model.predict([[years_of_experience]])

        # แสดงผลลัพธ์การทำนาย
        st.write(f"เงินเดือนที่คาดว่าจะได้รับ (USD) สำหรับผู้มีประสบการณ์ {years_of_experience} ปี: {predicted_salary[0]:,.2f} USD")

        # === 7. แสดงกราฟแสดงความสัมพันธ์ ===
        # สร้าง figure และ axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # พล็อตข้อมูลจริง
        sns.scatterplot(data=df_filtered, x="YearsCodePro", y="ConvertedCompYearly", alpha=0.5, label="Actual Data", ax=ax)

        # พล็อตเส้น Regression โดยใช้ค่าเรียงลำดับ
        X_sorted = np.sort(X.values.reshape(-1))
        y_sorted_pred = model.predict(X_sorted.reshape(-1, 1))
        sns.lineplot(x=X_sorted, y=y_sorted_pred, color="red", label="Regression Line", ax=ax)

        # เพิ่มเส้นทำนายเงินเดือนจากข้อมูลที่กรอก
        predicted_line = model.predict(np.array([[years_of_experience]]))  # คำนวณค่าเงินเดือนจากปีที่กรอก
        ax.plot([years_of_experience], predicted_line, 'go', label=f"Predicted Salary at {years_of_experience} years")

        # ตั้งค่าแกน x, y และชื่อกราฟ
        ax.set_xlabel("Years of Professional Coding Experience")
        ax.set_ylabel("Yearly Salary (USD)")
        ax.set_title("Linear Regression: Experience vs Salary")
        
        # แสดงกราฟใน Streamlit
        ax.legend()
        st.pyplot(fig)  # เรียกใช้เพียงครั้งเดียว
    
# หน้าบริการ (Neural Network)
def services_page():
    st.title("Neural Network")
    st.write("""
    ข้อมูลที่คุณใช้คือ Netflix Movies and TV Shows dataset ซึ่งมีข้อมูลเกี่ยวกับภาพยนตร์และรายการทีวีที่มีอยู่บน Netflix โดยคอลัมน์ (features) ใน dataset นี้อาจมีรายละเอียดต่าง ๆ ที่ให้ข้อมูลเกี่ยวกับภาพยนตร์หรือรายการทีวีที่มีอยู่ในแพลตฟอร์ม Netflix ซึ่งประกอบด้วย:
    ### features หลักใน dataset นี้:
    1. **show_id**: หมายเลขหรือรหัสเฉพาะที่ใช้ระบุแต่ละรายการ
    2. **type**: ประเภทของรายการ — เป็นการบ่งบอกว่าเป็น Movie หรือ TV Show (ภาพยนตร์หรือรายการทีวี)
    3. **title**: ชื่อของภาพยนตร์หรือรายการทีวี
    4. **director**: ชื่อของผู้กำกับ (ถ้ามี)
    5. **cast**: นักแสดงหรือบุคคลที่มีบทบาทในรายการ
    6. **country**: ประเทศที่ผลิตภาพยนตร์หรือรายการ
    7. **date_added**: วันที่รายการถูกเพิ่มเข้ามาใน Netflix
    8. **release_year**: ปีที่ภาพยนตร์หรือรายการถูกปล่อยออกมา
    9. **rating**: การจัดอันดับหรือเรตติ้งของภาพยนตร์หรือรายการ (เช่น PG, R, TV-MA)
    10. **duration**: ความยาวของภาพยนตร์หรือรายการในรูปแบบของ "X min" สำหรับภาพยนตร์ หรือ "X Seasons" สำหรับรายการทีวี
    11. **listed_in**: หมวดหมู่ที่ Netflix ใช้ในการจัดระเบียบภาพยนตร์หรือรายการ (เช่น ดราม่า, คอมเมดี้, สารคดี เป็นต้น)
    12. **description**: คำอธิบายเกี่ยวกับเนื้อหาของภาพยนตร์หรือรายการทีวี
    """)
    
    st.write(""" เริ่มจาก โหลดข้อมูล Netflix และ กำจัดข้อมูลที่หายไป""")
    st.image('in1.png')
    
    st.write(""" แปลง colum เป็นตัวเลข """)
    st.image('in2.png')
    
    st.write(""" เลือกว่าจะใช้ข้อมูลไหนในการแยกระหว่าง movie และ TV show ปรับและแบ่งข้อมูล """)
    st.image('in3.png')
    
    st.write(""" สร้าง model neural network """)
    st.image('in4.png')
    
    st.write(""" ทำการเทรนโมเดลและแสดงกราฟ """)
    st.image('in5.png')
    
    st.write(""" output แสดงความแม่นยำในการเทรน """)
    st.image('in6.png')

    st.write(""" โดยผมได้นำ data set มาจาก Kaggle """)
    st.markdown("Netflix Movies and TV Shows(https://www.kaggle.com/datasets/shivamb/netflix-shows)")

# หน้าติดต่อเรา (Demo Neural Network)
def contact_page():
    st.title("Demo Neural Network")
    st.image('ne.png')

    # โหลดข้อมูล Netflix
    df = pd.read_csv('C:/Users/Extended-AMD/Desktop/netflix_titles.csv')

    # กำจัดข้อมูลที่หายไป
    df.dropna(subset=['duration', 'release_year'], inplace=True)  # ลบข้อมูลที่มีค่าว่างใน duration หรือ release_year

    # แปลงคอลัมน์ 'type' (Movie / TV Show) เป็นตัวเลข (0: Movie, 1: TV Show)
    df['type'] = df['type'].map({'Movie': 0, 'TV Show': 1})

    # แปลงคอลัมน์ 'duration' จาก 'X min' หรือ 'X Seasons' เป็นตัวเลข
    def convert_duration(duration):
        if isinstance(duration, str):
            # ถ้าคือ 'min', แปลงเป็นตัวเลข
            if 'min' in duration:
                return int(duration.split(' ')[0]), 'Movie'  # เป็น Movie
            # ถ้าคือ 'Season', แปลงเป็นตัวเลข
            elif 'Season' in duration:
                return int(duration.split(' ')[0]), 'TV Show'  # เป็น TV Show
        return 0, 'Unknown'  # กรณีไม่พบข้อมูล

    # แปลง duration และประเภท
    df['duration_value'], df['category'] = zip(*df['duration'].apply(convert_duration))

    # การแยกประเภทโดยใช้ category
    movies_df = df[df['category'] == 'Movie']  # Movies
    tv_shows_df = df[df['category'] == 'TV Show']  # TV Shows

    # แสดงผล Movies
    st.write("Movies")
    st.write(movies_df[['title', 'duration', 'release_year']].head(10))

    # แสดงผล TV Shows
    st.write("TV Shows")
    st.write(tv_shows_df[['title', 'duration', 'release_year']].head(10))

    # สร้าง StandardScaler เพื่อปรับขนาดข้อมูล
    scaler = StandardScaler()

    # เลือกฟีเจอร์ที่ใช้ในการฝึกโมเดล
    X = df[['release_year', 'duration_value']]
    y = df['type']

    # การปรับขนาดข้อมูล
    X_scaled = scaler.fit_transform(X)

    # แบ่งข้อมูลเป็น Train และ Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # สร้างโมเดล Neural Network ด้วย Keras
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # ชั้นแรก 64 neurons
    model.add(Dense(32, activation='relu'))  # ชั้นที่สอง 32 neurons
    model.add(Dense(1, activation='sigmoid'))  # ชั้นสุดท้าย Sigmoid สำหรับการทำนายแบบ Binary Classification

    # คอมไพล์โมเดล
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ฝึกโมเดล
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # ทำนายผล
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)  # ปรับผลทำนายให้อยู่ในช่วง 0 หรือ 1

    # ประเมินผลด้วย Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")

    # การกรอกค่า duration แบบ Minutes หรือ Seasons
    duration_type = st.selectbox("Select type of duration:", ["Minutes (Movie)", "Seasons (TV Show)"])

    if duration_type == "Minutes (Movie)":
        duration = st.number_input("Enter the duration in minutes for Movie:", min_value=0)
        category = 'Movie'
    else:
        duration = st.number_input("Enter the duration in seasons for TV Show:", min_value=0)
        category = 'TV Show'

    # ทำนายประเภท (Movie or TV Show)
    if duration > 0:
        # ใช้ release_year ล่าสุดเป็น 2025 (หรือปีอื่น ๆ ที่ต้องการ)
        input_data = pd.DataFrame([[2025, duration]], columns=['release_year', 'duration_value'])  
        input_scaled = scaler.transform(input_data)
        
        # ทำนายผล
        prediction = model.predict(input_scaled)
        prediction = (prediction > 0.5).astype(int)

        # แสดงผลลัพธ์การทำนาย
        if prediction == 1:
            st.write(f"It is likely a {category}.")
        else:
            st.write(f"It is likely a {category}.")

# สร้างแถบเลือกหน้า
pages = {
    "Machine Learning": home_page,
    "Demo Machine Learning": about_page,
    "Neural Network": services_page,
    "Demo Neural Network": contact_page,
}

# ใช้ radio button เพื่อเลือกหน้า
page = st.sidebar.radio("Select a page", options=list(pages.keys()))

# แสดงเนื้อหาของหน้าที่เลือก
pages[page]()