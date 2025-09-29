import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from ta.trend import sma_indicator, macd, macd_signal
from ta.momentum import rsi
from ta.volatility import BollingerBands

def add_indicators(df):
    """Thêm các chỉ báo kỹ thuật cần thiết vào DataFrame"""
    df["SMA_20"] = sma_indicator(df["Close"], window=20)
    df["SMA_50"] = sma_indicator(df["Close"], window=50)
    df["RSI"] = rsi(df["Close"], window=14)
    df["MACD"] = macd(df["Close"])
    df["MACD_Signal"] = macd_signal(df["Close"])

    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    return df

st.set_page_config(
    page_title="Stock EDA & Forecast",
    layout="wide"
)

# ==== CSS ====
st.markdown(
    """
    <style>
    .red-text { color: red; font-size: 30px; }
    .edit-text_yellow { color: #D4F005; font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

company_codes = {
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Apple": "AAPL",
    "Coca-Cola": "KO",
    "Disney": "DIS",
    "Google": "GOOGL",
    "Intel": "INTC",
    "McDonald's": "MCD",
    "Meta": "META",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "NVIDIA": "NVDA",
    "Starbucks": "SBUX",
    "Tesla": "TSLA"
}

companies = list(company_codes.keys())

model_mapping = {
    "AAPL": (pickle.load(open('Model_data/Model_Apple.pkl', 'rb')),
             pickle.load(open('Scaler_data/Scaler_Apple.pkl', 'rb'))),
    "AMZN": (pickle.load(open('Model_data/Model_Amazon.pkl', 'rb')),
             pickle.load(open('Scaler_data/Scaler_Amazon.pkl', 'rb'))),
    "META": (pickle.load(open('Model_data/Model_Meta.pkl', 'rb')),
             pickle.load(open('Scaler_data/Scaler_Meta.pkl', 'rb'))),
    "GOOGL": (pickle.load(open('Model_data/Model_Google.pkl', 'rb')),
              pickle.load(open('Scaler_data/Scaler_Google.pkl', 'rb'))),
    "MSFT": (pickle.load(open('Model_data/Model_Microsoft.pkl', 'rb')),
             pickle.load(open('Scaler_data/Scaler_Microsoft.pkl', 'rb')))
}

if "history_predictions" not in st.session_state:
    st.session_state.history_predictions = []

# ==== Menu cấp 1 ====
main_menu = st.sidebar.radio("Main Menu", ["Home", "Dự báo với công ty có sẵn", "Tải dữ liệu lên"])

# ==== HOME ====
if main_menu == "Home":
    st.header("🏠 Trang chủ")
    st.write("Chọn 1 trong 2 luồng xử lý:")
    st.markdown("- **Lựa chọn 1:** Dự báo với công ty có sẵn")
    st.markdown("- **Lựa chọn 2:** Tải dữ liệu mới và xử lý")

# ==== DỰ BÁO VỚI CÔNG TY CÓ SẴN ====
elif main_menu == "Dự báo với công ty có sẵn":
    st.sidebar.markdown("### Lựa chọn 1")
    sub_menu = st.sidebar.radio("Chức năng", ["Tiền xử lý dữ liệu", "Phân tích mô tả", "Phân tích dự báo", "Lịch sử dự báo"
                                              , "Chiến lược đầu tư"])

    # ⚡ RÀNG BUỘC: chỉ cho phép vào các mục khác nếu đã có df_preprocessed
    if sub_menu != "Tiền xử lý dữ liệu":
        if "df_preprocessed" not in st.session_state:
            st.warning("⚠️ Vui lòng hoàn thành tiền xử lý dữ liệu trước khi tiếp tục.")
            st.stop()

    # --- TIỀN XỬ LÝ DỮ LIỆU ---
    if sub_menu == "Tiền xử lý dữ liệu":
        st.sidebar.markdown("#### 🛠 Các bước tiền xử lý")
        pre_option = st.sidebar.radio(
            "Chọn chức năng tiền xử lý",
            ["Giới thiệu", "Chọn công ty", "Xử lý giá trị thiếu", "Xử lý giá trị trùng lặp"],
            index=0
        )

        if pre_option == "Giới thiệu":
            st.subheader("ℹ️ Tiền xử lý dữ liệu là gì?")
            st.markdown("""
            - Tiền xử lý dữ liệu là bước **chuẩn bị dữ liệu trước khi phân tích / dự báo**.  
            - Bước này giúp xử lý dữ liệu bị thiếu và trùng lặp.  

            **Gồm các bước sau đây:**
            1. Xử lý giá trị thiếu. 
            2. Xử lý giá trị trùng lặp 

            👉 Hãy chọn công ty để bắt đầu.
            """)

        elif pre_option == "Chọn công ty":
            st.sidebar.markdown("#### 🔍 Cách chọn công ty")
            select_method = st.sidebar.radio("Chọn phương thức:", ["Theo tên công ty", "Theo mã chứng khoán"])

            selected_company = None
            selected_code = None

            # --- Chọn theo tên công ty ---
            if select_method == "Theo tên công ty":
                company_list = ["-- Chọn công ty --"] + companies
                selected_company = st.selectbox("Chọn công ty:", company_list)
                if selected_company != "-- Chọn công ty --":
                    selected_code = company_codes[selected_company]

            # --- Chọn theo mã chứng khoán ---
            else:
                input_code = st.text_input("Nhập mã chứng khoán:").strip().upper()
                if input_code:
                    selected_code = input_code
                    # Nếu có trong dict thì lấy tên công ty
                    selected_company = next((k for k, v in company_codes.items() if v == input_code), input_code)

            # --- Hàm reset ---
            def reset_forecast_state():
                """
                Reset trạng thái dự báo khi đổi công ty,
                nhưng vẫn giữ lại X, Y đã chọn của các công ty khác.
                """
                keys_to_remove = [
                    "test_size",
                    "X_train", "X_test", "y_train", "y_test",
                    "df_forecast_ready", "df_preprocessed",
                    "df_selected",
                    "scaler", "scale_method", "model"
                ]

                current_company_key = st.session_state.get("company_name", "default")
                keys_to_remove += [
                    f"x_vars_{current_company_key}",
                    f"y_var_{current_company_key}",
                    f"scaler_{current_company_key}",
                    f"scale_method_{current_company_key}"
                ]

                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]


            # --- Quản lý state khi đổi công ty ---
            if "last_company" not in st.session_state:
                st.session_state.last_company = None

            if selected_company != "-- Chọn công ty --":
                reset_forecast_state()
                st.session_state.last_company = selected_company

            # --- Nếu có mã thì tải dữ liệu ---
            if selected_code:
                try:
                    with st.spinner("⏳ Đang tải dữ liệu... Vui lòng chờ"):
                        ticker = yf.Ticker(selected_code)
                        df = ticker.history(period="max", auto_adjust=False)
                        #df = ticker.history(period="max", auto_adjust=False).iloc[:, :-2]

                    if df.empty:
                        st.warning("Không có dữ liệu cho công ty này.")
                    else:
                        start_date = df.index.min().date()
                        end_date = pd.Timestamp.today().date()

                        st.session_state.df_selected = df.copy()
                        st.session_state.selected_company_name = selected_company
                        st.session_state.selected_start_date = start_date
                        st.session_state.selected_end_date = end_date

                        st.session_state.company_name = selected_company
                        st.session_state.company_code = selected_code
                        st.subheader(f"📌 Dữ liệu từ {start_date} đến {end_date}")
                        st.write(df)

                except Exception as e:
                    st.error(f"Lỗi: {e}")

        elif pre_option == "Xử lý giá trị thiếu":
            if "df_selected" in st.session_state:
                df = st.session_state.df_selected
                missing_count = df.isna().sum().sum()

                st.subheader("📊 Thống kê giá trị thiếu")
                st.write(f"Số lượng giá trị bị thiếu: **{missing_count}**")

                if missing_count > 0:
                    method = st.radio(
                        "Chọn phương pháp xử lý giá trị thiếu:",
                        ["Xoá dòng có giá trị thiếu", "Điền bằng Mean", "Điền bằng Median", "Điền bằng Mode"]
                    )

                    if method == "Xoá dòng có giá trị thiếu":
                        cleaned = df.dropna()

                    elif method == "Điền bằng Mean":
                        cleaned = df.fillna(df.mean(numeric_only=True))

                    elif method == "Điền bằng Median":
                        cleaned = df.fillna(df.median(numeric_only=True))

                    elif method == "Điền bằng Mode":
                        # mode() có thể trả về nhiều giá trị → lấy giá trị đầu tiên
                        cleaned = df.fillna(df.mode().iloc[0])

                    st.subheader("✅ Dữ liệu sau khi xử lý giá trị thiếu")
                    st.write(cleaned)
                    st.session_state.df_preprocessed = cleaned
                else:
                    st.success("🎉 Dữ liệu không có giá trị thiếu.")
                    st.session_state.df_preprocessed = df.copy()

            else:
                st.warning("⚠️ Hãy chọn công ty trước.")

        elif pre_option == "Xử lý giá trị trùng lặp":
            if "df_selected" in st.session_state:
                # Lấy dữ liệu gốc
                df = st.session_state.df_selected.copy()

                # Đưa Date từ index thành cột để so sánh luôn
                df_reset = df.reset_index()

                # Đếm số dòng trùng (bao gồm cả Date + các cột giá trị)
                dup_count = df_reset.duplicated().sum()

                st.subheader("📊 Thống kê giá trị trùng lặp")
                st.write(f"Số lượng dòng trùng lặp: **{dup_count}**")

                if dup_count > 0:
                    st.subheader("📌 Các dòng trùng lặp:")
                    st.write(df_reset[df_reset.duplicated(keep=False)])

                    # Xóa trùng
                    deduplicated = df_reset.drop_duplicates()

                    # Đặt lại Date làm index để giữ format như ban đầu
                    deduplicated = deduplicated.set_index("Date")

                    st.subheader("✅ Dữ liệu sau khi loại bỏ giá trị trùng lặp")
                    st.write(deduplicated)

                    st.session_state.df_preprocessed = deduplicated

                    st.success("🎉 Đã loại bỏ các giá trị trùng lặp. Sẵn sàng cho phân tích mô tả.")
                else:
                    st.success("🎉 Dữ liệu không có giá trị trùng lặp.")
                    st.session_state.df_preprocessed = df.copy()
            else:
                st.warning("⚠️ Hãy chọn công ty trước.")

    # --- PHÂN TÍCH MÔ TẢ ---
    elif sub_menu == "Phân tích mô tả":
        st.sidebar.markdown("#### 📊 Phân tích mô tả")
        ana_option = st.sidebar.radio(
            "Chọn chức năng phân tích mô tả",
            ["Giới thiệu", "Phân tích đơn biến", "Phân tích đa biến"],
            index=0
        )

        # --- GIỚI THIỆU ---
        if ana_option == "Giới thiệu":
            st.subheader("ℹ️ Phân tích mô tả là gì?")
            st.markdown("""
            - **Phân tích mô tả** giúp bạn hiểu đặc trưng dữ liệu: xu hướng trung tâm, mức độ biến động, mối quan hệ giữa các biến.

            **Gồm 2 loại chính:**
            1. **Phân tích đơn biến (Univariate):** tập trung vào 1 biến duy nhất.  
            2. **Phân tích đa biến (Multivariate):** xem mối quan hệ giữa nhiều biến.  
            """)

        # --- PHÂN TÍCH ĐƠN BIẾN ---
        elif ana_option == "Phân tích đơn biến":
            if "df_preprocessed" in st.session_state:
                df = st.session_state.df_preprocessed
                st.subheader("📊 Phân tích đơn biến")

                column = st.selectbox("Chọn biến để phân tích:", df.columns)

                if column:
                    desc = df[column].describe()
                    # Thêm median và mode
                    median_val = df[column].median()
                    mode_val = df[column].mode().iloc[0] if not df[column].mode().empty else None

                    stats = {

                        "count": desc["count"],
                        "mean": desc["mean"] if "mean" in desc else None,
                        "std": desc["std"] if "std" in desc else None,
                        "min": desc["min"],
                        "25%": desc["25%"],
                        "50%": median_val,
                        "75%": desc["75%"],
                        "max": desc["max"],
                        "mode": mode_val
                    }
                    st.markdown("---")
                    st.subheader("📋 Bảng thống kê mô tả")
                    st.write(pd.DataFrame.from_dict(stats, orient="index", columns=[column]))

                    # Histogram
                    st.markdown("---")
                    st.subheader("📊 Biểu đồ Histogram")
                    fig_hist = px.histogram(df, x=column, nbins=30,
                                            title=f"Phân phối giá trị của {column}",
                                            color_discrete_sequence=["#1f77b4"])
                    fig_hist.update_layout(bargap=0.1, xaxis_title=column, yaxis_title="Tần suất")
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Boxplot
                    st.markdown("---")
                    st.subheader("📦 Biểu đồ Boxplot")
                    fig_box = px.box(df, y=column,
                                     title=f"Boxplot của {column}",
                                     color_discrete_sequence=["#ff7f0e"])
                    st.plotly_chart(fig_box, use_container_width=True)

                    # Vẽ biểu đồ
                    # Biểu đồ đường theo thời gian (Plotly)
                    st.markdown("---")
                    st.subheader("📈 Biểu đồ đường theo thời gian")
                    fig_line = px.line(df, x=df.index, y=column,
                                       title=f"Biểu đồ đường của {column} theo thời gian",
                                       labels={"x": "Thời gian", column: column})
                    st.plotly_chart(fig_line, use_container_width=True)

                    # Biểu đồ cột (Plotly)
                    st.markdown("---")
                    st.subheader("📊 Biểu đồ cột")
                    fig_bar = px.bar(df, x=df.index, y=column,
                                     title=f"Biểu đồ cột của {column}",
                                     labels={"x": "Thời gian", column: column})
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("⚠️ Vui lòng thực hiện tiền xử lý dữ liệu trước.")

        # --- PHÂN TÍCH ĐA BIẾN ---
        elif ana_option == "Phân tích đa biến":
            if "df_preprocessed" in st.session_state:
                df = st.session_state.df_preprocessed

                st.subheader("📈 Phân tích đa biến")
                st.markdown("🔍 Ma trận tương quan giữa các biến:")

                # Nút Heatmap
                if st.button("Biểu đồ Heatmap"):
                    corr = df.corr(numeric_only=True)
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Heatmap ma trận tương quan",
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Nút Scatter
                st.markdown("---")
                st.markdown("🔍 Biểu đồ phân tán giữa 2 biến:")
                col1, col2 = st.columns(2)

                with col1:
                    x_var = st.selectbox("Chọn biến độc lập (X):", df.columns, key="x_var")
                with col2:
                    y_var = st.selectbox("Chọn biến phụ thuộc (Y):", df.columns, key="y_var")

                if st.button("Biểu đồ Scatter"):
                    fig_scatter = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        title=f"Biểu đồ phân tán: {x_var} vs {y_var}",
                        opacity=0.7,
                        color_discrete_sequence=["#1f77b4"]
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Nút Candlestick
                st.markdown("---")
                st.markdown("🔍 Biểu đồ giá đóng cửa hàng ngày:")
                if {"Open", "High", "Low", "Close"}.issubset(df.columns):
                    if st.button("Biểu đồ Candlestick"):
                        fig_candle = go.Figure(data=[
                            go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close']
                            )
                        ])
                        fig_candle.update_layout(
                            title="Biểu đồ nến Nhật (Candlestick)",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            xaxis_rangeslider_visible=False,
                            width=1100,  # chiều rộng (pixels)
                            height=600  # chiều cao (pixels)
                        )
                        st.plotly_chart(fig_candle, use_container_width=True)
                else:
                    st.info("⚠️ Dữ liệu không có đủ cột Open/High/Low/Close để vẽ Candlestick.")
            else:
                st.warning("⚠️ Vui lòng thực hiện tiền xử lý dữ liệu trước.")

    # --- PHÂN TÍCH DỰ BÁO ---
    elif sub_menu == "Phân tích dự báo":
        st.sidebar.markdown("#### 📊 Phân tích dự báo")
        ana_option = st.sidebar.radio(
            "Thứ tự công việc cần thực hiện",
            ["Giới thiệu", "Xử lý giá trị ngoại lai (Tùy chọn)", "Chọn biến độc lập và phụ thuộc", "Chuẩn hóa dữ liệu",
             "Huấn luyện & Đánh giá", "Dự báo thủ công"],
            index=0
        )

        # --- Giới thiệu ---
        if ana_option == "Giới thiệu":
            st.subheader("ℹ️ Phân tích dự báo là gì?")
            st.markdown("""
            - Dựa trên dữ liệu lịch sử, mô hình sẽ **học mối quan hệ giữa các biến**.  
            - Sau đó ta có thể **dự báo giá trị tương lai**.  
            """)

        # --- Xử lý ngoại lai ---
        elif ana_option == "Xử lý giá trị ngoại lai (Tùy chọn)":
            if "df_preprocessed" in st.session_state:
                df = st.session_state.df_preprocessed.copy()

                st.subheader("📦 Xử lý ngoại lai")
                col = st.selectbox("Chọn biến cần xử lý ngoại lai:", df.columns)

                if col:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR

                    st.write(f"Ngưỡng dưới: {lower}, Ngưỡng trên: {upper}")

                    # Đếm số lượng ngoại lai
                    outliers = df[(df[col] < lower) | (df[col] > upper)]
                    st.write(f"Số lượng ngoại lai trong {col}: **{len(outliers)}**")

                    # Lựa chọn phương án xử lý
                    method = st.radio(
                        "Chọn phương pháp xử lý ngoại lai:",
                        ["Loại bỏ", "Thay bằng Median", "Thay bằng Mean", "Winsorization"]
                    )

                    if st.button("Thực hiện xử lý"):
                        if method == "Loại bỏ":
                            df = df[(df[col] >= lower) & (df[col] <= upper)]
                            st.success("✅ Đã loại bỏ ngoại lai.")
                        elif method == "Thay bằng Median":
                            median_val = df[col].median()
                            df[col] = np.where((df[col] < lower) | (df[col] > upper), median_val, df[col])
                            st.success("✅ Đã thay ngoại lai bằng Median.")
                        elif method == "Thay bằng Mean":
                            mean_val = df[col].mean()
                            df[col] = np.where((df[col] < lower) | (df[col] > upper), mean_val, df[col])
                            st.success("✅ Đã thay ngoại lai bằng Mean.")
                        elif method == "Winsorization":
                            df[col] = np.where(df[col] < lower, lower,
                                               np.where(df[col] > upper, upper, df[col]))
                            st.success("✅ Đã áp dụng Winsorization.")

                        # Lưu lại cho bước tiếp theo
                        st.session_state.df_forecast_ready = df
                        st.write(df)
            else:
                st.warning("⚠️ Vui lòng tiền xử lý dữ liệu trước.")

        elif ana_option == "Chọn biến độc lập và phụ thuộc":
            if "df_forecast_ready" in st.session_state or "df_preprocessed" in st.session_state:
                df = st.session_state.get("df_forecast_ready", st.session_state.df_preprocessed)

                st.subheader("📌 Chọn biến để huấn luyện mô hình")

                # Lấy tên công ty đang chọn để gắn key
                company_key = st.session_state.get("company_name", "default")
                x_key = f"x_vars_{company_key}"
                y_key = f"y_var_{company_key}"

                col1, col2 = st.columns(2)
                with col1:
                    x_vars = st.multiselect(
                        "Biến đầu vào (X):",
                        df.columns,
                        key=f"x_select_{company_key}"
                    )
                with col2:
                    available_y_cols = [c for c in df.columns if c not in x_vars]
                    y_var = st.selectbox(
                        "Biến đầu ra (Y):",
                        available_y_cols,
                        key=f"y_select_{company_key}"
                    )

                if x_vars and y_var:
                    st.session_state[x_key] = x_vars
                    st.session_state[y_key] = y_var

                    # Nhập tỷ lệ train/test
                    test_size = st.number_input(
                        "🔀 Nhập tỷ lệ tập kiểm tra (test size):",
                        min_value=0.0, max_value=1.0,
                        value=0.2, step=0.05, format="%.2f"
                    )

                    if 0 < test_size < 1:
                        st.session_state.test_size = test_size
                        st.success(f"✅ Đã chọn X = {x_vars}, Y = {y_var}, Test size = {test_size}")

                        from sklearn.model_selection import train_test_split

                        X = df[x_vars]
                        y = df[[y_var]]

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, shuffle=False
                        )

                        st.markdown("### 📊 Tập huấn luyện và kiểm tra")
                        st.write("**X_train:**", X_train)
                        st.write("**y_train:**", y_train)
                        st.write("**X_test:**", X_test)
                        st.write("**y_test:**", y_test)
                    else:
                        st.error("❌ Test size phải nằm trong khoảng (0,1).")
            else:
                st.warning("⚠️ Vui lòng tiền xử lý dữ liệu trước.")

        elif ana_option == "Chuẩn hóa dữ liệu":
            company_key = st.session_state.get("company_name", "default")
            x_key = f"x_vars_{company_key}"
            y_key = f"y_var_{company_key}"

            if x_key in st.session_state and y_key in st.session_state:
                st.subheader("⚖️ Chuẩn hóa dữ liệu X")

                df = st.session_state.get("df_forecast_ready", st.session_state.df_preprocessed)
                X = df[st.session_state[x_key]]
                y = df[[st.session_state[y_key]]]

                scale_method = st.radio(
                    "Chọn phương pháp chuẩn hóa:",
                    ["MinMaxScaler", "StandardScaler (Z-score)"],
                    key=f"scaler_method_{company_key}"
                )

                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import MinMaxScaler, StandardScaler

                test_size = st.session_state.get("test_size", 0.2)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=False
                )

                # Áp dụng chuẩn hóa
                scaler = MinMaxScaler() if scale_method == "MinMaxScaler" else StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Lưu lại vào session
                st.session_state[f"scaler_{company_key}"] = scaler
                st.session_state[f"scale_method_{company_key}"] = scale_method
                st.session_state.X_train = pd.DataFrame(X_train_scaled, columns=st.session_state[x_key],
                                                        index=X_train.index)
                st.session_state.X_test = pd.DataFrame(X_test_scaled, columns=st.session_state[x_key],
                                                       index=X_test.index)
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                st.success(f"✅ Đã chuẩn hóa dữ liệu bằng {scale_method}")

                # Hiển thị kết quả
                st.markdown("### 📊 Tập huấn luyện & kiểm tra sau chuẩn hóa")
                st.dataframe(st.session_state.X_train)
                st.dataframe(st.session_state.y_train)
                st.dataframe(st.session_state.X_test)
                st.dataframe(st.session_state.y_test)
            else:
                st.warning("⚠️ Vui lòng chọn biến X và Y trước khi chuẩn hóa.")

        # --- Huấn luyện & Dự báo ---
        elif ana_option == "Huấn luyện & Đánh giá":
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score

            if all(k in st.session_state for k in ["X_train", "X_test", "y_train", "y_test"]):
                # ✅ Dùng dữ liệu đã chuẩn hóa
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
            else:
                st.warning("⚠️ Phải thực hiện chuẩn hóa trước khi huấn luyện & đánh giá.")
                st.stop()

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # LƯU MODEL vào session để dùng lại trong phần Dự báo
            st.session_state.model = model

            # Dự báo
            y_pred = model.predict(X_test)
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np

            # --- Tính các chỉ số ---
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
            r2 = r2_score(y_test, y_pred)

            st.subheader("📊 Kết quả mô hình")
            st.write(
                f"📉 MSE: {mse:.4f} → Sai số bình phương trung bình. Con số càng nhỏ thì dự báo càng gần với thực tế.")
            st.write(
                f"📉 RMSE: {rmse:.4f} → Sai số trung bình theo đơn vị gốc dữ liệu. Trung bình mỗi dự báo lệch khoảng **{rmse:.2f}** đơn vị so với thực tế.")
            st.write(f"📉 MAE: {mae:.4f} → Sai số tuyệt đối trung bình. Dự báo lệch trung bình **{mae:.2f}** đơn vị.")
            st.write(
                f"📉 MAPE: {mape:.2f}% → Sai số phần trăm trung bình. Trung bình mỗi dự báo lệch khoảng **{mape:.2f}%** so với giá trị thực tế.")
            st.write(
                f"📉 R²: {r2:.4f} → Hệ số xác định. Mô hình giải thích được khoảng **{r2 * 100:.2f}%** biến động của dữ liệu.")

            # Vẽ kết quả
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test.values.flatten(), mode="lines", name="Thực tế"))
            fig.add_trace(go.Scatter(y=y_pred.flatten(), mode="lines", name="Dự báo"))

            y_var = st.session_state.get("y_var")
            if not y_var:
                y_var = "Giá trị dự báo"

            fig.update_layout(
                title="So sánh giá trị thực tế vs dự báo",
                xaxis_title="Index",
                yaxis_title=y_var
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- DỰ BÁO GIÁ TRỊ MỚI ---
        elif ana_option == "Dự báo thủ công":
            company_key = st.session_state.get("company_name", "default")
            x_key = f"x_vars_{company_key}"
            y_key = f"y_var_{company_key}"

            if "model" in st.session_state and x_key in st.session_state:
                st.subheader("🔮 Dự báo thủ công")

                # Lấy dữ liệu gốc
                df = st.session_state.get("df_forecast_ready", st.session_state.df_preprocessed)
                x_vars = st.session_state[x_key]

                # Giá trị mặc định = dòng cuối cùng
                last_row = df[x_vars].iloc[-1]

                #
                # --- Luôn hiển thị biểu đồ nến 15m ở bên ngoài ---
                company_code4 = st.session_state.get("company_code", None)
                name4 = st.session_state.get("company_name", None)
                if company_code4:
                    ticker = yf.Ticker(company_code4)
                    data_intraday_value = ticker.history(period="1d", interval="15m")

                    if not data_intraday_value.empty:
                        import plotly.graph_objects as go

                        fig = go.Figure(data=[go.Candlestick(
                            x=data_intraday_value.index,
                            open=data_intraday_value['Open'],
                            high=data_intraday_value['High'],
                            low=data_intraday_value['Low'],
                            close=data_intraday_value['Close'],
                            name="Intraday 15m"
                        )])

                        fig.update_layout(
                            title=f"📈 Biểu đồ nến intraday 15 phút của {name4}",
                            xaxis_title="Thời gian",
                            yaxis_title="Giá",
                            xaxis_rangeslider_visible=False,
                            template="plotly_dark"
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Không lấy được dữ liệu intraday 15m.")



                st.markdown("### 📥 Nhập giá trị các biến độc lập (X)")

                manual_input = {}
                for var in x_vars:
                    manual_input[var] = st.number_input(
                        f"{var}:",
                        value=float(last_row[var]),
                        key=f"manual_{var}"
                    )

                # Convert về DataFrame
                input_df = pd.DataFrame([manual_input])

                # Chuẩn hóa nếu có
                scaler_key = f"scaler_{company_key}"
                if scaler_key in st.session_state:
                    scaler = st.session_state[scaler_key]
                    input_scaled = scaler.transform(input_df)
                    input_df = pd.DataFrame(input_scaled, columns=x_vars)

                # Nút dự báo
                if st.button("🚀 Dự báo"):
                    check_vars = ["Open", "High", "Low", "Close", "Adj Close"]
                    values = {v: manual_input[v] for v in check_vars if v in manual_input}

                    errors = []
                    # 1. Kiểm tra Low phải là nhỏ nhất
                    if "Low" in values:
                        if values["Low"] != min(values.values()):
                            errors.append(f"❌ Giá trị Low không được lớn hơn các giá trị khác.")
                    # 2. Kiểm tra High phải là lớn nhất
                    if "High" in values:
                        if values["High"] != max(values.values()):
                            errors.append(f"❌ Giá trị High không được thấp hơn các giá trị khác.")
                    # 3. Kiểm tra không âm
                    for var, val in manual_input.items():
                        if val < 0:
                            errors.append(f"❌ {var} không được nhỏ hơn 0.")

                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        model = st.session_state.model
                        y_pred = model.predict(input_df)

                        y_var = st.session_state.get(y_key, "Giá trị dự báo")
                        st.success(f"🔮 Kết quả dự báo {y_var}: **{y_pred[0][0]:.4f}**")

                        from datetime import datetime
                        import json

                        # Sau khi có y_pred
                        company_name = st.session_state.get("company_name",
                                                            "Unknown")
                        y_var = st.session_state.get(y_key, "Giá trị dự báo")
                        history_entry = {
                            "company": company_name,
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "inputs": manual_input,
                            y_var: float(y_pred[0][0])
                        }

                        # Nếu chưa có lịch sử thì tạo mới
                        if "forecast_history" not in st.session_state:
                            st.session_state.forecast_history = []

                        st.session_state.forecast_history.append(history_entry)

                        # Lưu ra file JSON (vd: forecast_history.json)
                        with open("forecast_history.json", "w", encoding="utf-8") as f:
                            json.dump(st.session_state.forecast_history, f, ensure_ascii=False, indent=4)
                        st.info("📝 Lịch sử dự báo đã được lưu.")
            else:
                st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi dự báo.")

    elif sub_menu == "Lịch sử dự báo":
        st.subheader("📝 Lịch sử dự báo")

        if "forecast_history" in st.session_state and st.session_state.forecast_history:
            st.json(st.session_state.forecast_history)

            # Cho phép tải về
            with open("forecast_history.json", "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Tải xuống file JSON",
                    data=f.read(),
                    file_name="forecast_history.json",
                    mime="application/json"
                )

            # Nút reset lịch sử
            if st.button("♻️ Reset lịch sử"):
                st.session_state.forecast_history = []  # Xóa trong session
                import os, json

                if os.path.exists("forecast_history.json"):
                    with open("forecast_history.json", "w", encoding="utf-8") as f:
                        json.dump([], f, ensure_ascii=False, indent=4)
                st.success("✅ Đã reset lịch sử dự báo.")
                st.rerun()
        else:
            st.info("Chưa có lịch sử dự báo nào.")

    elif sub_menu == "Chiến lược đầu tư":
        if "df_preprocessed" in st.session_state:
            df = st.session_state.df_preprocessed.copy()
            df = add_indicators(df)

            strategy_option = st.selectbox(
                "📌 Chọn chiến lược đầu tư:",
                ["MA Crossover", "RSI Overbought/Oversold", "MACD", "Bollinger Bands"]
            )

            # --- Xác định Position ---
            df["Position"] = 0
            if strategy_option == "MA Crossover":
                df.loc[df["SMA_20"] > df["SMA_50"], "Position"] = 1
                df.loc[df["SMA_20"] < df["SMA_50"], "Position"] = -1

            elif strategy_option == "RSI Overbought/Oversold":
                df.loc[df["RSI"] < 30, "Position"] = 1
                df.loc[df["RSI"] > 70, "Position"] = -1

            elif strategy_option == "MACD":
                df.loc[df["MACD"] > df["MACD_Signal"], "Position"] = 1
                df.loc[df["MACD"] < df["MACD_Signal"], "Position"] = -1

            elif strategy_option == "Bollinger Bands":
                df.loc[df["Close"] < df["BB_Lower"], "Position"] = 1
                df.loc[df["Close"] > df["BB_Upper"], "Position"] = -1

            # --- Long-only Position ---
            df["Position_LO"] = df["Position"].clip(lower=0)

            # --- Xác định Entry/Exit ---
            df["Signal"] = df["Position"].diff().fillna(0)
            buys = df[(df["Signal"] == 1) | (df["Signal"] == 2)]  # Mua mới / đảo chiều lên
            sells = df[(df["Signal"] == -1) | (df["Signal"] == -2)]  # Bán mới / đảo chiều xuống

            # --- Vẽ biểu đồ ---
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            # --- Tạo layout subplot ---
            from plotly.subplots import make_subplots

            if strategy_option == "RSI Overbought/Oversold":
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Biểu đồ giá", "RSI")
                )
            elif strategy_option == "MACD":
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Biểu đồ giá", "Histogram")
                )
            else:
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=("Biểu đồ giá",)
                )

            # --- Vẽ nến giá (luôn ở hàng 1) ---
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="Giá"
            ), row=1, col=1)

            # --- Thêm đường chỉ báo ---
            if strategy_option == "MA Crossover":
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50"), row=1, col=1)

            elif strategy_option == "Bollinger Bands":
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", line=dict(color="lightblue"),
                                         name="BB Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", line=dict(color="lightblue"),
                                         name="BB Lower", fill='tonexty', fillcolor='rgba(173,216,230,0.2)'), row=1,
                              col=1)
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["BB_Middle"], mode="lines", line=dict(color="blue", dash="dash"),
                               name="Middle Band"), row=1, col=1)

            elif strategy_option == "RSI Overbought/Oversold":
                fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[70] * len(df), mode="lines",
                                         line=dict(dash="dash", color="red"), name="Overbought (70)"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[30] * len(df), mode="lines",
                                         line=dict(dash="dash", color="green"), name="Oversold (30)"), row=2, col=1)

            elif strategy_option == "MACD":
                fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="MACD Signal"), row=2,
                              col=1)
                fig.add_trace(go.Bar(x=df.index, y=df["MACD"] - df["MACD_Signal"],
                                     name="Histogram", marker_color='gray', opacity=0.5), row=2, col=1)

            # --- Điểm Mua/Bán (luôn trên chart giá chính) ---
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["Close"], mode="markers",
                marker=dict(color="green", size=10, symbol="triangle-up"),
                name="Mua"
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["Close"], mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-down"),
                name="Bán"
            ), row=1, col=1)

            # --- Layout ---
            # --- Layout ---
            layout_kwargs = dict(
                xaxis=dict(rangeslider=dict(visible=False)),
                showlegend=True
            )

            # Nếu có 2 chart (RSI/MACD) thì tăng chiều cao lên
            if strategy_option in ["RSI Overbought/Oversold", "MACD"]:
                layout_kwargs["height"] = 800

            fig.update_layout(**layout_kwargs)

            st.markdown("---")
            st.subheader("📈 Chiến lược đầu tư")
            st.plotly_chart(fig, use_container_width=True)

            # --- Hiệu suất chiến lược ---
            df["Return"] = df["Close"].pct_change().fillna(0)
            df["Strategy_LS"] = df["Position"].shift(1).fillna(0) * df["Return"]
            df["Strategy_LO"] = df["Position_LO"].shift(1).fillna(0) * df["Return"]

            # --- Cumulative Return ---
            cum_ret_LS = (1 + df["Strategy_LS"]).cumprod()
            cum_ret_LO = (1 + df["Strategy_LO"]).cumprod()

            # --- Max Drawdown ---
            cum_max_LS = cum_ret_LS.cummax()
            drawdown_LS = (cum_ret_LS - cum_max_LS) / cum_max_LS
            max_drawdown_LS = drawdown_LS.min()

            # Chỉ tính Max Drawdown những ngày thực sự có Position > 0
            cum_ret_LO_pos = cum_ret_LO[df["Position_LO"].shift(1) > 0]
            if not cum_ret_LO_pos.empty:
                cum_max_LO_pos = cum_ret_LO_pos.cummax()
                drawdown_LO = (cum_ret_LO_pos - cum_max_LO_pos) / cum_max_LO_pos
                max_drawdown_LO = drawdown_LO.min()
            else:
                max_drawdown_LO = 0

            # --- Hiển thị kết quả ---
            st.markdown("---")
            st.subheader("📊 Hiệu suất chiến lược")
            st.write(
                f"📌 Long/Short: Lợi suất tích lũy = **{(cum_ret_LS.iloc[-1] - 1) * 100:.2f}%**, Max Drawdown = **{max_drawdown_LS * 100:.2f}%**")
            st.write(
                f"📌 Long-only: Lợi suất tích lũy = **{(cum_ret_LO.iloc[-1] - 1) * 100:.2f}%**, Max Drawdown = **{max_drawdown_LO * 100:.2f}%**")

            # --- Biểu đồ so sánh ---
            from plotly.subplots import make_subplots
            fig2 = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.15,
                subplot_titles=("Cumulative Return", "Drawdown")
            )

            # Chart 1: Cumulative Return
            fig2.add_trace(
                go.Scatter(x=df.index, y=cum_ret_LS - 1, mode="lines", name="CumReturn LS", line=dict(color="blue")),
                row=1, col=1
            )
            fig2.add_trace(
                go.Scatter(x=df.index, y=cum_ret_LO - 1, mode="lines", name="CumReturn LO", line=dict(color="green")),
                row=1, col=1
            )

            # Chart 2: Drawdown
            fig2.add_trace(
                go.Scatter(x=df.index, y=drawdown_LS, mode="lines", name="Drawdown LS",
                           line=dict(color="red")),
                row=2, col=1
            )
            if not cum_ret_LO_pos.empty:
                drawdown_full_LO = pd.Series(index=df.index, dtype=float)
                drawdown_full_LO[cum_ret_LO_pos.index] = drawdown_LO
                fig2.add_trace(
                    go.Scatter(x=drawdown_full_LO.index, y=drawdown_full_LO, mode="lines",
                               name="Drawdown LO", line=dict(color="orange")),
                    row=2, col=1
                )

            fig2.update_layout(
                height=800,
                showlegend=True,
                yaxis_title="Cumulative Return",
                yaxis2_title="Drawdown"
            )

            st.plotly_chart(fig2, use_container_width=True)

            # --- Backtest bổ sung ---
            buy_signals = df[df["Signal"] > 0]  # chỉ lấy tín hiệu mua

            # --- Holding Return ---
            holding_gia_von = None
            holding_return = None

            if not buy_signals.empty:
                first_buy_index = buy_signals.index[0]

                # Lấy vị trí số nguyên (integer position) thay vì Timestamp index
                first_buy_pos = df.index.get_loc(first_buy_index)

                if first_buy_pos + 1 < len(df):
                    holding_gia_von = df["Open"].iloc[first_buy_pos + 1]
                    holding_gia_cuoi = df["Close"].iloc[-1]
                    holding_return = (holding_gia_cuoi / holding_gia_von - 1) * 100
                else:
                    holding_gia_von = None
                    holding_return = None

            # --- Buy & Hold ---
            buyhold_gia_von = df["Open"].iloc[0]  # Mua vào giá mở cửa ngày đầu tiên
            buyhold_gia_cuoi = df["Close"].iloc[-1]  # Bán ra giá đóng cửa ngày cuối
            buyhold_return = (buyhold_gia_cuoi / buyhold_gia_von - 1) * 100

            # --- Hiển thị đẹp ---
            st.markdown("---")
            st.subheader("📌 So sánh các phương án đầu tư")

            if holding_gia_von is not None:
                st.markdown(f"**🎯 Holding Return:** Giá vốn = {holding_gia_von:.2f}, Hiệu suất = {holding_return:.2f}%")
            else:
                st.markdown("⚠️ Không có tín hiệu mua trong dữ liệu để tính Holding Return.")

            st.markdown(f"**📈 Buy & Hold:** Giá vốn = {buyhold_gia_von:.2f}, Hiệu suất = {buyhold_return:.2f}%")

        else:
            st.warning("⚠️ Vui lòng xử lý dữ liệu trước khi chạy chiến lược.")



# ==== TẢI DỮ LIỆU LÊN ====
else:
    st.sidebar.markdown("### Lựa chọn 2")
    sub_menu = st.sidebar.radio(
        "Chức năng",
        ["Tiền xử lý dữ liệu", "Phân tích mô tả", "Phân tích dự báo", "Lịch sử dự báo"]
    )

    # ⚡ RÀNG BUỘC: chỉ cho phép vào các mục khác nếu đã TIỀN XỬ LÝ XONG
    if sub_menu != "Tiền xử lý dữ liệu":
        if "preprocess_done" not in st.session_state or not st.session_state.preprocess_done:
            st.warning("⚠️ Vui lòng hoàn thành tiền xử lý dữ liệu trước khi tiếp tục.")
            st.stop()

    # --- TIỀN XỬ LÝ DỮ LIỆU ---
    if sub_menu == "Tiền xử lý dữ liệu":
        st.sidebar.markdown("#### 🛠 Các bước tiền xử lý")
        pre_option = st.sidebar.radio(
            "Chọn chức năng tiền xử lý",
            ["Giới thiệu", "Tải dữ liệu lên", "Xử lý giá trị thiếu", "Xử lý giá trị trùng lặp"],
            index=0
        )

        if pre_option == "Giới thiệu":
            st.subheader("ℹ️ Tiền xử lý dữ liệu là gì?")
            st.markdown("""
                - Tiền xử lý dữ liệu là bước **chuẩn bị dữ liệu trước khi phân tích / dự báo**.  
                - Bước này giúp xử lý dữ liệu bị thiếu và trùng lặp.  

                **Gồm các bước sau đây:**
                1. Xử lý giá trị thiếu  
                2. Xử lý giá trị trùng lặp  

                👉 Hãy tải dữ liệu lên để bắt đầu. 
            """)

        elif pre_option == "Tải dữ liệu lên":
            st.subheader("📂 Tải dữ liệu công ty")

            st.info("ℹ️ Vui lòng tải dữ liệu dạng **OHLCV** (Open, High, Low, Close, Volume). "
                    "Nếu sử dụng dữ liệu khác, mô hình có thể hoạt động không chính xác.")

            uploaded_file = st.file_uploader(
                "Chọn file dữ liệu (.csv, .xlsx hoặc .txt)",
                type=["csv", "xlsx", "txt"]
            )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith(".txt"):
                        df = pd.read_csv(uploaded_file, delimiter=r'[,\t;]', engine="python")


                    # Lưu vào session_state
                    st.session_state.df_raw = df.copy()
                    st.session_state.df_preprocessed1 = df.copy()
                    st.session_state.missing_done = False
                    st.session_state.duplicate_done = False
                    st.session_state.preprocess_done = False

                    # Reset các trạng thái huấn luyện/dự báo cũ
                    for key1 in ["x_vars1", "y_var1", "test_size1",
                                "X_train1", "X_test1", "y_train1", "y_test1",
                                 "x_select1", "y_select1",
                                "scaler1", "scale_method1", "model1"]:
                        if key1 in st.session_state:
                            del st.session_state[key1]

                    st.success("✅ Dữ liệu đã được tải thành công!")

                    st.write("📊 Xem trước dữ liệu:")
                    st.dataframe(df)

                    st.write("**Kích thước dữ liệu:**", df.shape)
                    st.write("**Các cột dữ liệu:**", list(df.columns))

                except Exception as e:
                    st.error(f"Lỗi khi đọc file: {e}")
            else:
                st.info("⬆️ Hãy tải lên file dữ liệu để bắt đầu.")

        elif pre_option == "Xử lý giá trị thiếu":

            if "df_preprocessed1" not in st.session_state:
                st.warning("⚠️ Vui lòng tải dữ liệu lên trước khi xử lý giá trị thiếu.")
                st.stop()
            else:
                df = st.session_state.df_preprocessed1

                missing_count = df.isnull().sum().sum()
                st.subheader("📊 Thống kê giá trị thiếu")
                st.write(f"Số lượng giá trị bị thiếu: **{missing_count}**")

                if missing_count == 0:
                    st.success("✅ Dữ liệu không có giá trị thiếu.")
                    st.session_state.missing_done = True
                else:
                    method = st.radio(
                        "Chọn phương pháp xử lý:",
                        ["Xóa hàng chứa giá trị thiếu", "Điền bằng Mean", "Điền bằng Median", "Điền bằng Mode"]
                    )

                    if st.button("Thực hiện xử lý"):
                        if method == "Xóa hàng chứa giá trị thiếu":
                            df = df.dropna()
                        elif method == "Điền bằng Mean":
                            df = df.fillna(df.mean(numeric_only=True))
                        elif method == "Điền bằng Median":
                            df = df.fillna(df.median(numeric_only=True))
                        elif method == "Điền bằng Mode":
                            for col in df.columns:
                                if df[col].isnull().any():
                                    m = df[col].mode()
                                    if not m.empty:
                                        df[col] = df[col].fillna(m.iloc[0])

                        st.session_state.df_preprocessed1 = df
                        st.session_state.missing_done = True

                        st.success("✅ Đã xử lý giá trị thiếu!")
                        st.dataframe(df)

            if st.session_state.missing_done and st.session_state.duplicate_done:
                st.session_state.preprocess_done = True

        elif pre_option == "Xử lý giá trị trùng lặp":

            if "df_preprocessed1" in st.session_state:
                df = st.session_state.df_preprocessed1

                dup_count = df.duplicated().sum()
                st.subheader("📊 Thống kê giá trị trùng lặp")
                st.write(f"Số lượng dòng trùng lặp: **{dup_count}**")

                if dup_count == 0:
                    st.success("✅ Dữ liệu không có giá trị trùng lặp.")
                    st.session_state.duplicate_done = True
                else:
                    if st.button("Xóa các dòng trùng lặp"):
                        df = df.drop_duplicates()
                        st.session_state.df_preprocessed1 = df
                        st.session_state.duplicate_done = True
                        st.success("✅ Đã xóa các dòng trùng lặp!")
                        st.dataframe(df)

                if st.session_state.missing_done and st.session_state.duplicate_done:
                    st.session_state.preprocess_done = True

            else:
                st.warning("⚠️ Vui lòng tải dữ liệu lên trước khi xử lý giá trị trùng lặp.")

    # --- PHÂN TÍCH MÔ TẢ ---
    elif sub_menu == "Phân tích mô tả":
        st.sidebar.markdown("#### 📊 Phân tích mô tả")
        ana_option = st.sidebar.radio(
            "Chọn chức năng phân tích mô tả",
            ["Giới thiệu", "Phân tích đơn biến", "Phân tích đa biến"],
            index=0
        )

        # --- GIỚI THIỆU ---
        if ana_option == "Giới thiệu":
            st.subheader("ℹ️ Phân tích mô tả là gì?")
            st.markdown("""
            - **Phân tích mô tả** giúp bạn hiểu đặc trưng dữ liệu: xu hướng trung tâm, mức độ biến động, mối quan hệ giữa các biến.

            **Gồm 2 loại chính:**
            1. **Phân tích đơn biến (Univariate):** tập trung vào 1 biến duy nhất.  
            2. **Phân tích đa biến (Multivariate):** xem mối quan hệ giữa nhiều biến.  
            """)

        # --- PHÂN TÍCH ĐƠN BIẾN ---
        elif ana_option == "Phân tích đơn biến":
            if "df_preprocessed1" in st.session_state:
                df = st.session_state.df_preprocessed1

                st.subheader("📊 Phân tích đơn biến")

                # Loại bỏ các cột có khả năng là thời gian
                import re

                # Danh sách từ khóa "cột thời gian"
                time_keywords = ["date", "time", "ngày", "timestamp", "datetime", "thời gian", "d", "t"]

                def is_time_col(col_name: str) -> bool:
                    col_lower = col_name.lower().strip()
                    for kw in time_keywords:
                        # chỉ match nếu trùng hẳn hoặc khớp nguyên từ
                        if col_lower == kw or re.search(rf"\b{kw}\b", col_lower):
                            return True
                    return False

                # Xác định cột thời gian
                time_cols = [c for c in df.columns if is_time_col(c)]

                # Chỉ lấy cột số và loại bỏ cột thời gian
                numeric_cols = [
                    c for c in df.select_dtypes(include=['number']).columns.tolist()
                    if c not in time_cols
                ]

                # Khởi tạo mặc định
                column = None

                if not numeric_cols:
                    st.warning("⚠️ Dữ liệu không có cột số để phân tích đơn biến (đã loại bỏ cột thời gian).")
                else:
                    column = st.selectbox("Chọn biến để phân tích:", numeric_cols)

                    if column:
                        # Thống kê mô tả
                        desc = df[column].describe()
                        median_val = df[column].median()
                        mode_val = df[column].mode().iloc[0] if not df[column].mode().empty else None

                        stats = {
                            "count": desc["count"],
                            "mean": desc["mean"] if "mean" in desc else None,
                            "std": desc["std"] if "std" in desc else None,
                            "min": desc["min"],
                            "25%": desc["25%"],
                            "50%": median_val,
                            "75%": desc["75%"],
                            "max": desc["max"],
                            "mode": mode_val
                        }
                        st.markdown("---")
                        st.subheader("📋 Bảng thống kê mô tả")
                        st.write(pd.DataFrame.from_dict(stats, orient="index", columns=[column]))

                        # Biểu đồ Histogram
                        st.markdown("---")
                        st.subheader("📊 Biểu đồ Histogram")
                        fig_hist = px.histogram(df, x=column, nbins=30,
                                                title=f"Phân phối giá trị của {column}",
                                                color_discrete_sequence=["#1f77b4"])
                        fig_hist.update_layout(bargap=0.1, xaxis_title=column, yaxis_title="Tần suất")
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Biểu đồ Boxplot
                        st.markdown("---")
                        st.subheader("📦 Biểu đồ Boxplot")
                        fig_box = px.box(df, y=column,
                                         title=f"Boxplot của {column}",
                                         color_discrete_sequence=["#ff7f0e"])
                        st.plotly_chart(fig_box, use_container_width=True)

                if time_cols:
                    # Vì chỉ có duy nhất 1 cột thời gian, tự động chọn
                    date_col = time_cols[0]

                    # Cột số để vẽ theo thời gian: dùng chính cột đã chọn ở phần trên
                    y_col = column if column is not None else None

                    df_temp = df.copy()
                    from pandas.api.types import is_numeric_dtype

                    s = df_temp[date_col]

                    if is_numeric_dtype(s):
                        s_num = pd.to_numeric(s, errors="coerce")
                        unit = "ms" if s_num.dropna().median() > 1e11 else "s"
                        df_temp["time_index"] = pd.to_datetime(s_num, unit=unit, errors="coerce")
                    else:
                        s_str = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
                        t1 = pd.to_datetime(s_str, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                        df_temp["time_index"] = t1.fillna(pd.to_datetime(s_str, errors="coerce"))

                    df_temp = df_temp.dropna(subset=["time_index"]).sort_values("time_index")

                    if df_temp.empty:
                        st.warning(f"⚠️ Cột {date_col} không có giá trị thời gian hợp lệ để vẽ biểu đồ.")
                    else:
                        if y_col:
                            st.subheader("📈 Biểu đồ đường theo thời gian")
                            fig_line = px.line(
                                df_temp, x="time_index", y=y_col,
                                title=f"Biểu đồ đường theo thời gian ({y_col})",
                                labels={"time_index": "Thời gian", y_col: y_col}
                            )
                            st.plotly_chart(fig_line, use_container_width=True)

                            st.subheader("📊 Biểu đồ cột theo thời gian")
                            fig_bar = px.bar(
                                df_temp, x="time_index", y=y_col,
                                title=f"Biểu đồ cột theo thời gian ({y_col})",
                                labels={"time_index": "Thời gian", y_col: y_col}
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.warning("⚠️ Vui lòng thực hiện tiền xử lý dữ liệu trước.")

        # --- PHÂN TÍCH ĐA BIẾN ---
        elif ana_option == "Phân tích đa biến":
            if "df_preprocessed1" in st.session_state:
                df = st.session_state.df_preprocessed1

                st.subheader("📈 Phân tích đa biến")
                st.markdown("🔍 Ma trận tương quan giữa các biến:")

                import re

                # Danh sách từ khóa "cột thời gian"
                time_keywords = ["date", "time", "ngày", "timestamp", "datetime", "thời gian", "d", "t"]


                def is_time_col(col_name: str) -> bool:
                    col_lower = col_name.lower().strip()
                    for kw in time_keywords:
                        if col_lower == kw or re.search(rf"\b{kw}\b", col_lower):
                            return True
                    return False

                # Xác định cột thời gian
                time_cols = [c for c in df.columns if is_time_col(c)]

                # Chỉ lấy cột số và loại bỏ cột thời gian
                numeric_cols = [
                    c for c in df.select_dtypes(include=['number']).columns.tolist()
                    if c not in time_cols
                ]

                if not numeric_cols:
                    st.warning("⚠️ Dữ liệu không có cột số để phân tích tương quan.")
                else:
                    # Heatmap
                    if st.button("Biểu đồ Heatmap"):
                        corr = df[numeric_cols].corr()
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title="Heatmap ma trận tương quan",
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("🔍 Biểu đồ phân tán giữa 2 biến:")

                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("Chọn biến X:", numeric_cols, key="x_var")
                    with col2:
                        y_var = st.selectbox("Chọn biến Y:", numeric_cols, key="y_var")

                    if st.button("Biểu đồ Scatter"):
                        fig_scatter = px.scatter(
                            df,
                            x=x_var,
                            y=y_var,
                            title=f"Biểu đồ phân tán: {x_var} vs {y_var}",
                            opacity=0.7,
                            color_discrete_sequence=["#1f77b4"]
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("⚠️ Vui lòng thực hiện tiền xử lý dữ liệu trước.")

    # --- PHÂN TÍCH DỰ BÁO ---
    elif sub_menu == "Phân tích dự báo":
        st.sidebar.markdown("#### 📊 Phân tích dự báo")
        ana_option = st.sidebar.radio(
            "Thứ tự công việc cần thực hiện",
            ["Giới thiệu", "Xử lý giá trị ngoại lai (Tùy chọn)", "Chọn biến độc lập và phụ thuộc", "Chuẩn hóa dữ liệu",
             "Huấn luyện & Đánh giá", "Dự báo thủ công"],
            index=0
        )

        # --- Giới thiệu ---
        if ana_option == "Giới thiệu":
            st.subheader("ℹ️ Phân tích dự báo là gì?")
            st.markdown("""
            - Dựa trên dữ liệu lịch sử, mô hình sẽ **học mối quan hệ giữa các biến**.  
            - Sau đó ta có thể **dự báo giá trị tương lai**.  
            """)

        # --- Xử lý ngoại lai ---
        elif ana_option == "Xử lý giá trị ngoại lai (Tùy chọn)":
            if "df_preprocessed1" in st.session_state:
                df = st.session_state.df_preprocessed1.copy()

                st.subheader("📦 Xử lý ngoại lai")
                # Lọc chỉ cột số
                import re

                # Danh sách từ khóa "cột thời gian"
                time_keywords = ["date", "time", "ngày", "timestamp", "datetime", "thời gian", "d", "t"]

                def is_time_col(col_name: str) -> bool:
                    col_lower = col_name.lower().strip()
                    for kw in time_keywords:
                        if col_lower == kw or re.search(rf"\b{kw}\b", col_lower):
                            return True
                    return False

                # Xác định cột thời gian
                time_cols = [c for c in df.columns if is_time_col(c)]

                # Chỉ lấy cột số và loại bỏ cột thời gian
                numeric_cols = [
                    c for c in df.select_dtypes(include=['number']).columns.tolist()
                    if c not in time_cols
                ]

                if not numeric_cols:
                    st.warning("⚠️ Không có cột số nào để xử lý ngoại lai.")
                else:
                    col = st.selectbox("Chọn biến cần xử lý ngoại lai:", numeric_cols)

                    if col:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR

                        st.write(f"Ngưỡng dưới: {lower}, Ngưỡng trên: {upper}")

                        # Đếm số lượng ngoại lai
                        outliers = df[(df[col] < lower) | (df[col] > upper)]
                        st.write(f"Số lượng ngoại lai trong {col}: **{len(outliers)}**")

                        # Lựa chọn phương án xử lý
                        method = st.radio(
                            "Chọn phương pháp xử lý ngoại lai:",
                            ["Loại bỏ", "Thay bằng Median", "Thay bằng Mean", "Winsorization"]
                        )

                        if st.button("Thực hiện xử lý"):
                            if method == "Loại bỏ":
                                df = df[(df[col] >= lower) & (df[col] <= upper)]
                                st.success("✅ Đã loại bỏ ngoại lai.")
                            elif method == "Thay bằng Median":
                                median_val = df[col].median()
                                df[col] = np.where((df[col] < lower) | (df[col] > upper), median_val, df[col])
                                st.success("✅ Đã thay ngoại lai bằng Median.")
                            elif method == "Thay bằng Mean":
                                mean_val = df[col].mean()
                                df[col] = np.where((df[col] < lower) | (df[col] > upper), mean_val, df[col])
                                st.success("✅ Đã thay ngoại lai bằng Mean.")
                            elif method == "Winsorization":
                                df[col] = np.where(df[col] < lower, lower,
                                                   np.where(df[col] > upper, upper, df[col]))
                                st.success("✅ Đã áp dụng Winsorization.")

                            # Lưu lại cho bước tiếp theo
                            st.session_state.df_forecast_ready1 = df
                            st.write(df)
            else:
                st.warning("⚠️ Vui lòng tiền xử lý dữ liệu trước.")

        # --- Chọn biến X và Y ---
        elif ana_option == "Chọn biến độc lập và phụ thuộc":
            if "df_forecast_ready1" in st.session_state or "df_preprocessed1" in st.session_state:
                df = st.session_state.get("df_forecast_ready1", st.session_state.df_preprocessed1)

                st.subheader("📌 Chọn biến để huấn luyện mô hình")

                # Lọc cột số (loại bỏ Date, chuỗi text, categorical)
                import re

                # Danh sách từ khóa "cột thời gian"
                time_keywords = ["date", "time", "ngày", "timestamp", "datetime", "thời gian", "d", "t"]


                def is_time_col(col_name: str) -> bool:
                    col_lower = col_name.lower().strip()
                    for kw in time_keywords:
                        if col_lower == kw or re.search(rf"\b{kw}\b", col_lower):
                            return True
                    return False

                # Xác định cột thời gian
                time_cols = [c for c in df.columns if is_time_col(c)]

                # Chỉ lấy cột số và loại bỏ cột thời gian
                numeric_cols = [
                    c for c in df.select_dtypes(include=['number']).columns.tolist()
                    if c not in time_cols
                ]

                col1, col2 = st.columns(2)
                with col1:
                    x_vars1 = st.multiselect(
                        "Biến đầu vào (nguyên nhân, yếu tố ảnh hưởng - X):",
                        numeric_cols, key="x_select1"
                    )
                with col2:
                    # Chỉ hiển thị những cột chưa được chọn ở X
                    available_y_cols = [c for c in numeric_cols if c not in x_vars1]
                    y_var1 = st.selectbox(
                        "Biến đầu ra (giá trị cần dự đoán - Y):",
                        available_y_cols, key="y_select1"
                    )

                if x_vars1 and y_var1:
                    st.session_state["x_vars1"] = x_vars1
                    st.session_state["y_var1"] = y_var1


                    # Nhập tỷ lệ train/test
                    test_size1 = st.number_input(
                        "🔀 Nhập tỷ lệ tập kiểm tra (test size):",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        step=0.05,
                        format="%.2f"
                    )

                    # Kiểm tra giá trị nhập hợp lệ (0 < test_size < 1)
                    if 0 < test_size1 < 1:
                        st.session_state.test_size1 = test_size1
                        st.success(f"✅ Đã chọn X = {x_vars1}, Y = {y_var1}, Test size = {test_size1}")

                        # Tách tập train/test
                        from sklearn.model_selection import train_test_split

                        X = df[x_vars1]
                        y = df[[y_var1]]

                        X_train1, X_test1, y_train1, y_test1 = train_test_split(
                            X, y, test_size=test_size1, shuffle=False
                        )

                        # Hiển thị tập X và Y
                        st.markdown("### 📊 Tập huấn luyện và kiểm tra")
                        st.write("**X_train:**")
                        st.write(X_train1)
                        st.write("**y_train:**")
                        st.write(y_train1)
                        st.write("**X_test:**")
                        st.write(X_test1)
                        st.write("**y_test:**")
                        st.write(y_test1)
                    else:
                        st.error("❌ Test size phải nằm trong khoảng (0,1).")
            else:
                st.warning("⚠️ Vui lòng tiền xử lý dữ liệu trước.")

        # --- CHUẨN HÓA DỮ LIỆU ---
        elif ana_option == "Chuẩn hóa dữ liệu":
            if "x_vars1" in st.session_state and "y_var1" in st.session_state:
                st.subheader("⚖️ Chuẩn hóa dữ liệu X")

                # Lấy lại dữ liệu gốc đã chọn X và Y
                df = st.session_state.get("df_forecast_ready1", st.session_state.df_preprocessed1)
                X = df[st.session_state.x_vars1]
                y = df[[st.session_state.y_var1]]

                scale_method1 = st.radio(
                    "Chọn phương pháp chuẩn hóa:",
                    ["MinMaxScaler", "StandardScaler (Z-score)"]
                )

                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import MinMaxScaler, StandardScaler

                test_size1 = st.session_state.get("test_size1", 0.2)

                X_train1, X_test1, y_train1, y_test1 = train_test_split(
                    X, y, test_size=test_size1, shuffle=False
                )

                # Áp dụng chuẩn hóa
                if scale_method1 == "MinMaxScaler":
                    scaler1 = MinMaxScaler()
                else:
                    scaler1 = StandardScaler()

                X_train_scaled1 = scaler1.fit_transform(X_train1)
                X_test_scaled1 = scaler1.transform(X_test1)

                # LƯU SCALER VÀ PHƯƠNG PHÁP VỀ SESSION để sử dụng khi dự báo
                st.session_state.scaler1 = scaler1
                st.session_state.scale_method1 = scale_method1

                # Trả lại DataFrame với tên cột gốc
                X_train1 = pd.DataFrame(X_train_scaled1, columns=st.session_state.x_vars1, index=X_train1.index)
                X_test1 = pd.DataFrame(X_test_scaled1, columns=st.session_state.x_vars1, index=X_test1.index)

                st.success(f"✅ Đã chuẩn hóa dữ liệu bằng {scale_method1}")

                # Hiển thị kết quả
                st.markdown("### 📊 Tập huấn luyện & kiểm tra sau chuẩn hóa")
                st.write("**X_train:**")
                st.dataframe(X_train1)
                st.write("**y_train:**")
                st.dataframe(y_train1)
                st.write("**X_test:**")
                st.dataframe(X_test1)
                st.write("**y_test:**")
                st.dataframe(y_test1)

                # Lưu vào session cho bước huấn luyện
                st.session_state.X_train1 = X_train1
                st.session_state.X_test1 = X_test1
                st.session_state.y_train1 = y_train1
                st.session_state.y_test1 = y_test1

            else:
                st.warning("⚠️ Vui lòng chọn biến X và Y trước khi chuẩn hóa.")

        # --- Huấn luyện & Dự báo ---
        elif ana_option == "Huấn luyện & Đánh giá":
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score

            if all(k in st.session_state for k in ["X_train1", "X_test1", "y_train1", "y_test1"]):
                # ✅ Dùng dữ liệu đã chuẩn hóa
                X_train1 = st.session_state.X_train1
                X_test1 = st.session_state.X_test1
                y_train1 = st.session_state.y_train1
                y_test1 = st.session_state.y_test1
            else:
                st.warning("⚠️ Phải thực hiện chuẩn hóa trước khi huấn luyện & đánh giá.")
                st.stop()

            # Train model
            model1 = LinearRegression()
            model1.fit(X_train1, y_train1)

            # LƯU MODEL vào session để dùng lại trong phần Dự báo
            st.session_state.model1 = model1

            # Dự báo
            y_pred1 = model1.predict(X_test1)
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np

            # --- Tính các chỉ số ---
            mse1 = mean_squared_error(y_test1, y_pred1)
            rmse1 = np.sqrt(mse1)
            mae1 = mean_absolute_error(y_test1, y_pred1)
            mape1 = np.mean(np.abs((y_test1 - y_pred1) / np.where(y_test1 != 0, y_test1, 1))) * 100
            r21 = r2_score(y_test1, y_pred1)

            st.subheader("📊 Kết quả mô hình")
            st.write(
                f"📉 MSE: {mse1:.4f} → Sai số bình phương trung bình. Con số càng nhỏ thì dự báo càng gần với thực tế.")
            st.write(
                f"📉 RMSE: {rmse1:.4f} → Sai số trung bình theo đơn vị gốc dữ liệu. Trung bình mỗi dự báo lệch khoảng **{rmse1:.2f}** đơn vị so với thực tế.")
            st.write(f"📉 MAE: {mae1:.4f} → Sai số tuyệt đối trung bình. Dự báo lệch trung bình **{mae1:.2f}** đơn vị.")
            st.write(
                f"📉 MAPE: {mape1:.2f}% → Sai số phần trăm trung bình. Trung bình mỗi dự báo lệch khoảng **{mape1:.2f}%** so với giá trị thực tế.")
            st.write(
                f"📉 R²: {r21:.4f} → Hệ số xác định. Mô hình giải thích được khoảng **{r21 * 100:.2f}%** biến động của dữ liệu.")

            # Vẽ kết quả
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test1.values.flatten(), mode="lines", name="Thực tế"))
            fig.add_trace(go.Scatter(y=y_pred1.flatten(), mode="lines", name="Dự báo"))

            y_var1 = st.session_state.get("y_var1")
            if not y_var1:
                y_var1 = "Giá trị dự báo"

            fig.update_layout(
                title="So sánh giá trị thực tế vs dự báo",
                xaxis_title="Index",
                yaxis_title=y_var1
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- DỰ BÁO GIÁ TRỊ MỚI ---
        elif ana_option == "Dự báo thủ công":
            if "model1" in st.session_state and "x_vars1" in st.session_state:
                st.subheader("🔮 Dự báo thủ công")

                # Lấy dữ liệu gốc
                df = st.session_state.get("df_forecast_ready1", st.session_state.df_preprocessed1)
                x_vars1 = st.session_state.x_vars1

                last_row1 = df[x_vars1].iloc[-1]

                st.markdown("### 📥 Nhập giá trị các biến độc lập (X)")
                manual_input1 = {}
                for var in x_vars1:
                    manual_input1[var] = st.number_input(
                        f"{var}:",
                        value=float(last_row1[var]),
                        key=f"manual_{var}"
                    )

                # Convert về DataFrame
                input_df1 = pd.DataFrame([manual_input1])

                # Chuẩn hóa nếu có
                if "scaler1" in st.session_state:
                    input_scaled1 = st.session_state.scaler1.transform(input_df1)
                    input_df1 = pd.DataFrame(input_scaled1, columns=x_vars1)

                # Nút dự báo
                if st.button("🚀 Dự báo"):
                    ohlcv_alias_map = {
                        "open": ["open", "op", "o"],
                        "high": ["high", "hi", "h"],
                        "low": ["low", "lo", "l"],
                        "close": ["close", "cl", "c"],
                        "adj close": ["adj close", "adjusted close", "adj_cl"]
                    }


                    def find_key(col_name, alias_dictionary):
                        name = col_name.strip().lower()
                        suffix = name.split("_")[-1]
                        for k, aliases in alias_dictionary.items():
                            if suffix in [a.lower() for a in aliases]:
                                return k
                        return None


                    # Chuẩn hóa input
                    normalized_inputs = {}
                    for k, v in manual_input1.items():
                        main_key = find_key(k, ohlcv_alias_map)
                        if main_key:
                            normalized_inputs[main_key] = v

                    # Lấy giá trị cần check
                    values1 = {v: normalized_inputs[v] for v in ["open", "high", "low", "close", "adj close"] if
                               v in normalized_inputs}

                    errors = []
                    # 1. Low phải là nhỏ nhất
                    if "low" in values1:
                        if values1["low"] != min(values1.values()):
                            errors.append("❌ Giá trị Low không được lớn hơn các giá trị khác.")
                    # 2. High phải là lớn nhất
                    if "high" in values1:
                        if values1["high"] != max(values1.values()):
                            errors.append("❌ Giá trị High không được thấp hơn các giá trị khác.")
                    # 3. Không âm
                    for var, val in manual_input1.items():
                        if val < 0:
                            errors.append(f"❌ {var} không được nhỏ hơn 0.")

                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        model1 = st.session_state.model1
                        y_pred1 = model1.predict(input_df1)
                        y_var1 = st.session_state.get("y_var1", "Giá trị dự báo")

                        st.success(f"🔮 Kết quả dự báo {y_var1}: **{y_pred1[0][0]:.4f}**")

                        from datetime import datetime
                        import json

                        # Sau khi có y_pred
                        company_name = st.session_state.get("company_name_kiva",
                                                            "Unknown")
                        y_var1 = st.session_state.get("y_var1", "Giá trị dự báo")

                        history_entry1 = {
                            "company": company_name,
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "inputs": manual_input1,
                            y_var1: float(y_pred1[0][0])
                        }

                        # Nếu chưa có lịch sử thì tạo mới
                        if "forecast_history1" not in st.session_state:
                            st.session_state.forecast_history1 = []

                        st.session_state.forecast_history1.append(history_entry1)

                        # Lưu ra file JSON (vd: forecast_history.json)
                        with open("forecast_history_upload.json", "w", encoding="utf-8") as f:
                            json.dump(st.session_state.forecast_history1, f, ensure_ascii=False, indent=4)

                        st.info("📝 Lịch sử dự báo đã được lưu.")
            else:
                st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi dự báo.")

    elif sub_menu == "Lịch sử dự báo":
        st.subheader("📝 Lịch sử dự báo")

        if "forecast_history1" in st.session_state and st.session_state.forecast_history1:
            st.json(st.session_state.forecast_history1)

            # Cho phép tải về
            with open("forecast_history_upload.json", "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Tải xuống file JSON",
                    data=f.read(),
                    file_name="forecast_history_upload.json",
                    mime="application/json"
                )

            # Nút reset lịch sử
            if st.button("♻️ Reset lịch sử"):
                st.session_state.forecast_history1 = []
                import os, json

                if os.path.exists("forecast_history_upload.json"):
                    with open("forecast_history_upload.json", "w", encoding="utf-8") as f:
                        json.dump([], f, ensure_ascii=False, indent=4)
                st.success("✅ Đã reset lịch sử dự báo.")
                st.rerun()
        else:
            st.info("Chưa có lịch sử dự báo nào.")