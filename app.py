import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title='Neural Search',
    layout='centered'
)

st.title('Поисковик по русскоязычному корпусу MIRACL')
st.caption('Двухэтапный поиск с ранжированием по русскоязычному корпусу MIRACL')

with st.sidebar:
    st.header('Настройки поиска')
    top_k = st.slider('Количество ответов', min_value=1, max_value=10, value=3)
    bm25_k = st.slider('Кандидаты для первичного отбора', min_value=10, max_value=100, value=50)


examples = [
    'Кто такой Юрий Гагарин?',
    'Кто и когда основал Москву?',
    'Что такое планета Земля?'
]

selected_example = st.selectbox('Или выберите готовый пример', ["Свой запрос"] + examples)
default_query = "" if selected_example == "Свой запрос" else selected_example
query = st.text_input("Введите поисковый запрос", value=default_query, placeholder="Например: кто такой Юрий Гагарин?")

if st.button("Найти ответы", type='primary') or (query and selected_example != 'Свой запрос'):
    if not query.strip():
        st.warning("Введите текст запроса.")
    else:
        with st.spinner("Ищем ответы..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    json={"query": query, "top_k": top_k, "bm25_k": bm25_k},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    st.info("По вашему запросу ничего не найдено.")
                else:
                    st.subheader(f"Топ-{len(results)} релевантных ответов:")
                    for item in results:
                        lines = item["text"].split("\n", 1)
                        title = lines[0]
                        body = lines[1] if len(lines) > 1 else ""

                        with st.container(border=True):
                            col1, col2 = st.columns([4, 1])
                            col1.markdown(f"#{item['rank']} - {title}")
                            col2.markdown(f"score: {item['score']:.4f}")
                            if body:
                                st.write(body)
                            st.caption(f"docid: {item['docid']}")
            except requests.exceptions.ConnectionError:
                st.error("Не удалось подключиться к FastAPI.")
            except requests.exceptions.HTTPError as err:
                st.error(f"Ошибка сервера API: {err}")