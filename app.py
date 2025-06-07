import streamlit as st
import pandas as pd
import io
import csv
from PIL import Image
from model_utils import load_model, preprocess_image, analyze_single_image

def main():
    st.title("Анализ рентгеновских снимков для выявления пневмонии")
    st.write("Загрузите один или несколько рентгеновских снимков грудной клетки для анализа")

    # Загрузка модели
    try:
        model = load_model()
    except:
        st.error("Ошибка загрузки модели. Убедитесь, что файл best_model.h5 существует.")
        return

    uploaded_files = st.file_uploader(
        "Выберите рентгеновские снимки",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help='Перетащите файлы сюда или кликните для выбора файлов.'
    )

    # Сброс результатов анализа при изменении списка файлов
    file_names = [f.name for f in uploaded_files] if uploaded_files else []
    if 'last_files' not in st.session_state:
        st.session_state['last_files'] = []
    if file_names != st.session_state['last_files']:
        if 'results' in st.session_state:
            del st.session_state['results']
        st.session_state['last_files'] = file_names

    if uploaded_files:
        if len(uploaded_files) == 1:
            image = Image.open(uploaded_files[0])
            st.image(image, caption='Загруженный рентгеновский снимок', use_column_width=True)

            if st.button('Анализировать'):
                with st.spinner('Анализ изображения...'):
                    result = analyze_single_image(image, model)
                    st.write("---")
                    st.subheader("Результат анализа:")
                    if result["Результат"] == "Норма":
                        st.success(f"Норма (уверенность: {result['Уверенность']})")
                    else:
                        st.error(f"Обнаружена пневмония (уверенность: {result['Уверенность']})")
        else:
            if st.button('Анализировать все изображения') or 'results' in st.session_state:
                if 'results' not in st.session_state:
                    with st.spinner('Анализ изображений...'):
                        results = []
                        for uploaded_file in uploaded_files:
                            image = Image.open(uploaded_file)
                            result = analyze_single_image(image, model)
                            result['Имя файла'] = uploaded_file.name
                            result['Изображение'] = image
                            results.append(result)
                        st.session_state['results'] = results
                else:
                    results = st.session_state['results']

                st.write("---")
                st.subheader("Результаты анализа:")

                # Фильтрация
                if 'filter' not in st.session_state:
                    st.session_state['filter'] = 'all'

                cols = st.columns([1, 2, 1, 2, 1, 2, 1])
                with cols[1]:
                    show_normal = st.button('Только Норма', key='show_normal')
                with cols[3]:
                    show_pneumonia = st.button('Только Пневмония', key='show_pneumonia')
                with cols[5]:
                    show_all = st.button('Показать все', key='show_all')

                if show_normal:
                    st.session_state['filter'] = 'normal'
                if show_pneumonia:
                    st.session_state['filter'] = 'pneumonia'
                if show_all:
                    st.session_state['filter'] = 'all'

                # Применяем фильтр
                if st.session_state['filter'] == 'normal':
                    filtered = [r for r in results if r['Результат'] == 'Норма']
                elif st.session_state['filter'] == 'pneumonia':
                    filtered = [r for r in results if r['Результат'] == 'Пневмония']
                else:
                    filtered = results

                for r in filtered:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(r['Изображение'], caption=r['Имя файла'], use_column_width=True)
                    with col2:
                        if r["Результат"] == "Норма":
                            st.success(f"Норма (уверенность: {r['Уверенность']})")
                        else:
                            st.error(f"Обнаружена пневмония (уверенность: {r['Уверенность']})")
                    st.write("---")

                # Кнопка для скачивания отчета
                if results:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(['Имя файла', 'Результат', 'Уверенность', 'Вероятность нормы', 'Вероятность пневмонии'])
                    for r in results:
                        writer.writerow([
                            r['Имя файла'],
                            r['Результат'],
                            r['Уверенность'],
                            r['Вероятность нормы'],
                            r['Вероятность пневмонии']
                        ])
                    st.download_button(
                        label='Скачать отчет (CSV)',
                        data=output.getvalue(),
                        file_name='pneumonia_analysis_report.csv',
                        mime='text/csv'
                    )

                # Статистика
                st.write("---")
                st.subheader("Статистика:")
                total = len(results)
                normal_count = sum(1 for r in results if r['Результат'] == 'Норма')
                pneumonia_count = total - normal_count

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Всего снимков", total)
                    st.metric("Норма", normal_count)
                with col2:
                    st.metric("Пневмония", pneumonia_count)
                    st.metric("Процент пневмонии", f"{(pneumonia_count/total)*100:.1f}%")

if __name__ == "__main__":
    main()
