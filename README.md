# LLM Visualizer RU

Русскоязычный интерактивный сайт, объясняющий как работают LLM (по мотивам формата bbycroft.net/llm, но с собственной реализацией).

## Что внутри
- `index.html` — основной экран
- `src/styles.css` — стили
- `src/app.js` — интерактив (tokenization, attention demo, generation loop)

## Запуск локально
Открой `index.html` в браузере.

## План v1
- Блоки: токены, эмбеддинги, attention, next-token prediction
- Режимы: базовый / продвинутый
- Полная русская локализация


## Деплой (GitHub Pages)
После push в `main` сайт деплоится автоматически через GitHub Actions (`.github/workflows/pages.yml`).
