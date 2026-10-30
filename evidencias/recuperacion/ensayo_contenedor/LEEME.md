# Ensayo previo en contenedor (no reemplaza la prueba oficial)

Ensayo ejecutado el 2026-10-30 en un contenedor Linux limpio antes de publicar, para detectar
fallos del procedimiento. Pasos reales: `git clone` desde una copia *bare* del repositorio,
entorno nuevo con `uv` y `requirements.txt`, caché DVC vacía, `dvc pull` por HTTP desde una
copia local del almacenamiento (`dvc push`), `python -m src.verificar_datos` (OK) y
`python -m src.recuperacion verificar` para v1 y v2 con las carpetas que luego se suben a
Hugging Face.

Diferencias con la prueba oficial: GitHub y Hugging Face se simularon con copias locales,
porque el contenedor no tiene acceso de red a huggingface.co. La prueba oficial
(`scripts/recuperar.sh`, desde GitHub y Hugging Face por SHA) deja su evidencia en
`../entorno_nuevo/`.
