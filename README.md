# Static version of the app (GitHub Pages)

This branch holds one page, `index.html`, that runs the Streamlit app entirely in
the visitor's browser with [stlite](https://github.com/whitphx/stlite). There is
no server, so the page never goes to sleep.

- The page loads `app.py` and the files in `data/dayahead/` from the `main`
  branch each time it opens, so the forecast committed by the daily GitHub
  Action appears automatically. Nothing on this branch needs to change.
- The Streamlit Community Cloud app keeps running from `main` as before.
- If `app.py` starts reading a new data file, add its name to `DATA_FILES`
  in `index.html`.
- Anything `app.py` imports must run in the browser (Pyodide). Today it only
  needs pandas, numpy, altair and streamlit, which all do.
