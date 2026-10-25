from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=120).run()
print("exceptions:", [e.value for e in at.exception])
print("success:", [s.value for s in at.success])
at.selectbox[0].select("payments").run()
print("exceptions:", [e.value for e in at.exception]); print("success:", [s.value for s in at.success])
