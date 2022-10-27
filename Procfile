# important to specify the --workers=1 argument, otherwise the app gets started
# twice
# https://stackoverflow.com/questions/68549612/thread-in-python-gets-started-twice-on-heroku
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000} --workers=1