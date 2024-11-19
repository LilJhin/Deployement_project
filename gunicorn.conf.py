# gunicorn.conf.py
bind = "0.0.0.0:10000"
workers = 1
threads = 2
timeout = 120
max_requests = 100
max_requests_jitter = 50
