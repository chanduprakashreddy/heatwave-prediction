import os

# Render sets the PORT environment variable (default 10000)
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 1
threads = 4
timeout = 120  # Give the app 2 minutes to start up if needed