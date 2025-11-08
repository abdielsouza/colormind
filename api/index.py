from http.server import BaseHTTPRequestHandler
import subprocess
class handler(BaseHTTPRequestHandler):
   def do_GET(self):
       # Start the Streamlit server
       subprocess.Popen(["streamlit", "run", "../app.py", "--server.port=8501"])
       # Forward requests to the Streamlit server
       self.send_response(200)
       self.send_header('Content-type', 'text/plain')
       self.end_headers()
       self.wfile.write("Streamlit app is running!".encode())