import base64
with open('google.json', 'r') as f:
       json_str = base64.b64encode(f.read().encode()).decode()
print(f"GOOGLE_CREDENTIALS_B64={json_str}")
   