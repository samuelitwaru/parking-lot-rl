import requests
import os
import sys

# PythonAnywhere API configuration
username = os.environ.get('USERNAME')
api_token = os.environ.get('API_TOKEN') 
domain = os.environ.get('DOMAIN_NAME')


if not all([username, api_token, domain]):
    print("Missing required environment variables")
    sys.exit(1)


base_url = f'https://www.pythonanywhere.com/api/v0/user/{username}/'
headers = {'Authorization': f'Token {api_token}'}

print('base_url:', base_url)

print("Starting deployment to PythonAnywhere...")

# 1. Pull latest code from repository
print("Pulling latest code from: " + f'{base_url}consoles/43517576/send_input/')
pull_response = requests.post(
    f'{base_url}consoles/43517576/send_input/',
    headers=headers,
    json={'input': 'cd ~/parking-lot-rl && git pull origin main\n'}
)


if pull_response.status_code == 200:
    print("✓ Code pulled successfully")
else:
    print(f"✗ Failed to pull code: {pull_response.status_code}")
    exit()
    
# 2. Install/update dependencies
print("Installing dependencies...")
pip_response = requests.post(
    f'{base_url}consoles/43517576/send_input/',
    headers=headers,
    json={'input': 'cd ~/parking-lot-rl && pip3.11 install --user -r requirements.txt\n'}
)

if pip_response.status_code == 200:
    print("✓ Dependencies installed successfully")
else:
    print(f"✗ Failed to install dependencies: {pip_response.status_code}")
    exit()

# 3. Reload web app
print(f"Reloading web app: {domain}")
reload_response = requests.post(
    f'{base_url}webapps/{domain}/reload/',
    headers=headers
)

if reload_response.status_code == 200:
    print("✓ Web app reloaded successfully")
else:
    print(f"✗ Failed to reload web app: {reload_response.status_code}")
    print(f"Response: {reload_response.text}")
    exit()
    
# 4. Check web app status
print("Checking web app status...")
status_response = requests.get(
    f'{base_url}webapps/{domain}/',
    headers=headers
)

if status_response.status_code == 200:
    app_info = status_response.json()
    print(f"✓ Web app status: {app_info.get('enabled', 'unknown')}")
    print(f"✓ Python version: {app_info.get('python_version', 'unknown')}")
else:
    print(f"✗ Failed to get web app status: {status_response.status_code}")
    exit()

print("Deployment completed!")