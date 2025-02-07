import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

INSTALLED_APPS = [
    ...,
    'corsheaders',
    'science_fair',  # Correct app name
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    ...,
]

# CORS settings
CORS_ALLOW_ALL_ORIGINS = True  # For testing; restrict in production

# Allowed hosts
ALLOWED_HOSTS = ['*']  # Restrict to Render domain in production
