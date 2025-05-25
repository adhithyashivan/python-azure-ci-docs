from flask import Flask
import random
import os

app = Flask(__name__)


@app.route('/')
def hello():
    """
    A simple greeting endpoint.
    Returns a friendly message.
    """
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    operation = random.choice(['add', 'subtract'])
    if operation == 'add':
        result = num1 + num2
        op_symbol = '+'
    else:
        result = num1 - num2
        op_symbol = '-'

    custom_message = os.environ.get(
        "CUSTOM_MESSAGE", "Welcome to the App deployed via GitHub Actions!")

    return f"<h1>{custom_message}</h1><p>Random Logic: {num1} {op_symbol} {num2} = {result}</p>"


@app.route('/status')
def status():
    """
    Provides the operational status of the application.
    Indicates if the app is running.
    """
    return {"status": "OK", "message": "Application is running!"}

# For Azure App Service with Gunicorn (which it often uses by default for Python)
# main.py is often the entry point. The 'app' variable is the Flask instance.
# The startup command will be configured in Azure App Service.
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))
