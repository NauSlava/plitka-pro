from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test App for Web-Eval-Agent</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }
            .button:hover { background: #0056b3; }
            .form-group { margin: 20px 0; }
            input { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Test Application</h1>
            <p>This is a test application for web-eval-agent MCP server.</p>
            
            <div class="form-group">
                <h2>Login Form</h2>
                <input type="email" placeholder="Email" id="email">
                <input type="password" placeholder="Password" id="password">
                <button class="button" onclick="login()">Login</button>
            </div>
            
            <div class="form-group">
                <h2>Navigation</h2>
                <button class="button" onclick="navigateTo('about')">About</button>
                <button class="button" onclick="navigateTo('contact')">Contact</button>
                <button class="button" onclick="navigateTo('settings')">Settings</button>
            </div>
            
            <div class="form-group">
                <h2>Actions</h2>
                <button class="button" onclick="showAlert()">Show Alert</button>
                <button class="button" onclick="changeColor()">Change Color</button>
                <button class="button" onclick="addItem()">Add Item</button>
            </div>
            
            <div id="items"></div>
        </div>
        
        <script>
            function login() {
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                console.log('Login attempt:', { email, password });
                alert('Login functionality would be implemented here');
            }
            
            function navigateTo(page) {
                console.log('Navigating to:', page);
                alert('Navigation to ' + page + ' would be implemented here');
            }
            
            function showAlert() {
                alert('This is a test alert from the web-eval-agent test app!');
            }
            
            function changeColor() {
                document.body.style.backgroundColor = '#' + Math.floor(Math.random()*16777215).toString(16);
            }
            
            function addItem() {
                const itemsDiv = document.getElementById('items');
                const item = document.createElement('div');
                item.innerHTML = '<p>Item ' + (itemsDiv.children.length + 1) + ' added at ' + new Date().toLocaleTimeString() + '</p>';
                itemsDiv.appendChild(item);
            }
        </script>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
