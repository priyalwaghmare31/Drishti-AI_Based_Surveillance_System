// Authentication functions
function showPopup(message) {
    const popup = document.getElementById('errorPopup');
    popup.textContent = message;
    popup.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(function() {
        popup.style.display = 'none';
    }, 5000);
}

function hidePopup() {
    document.getElementById('errorPopup').style.display = 'none';
}

// Login form handler
document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: email,
                    password: password
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    window.location.href = '/';
                } else {
                    showPopup(data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showPopup('Login failed. Please try again.');
            });
        });
    }

    // Signup form handler
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            fetch('/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: name,
                    email: email,
                    password: password
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showPopup('Account created successfully! Please login.');
                    setTimeout(function() {
                        window.location.href = '/login';
                    }, 2000);
                } else {
                    showPopup(data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showPopup('Signup failed. Please try again.');
            });
        });
    }
});
