document.addEventListener('DOMContentLoaded', () => {
    // --- UTILITY FUNCTIONS ---
    function getUsers() {
        return JSON.parse(localStorage.getItem('users')) || [];
    }

    function saveUsers(users) {
        localStorage.setItem('users', JSON.stringify(users));
    }

    function getLoggedInUser() {
        return sessionStorage.getItem('loggedInUser'); // Stores username
    }

    function setLoggedInUser(username) {
        sessionStorage.setItem('loggedInUser', username);
    }

    function clearLoggedInUser() {
        sessionStorage.removeItem('loggedInUser');
    }

    // --- PAGE SPECIFIC LOGIC ---
    const currentPage = window.location.pathname.split('/').pop();

    // --- AUTHENTICATION PROTECTION & HEADER UPDATE ---
    // For pages that require login
    if (['dashboard_page.html', 'cctv_page.html', 'pengaturan_page.html'].includes(currentPage)) {
        const activeUserUsername = getLoggedInUser();
        if (!activeUserUsername) {
            window.location.href = 'login_page.html';
        } else {
            const users = getUsers();
            const userDetails = users.find(user => user.username === activeUserUsername);
            if (userDetails) {
                const userNameHeader = document.getElementById('loggedInUserNameHeader');
                if (userNameHeader) {
                    userNameHeader.textContent = userDetails.fullName || userDetails.username;
                }
            } else { // Should not happen if session is set correctly
                clearLoggedInUser();
                window.location.href = 'login_page.html';
            }
        }
    }

    // --- SIGNUP PAGE ---
    if (currentPage === 'signup_page.html') {
        const signupForm = document.getElementById('signupForm');
        const signupErrorEl = document.getElementById('signupError');
        const signupSuccessEl = document.getElementById('signupSuccess');

        if (signupForm) {
            signupForm.addEventListener('submit', (e) => {
                e.preventDefault();
                signupErrorEl.style.display = 'none';
                signupSuccessEl.style.display = 'none';

                const fullName = document.getElementById('fullname').value;
                const username = document.getElementById('new-username').value;
                const email = document.getElementById('email').value;
                const password = document.getElementById('new-password').value;
                const confirmPassword = document.getElementById('confirm-password').value;

                if (password !== confirmPassword) {
                    signupErrorEl.textContent = 'Password dan konfirmasi password tidak cocok!';
                    signupErrorEl.style.display = 'block';
                    return;
                }

                let users = getUsers();
                if (users.find(user => user.username === username)) {
                    signupErrorEl.textContent = 'Username sudah digunakan!';
                    signupErrorEl.style.display = 'block';
                    return;
                }
                if (users.find(user => user.email === email)) {
                    signupErrorEl.textContent = 'Email sudah terdaftar!';
                    signupErrorEl.style.display = 'block';
                    return;
                }

                // Di dunia nyata, password harus di-hash sebelum disimpan
                users.push({ fullName, username, email, password });
                saveUsers(users);

                signupSuccessEl.textContent = 'Akun berhasil dibuat! Anda akan diarahkan ke halaman login.';
                signupSuccessEl.style.display = 'block';
                signupForm.reset();

                setTimeout(() => {
                    window.location.href = 'login_page.html';
                }, 2000);
            });
        }
    }

    // --- LOGIN PAGE ---
    if (currentPage === 'login_page.html') {
        const loginForm = document.getElementById('loginForm');
        const loginErrorEl = document.getElementById('loginError');

        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                loginErrorEl.style.display = 'none';

                const usernameOrEmail = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                const users = getUsers();

                const foundUser = users.find(user =>
                    (user.username === usernameOrEmail || user.email === usernameOrEmail) &&
                    user.password === password // Perbandingan password langsung (TIDAK AMAN!)
                );

                if (foundUser) {
                    setLoggedInUser(foundUser.username);
                    window.location.href = 'dashboard_page.html';
                } else {
                    loginErrorEl.textContent = 'Username/Email atau password salah!';
                    loginErrorEl.style.display = 'block';
                }
            });
        }
    }

    // --- PENGATURAN PAGE ---
    if (currentPage === 'pengaturan_page.html') {
        const activeUserUsername = getLoggedInUser();
        if (activeUserUsername) {
            const users = getUsers();
            const userDetails = users.find(user => user.username === activeUserUsername);

            if (userDetails) {
                document.getElementById('accountFullName').textContent = userDetails.fullName;
                document.getElementById('accountUsername').textContent = userDetails.username;
                document.getElementById('accountEmail').textContent = userDetails.email;
            }
        }

        const logoutButton = document.getElementById('logoutButton');
        if (logoutButton) {
            logoutButton.addEventListener('click', () => {
                clearLoggedInUser();
                window.location.href = 'login_page.html';
            });
        }
    }
});