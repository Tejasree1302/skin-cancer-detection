from flask import session, redirect, url_for
from functools import wraps

def login_required(func):
    """
    Decorator to restrict access to routes that require login.
    If the user is not logged in, they will be redirected to the login page.
    """
    @wraps(func)
    def decorator(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return decorator

def set_session(username: str, email: str, remember_me: bool = False) -> None:
    """
    Set session variables for the logged-in user.
    
    Args:
        username (str): Username of the logged-in user
        email (str): Email of the logged-in user
        remember_me (bool): If True, keeps session persistent
    """
    session['username'] = username
    session['email'] = email
    session.permanent = remember_me
